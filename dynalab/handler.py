# Copyright (c) Facebook, Inc. and its affiliates.

"""
Instructions:
Please work through this file to construct your handler. Here are things
to watch out for:
- TODO blocks: you need to fill or modify these according to the instructions.
   The code in these blocks are for demo purpose only and they may not work.
- NOTE inline comments: remember to follow these instructions to pass the test.
For expected task I/O, please check dynalab/tasks/README.md
"""

import json
import os
import sys
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from zmq import device

from dynalab.handler.base_handler import BaseDynaHandler
from dynalab.tasks.task_io import TaskIO, ROOTPATH


# NOTE: use the following line to import modules from your repo
sys.path.append(ROOTPATH)
label_dict = {0: 'agrees_with_the_post', 1: 'disagrees_with_the_post', 2: 'other'}

class Handler(BaseDynaHandler):
    def initialize(self, context):
        """
        load model and extra files
        """
        model_pt_path, model_file_dir, device_str = self._handler_initialize(context)
        print(model_pt_path, model_file_dir, device_str)
        self.taskIO = TaskIO("cs")

        # ############TODO 1: Initialize model ############
        """
        Load model and read relevant files here.
        """
        config = AutoConfig.from_pretrained(model_pt_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_pt_path, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_pt_path)

        # with open(os.path.join(model_file_dir, "config")) as f:
        #     config = json.load(f)
        # self.model = MyModel(config)
        # self.model.load_state_dict(torch.load(model_pt_path, map_location=device_str))
        # self.model.to(torch.device(device_str))
        self.model.eval()
        # #################################################

        self.initialized = True

    def preprocess(self, data):
        """
        preprocess data into a format that the model can do inference on
        """
        example = self._read_data(data)

        # ############TODO 2: preprocess data #############
        """
        You can extract the key and values from the input data like below
        example is a always json object. Yo can see what an example looks
        in a Python interpreter by
        ```
        >>> from dynalab.tasks.task_io import TaskIO
        >>> task_io = TaskIO("{your_task}")
        >>> task_io.mock_datapoints[0]
        ```
        """
        context = example["context"]
        response = example["response"]
        # input_data = len(context) + len(response)
        input_encoding = self.tokenizer(context, response, return_tensors="pt")
        # #################################################

        # input_encoding = self.tokenizer.encode_plus(
        #     question, context, max_length=512, return_tensors="pt"
        # )
        # input_ids = input_encoding["input_ids"].tolist()[0]
        # return (input_encoding, input_ids)

        return input_encoding

    def inference(self, input_data):
        """
        do inference on the processed example
        """

        # ############TODO 3: inference ###################
        """
        Run model prediction using the processed data
        """
        with torch.no_grad():
            # inference_output = self.model(input_data)
            logits = self.model(**input_data).logits
            scores = torch.softmax(logits, dim=1)
            pred_prob_list = scores.tolist()[0]
            pred_indice = scores.argmax().item() # indices of the predicted label
            pred_label, conf = label_dict[pred_indice], pred_prob_list[pred_indice]
        # #################################################

        return (pred_label, conf)

    def postprocess(self, inference_output, data):
        """
        post process inference output into a response.
        response should be a single element list of a json
        the response format will need to pass the validation in
        ```
        dynalab.tasks.TaskIO("{your_task}").verify_response(response, data)
        ```
        """
        response = dict()
        answer, conf = inference_output
        example = self._read_data(data)
        response["id"] = example["uid"]
        # ############TODO 4: postprocess response ########
        """
        Add attributes to response
        """
        response["answer"] = answer if answer != '[CLS]' else ''
        response["conf"] = conf
        # #################################################
        self.taskIO.sign_response(response, example)
        return [response]


_service = Handler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)
    if data is None:
        return None

    # ############TODO 5: assemble inference pipeline #####
    """
    Normally you don't need to change anything in this block.
    However, if you do need to change this part (e.g. function name, argument, etc.),
    remember to make corresponding changes in the Handler class definition.
    """
    input_data = _service.preprocess(data)
    output = _service.inference(input_data)
    response = _service.postprocess(output, data)
    # #####################################################

    return response