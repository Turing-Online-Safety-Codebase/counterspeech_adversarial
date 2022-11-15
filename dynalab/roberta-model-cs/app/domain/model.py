# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


class ModelController:
    def __init__(self) -> None:
        config = AutoConfig.from_pretrained("./app/resources/config.json")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "./app/resources/pytorch_model.bin", config=config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "./app/resources/", local_files_only=True
        )
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def single_evaluation(self, context, response):

        args = (context, response)
        input_data = self.tokenizer(*args, return_tensors="pt")
        input_data = input_data.to(self.device)
        with torch.no_grad():
            inference_output = self.model(**input_data)
            inference_output = F.softmax(inference_output[0]).squeeze()
        response = dict()
        response["label"] = {0: "not_counter_speech", 1: "counter_speech", 2: "other"}[
            int(inference_output.argmax())
        ]
        response["prob"] = {
            "not_counter_speech": float(inference_output[0]),
            "counter_speech": float(inference_output[1]),
            "other": float(inference_output[2]),
        }
        return response


    def batch_evaluation(self, dataset_samples):
        response_list = []
        for example in dataset_samples:
            # print("______________")
            # print(type(dataset_samples))
            # print(type(example))
            # print(example)
            # print("______________")
            args = (example.context, example.response)
            input_data = self.tokenizer(*args, return_tensors="pt")
            input_data = input_data.to(self.device)
            with torch.no_grad():
                inference_output = self.model(**input_data)
                inference_output = F.softmax(inference_output[0]).squeeze()
            response = dict()
            response["label"] = {0: "not_counter_speech", 1: "counter_speech", 2: "other"}[
                int(inference_output.argmax())
            ]
            response["prob"] = {
                "not_counter_speech": float(inference_output[0]),
                "counter_speech": float(inference_output[1]),
                "other": float(inference_output[2]),
            }
            response_list.append(response)
        return response_list
