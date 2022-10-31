# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, pipeline

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

    def single_evaluation(self, context, hypothesis):
        args = (context, hypothesis)
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
    
    def batch_evaluation(self, dataset_samples: List(List)):
        # dataset_samples should be like this [[context, hypothesis], [context, hypothesis]]
        predict = self.tokenizer(dataset_samples, return_tensors="pt")    
        logits = self.model(**predict).logits
        scores = F.softmax(logits, dim=1)

        responses = []
        for pred in scores:
            result = dict()
            result["label"] = {0: "not_counter_speech", 1: "counter_speech", 2: "other"}[int(pred.argmax())]
            result["prob"] = {
                "not_counter_speech": float(pred[0]),
                "counter_speech": float(pred[1]),
                "other": float(pred[2]),
            }
            responses.append(result)
        return responses

    # simple version
    # def batch_evaluation(self, dataset_samples: list):
    #     final_predictions = []
    #     for dataset_sample in dataset_samples:
    #         final_predictions.append(self.single_evaluation(dataset_sample))
    #     return final_predictions