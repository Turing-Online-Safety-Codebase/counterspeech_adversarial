# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pydantic import BaseModel
from typing import List, Dict

class ModelSingleInput(BaseModel):
    context: str
    response: str

class ModelSingleOutputProbabilities(BaseModel):
    counter_speech: float
    not_counter_speech: float
    other: float

class ModelSingleOutput(BaseModel):
    label: str
    prob: ModelSingleOutputProbabilities

class ModelBatchInput(BaseModel):
    dataset_samples: List[List]

# added
class ModelBatchOutput(BaseModel):
    dataset_samples: List[Dict]
