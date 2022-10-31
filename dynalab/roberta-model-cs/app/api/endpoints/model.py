# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fastapi import APIRouter

from app.api.schemas.model import ModelSingleInput, ModelSingleOutput, ModelBatchInput, ModelBatchOutput
from app.domain.model import ModelController


router = APIRouter()


@router.get("/")
async def hello():
    model = ModelController()
    text = model.single_evaluation("I hate lambda")
    return {"message": text}


# @router.post("/single_evaluation")
# async def single_evaluation(data: ModelSingleInput):
#     model = ModelController()
#     text = model.single_evaluation(data.context, data.hypothesis)
#     return {"message": text}

@router.post("/single_evaluation", response_model=ModelSingleOutput)
async def single_evaluation(data: ModelSingleInput):
    model = ModelController()
    asnwer = model.single_evaluation(data.context, data.response)
    return asnwer

# added
@router.post("/batch_evaluation", response_model=ModelBatchOutput)
async def batch_evaluation(data: ModelBatchInput):
    model = ModelController()
    asnwer = model.batch_evaluation(data.dataset_samples)
    return asnwer