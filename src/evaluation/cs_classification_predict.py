#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

""" This script is adapted from run_glue.py by HuggingFace for model inference."""

import transformers
import logging
import os
import random
import time
import datetime
import re
import sys
import numpy as np
import wandb

from dataclasses import dataclass, field
from typing import Optional, Union#, Protocol
from datasets import load_dataset, load_metric
from ..evaluation import compute_results
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

TASK = 'cs_classification'
# wandb.init()
wandb.login()
logger = logging.getLogger(__name__)

def is_main_process(local_rank):
    """
    Whether or not the current process is the local process, based on `xm.get_ordinal()` (for TPUs) first, then on
    `local_rank`.
    """
    return local_rank in [-1, 0]

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    
    pred_file: Optional[str] = field(default=None, metadata={"help": "The name of the prediction file"})

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def main():

    datetime_str = str(datetime.datetime.now())
    run_time = 0

    # See all possible arguments at https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
    # or by passing the --help flag to this script.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(f"experiments/experiment_logs/{training_args.run_name}/{datetime_str}.log")
    format = logging.Formatter('%(asctime)s  %(name)s %(levelname)s: %(message)s')
    handler.setFormatter(format)
    logger.addHandler(handler)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    data_files = {"test": data_args.test_file}
    datasets = load_dataset("csv", data_files=data_files)
    logger.info(f"data structure is: {datasets}")

    # Labels
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
    label_list = datasets["test"].unique("label")
    # print(label_list, type("label_list"))
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Preprocessing the datasets, set the column names for model inputs
    non_label_column_names = [name for name in datasets["test"].column_names if name != "label"]
    if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    elif 'hateSpeech' in non_label_column_names and 'counterSpeech' in non_label_column_names:
        sentence1_key, sentence2_key = "hateSpeech", "counterSpeech"
    else:
        sentence1_key, sentence2_key = "abusive_speech", "counter_speech"

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    else:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[l] for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)
    test_dataset = datasets["test"]

    # define metric (use accuracy and f1)
    # Custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        metric_f1 = load_metric('f1') #load_metric('glue', "mrpc")
        metric_pre = load_metric('precision')
        metric_rec = load_metric('recall')

        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        f1_scr = metric_f1.compute(predictions=preds, references=p.label_ids, average="weighted")["f1"]
        pre_scr = metric_pre.compute(predictions=preds, references=p.label_ids, average="weighted")["precision"]
        rec_scr = metric_rec.compute(predictions=preds, references=p.label_ids, average="weighted")["recall"]
        accuracy = (preds == p.label_ids).astype(np.float32).mean().item()
        logger.info(f"Results: f1 = {f1_scr}, precision = {pre_scr}, recall = {rec_scr}")
        return {"f1": f1_scr, "precision": pre_scr, "recall": rec_scr, "accuracy": accuracy}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset= None,
        eval_dataset= None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Predict
    if training_args.do_predict:
        logger.info("\n*** Prediction on test set ***")

        pred_file = data_args.pred_file if data_args.pred_file is not None else "test_results_on_test_set.csv"

        test_gold_labels = test_dataset['label']
        test_dataset.remove_columns("label")
        test_output = trainer.predict(test_dataset=test_dataset)
        test_predictions, test_results = test_output.predictions, test_output.metrics
        test_predictions = np.argmax(test_predictions, axis=1)
        logger.info(f"{test_results}")

        eval_gold_labels=[], eval_predictions=np.array([])
        test_results = compute_results.compute_classification(test_gold_labels, test_predictions.tolist())
        results_dict = compute_results.get_cls_results_dict(TASK, model_args.model_name_or_path, run_time,
                    test_gold_labels, test_predictions,
                    eval_gold_labels, eval_predictions,
                    datetime_str)
        compute_results.save_results(f"experiments/experiment_logs/{training_args.run_name}", datetime_str, results_dict)

        output_test_file = os.path.join(training_args.output_dir, pred_file)
        if trainer.is_world_process_zero():
            with open(output_test_file, "w") as writer:
                writer.write("index \t abusive_text \t counter_speech \t label_id \t label \t prediction \n")
                for index, item in enumerate(test_predictions):
                    item = label_list[item]
                    writer.write(f"{index} \t {test_dataset[sentence1_key][index]} \t {test_dataset[sentence2_key][index]} \t {label_list[test_gold_labels[index]]} \t {test_gold_labels[index]} \t {item} \n")
                logger.info("***** Prediction finished *****")


if __name__ == "__main__":
    main()

