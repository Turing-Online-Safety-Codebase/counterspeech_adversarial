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

""" This script is adapted from run_glue.py from HuggingFace for counter speech classification."""

import transformers
import logging
import os
import random
import re
import sys
import numpy as np
import wandb

from dataclasses import dataclass, field
from typing import Optional, Union#, Protocol
from datasets import load_dataset, load_metric
from src.evaluation.compute_metrics import compute_classification
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
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    
    pred_file: Optional[str] = field(default=None, metadata={"help": "The name of the prediction file"})

    def __post_init__(self):
        train_extension = self.train_file.split(".")[-1]
        assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        validation_extension = self.validation_file.split(".")[-1]
        assert (
            validation_extension == train_extension
        ), "`validation_file` should have the same extension (csv or json) as `train_file`."

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

    # See all possible arguments at https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
    # or by passing the --help flag to this script.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(f"experiments/experiment_logs/{training_args.run_name}.log")
    format = logging.Formatter('%(asctime)s  %(name)s %(levelname)s: %(message)s')
    handler.setFormatter(format)
    logger.addHandler(handler)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the test dataset: CSV/JSON test file
    # Use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' or 'abusive_speech' and 'counter_speech'.
    data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

    if training_args.do_predict:
        if data_args.test_file is not None:
            train_extension = data_args.train_file.split(".")[-1]
            test_extension = data_args.test_file.split(".")[-1]
            assert (
                test_extension == train_extension
            ), "`test_file` should have the same extension (csv or json) as `train_file`."
            data_files["test"] = data_args.test_file
        else:
            raise ValueError("Need a test file for `do_predict`.")

    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    datasets = load_dataset("csv", data_files=data_files)
    logger.info(f"data structure is: {datasets}")

    # Labels
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
    label_list = datasets["train"].unique("label")
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
    non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
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
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]
    if data_args.test_file is not None:
        test_dataset = datasets["test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the train set: {train_dataset[index]}.")
        # logger.info(f"Sample {index} of the validation set: {eval_dataset[index]}.")

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
        logger.info(f"Results: f1 = {f1_scr}, precision = {pre_scr}, recall = {rec_scr}, accuracy = {accuracy}")
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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics

        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("\n***** Train results *****")
                for key, value in sorted(metrics.items()):
                    logger.info(f" {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    if training_args.do_eval:
        logger.info("\n*** Evaluate on validation set ***")

        # eval_datasets = [eval_dataset]
        eval_gold_labels = eval_dataset['label']
        eval_dataset.remove_columns("label")
        eval_output = trainer.predict(test_dataset=eval_dataset)
        eval_predictions, eval_results = eval_output.predictions, eval_output.metrics
        eval_predictions = np.argmax(eval_predictions, axis=1)

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_on_val_set.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(eval_results.items()):
                    key = key.replace("test", "eval")
                    logger.info(f" {key} = {value}")
                    writer.write(f"{key} = {value}\n")

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

        compute_classification(test_gold_labels, predictions.tolist())

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

