#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Runs Prompt engineering experiments. (in progress)
"""

import argparse
import torch
import pandas
import time
import logging
import numpy
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptForClassification, PromptDataLoader
from transformers import AdamW
from evaluation.compute_metrics import compute_classification
from utils import convert_labels, load_balanced_n_samples

TASK = 'binary_abuse'
TECH = 'promting'

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Prompt learning")
    parser.add_argument('--n_examples', type=str, default='15,20,32,50,70,100,150,200,320,500,750', help='num of training points tested (seperated by ",", e.g., "15,20")')
    parser.add_argument('--run_name', type=str, default='', required=True, help='name of the experiment')
    parser.add_argument('--model_name', type=str, default='bert', help='name of the model')
    parser.add_argument('--model_path', type=str, default='bert-base-cased', help='path to the model')
    parser.add_argument('--use_cuda', type=bool, default=True, help='if using cuda')
    parser.add_argument('--eval_steps', type=int, default='4', help='num of update steps between two evaluations')
    pars_args = parser.parse_args()

    print("the inputs are:")
    for arg in vars(pars_args):
        print(f"{arg} is {getattr(pars_args, arg)}")
    return pars_args

def main(run_name, template, model_name, model_path, use_cuda, n_examples):

    # Setup logging
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(f"experiments/experiment_logs/prompt_engineering/{run_name}.log")
    # format = logging.Formatter('%(asctime)s  %(name)s %(levelname)s: %(message)s')
    # handler.setFormatter(format)
    logger.addHandler(handler)

    # Measure training run time, run_time will be 0 if n = 0 (i.e. no training)
    run_time = 0

    # Load a Pre-trained Language Model (PLM).
    plm, tokenizer, model_config, WrapperClass = load_plm(model_name, model_path)
    
    # Define a Template.
    promptTemplate = ManualTemplate(
        text=template,
        tokenizer=tokenizer,
    )

    # Define a Verbalizer
    promptVerbalizer = ManualVerbalizer(
        classes= [0, 1, 2],    # Three classes: 0 for not agreement, 1 for disagreement, and 2 for others
        label_words={
            0: ["agreement"],
            1: ["disagreement"],
            2: ['other'],
        },
        tokenizer=tokenizer,
    )

    # Combine them into a PromptModel
    promptModel = PromptForClassification(
        template=promptTemplate,
        plm=plm,
        verbalizer=promptVerbalizer,
    )
    if use_cuda:
        promptModel = promptModel.cuda()

    # Set up an optimizer
    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in promptModel.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in promptModel.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)

    def inference(model, dataset):
        model.eval()
        preds = numpy.array([])
        gold_labels = numpy.array([])
        with torch.no_grad():
            for step, inputs in enumerate(dataset):
                if use_cuda:
                    inputs = inputs.cuda()
                logits = model(inputs)
                labels = inputs['label']
                gold_labels = numpy.concatenate((gold_labels, labels.cpu()), axis=0)
                preds = numpy.concatenate((preds, torch.argmax(logits, dim=-1).cpu()), axis=0)

        # Compute scores
        results = compute_classification(gold_labels.tolist(), preds.tolist())
        for key, value in sorted(metrics.items()):
            logger.info(f" {key} = {value}")
        return gold_labels, preds, results

    # Prepare dataset
    raw_dataset = {}
    raw_dataset['train'] = pandas.read_csv("data/final_modeling_data/train_labelled.csv")
    raw_dataset['val'] = pandas.read_csv("data/final_modeling_data/val_labelled.csv")
    raw_dataset['test'] = pandas.read_csv("data/final_modeling_data/test_labelled.csv")
    raw_dataset['train'], n_train_examples = convert_labels(raw_dataset['train'])
    raw_dataset['val'], n_dev_examples = convert_labels(raw_dataset['val'])
    raw_dataset['test'], n_test_examples = convert_labels(raw_dataset['test'])
    # raw_dataset['train'] = load_balanced_n_samples('data', 'binary_abuse', 'train', int(n_examples))
    logger.info(f"--label distribution for train set--\n{raw_dataset['train']['label'].value_counts()}")
    logger.info(f"Num of examples for train/dev/test: {n_train_examples}, {n_dev_examples}, {n_test_examples}")

    dataset = {}
    for split in ['train', 'val', 'test']:
        dataset[split] = []
        for index, row in raw_dataset[split].iterrows():
            input_example = InputExample(text_a=row['abusive_speech'], text_b=row['counter_speech'], label=row['label'])
            dataset[split].append(input_example)

    train_dataloader = PromptDataLoader(
        dataset=dataset['train'],
        template=promptTemplate,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=256,
        batch_size=4,
        shuffle=False,
        teacher_forcing=False,
        predict_eos_token=False,
        truncate_method="tail")

    test_dataloader = PromptDataLoader(
        dataset=dataset["test"],
        template=promptTemplate,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=256,
        batch_size=4,
        shuffle=False,
        teacher_forcing=False,
        predict_eos_token=False,
        truncate_method="tail")

    validation_dataloader = PromptDataLoader(
        dataset=dataset["val"],
        template=promptTemplate,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=256,
        batch_size=4,
        shuffle=False,
        teacher_forcing=False,
        predict_eos_token=False,
        truncate_method="tail")

    start = time.time()
    for epoch in range(1, 4):
        promptModel.train()
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            optimizer.zero_grad()
            labels = inputs['label']
            logits = promptModel(inputs)
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            logger.info(f"Epoch {epoch}, step {step}, average loss: {tot_loss/(step+1)}")
        logger.info("--Perform validation--")
        val_gold_labels, val_preds, val_result = inference(promptModel, validation_dataloader)
        promptModel.train()
    end = time.time()
    run_time = end - start

    # Inference and evalute
    logger.info("--Perform testing--")
    test_gold_labels, test_preds, test_result = inference(promptModel, test_dataloader)


if __name__ == '__main__':
    args = parse_args()

    num_examples = args.n_examples.split(',')
    templates = 'Response: {"placeholder":"text_b"} | {"mask"}, {"placeholder":"text_a"}'

    main(args.run_name, templates, args.model_name, args.model_path, args.use_cuda, n_examples=5565)