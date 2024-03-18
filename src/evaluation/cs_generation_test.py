#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

""" 
This script is adapted from run_generation.py from HuggingFace for counter speech generation.
Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""


import argparse
import logging

import numpy as np
import torch
import datetime
import pandas
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
    return prompt_text

PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    # added
    parser.add_argument("--test_file", type=str, default="")
    parser.add_argument("--output_raw_file_path", type=str, default="")
    parser.add_argument("--output_clean_file_path", type=str, default="")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument("--temperature", type=float, default=1.0, help="temperature of 1.0 has no effect, lower tend toward greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--seq2seqlm", action="store_true", help="If using sequence-to-sequence LMs. Default to False.")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

    set_seed(args)

    # Initialize the model and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.seq2seqlm:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    model.to(args.device)

    if args.fp16:
        model.half()

    logger.info(args)

    output_raw_file = open(args.output_raw_file_path, "w")
    output_clean_file = open(args.output_clean_file_path, "w")
    with open(args.test_file) as test_f:
        for line in test_f:
            text = ""
            prompt_text = line.replace("\n", "")

            # Different models need different input formatting and/or extra arguments
            requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
            if requires_preprocessing:
                prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
                preprocessed_prompt_text = prepare_input(args, model, tokenizer, prompt_text)

                if model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
                    tokenizer_kwargs = {"add_space_before_punct_symbol": True}
                else:
                    tokenizer_kwargs = {}

                encoded_prompt = tokenizer.encode(
                    preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", max_length=1024, **tokenizer_kwargs
                )
            else:
                prefix = args.prefix if args.prefix else args.padding_text
                encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=True, return_tensors="pt", max_length=1024)
            encoded_prompt = encoded_prompt.to(args.device)

            if encoded_prompt.size()[-1] == 0:
                input_ids = None
            else:
                input_ids = encoded_prompt

            num_attempts = 0
            while "<END>" not in text:
                num_attempts += 1

                output_sequences = model.generate(
                    input_ids=input_ids,
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=args.p,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=True,
                    min_length=10, 
                    # max_length=args.length + len(encoded_prompt[0]),
                    max_length=128,
                    num_return_sequences=args.num_return_sequences,
                )

                # Remove the batch dimension when returning multiple sequences
                if len(output_sequences.shape) > 2:
                    output_sequences.squeeze_()

                generated_sequences = []

                for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                    print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1}, NUM ATTEMPTS: {num_attempts} ===")
                    generated_sequence = generated_sequence.tolist()

                    # Decode text
                    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

                    # Remove all text after the stop token
                    text = text[: text.find(args.stop_token) if args.stop_token else None]

                    output = text.replace(prompt_text, "")
                    clean_output = output.split(' <END>')[0].replace('</s>', '').replace('<pad>', '').strip()

                if "<END>" in output or num_attempts > 3:
                    break

            print("====== New Entry======")
            output_raw_file.write("====== New Entry======" + "\n")
            output_raw_file.write(text + "\n")
            
            output_clean_file.write(clean_output + "\n")

    output_raw_file.close()
    output_clean_file.close()

    return generated_sequences


if __name__ == "__main__":
    main()