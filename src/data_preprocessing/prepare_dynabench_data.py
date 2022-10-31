#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Prepare contexts and datasets in jsonl format for Dynabench.
"""

import json
import argparse
import pandas as pd

label_dic = {"agrees_with_the_post": 0, "disagrees_with_the_post": 1, "other": 2}

def parse_args():
    """Parses Command Line Args"""
    parser = argparse.ArgumentParser(description="Process labelled data for modeling")
    parser.add_argument('--dataset_path', type=str, default='data/twitter_plf_data/twitter_plf_labelled/final_data/test_labelled.csv', help='Path to test data for evaluation on Dynabench')
    parser.add_argument('--output_dataset_path', type=str, default='data/dynabench_data/indomain_test_dataset.jsonl', help='Path to output test data')
    parser.add_argument('--contexts_path', type=str, default='data/twitter_plf_data/twitter_plf_labelled/final_data/train_labelled.csv', help='Path to contexts data for adversarial data collection')
    parser.add_argument('--output_contexts_path', type=str, default='data/dynabench_data/round1_contexts.jsonl', help='Path to output contexts data')
    args = parser.parse_args()

    print("the inputs are:")
    for arg in vars(args):
        print(f"{arg} is {getattr(args, arg)}")
    return args

def label_to_id(label):
    return label_dic[label]

def create_contexts(input_df, output):
    """
    output format should look like this:    
    {"context": {"context": "Some abusive example1"}, "tag": null, "metadata": {}}
    {"context": {"context": "Some abusive example2"}, "tag": null, "metadata": {}}
    """
    with open(output, "a") as f:
        for index, row in input_df.iterrows():
            data = {}
            data["context"] = {}
            data["context"]["context"] = row['abusive_speech']
            data["context"]["tag"] = 'round1'
            data["context"]["metadata"] = {}
            json.dump(data, f)
            f.write("\n")

def create_datasets(input_df, output):
    """
    output format should look like this:    
    {"uid": "1", "context": "Hello world", "response": "How are you?", "label": "1"}
    {"uid": "2", "context": "Foo bar", "response": "hmmm....", "label": "2"}
    {"uid": "3", "context": "Foo bar", "response": "Some bad responses", "label": "0"}
    """
    input_df['label'] = input_df.apply(lambda x: label_to_id(x['label']), axis=1)
    input_df.rename(columns = {'Rep_ID':'uid', 'abusive_speech':'context', 'counter_speech':'response'}, inplace = True)
    df_new = input_df[['uid', 'context', 'response', 'label']]
    df_new.to_json(output, orient='records', lines=True)

if __name__ == '__main__':
    args = parse_args()

    df_data = pd.read_csv(args.dataset_path)
    df_data = df_data.astype({'Rep_ID': int})
    create_datasets(df_data, args.output_dataset_path)

    df_context = pd.read_csv(args.contexts_path)
    df_context = df_context.astype({'Rep_ID': int})
    create_contexts(df_context, args.output_contexts_path)
