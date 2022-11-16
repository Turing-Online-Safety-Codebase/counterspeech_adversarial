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
    parser.add_argument('--output_dataset_path', type=str, default='data/dynabench_data/indomain_test_dataset_temp.jsonl', help='Path to output test data')
    parser.add_argument('--contexts_path', type=str, default='data/twitter_plf_data/twitter_plf_labelled/final_data/train_labelled.csv', help='Path to contexts data for adversarial data collection')
    parser.add_argument('--output_contexts_path', type=str, default='data/dynabench_data/round1_contexts_temp.jsonl', help='Path to output contexts data')
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
            data["tag"] = 'round1'
            data["metadata"] = {}
            json.dump(data, f)
            f.write("\n")

def convert_to_str(input_id):
    return str(input_id)

def convert_to_emojis(text):
    return text.encode('utf-8', 'replace').decode() #text.encode('latin-1').decode("utf-8")

def create_datasets(input_df, output):
    """
    output format should look like this:    
    {"uid": "1", "context": "Hello world", "response": "How are you?", "label": "1"}
    {"uid": "2", "context": "Foo bar", "response": "hmmm....", "label": "2"}
    {"uid": "3", "context": "Foo bar", "response": "Some bad responses", "label": "0"}
    """
    input_df['label'] = input_df.apply(lambda x: label_to_id(x['label']), axis=1)
    input_df['Rep_ID'] = input_df.apply(lambda x: convert_to_str(x['Rep_ID']), axis=1)
    # input_df['abusive_speech'] = input_df.apply(lambda x: convert_to_emojis(x['abusive_speech']), axis=1)
    # input_df['counter_speech'] = input_df.apply(lambda x: convert_to_emojis(x['counter_speech']), axis=1)
    input_df.rename(columns = {'Rep_ID':'uid', 'abusive_speech':'context', 'counter_speech':'response'}, inplace = True)
    df_new = input_df[['uid', 'context', 'response', 'label']]
    df_new.to_json(output, orient='records', lines=True)

def convert_jsonl_to_emoji(input_js, output_js):
    df = pd.read_json(input_js, lines=True)
    df.to_json(output_js, orient='records', lines=True, force_ascii=False)


if __name__ == '__main__':
    args = parse_args()

    df_data = pd.read_csv(args.dataset_path, encoding='utf8')
    df_data = df_data.astype({'Rep_ID': int})
    create_datasets(df_data, args.output_dataset_path)

    df_context = pd.read_csv(args.contexts_path, encoding='utf8')
    df_context = df_context.astype({'Rep_ID': int})
    create_contexts(df_context, args.output_contexts_path)

    convert_jsonl_to_emoji(args.output_dataset_path, 'data/dynabench_data/indomain_test_dataset_final.jsonl')
    convert_jsonl_to_emoji(args.output_contexts_path, 'data/dynabench_data/round1_contexts_final.jsonl')
