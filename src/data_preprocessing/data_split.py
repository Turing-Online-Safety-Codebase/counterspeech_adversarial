#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Splits labelled data into train, val and test sets of predefined size.
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_args():
    """Parses Command Line Args"""
    parser = argparse.ArgumentParser(description="Process labelled data for modeling")
    parser.add_argument('--input_data_path', type=str, default='', help='Path to data for splitting')
    parser.add_argument('--output_data_path', type=str, default='', help='Path to data for splitting')
    parser.add_argument('--data_split_size', type=str, default="0.7_0.334", help='Path to data for splitting')
    args = parser.parse_args()

    print("the inputs are:")
    for arg in vars(args):
        print(f"{arg} is {getattr(args, arg)}")
    return args


def main(data_split_size, input_data_path, output_data_path):
    """
    Runs process for train_test_split
    Returns:
        train, val, test csvs with columns ['1M_id', 'orig_id', 'text', 'label']
    """

    file_path, filename = os.path.split(input_data_path)  # get file path and filename
    df = pd.read_csv(file_path)

    train_size, val_size = data_split_size.split("_")
    train, tmp = train_test_split(df, train_size=float(train_size), random_state=43)
    val, test = train_test_split(tmp, train_size=float(val_size), random_state=43)
    train.to_csv(f'{output_data_path}/train_labelled.csv')
    val.to_csv(f'{output_data_path}/val_labelled.csv')
    test.to_csv(f'{output_data_path}/test_labelled.csv')

if __name__ == '__main__':
    args = parse_args()

    main(args.data_split_size, args.input_data_path, args.output_data_path)