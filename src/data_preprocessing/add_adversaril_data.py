"""Add adversarial data to existing training data"""

import argparse
import pandas

def parse_args():
    parser = argparse.ArgumentParser(description="Process labelled data for modeling")
    parser.add_argument('--adversarial_data', type=str, default='pseudo_adversarial_iter1.csv', help='name of reviewed file')
    parser.add_argument('--current_batch', type=str, default='iter0', help='name of file for labelling, e.g., iter0')
    pars_args = parser.parse_args()

    print("the inputs are:")
    for arg in vars(pars_args):
        print(f"{arg} is {getattr(pars_args, arg)}")
    return pars_args

def main(df, df_adversaril, current_batch):
    df_1_selected = df[["Rep_ID", "abusive_speech", "counter_speech", "label"]]
    df_new = df_1_selected.append(df_adversaril, ignore_index=True)

    next_batch = int(current_batch[-1])+ 1
    df_new.to_csv(f'data/iter{str(next_batch)}/train_labelled.csv', index=False)

if __name__ == "__main__":
    args = parse_args()

    df_1 = pandas.read_csv(f'data/{args.current_batch}/train_labelled.csv')
    df_adv = pandas.read_csv(f'data/adversarial_data/{args.adversarial_data}')

    main(df_1, df_adv, args.current_batch)