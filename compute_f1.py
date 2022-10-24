"""compute F1, precision, recall of a model prediction file"""

import pandas
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import argparse
import pandas

def parse_args():
    parser = argparse.ArgumentParser(description="Process labelled data for modeling")
    parser.add_argument('--gold_data_path', type=str, default='mps_hatemoji_plf_prsp_500.csv', help='data path of model prediction')
    parser.add_argument('--pred_data_path', type=str, default='', help='data path of gold file')
    pars_args = parser.parse_args()

    print("the inputs are:")
    for arg in vars(pars_args):
        print(f"{arg} is {getattr(pars_args, arg)}")
    return pars_args

def main(gold_data_path, pred_data_path):
    df_gold = pandas.read_csv(gold_data_path)
    df_pred = pandas.read_csv(pred_data_path)
    y_true = df_gold['label']
    y_pred = df_pred['label']

    f1_pred = f1_score(y_true, y_pred, average='macro')    
    print(f'F1 score for prediction is {f1_pred}')

    recall_pred = recall_score(y_true, y_pred, average='macro')
    print(f'recall score for hatemoji is {recall_pred}')
    
    precision_pred = precision_score(y_true, y_pred, average='macro')
    print(f'precision score for hatemoji is {precision_pred}')
    
if __name__ == "__main__":
    args = parse_args()
    main(args.gold_data_path, args.pred_data_path)