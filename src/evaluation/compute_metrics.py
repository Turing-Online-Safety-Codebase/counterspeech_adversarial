"""Evaluate a model prediction file based on metrics: 
    compute F1, precision, recall, accuracy, and confusion matrix"""

import argparse
import pandas
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

logger = logging.getLogger(__name__)
label = ['Agree with the Post', 'Disagree with the Post', 'Other']

def parse_args():
    parser = argparse.ArgumentParser(description="Model evaluation")
    parser.add_argument('--gold_data_path', type=str, default='mps_hatemoji_plf_prsp_500.csv', help='data path of model prediction')
    parser.add_argument('--pred_data_path', type=str, default='', help='data path of gold file')
    parser.add_argument('--output_path', type=str, required=True, default='', help='data path of output file')
    pars_args = parser.parse_args()

    print("the inputs are:")
    for arg in vars(pars_args):
        print(f"{arg} is {getattr(pars_args, arg)}")
    return pars_args

def plot_confusion_matrix(truth, prediction, title='confusion metric', normalise='true'):
    ax= plt.subplot()
    matrix = confusion_matrix(truth, prediction, labels=label)    
    group_counts = [str(value) for value in matrix.flatten()]        
    
    matrix_p = confusion_matrix(truth, prediction, labels=label, normalize=normalise)
    group_percentages = [round(value, 2) for value in matrix_p.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_percentages, group_counts)]
    labels = np.asarray(labels).reshape(3,3)    
    sns.heatmap(matrix_p, annot=labels, xticklabels=label, yticklabels=label, fmt='', linewidths=.5, cmap='YlGnBu')
    
    # labels, title and ticks
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Gold label')
    ax.set_title(title)

def main(df_gold, df_pred, output_path):
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(f"{output_path}.log")
    format = logging.Formatter('%(asctime)s  %(name)s %(levelname)s: %(message)s')
    handler.setFormatter(format)
    logger.addHandler(handler)

    y_true = df_gold['label']
    y_pred = df_pred['label']

    results = {}
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['f1'] = f1_score(y_true, y_pred, average='macro', zero_division = 0)
    results['precision'] = precision_score(y_true, y_pred, average = 'macro', zero_division = 0)
    results['recall'] = recall_score( y_true, y_pred, average = 'macro', zero_division = 0)
    results['cm'] = confusion_matrix(y_true, y_pred, labels=label, normalize='true').ravel()    #tn, fp, fn, tp

    for key in results.keys():
        logger.info(f"Result for {key}: {results[key]}")
    plot_confusion_matrix(y_true, y_pred)
    return results

if __name__ == "__main__":
    args = parse_args()
    df_gold = pandas.read_csv(args.gold_data_path)
    df_pred = pandas.read_csv(args.pred_data_path)
    main(df_gold, df_pred, args.output_path)