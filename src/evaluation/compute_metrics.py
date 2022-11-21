"""Evaluate a model prediction file based on metrics: 
    compute F1, precision, recall, accuracy, and confusion matrix"""

import argparse
import pandas
import logging
import evaluate
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

def compute_classification(true, pred, target_names=["agree_with_the_posts", "disagree_with_the_posts", "other"]):
    """Computes the number of votes received for each class labelled for an entry.
        Three classes are included: disagree, agree, other.
    Args:
        true (list): ground truth (correct) target values
        pred (list): model prediction
        target_names (list): display names matching the labels (same order)
    Returns:
        a dictionary of the evaluation results including accuracy, f1, precision, recall,
        and confusion metric
    """
    results = {}
    results['accuracy'] = accuracy_score(true, pred)
    results['precision'] = precision_score(true, pred, average='weighted', zero_division=0)
    results['recall'] = recall_score(true, pred, average='weighted', zero_division=0)
    results['f1'] = f1_score(true, pred, average='weighted', zero_division=0)
    print(f'--confusion metric-- \n {confusion_matrix(true, pred)}')
    results['tn, fp, fn, tp'] = confusion_matrix(true, pred, normalize='true').ravel()
    print("\n--full report--")
    print(classification_report(true, pred, output_dict=False, target_names=target_names))
    return results

def plot_confusion_matrix(true_labels, prediction, title='confusion metric', normalise='true'):
    """Plot confusion matrix give true labels and prediction
        Three classes are included: disagree, agree, other.
    Args:
        true_labels (list): ground truth (correct) target values
        pred (list): model prediction
        target_names (list): display names matching the labels (same order)
    """
    ax= plt.subplot()
    matrix = confusion_matrix(true_labels, prediction, labels=label)    
    group_counts = [str(value) for value in matrix.flatten()]        
    
    matrix_p = confusion_matrix(true_labels, prediction, labels=label, normalize=normalise)
    group_percentages = [round(value, 2) for value in matrix_p.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_percentages, group_counts)]
    labels = np.asarray(labels).reshape(3,3)    
    sns.heatmap(matrix_p, annot=labels, xticklabels=label, yticklabels=label, fmt='', linewidths=.5, cmap='YlGnBu')
    
    # labels, title and ticks
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Gold label')
    ax.set_title(title)

def evalute_classification(df_true, df_pred, output_path):
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(f"{output_path}.log")
    log_format = logging.Formatter('%(asctime)s  %(name)s %(levelname)s: %(message)s')
    handler.setFormatter(log_format)
    logger.addHandler(handler)

    y_true = df_true['label']
    y_pred = df_pred['label']

    return compute_classification(y_true, y_pred, target_names=["agree_with_the_posts", "disagree_with_the_posts", "other"])

def evalute_generation(predictions, references):
    """Computes evaluation for generated text.
    Args:
        predictions (list): ground truth (correct) target values
        references (list): model prediction
    Returns:
        a dictionary of the evaluation results including bleu, rouge, mauve, and bleurt
    """
    # TODO: add vocav_size, check the type of prediction and references (for rouge and bleu)
    # predictions = ["hello there general kenobi","foo bar foobar"]
    # references = [["hello there general kenobi"], ["foo bar foobar"],] or multiple references = [[["hello there general kenobi"], ["hello there!"]], [["foo bar foobar"]]]
    
    results = {}
    bleu = evaluate.load("bleu")
    rouge = evaluate.load('rouge')
    mauve = evaluate.load('mauve')
    bleurt = evaluate.load("bleurt", module_type="metric", checkpoint="bleurt-base-128")

    results['bleu_results'] = bleu.compute(predictions=predictions, references=references)
    results['rouge_results'] = rouge.compute(predictions=predictions, references=references)
    results['bleurt_results'] = bleurt.compute(predictions=predictions, references=references)
    results['mauve_results'] = mauve.compute(predictions=predictions, references=references)
    return results

if __name__ == "__main__":
    args = parse_args()
    df_gold = pandas.read_csv(args.gold_data_path)
    df_pred = pandas.read_csv(args.pred_data_path)
    evalute_classification(df_gold, df_pred, args.output_path)