"""Evaluate a model prediction file based on metrics: 
    compute F1, precision, recall, accuracy, and confusion matrix"""

import argparse
import json
import pandas
import logging
import evaluate
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

logger = logging.getLogger(__name__)
label = ['Agree with the Post', 'Disagree with the Post', 'Other']

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
    results['tn, fp, fn, tp'] = confusion_matrix(true, pred, normalize='true').ravel().tolist()
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

def get_cls_results_dict(task, model_name, runtime,
                      test_true, test_pred,
                      dev_true, dev_pred,
                      datetime_str):
    """Standardizes classification results dictionary.

    Args:
        task (str): The current task e.g., binary_abuse.
        model_name (str): The model name (if applicable).
        runtime (str): The training runtime of the technique in seconds.
        test_true (np.array): True labels for test set.
        test_pred (np.array): Pred labels for test set.
        dev_true (np.array): True labels for dev set.
        dev_pred (np.array): Pred labels for dev set.
        datetime_str (str): Current datetime.

    Returns:
        dict: Dictionary of results.
    """
    results_dict = {}
    results_dict['task'] = task
    results_dict['model'] = model_name
    results_dict['train_runtime'] = runtime
    results_dict['datetime'] = datetime_str
    results_dict['test_true'] = test_true.tolist()
    results_dict['test_pred'] = test_pred.tolist()
    results_dict['dev_true'] = dev_true.tolist()
    results_dict['dev_pred'] = dev_pred.tolist()
    return results_dict


def save_results(output_dir, datetime_str, results_dict):
    """Saves results dictionary as a json.

    Args:
        output_dir (str): Filepath to store results.
        datetime_str (str): Current datetime for filename.
        results_dict (dict): Dictionary of results
    """
    with open(f'{output_dir}/result_{datetime_str}.json', 'w', encoding="utf-8") as file:
        json.dump(results_dict, file)
