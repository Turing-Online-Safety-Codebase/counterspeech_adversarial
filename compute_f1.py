"""compute F1, precision, recall of 3 abuse classifier"""

import pandas
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


df = pandas.read_excel("counterspeech_mps/mps_hatemoji_plf_prsp_500.xlsx")
y_true = df['Is the tweet abusive?']
y_hatemoji = df['hatemoji_label']
y_plf = df['plf_label']
y_prsp = df['prsp_label']
hatemoji = f1_score(y_true, y_hatemoji, average='macro')
plf = f1_score(y_true, y_plf, average='macro')
prsp = f1_score(y_true, y_prsp, average='macro')

print(f'F1 score for hatemoji is {hatemoji}')
print(f'F1 score for OHO footballer is {plf}')
print(f'F1 score for perspective API is {prsp}')

recall_hatemoji = recall_score(y_true, y_hatemoji, average='macro')
recall_plf = recall_score(y_true, y_plf, average='macro')
recall_prsp = recall_score(y_true, y_prsp, average='macro')

print(f'recall score for hatemoji is {recall_hatemoji}')
print(f'recall score for OHO footballer is {recall_plf}')
print(f'recall score for perspective API is {recall_prsp}')

precision_hatemoji = precision_score(y_true, y_hatemoji, average='macro')
precision_plf = precision_score(y_true, y_plf, average='macro')
precision_prsp = precision_score(y_true, y_prsp, average='macro')

print(f'precision score for hatemoji is {precision_hatemoji}')
print(f'precision score for OHO footballer is {precision_plf}')
print(f'precision score for perspective API is {precision_prsp}')
