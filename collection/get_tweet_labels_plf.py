import sys, os
import pandas as pd
import numpy as np
import json
import datetime as dt
from io import StringIO

from tqdm import tqdm, tqdm_pandas
#tqdm.pandas()

from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
import torch

from cs_config import *
from cs_tools import *

df = pd.read_csv('data/mps_valid_ac_tweets_idtxt_rc_nosu_anyreplies.csv', converters={'id':object})

print(f"{df.shape[0]} tweets to label")

data = df['text_replaced_b.tolist()']

model = HuggingfaceInferenceModel('../models/footballer_abuse_model', 'temp/', 32)

probs,labels = model(data)

df['probs'] = probs
df['labels'] = labels

df.to_csv('data/mps_valid_ac_tweets_idtxt_rc_nosu_anyreplies_labelled.csv')