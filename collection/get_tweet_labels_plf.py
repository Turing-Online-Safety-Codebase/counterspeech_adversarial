import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

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

df = pd.read_csv('../data/mps_valid_ac_tweets_idtxt_rc_nosu_anyreplies.csv', dtype={'id':object})

print(f"{df.shape[0]} tweets to label")

data = df['text_replaced_b'].tolist()

print('Loading model', dt.datetime.now())
model = HuggingfaceInferenceModel('../models/footballer_abuse_model', '../temp/', 50)

print('Doing inference', dt.datetime.now())
probs,labels = model(data)

df['probs'] = probs
df['labels'] = labels

print('Saving results', dt.datetime.now())
df.to_csv('../data/mps_valid_ac_tweets_idtxt_rc_nosu_anyreplies_labelled.csv', index=False, encoding='utf-8-sig')