import sys, os
import pandas as pd
import numpy as np
import json
import datetime as dt
from io import StringIO

from tqdm import tqdm, tqdm_pandas
#tqdm.pandas()

import azure.cosmos.cosmos_client as azurecosmos
import azure.storage.blob as azureblob

from cs_config import *
from cs_tools import *

cosmos = get_cosmos_client(Cosmos.host, Cosmos.key, Cosmos.mps_db, Cosmos.mps_container)

df = pd.concat([
    pd.read_csv('mps_valid_ac_tweets_idtxt_0.csv'),
    pd.read_csv('mps_valid_ac_tweets_idtxt_1.csv'),
    pd.read_csv('mps_valid_ac_tweets_idtxt_2.csv'),
    pd.read_csv('mps_valid_ac_tweets_idtxt_3.csv')
])

print(f"{df.shape[0]} tweets to get reply counts for")

#df['replycount'] = df['id'].progress_apply(lambda x: get_tweet_reply_count(cosmos, x))

for i,id in tqdm.tqdm(enumerate(df['id']), total=df.shape[0]):
    try: df.loc[df['id']==id, 'replycount'] = get_tweet_reply_count(cosmos, id)
    except Exception as e: print(e)
    if i%5000==0: df.to_csv('mps_valid_ac_tweets_idtxt_rc.csv', index=False, encoding='utf-8-sig')

df.to_csv('mps_valid_ac_tweets_idtxt_rc.csv', index=False, encoding='utf-8-sig')