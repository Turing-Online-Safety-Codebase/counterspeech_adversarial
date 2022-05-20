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

from tqdm import tqdm, tqdm_pandas
#tqdm.pandas()

import azure.cosmos.cosmos_client as azurecosmos
import azure.storage.blob as azureblob

from cs_config import *
from cs_tools import *

PATH = "../data/mp_replies/"

cosmos = get_cosmos_client(Cosmos.host, Cosmos.key, Cosmos.mps_db, Cosmos.mps_container)

df = pd.read_csv('../data/mps_valid_ac_tweets_idtxt_rc_nosu_anyreplies_labelled.csv', dtype={'id':object})
df = df[df.labels==1].reset_index(drop=True)

print(f"{df.shape[0]} tweets to get replies for")

for i, id in tqdm.tqdm(enumerate(df['id'].tolist()), total=df.shape[0]):
    tweet_dir = PATH+str(id)
    os.mkdir(tweet_dir)

    root_tweet = query_cosmos(cosmos, filter=f"c.id=\"{id}\"")[0]
    with open(f"{tweet_dir}/root.json", "w") as f: json.dump(root_tweet, f, indent=4)

    replies = get_tweet_replies(cosmos, id)
    for j, reply in enumerate(replies):
        with open (f"{tweet_dir}/reply_{reply['id']}.json", "w") as f: json.dump(reply, f, indent=4)
