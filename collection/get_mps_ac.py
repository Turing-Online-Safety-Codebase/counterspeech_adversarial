import sys, os
import pandas as pd
import numpy as np
import json
import tqdm
import datetime as dt
from io import StringIO

import azure.cosmos.cosmos_client as azurecosmos
import azure.storage.blob as azureblob

from cs_config import *
from cs_tools import *

cosmos = get_cosmos_client(Cosmos.host, Cosmos.key, Cosmos.mps_db, Cosmos.mps_container)

# get tweets month by month to reduce risk of memory error
months = [
    '2022-01-13T00:00:00.0000000Z',
    '2022-02-14T00:00:00.0000000Z',
    '2022-03-14T00:00:00.0000000Z',
    '2022-04-14T00:00:00.0000000Z',
    '2022-05-14T00:00:00.0000000Z',
]

start_month = months[0]

for i, month in enumerate(months[1:]):
    print(f"--- {start_month[:10]}:{month[:10]} --- {dt.datetime.now()}")
    # get all valid MP AC tweets
    tweets = query_cosmos(
        cosmos,
        'c.id, c.text_replaced_b',
        dt_start=start_month,
        dt_end=month,
        filter='c.bucket=\"audience_contact\" and c.valid=true',
        print_info=True
    )
    
    pd.DataFrame(tweets).to_csv(f'./mps_valid_ac_tweets_idtxt_{i}.csv', index=False, encoding='utf-8-sig')

    start_month = month