import sys, os
import pandas as pd
import numpy as np
import json
import tqdm
import datetime as dt
import random
from io import StringIO
from typing import Callable

import azure.cosmos.cosmos_client as azurecosmos
import azure.storage.blob as azureblob

from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
import torch

### Azure Cosmos DB Functions ###

def get_cosmos_client(
    host: str, 
    key: str, 
    db: str, 
    container: str) -> azurecosmos.CosmosClient:
    """ 
    Get Cosmos DB container client

    Args:
        host (str): Host URL for Cosmos DB account
        key (str): Access key for Cosmos DB account
        db (str): Cosmos DB database name
        container (str): Cosmos DB container name

    Returns:
        azurecosmos.CosmosClient: Cosmos DB container client
    """
    return azurecosmos.CosmosClient(
        url=host, 
        credential={'masterKey': key}
    ).get_database_client(db).get_container_client(container)


def query_cosmos(
    container, 
    select: str = '*', 
    dt_start: str = None, 
    dt_end: str = None, 
    filter: str = None, 
    orderby: str = None,
    offset: int = 0,
    limit: int = None,
    print_info: bool = False) -> list:
    """
    Query cosmos container based on parameters passed

    Args:
        container : Cosmos DB container client to query from
        select (str) : Select statement to use, eg. '*' for whole documents
        dt_start (str) : Datetime string (in DT_COSMOS) to use as lower datetime bound for query
        dt_end (str) : Datetime string (in DT_COSMOS) to use as upper datetime bound for query
        filter (str) : Additional query filters, eg. 'c.bucket="audience_contact AND c.valid=true'
        orderby (str) : Orderby statement, eg. 'c.id DESC' 
        offset (int) : Offset for query, defaults to 0 - only applicable if limit is set
        limit (int) : Limit for query
        print_info (bool) : flag indicating whether to print outputs or not

    Returns:
        list_results (list) : Results of query as list
    """
    q = f"SELECT {select} FROM c"

    # Construct query from date filter, orderby & limit parameters
    q_where = []
    if dt_start is not None:
        q_where.append(f"c.datetime>=\"{dt_start}\"")
    if dt_end is not None:
        q_where.append(f"c.datetime<\"{dt_end}\"")
    if filter is not None:
        q_where.append(f"({filter})")

    if len(q_where)>0: q = q + " WHERE " + ' AND '.join(q_where)

    if orderby is not None: q = q + " ORDER BY " + orderby

    if limit is not None: q = q + f" OFFSET {offset} LIMIT {limit}"

    if print_info: 
        print(f"Querying {container.id}: {q}")
        t1 = dt.datetime.now()

    list_results = list(container.query_items(query=q, enable_cross_partition_query=True))
    
    if print_info: 
        print(f"{len(list_results)} results returned in {dt.datetime.now()-t1}")

    return list_results


def query_cosmos_count(
    container, 
    dt_start: str = None, 
    dt_end: str = None, 
    filter: str = None, 
    print_info: bool = False) -> list:
    """
    Query cosmos document count based on parameters passed

    Args:
        container : Cosmos DB container client to query from
        dt_start (str) : Datetime string (in DT_COSMOS) to use as lower datetime bound for query
        dt_end (str) : Datetime string (in DT_COSMOS) to use as upper datetime bound for query
        filter (str) : Additional query filters, eg. 'c.bucket="audience_contact AND c.valid=true'
        print_info (bool) : flag indicating whether to print outputs or not

    Returns:
        (int) : Document count 
    """
    return query_cosmos(
        container, 
        select='VALUE COUNT(1)', 
        dt_start=dt_start, 
        dt_end=dt_end, 
        filter=filter, 
        print_info=print_info)[0]


def query_cosmos_field(
    container, 
    field: str,
    dt_start: str = None, 
    dt_end: str = None, 
    filter: str = None, 
    orderby: str = None,
    offset: int = 0,
    limit: int = None,
    print_info: bool = False) -> list:
    """
    Query cosmos document count

    Args:
        container : Cosmos DB container client to query from
        field (str) : Name of field to select in query, eg.'id'
        dt_start (str) : Datetime string (in DT_COSMOS) to use as lower datetime bound for query
        dt_end (str) : Datetime string (in DT_COSMOS) to use as upper datetime bound for query
        filter (str) : Additional query filters, eg. 'c.bucket="audience_contact AND c.valid=true'
        print_info (bool) : flag indicating whether to print outputs or not

    Returns:
        (list) : List of field values
    """
    return [
        x[field] 
        for x in query_cosmos(
            container, 
            f"c.{field}", 
            dt_start, 
            dt_end, 
            filter,
            orderby,
            offset,
            limit,
            print_info
        )
    ]


def query_cosmos_by_ids(container, ids: list, select: str = '*') -> list:
    """
    Query cosmos container using a list of ids

    Args:
        container : Cosmos DB container client to query from
        ids (list) : List of document ids to retrieve
        select (str) : Select statement to use, eg. '*' for whole documents

    Returns:
        list_results (list) : Results of query as list
    """
    # batch a list of ids into valid queries (<256kb)
    q = f"SELECT {select} FROM c WHERE"
    queries = [f"SELECT {select} FROM c WHERE c.id=\"{ids[0]}\""]
    q_size = [1]
    q_i = 0

    for doc_id in tqdm.tqdm(ids[1:], desc='Batching IDs'):
        # if still space in this query, add another id
        if len(queries[q_i].encode('utf-8')) < 250000:
            queries[q_i] = queries[q_i] + f" OR c.id=\"{doc_id}\""
            q_size[q_i] += 1
        # otherwise add new query to list and update index
        else:
            queries.append(q + " c.id=\"{doc_id}\"")
            q_i += 1
            q_size.append(1)

    print(q_size)

    all_docs = []
    for query in tqdm.tqdm(queries, desc='Querying batches'):
        start = dt.datetime.now()
        list_r = list(container.query_items(query=query,enable_cross_partition_query=True))
        print(f'{len(list_r)} docs retrieved in {dt.datetime.now()-start}')
        all_docs.extend(list_r)

    return all_docs


def get_random_cosmos_sample_ids(
    container, 
    n: int, 
    seed: int = None,
    dt_start: str = None, 
    dt_end: str = None, 
    filter: list = None,
    exclude_ids: list = None):
    """
    Get n random document ids from a cosmos container that match provided dates/filters
    
    !!! this can be very cpu/memory hungry if the pool of documents is large !!!

    Args:
        container : Cosmos DB container client to get docs from
        n (int) : Size of sample to return
        seed (int) : Seed for random number generator
        dt_start (str) : Datetime string (DT_COSMOS) represeting beginning of query window
        dt_end (str) : Datetime string (DT_COSMOS) represeting end of query window
        filter (str) : Additional query filters, eg. 'c.bucket="audience_contact AND c.valid=true'
        exclude_ids (list) : List of tweet ids to exclude from sample

    Returns: 
        all_ids ([str]) : List of all ids matching query
        valid_ids ([str]]) : List of all ids matching query that are not in exclude_ids
        sample_ids ([str]]) : List of all n random sampled ids
    """
    # get all ids that match query parameters
    all_ids = query_cosmos_field(container, 'id', dt_start, dt_end, filter)
    if exclude_ids is not None: valid_ids = [x for x in all_ids if x not in exclude_ids]
    else: valid_ids = all_ids

    print(f"- Total Pool Size  : {len(all_ids)}")
    print(f"- Excluded IDs     : {len(exclude_ids) if exclude_ids is not None else 'N/A'}")
    print(f"- Valid Pool Size  : {len(valid_ids)}")
    print(f"- Sample size      : {n}")

    # the number of samples needs can't exceed the number available
    if n>len(valid_ids): raise Exception(f'Too many samples requested')

    # get random sample of n ids
    if seed is None: seed=dt.datetime.now()
    random.seed(seed)
    sample_ids = random.sample(valid_ids, n)
    
    print(f"Retrieved {n} samples from {len(valid_ids)} ids")

    return all_ids, valid_ids, sample_ids


def create_random_cosmos_sample(
    cosmos_container, 
    blob_container,
    n: int,
    seed: int = None,
    dt_start: str = None,
    dt_end: str = None,
    processing: Callable = None,
    filter: list = None,
    exclude_ids: list = None,
    blob_prefix : str = '',
    save_prefix : str = '',
    return_ids : bool = True,
    return_docs : bool = True):
    """"
    Get n size sample of documents from cosmos container
    Uploads sampled docs & ids to azure blob storage container

    !!! this can be very cpu/memory hungry if the pool of documents or sample size is large !!!

    Args:
        cosmos_container : Azure Cosmos DB container to get samples from
        blob_container : Azure Blob Storage container to upload samples to
        n (int) : Number of samples to draw
        seed (int) : Seed for random number generator
        dt_start (str) : Datetime string (DT_COSMOS) represeting beginning of query window
        dt_end (str) : Datetime string (DT_COSMOS) represeting end of query window
        processing (Callable) : extra processing function to pass documents through 
        filter (str) : Additional query filters, eg. 'c.bucket="audience_contact AND c.valid=true'
        blob_prefix (str) : Path within blob container to upload to
        save_prefix (str) : Local path to save all ids and sampled ids to
        exclude_ids (list) : List of tweet ids to exclude from sample
        return_ids (bool) : flag indicating whether to return ids or not
        return_docs (bool) : flag indicating whether to return docs or not

    Returns:
        all_ids ([str]) : List of all ids matching query
        valid_ids ([str]]) : List of all ids matching query that are not in exclude_ids
        sample_ids ([str]]) : List of all n random sampled ids
        sampled_docs ([dict]) : List of sampled documents
    """
    # get doc ids based on query params, and n samples of ids
    print("### Getting ids")
    all_ids, valid_ids, sample_ids = get_random_cosmos_sample_ids(
        cosmos_container, 
        n, 
        seed,
        dt_start, 
        dt_end, 
        filter,
        exclude_ids
    )
    
    # save ids locally
    print(f"### Saving ids locally to '{save_prefix}' (local)")
    json_all = json.dumps(all_ids)
    json_valid = json.dumps(valid_ids)
    json_sample = json.dumps(sample_ids)
    with open(f"{save_prefix}all_ids.json", 'w') as file: json.dump(json_all,file)
    with open(f"{save_prefix}valid_ids.json", 'w') as file: json.dump(json_valid,file)
    with open(f"{save_prefix}sample_ids.json", 'w') as file: json.dump(json_sample,file)

    # save ids to blob storage
    print(f"### Saving ids to {blob_container.container_name}:{blob_prefix}")
    upload_data_to_blob(blob_container, json_all, 'all_ids', 'json', blob_prefix)
    upload_data_to_blob(blob_container, json_valid, 'valid_ids', 'json', blob_prefix)
    upload_data_to_blob(blob_container, json_sample, 'sample_ids', 'json', blob_prefix)

    # get documents from sample_ids
    print("### Getting sample docs:")
    sample_docs = query_cosmos_by_ids(
        cosmos_container,
        sample_ids,
        select='*',
    )

    # do extra processing if passed
    if processing is not None:
        print(f"### Doing extra processing using '{processing.__name__}'")
        sample_docs = processing(sample_docs)

    # save sample docs to blob storage
    print(f"### Uploading sample docs to blob container {blob_container.container_name}:{blob_prefix}")
    for doc in tqdm.tqdm(sample_docs, desc='Blob Upload'): 
        upload_dict_to_blob_as_json(blob_container, doc['id'], doc, blob_prefix)

    if return_ids and return_docs:
        return all_ids, valid_ids, sample_ids, sample_docs
    elif return_ids and not return_docs:
        return all_ids, valid_ids, sample_ids
    elif not return_ids and return_docs:
        return sample_docs


def get_tweet_reply_count(
    container, 
    tweet_id: str, 
    valid_only: bool = True, 
    filter: str = None) -> int:
    """
    Get the number of replies for a list of tweet ids

    Args:
        container : Azure Cosmos container
        tweet_id (str) : The tweet ID to get the number of replies for
        valid_only (bool) : Flag indicating whether to only include "valid" tweets in replies
        filter (str) : Additional query filters
    Returns:
        (int) : Number of replies 
    """
    qf = f"c.in_reply_to_status_id_str=\"{tweet_id}\""
    if valid_only: qf = qf + " AND c.valid=true"
    if filter is not None: qf = qf + " AND " + filter
    return query_cosmos_count(container, filter=qf)


def get_tweet_replies(
    container, 
    tweet_id: str,
    select: str = '*',
    valid_only: bool = True, 
    filter: str = None) -> int:
    """
    Get the replies for a list of tweet ids

    Args:
        container : Azure Cosmos container
        tweet_id (str) : The tweet ID to get replies for
        select (str) : Fields to select from cosmos
        valid_only (bool) : Flag indicating whether to only include "valid" tweets in replies
        filter (str) : Additional query filters
    Returns:
        (list) : List of reply tweets
    """
    qf = f"c.in_reply_to_status_id_str=\"{tweet_id}\""
    if valid_only: qf = qf + " AND c.valid=true"
    if filter is not None: qf = qf + " AND " + filter
    return query_cosmos(container, select=select, filter=qf)


### Azure Blob Storage Functions ###

def get_blob_client(connect_str: str, container: str) -> azureblob.ContainerClient:
    """
    Get client for Azure Blob Storage container

    Args:
        connect_str (str): connection string for Azure Blob Storage account
        container (str): name of Azure Blob Storage container

    Returns:
        azureblob.ContainerClient: Azure Blob Storage container client
    """
    return azureblob.BlobServiceClient.from_connection_string(
        connect_str).get_container_client(container)


def upload_data_to_blob(container, data, name: str, extension: str, prefix: str = ''):
    """
    Upload generic Python object to Azure Blob Storage container

    Args:
        container : Azure Blob Storage container client
        data : Python object to upload
        name (str): Name of blob to upload to
        extension (str): Extension of blob to upload to
        prefix (str, optional): Path to upload blob into, eg. folder name
    """
    container.upload_blob(
        name=f'{prefix}{name}.{extension}',
        data=data,
        overwrite=True, 
    )


def upload_file_to_blob(container, path: str, name: str, prefix: str = ''):
    """ 
    Upload local file to Azure Blob Storage container

    Args:
        container : Azure Blob Storage container client
        path (str): Path to file to upload
        name (str): Name of blob to upload to
        prefix (str, optional): Path to upload blob into, eg. folder name
    """
    extension = path.split('.')[-1]
    upload_data_to_blob(container, open(path, 'rb'), name, extension, prefix)


def download_data_from_blob(container, blob_path: str):
    """
    Download file from Azure Blob Storage to Python bytes object

    Args:
        container : Azure Blob Storage container client
        blob_path (str): Path to blob to download
    """
    return container.download_blob(blob_path).readall()


def download_file_from_blob(container, blob_path: str, out_path: str):
    """
    Download file from Azure Blob Storage to local file

    Args:
        container : Azure Blob Storage container client
        blob_path (str): Path to blob to download
        out_path (str): Path to local file to write to
    """
    with open(out_path, 'wb') as f:
        f.write(download_data_from_blob(container, blob_path))


def upload_folder_to_blob(container, folder_path: str, prefix: str = ''):
    """
    Upload local folder to Azure Blob Storage container

    Args:
        container : Azure Blob Storage container client
        folder_path (str): Path to folder to upload
        prefix (str, optional): Path to upload blob into, eg. folder name
    """
    i=0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            i+=1
            full_path = os.path.join(root, file)
            upload_file_to_blob(container, full_path, full_path, prefix)

    print(f'{i} files uploaded')


def download_folder_from_blob(container, folder_path: str, out_path: str):
    """"
    Download folder from Azure Blob Storage to local folder

    Args:
        container : Azure Blob Storage container client
        folder_path (str): Path to folder to download
        out_path (str): Path to local folder to write to
    """
    # if output folder doesn't exist, create
    if not os.path.isdir(out_path): os.makedirs(out_path)

    # for all blobs in chosen folder
    i = 0
    for blob in container.list_blobs(name_starts_with=folder_path):
        i+=1
        # if there's another subfolder
        if len(blob.name.split('/'))>2:
            # for all subfolders in path
            sf_path = out_path
            for subfolder in blob.name.split('/')[1:-1]:
                # if subfolder doesn't exist, create
                sf_path = sf_path+subfolder+'/'
                if not os.path.isdir(sf_path): 
                    os.makedirs(sf_path)

        # create local path for blob
        path = out_path + '/'.join(blob.name.split('/')[1:])
        # download blob
        download_file_from_blob(container, blob.name, path)

    print(f"{i} blobs downloaded from {folder_path}")


def upload_df_to_blob_as_csv(container, name: str, df: pd.DataFrame, prefix: str = '', index: bool = False):
    """
    Upload dataframe to Azure Blob Storage container as csv file

    Args:
        container : Azure Blob Storage container client
        name (str): File name to use for csv file
        df (pd.DataFrame): Dataframe to upload
        prefix (str, optional): File prefix, use to upload files to subfolders, eg. "folder/"
        index (bool, optional): Whether to include index from dataframe
    """
    upload_data_to_blob(container, df.to_csv(index=index, encoding='utf-8-sig'), name, 'csv', prefix)


def upload_dict_to_blob_as_json(container, name: str, data: dict, prefix: str = ''):
    """
    Upload dictionary to Azure Blob Storage container as json file

    Args:
        container : Azure Blob Storage container client
        name (str): File name to use for json file
        data (dict): Dictionary to upload
        prefix (str, optional): File prefix, use to upload files to subfolders, eg. "folder/"
    """
    upload_data_to_blob(container, json.dumps(data), name, 'json', prefix)


def download_csv_from_blob_to_df(container, blob: str) -> pd.DataFrame:
    """
    Download csv from Azure Blob Storage container to dataframe

    Args:
        container : Azure Blob Storage container client
        blob (str): Path to blob to download

    Returns:
        pd.DataFrame: Dataframe of csv file
    """
    return pd.read_csv(StringIO(container.download_blob(blob).content_as_text()))


def download_json_from_blob_to_dict(container, blob: str) -> dict:
    """
    Download json from Azure Blob Storage container to dictionary

    Args:
        container : Azure Blob Storage container client
        blob (str): Path to blob to download

    Returns:
        dict: Dictionary of json file
    """
    return json.loads(container.download_blob(blob).content_as_text())



### Hugging Face Model Inference functions/classes ###

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    
class HuggingfaceInferenceModel():
    """ Class to simplify process of doing inference on language model using Huggingface"""

    def __init__(self, path: str, traindir: str, batchsize: int):
        """
        Args:
            path (str) : Path to either local model files, or to HuggingFace hosted model
            traindir (str) : Path to directory for storing training logs (nothing written but this must be specified)
            batchsize (int) : Batch size for inference
        """
        self.path = path
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        
        self.traindir = traindir
        self.batchsize = batchsize
        self.trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=self.traindir,
                per_device_eval_batch_size=self.batchsize,
        ))


    def __call__(self, data, return_df: bool = False):
        """
        Process passed text and do inference
        
        Args:
            data (list) : List of strings to do inference on
            return_df (bool) : Flag indicating whether to return results as pandas dataframe or not

        Returns:
            (list, list) OR pd.DataFrame : resulting probabilities and labels, either as two lists or as df
        """
        probs, labels = self.do_inference(self.process_input(data))

        if return_df:
            return pd.DataFrame(data={'text': data, 'probs': probs, 'labels': labels})
        else:
            return probs, labels


    def process_input(self, data):
        """
        Create Torch Dataset of encodings of input data

        Args: 
            data (list) : list of input strings
        
        Returns:
            dataset (TorchDataset) : Torch dataset containing encodings of input strings
        """
        # apply tokenizer
        encodings = self.tokenizer(data, truncation=True, padding=True)
        # create torch dataset from text encodings + labels
        dataset = TorchDataset(encodings, [0 for d in data])
        return dataset
    

    def do_inference(self, dataset):
        """
        Do inference on Torch Dataset

        Args:
            dataset (TorchDataset) : Torch Dataset to do inference on

        Returns:
            probs,labels (list, list) : resulting probabilities and labels
        """
        results = self.trainer.predict(dataset)

        preds_prob = list(map(
            (lambda x: float(torch.nn.functional.softmax(torch.from_numpy(x), dim=-1)[1])),
            results[0]
        ))
        preds_bin = list(map(
            (lambda x: int(np.argmax(x))),
            results[0]
        ))

        return preds_prob, preds_bin


### Other Functions ###

class DatetimeFormats:
    DT_COSMOS = "%Y-%m-%dT%H:%M:%S.%f0Z" # eg. '2021-11-02T16:35:40.0000000Z'
    DT_TWITTER = '%a %b %d %X %z %Y' # eg. 'Tue Nov 02 16:35:40 +0000 2021'
    DT_DATE = '%d/%m/%y' # eg. '02/11/21


def dt_string_conversion(date: str, informat: str, outformat: str):
    """
    Convert string representation of datetime to another string representation

    Args:
        date (str): date/time to be converted
        informat (str): strftime format code of input date
        outformat (str): strftime format code to output

    Returns:
        outdate (str): date in specified output format
    """
    outdate = dt.datetime.strftime(
        dt.datetime.strptime(date,informat), 
        outformat)
    return outdate
    

def create_tweet_df(
    tweet_list: list,
    cols: list = [
        'id','user_id','text','text_replaced','datetime','seed_BODY','seed_CLUB','seed_PLAYER',
        'non_seed_USER', 'BODY', 'CLUB', 'PLAYER', 'perspective', 'retweeted_status', 
        'in_reply_to_status_id', 'is_quote_status'
    ]) -> pd.DataFrame:
    """
    Turn list of tweets into dataframe

    Args:
        tweet_list ([dict]) : List of tweets as dictionaries
        cols ([str]) : List of columns to include in dataframe

    Returns:
        df_tweets (pd.DataFrame): Dataframe of tweets from tweet_list
    """
    df_tweets = pd.DataFrame(tweet_list, columns=cols)
    
    # check that no tweets were lost
    assert df_tweets.shape[0]==len(tweet_list)

    return df_tweets