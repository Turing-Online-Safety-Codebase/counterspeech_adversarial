import json
import pandas
from sklearn.utils import shuffle

def convert_labels(df):
    """Converts string labels to integers.
    Args:
        df (pd.Dataframe): Input dataframe.
    Returns:
        pd.DataFrame: Output dataframe with label columns.
        int: Number of examples in the dataframe.
    """
    # Replace label column with int values
    df['label'] = pandas.Categorical(df['label']).codes
    n_examples = len(df['label'])
    return df, n_examples

def load_balanced_n_samples(data_dir, task, split, n_entries):
    """Loads balanced first n entries of training dataset split across 2 classes.
    Args:
        data_dir (str): Directory with data.
        task (str): Task name e.g. abuse
        split (str): Split from [train, test, dev] to be sampled from.
        n_entries (int): Number of entries to sample in total.
    Returns:
        pd.DataFrame: Dataset of n rows.
    """
    SEED = 123
    balanced_n = int(n_entries/2)
    df = pandas.read_csv(f'{data_dir}/{task}/clean_data/{task}_{split}.csv')
    df_agree = df[df['label'] == 0].head(balanced_n)
    df_disagree = df[df['label'] == 1].head(balanced_n)
    df_other = df[df['label'] == 2].head(balanced_n)
    df_concat = pandas.concat([df_agree, df_disagree, df_other])
    shuffled_df = shuffle(df_concat, random_state = SEED)
    return shuffled_df