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

def load_balanced_n_samples(data_dir, n_entries):
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
    balanced_n = int(n_entries/3)
    df = pandas.read_csv(data_dir)
    df_not_cs = df[df['label'] == 'agrees_with_the_post'].head(balanced_n)
    df_cs = df[df['label'] == 'disagrees_with_the_post'].head(balanced_n)
    df_other = df[df['label'] == 'other'].head(balanced_n)
    df_concat = pandas.concat([df_not_cs, df_cs, df_other])
    shuffled_df = shuffle(df_concat, random_state = SEED)
    return shuffled_df