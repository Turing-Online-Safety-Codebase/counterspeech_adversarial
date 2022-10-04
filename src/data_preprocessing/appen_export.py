"""
Take labelled data from Appen, do quality control and calculate annotator agreement
Output results to csv and create a csv for tweets required extra review
"""

import argparse
import pandas
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(description="Process labelled data for modeling")
    parser.add_argument('--data_dir', type=str, default='data/counterspeech_plf', help='')
    parser.add_argument('--annotation_filename', default='R6_full.csv', type=str, help='name of labelled data')
    parser.add_argument('--gold_filename', type=str, default='R6_Appen_test_quiz.csv', help='name of gold standard')
    parser.add_argument('--output_file', default='data/counterspeech_plf/2.Appen_export_rangled/R6_rangled.csv', type=str, help='')
    parser.add_argument('--column_reply_category', type=str, default='which_category_best_describes_the_reply',
                        help='column name for reply_category annotation in Task 1')
    parser.add_argument('--column_reply_abusive', type=str, default='is_the_reply_abusive',
                        help='column name for reply_abusive annotation in Task 2')
    parser.add_argument('--column_reply_support', type=str, default='is_the_reply_supporting_the_football_player',
                        help='column name for reply_support annotation in Task 2')
    parser.add_argument('--flag_reply_category', type=bool, default=True, help='flag reply_category')
    parser.add_argument('--flag_reply_abusive', type=bool, default=True, help='flag reply_abusive')
    parser.add_argument('--flag_reply_support', type=bool, default=True, help='flag reply_abusive')
    parser.add_argument('--flag_time', type=bool, default=False, help='flag time')
    parser.add_argument('--reply_category_bar', type=float, default='1.0', help='threshold for reply_category check')
    parser.add_argument('--reply_abusive_bar', type=float, default='1.0', help='threshold for reply_abusive check')
    parser.add_argument('--reply_support_bar', type=float, default='1.0', help='threshold for reply_abusive check')
    parser.add_argument('--time_bar', type=float, default='10', help='threshold for time check (seconds)')
    parser_args = parser.parse_args()

    print("the inputs are:")
    for arg in vars(parser_args):
        print(f"{arg} is {getattr(parser_args, arg)}")
    return parser_args

def remove_space(text):
    """Remove extra space in a text."""
    return " ".join(text.split())

def get_text(df, unit_id):
    """Given a unit_id from appen report, retrieve its corresponding text in the same dataframe."""
    return remove_space((unit_id, df[df['_unit_id'] == int(unit_id)]["text"].iloc[0])[1])

def get_data(df, text, column_name, gold=False):
    """Given a text, retrieve corresponding info of a defined column name in a dataframe.
        If nothing is found, return Found no data.
    Args:
        df (pandas.DataFrame): dataframe to retrieve inforomation from
        text (str): text used for retrieving desired information
        column_name (str): the column name of desired information in df
        gold (bool): wether retrieving gold standard or not
    Returns:
        result (str): the desireed information
    """
    if gold:  # in case no info is found
        result = "no gold standard"
    else:
        result = "no " + column_name
    try:
        return df[df['text'] == text][column_name].iloc[0]
    except:
        return result

def same_with_gold(annotation, gold):
    """Check if an annotation is the same as gold standard - yes/no."""
    gold_label = gold
    if gold == "no gold standard":
        return gold
    if annotation == gold_label:
        return "yes"
    return "no"

def get_annotation(df, unit_id, column_name):
    """Retrieve all annotations for an entry.
    Return pandas.Series of all 7 annotations
    If the num of annotations < max annotations, 7, append N/A till the len of annotations is 7.
    Args:
        df (pandas.DataFrame): dataframe to retrieve inforomation from
        unit_id (int): unit_id used for retrieving desired information
        column_name (str): the column name of desired information in df
        max_annotion (bool): wether retrieving gold standard or not
    Returns:
        annotations (pandas.Series): 7 annotations (judgements) for an entry
    """
    df1 = df[df['rep_id'] == unit_id]
    votes = df1[column_name].tolist()

    while len(votes) < 7:
        votes.append("N/A")
    if len(votes) > 7:
        votes = votes[:7]
    return pd.Series(votes)

def cnt_labels(annotation1, annotation2, annotation3, annotation4, annotation5, annotation6, annotation7):
    """Compute the number of votes received for each class labelled for an entry.
        Three classes are included: disagree, agree, other.
    Args:
        annotation1-7 (str): annotations from 7 contributors
    Returns:
        num_uniq_label: number of uniq label for an entry
        disagree: number of disagree annotation for an entry
        agree: number of agree annotation for an entry
        other: number of other annotation for an entry
        num_label: total number of annotation received for an entry
    """
    annotations = [annotation1, annotation2, annotation3, annotation4, annotation5, annotation6, annotation7]
    disagree = annotations.count('disagrees_with_the_post')
    agree = annotations.count('agrees_with_the_post')
    other = annotations.count('other')
    num_label = disagree + agree + other

    labels = [disagree, agree, other]
    num_uniq_label = 3 - labels.count(0)
    return pd.Series([num_uniq_label, disagree, agree, other, num_label])

def normalise_category(text):
    try:
        t = text.replace("Attacking the Text", 'attacks_the_text').replace("Agreeing with the Text", 'agrees_with_the_text')
        t = t.replace("Supporting the victims", 'supports_the_football_players').replace("Expressing opinions", 'expresses_opinions')
        t = t.replace("Other", 'other').replace("Yes", 'yes').replace("No", "no")
        return t
    except:
        return text

def get_majority_vote_binary(annotation1, annotation2, annotation3, annotation4, annotation5, annotation6, annotation7, cls1,
                      cls2):
    """Compute majority vote based on binary classes and its percentage.
        Return the majority vote, % for the majority vote, and total number of annotations received for an entry
    Args:
        annotation1-7 (str): annotations from 7 contributors
        cls1 (str): "yes", "abusive"
        cls2 (str): "no", "non_abusive"
    Returns:
        majority vote (str): majority vote across annotators for an entry (yes/no/tie/no record)
        percentage (float): the percentage of corresponding majority vote (i.e. level of agreement)
        total_cnt (int): total number of annotations received for an entry
        number of annotation for majority vote (int)
    """
    annotations = [annotation1, annotation2, annotation3, annotation4, annotation5, annotation6, annotation7]
    cls1_cnt = annotations.count(cls1)
    cls2_cnt = annotations.count(cls2)
    # print(cls1_cnt, cls2_cnt)
    total_cnt = cls1_cnt + cls2_cnt
    if cls1_cnt > cls2_cnt:
        return pd.Series([cls1, round(cls1_cnt / total_cnt, 2), total_cnt, cls1_cnt])
    elif cls1_cnt == cls2_cnt:
        return pd.Series(['tie', 0.50, total_cnt, cls1_cnt])
    elif cls2_cnt > cls1_cnt:
        return pd.Series([cls2, round(cls2_cnt / total_cnt, 2), total_cnt, cls2_cnt])
    else:  # cls1_cnt == 0 and cls2_cnt == 0, when no annotation is available
        return pd.Series(['no record', 0.00, 0, 0])

def get_majority_category(annotation1, annotation2, annotation3, annotation4, annotation5, annotation6, annotation7, tot_cnt):
    """Compute majority stance based on 3 classes and calculate the percentage for the label.
    Args:
        annotation1-7 (str): annotations from 7 contributors for an entry
        tot_cnt (int): total number of annotations received for an entry

    Returns:
        majority vote (str): majority vote across annotators for an entry for the selected annotation type
        percentage (float): the percentage of corresponding majority vote (i.e. level of agreement)
    """
    annotations = [annotation1, annotation2, annotation3, annotation4, annotation5, annotation6, annotation7]
    annotations = [e for e in annotations if e != 'N/A']
    c = Counter(annotations)
    # rank = c.most_common()
    value0, count0 = c.most_common()[0]
    try:
        value1, count1 = c.most_common()[1]
        if count0 == count1:
            return pd.Series(['tie', count0 / tot_cnt, count0])
        if count0 == 0.5:
            return pd.Series(['tie', count0 / tot_cnt, count0])
        return pd.Series([value0, count0 / len(annotations), count0])
    except:
        return pd.Series([value0, count0 /len(annotations), count0])

def irr_analysis_multi(df1, column_name, gold=None):
    """Compute Inter-rater reliability using cohen_kappa_score
    """
    if gold is not None:
        r1 = df1[column_name].tolist()
        r2 = df1[gold].tolist()
        print(f'Cohen’s kappa for {column_name} and {gold} is {round(cohen_kappa_score(r1, r2), 3)}')
    else:
        r1 = df1[column_name+"1"].tolist()
        r2 = df1[column_name+"2"].tolist()
        print(f'Cohen’s kappa for {column_name} among annotators is {round(cohen_kappa_score(r1, r2), 3)}')

def compute_avg_time(ant1, ant2, ant3, ant4, ant5, ant6, ant7, delta1, delta2, delta3, delta4, delta5, delta6, delta7,
                     num_annotation):
    """Compute average time spent per entry.
    Not consider the time spent on an entry where no annotation is recorded.
    Args:
        ant1-an7 (str): annotationa by contributors (1-7)
        delta1-delta7 (float): time spent by contributors (1-7)
        num_annotation: number of annotations received
    Returns:
        avg_second (float): average time in seconds spent per entry
        avg_minute (float): average time in minutes spent per entry
    """
    delta = [delta1, delta2, delta3, delta4, delta5, delta6, delta7]
    anntoation = [ant1, ant2, ant3, ant4, ant5, ant6, ant7]
    new_delta = []

    if num_annotation == 0:
        return pd.Series([0.00, 0.00])
    for ind, t in enumerate(delta):
        if not pd.isna(anntoation[ind]) and anntoation[ind] != 'N/A':
            if not pd.isna(t):
                new_delta.append(t)
    avg_second = round((sum(new_delta) / len(new_delta)), 2)
    avg_minute = round(avg_second / 60, 2)
    return pd.Series([avg_second, avg_minute])

def time_group(time):
    """Group a given time (seconds) into 10 time interval.
    Args:
        time (float): time in seconds
    Returns:
        result (str): 10 time intetrval: <10, 10-20, 20-30, ..., >90.
    """
    result = time  # in case the time is nan
    if float(time) <= 10:
        result = "<= 10"
    elif float(time) > 10 and float(time) <= 20:
        result = "10 < t <= 20"
    elif float(time) > 20 and float(time) <= 30:
        result = "20 < t <= 30"
    elif float(time) > 30 and float(time) <= 40:
        result = "30 < t <= 40"
    elif float(time) > 40 and float(time) <= 50:
        result = "40 < t <= 50"
    elif float(time) > 50 and float(time) <= 60:
        result = "50 < t <= 60"
    elif float(time) > 60 and float(time) <= 70:
        result = "60 < t <= 70"
    elif float(time) > 70 and float(time) <= 80:
        result = "70 < t <= 80"
    elif float(time) > 80 and float(time) <= 90:
        result = "80 < t <= 90"
    elif float(time) > 90:
        result = "t > 90"
    return result

def mark_annotator_time(time):
    """Given annotator's time spent per entry, group it into 10 time interval.
    Return time interval or nan if no response is recorded.
    Args:
        time (float): time spent by a contributor
    Returns:
        time_interval (str): <10, 10-20, 20-30, ..., >90, or nan if no response is recorded.
    """
    time_interval = time
    if type(time) == float:
        time_interval = time_group(time)
    return time_interval

def flag_extra_review(orig_id, num_rep_category, num_rep_abusive, num_rep_support, rep_category_agreement, rep_abusive_agreement, rep_support_agreement,
                      avg_time, rep_category_bar, rep_abusive_bar, rep_support_bar, time_bar, rep_category, rep_abusive, rep_support, time):
    """Flag an entry requiring expert review based on English, stance, person/identity-directed and time.
    Provide a reason explaining what reviews are required.
    Args:
        orig_id (int): reply ID
        num_rep_category (int): number of reply_category annotations received
        num_rep_abusive (int): number of reply_abusive annotations received
        num_rep_support (int): number of reply_support annotations received
        rep_category_agreement (float): level of agreement for reply_category
        rep_abusive_agreement (float): level of agreement for reply_abusive
        rep_support_agreement (float): level of agreement for reply_support
        avg_time (float): average time spent for an entry
        rep_category_bar (float): threshold for flagging reply_category
        rep_abusive_bar (float): threshold for flagging reply_abusive
        rep_support_bar (float): threshold for flagging reply_support
        time_bar (float): threshold for flagging time
        rep_category (bool): whether to flag reply_category (default=True)
        rep_abusive (bool): whether to flag reply_abusive (default=True)
        rep_support (bool): whether to flag reply_support (default=True)
        time (bool): whether to flag time (default=False)
    Returns:
        check_rep_cat (str): whether check on category class is needed (yes/no)
        check_rep_abu (str): whether check on abuse class is needed (yes/no)
        check_rep_sup (str): whether check on support class is needed (yes/no)
        checktime (str): whether time check is needed (yes/no)
        reason (str): what review is required (i.e. category, abuse, support, time)
    """
    check_rep_cat, check_rep_abu, check_rep_sup, checktime = 'no', 'no', 'no', 'no'
    reason = ''
    if rep_category:
        if rep_category_agreement < float(rep_category_bar) or num_rep_category < 3:
            reason += ", category"
            check_rep_cat = 'yes'
    if orig_id == 'no id':
        reason += ", no id"
    if rep_abusive:
        if rep_abusive_agreement < float(rep_abusive_bar) or num_rep_abusive < 3:
            reason += ", abusive"
            check_rep_abu = 'yes'
    if rep_support:
        if rep_support_agreement < float(rep_support_bar) or num_rep_support < 3:
            reason += ", support"
            check_rep_sup = 'yes'
    if time:
        if float(avg_time) < time_bar:
            reason += ", time"
            checktime = 'yes'
    return pd.Series([check_rep_cat, check_rep_abu, check_rep_sup, checktime, reason.lstrip(",").lstrip(" ")])

def mark_replicate(df):
    text_list = []
    repeated_text = []
    for index, row in df.iterrows():
        if row['text'] in text_list:
            repeated_text.append(row['text'])
        text_list.append(row['rep_id'])
    return repeated_text

def main(output_file, data_dir, annotation_filename, gold_file, column_reply_category, column_reply_support, column_reply_abusive,
     flag_reply_category, flag_reply_support, flag_reply_abusive, flag_time, reply_category_bar, reply_support_bar, reply_abusive_bar, time_bar):
    # extra_review_path = f'{data_dir}/extra_review'

    batch = annotation_filename.split("_")[0]
    df_gold = pd.read_csv(f'{data_dir}/1.Appen_annotation/{batch}/{gold_file}', dtype={'Rep_ID': str, '_golden':str, 'ACT_ID':str})
    df_t1 = pd.read_csv(f'{data_dir}/1.Appen_annotation/{batch}/{annotation_filename}', dtype={'rep_id': str})
    df_t1 = df_t1.astype({'_golden': str})  # convert numpy boolean to string
    df_gold = df_gold.astype({'Rep_ID': str})  # convert numpy boolean to string

    # unique_id = list(set(df_t1[df_t1['_golden'] == "False"]['_unit_id']))
    df_merged = df_gold[(df_gold['_golden'] == "FALSE") | (df_gold['rounds'] == 'R6 ( using R4 as gold)')]
    print("number of rows: ", len(df_merged))
    # df_merged = pd.DataFrame({'unit_id': unique_id})

    uniq_text, size_text = len(set(df_merged['ACT_text_normalised'].tolist())), len(df_merged['ACT_text_normalised'].tolist())
    rep_uniq_text, rep_size_text = len(set(df_merged['Rep_text_normalised'].tolist())), len(df_merged['Rep_text_normalised'].tolist())
    print(f'There are {size_text} rows in total, and {uniq_text} uniq abusive content.')
    print(f'There are {rep_size_text} rows in total, and {rep_uniq_text} uniq replies.')

    # get all annotations for each entry and output it to a separate column
    df_merged[
        ['rep_category1', 'rep_category2', 'rep_category3', 'rep_category4', 'rep_category5', 'rep_category6', 'rep_category7']] = \
        df_merged.apply(lambda x: get_annotation(df_t1, x['Rep_ID'], column_reply_category), axis=1)
    df_merged[
        ['rep_abusive1', 'rep_abusive2', 'rep_abusive3', 'rep_abusive4', 'rep_abusive5', 'rep_abusive6', 'rep_abusive7']] = \
        df_merged.apply(lambda x: get_annotation(df_t1, x['Rep_ID'], column_reply_abusive), axis=1)
    df_merged[
        ['rep_support1', 'rep_support2', 'rep_support3', 'rep_support4', 'rep_support5', 'rep_support6',
         'rep_support7']] = \
        df_merged.apply(lambda x: get_annotation(df_t1, x['Rep_ID'], column_reply_support), axis=1)
    df_merged[
        ['# of unique labels', 'disagree', 'agree', 'other', '# of labels']] = df_merged.apply(
        lambda x: cnt_labels(x['rep_category1'], x['rep_category2'], x['rep_category3'], x['rep_category4'],
                             x['rep_category5'], x['rep_category6'], x['rep_category7']), axis=1)
    df_merged[['start_time_second1', 'start_time_second2', 'start_time_second3', 'start_time_second4',
               'start_time_second5', 'start_time_second6', 'start_time_second7']] = df_merged.apply(
        lambda x: get_annotation(df_t1, x['Rep_ID'], '_started_at'), axis=1)
    df_merged[['end_time_second1', 'end_time_second2', 'end_time_second3', 'end_time_second4', 'end_time_second5',
               'end_time_second6', 'end_time_second7']] = df_merged.apply(
        lambda x: get_annotation(df_t1, x['Rep_ID'], '_created_at'), axis=1)

    df_merged[['rep_abusive_annotation', '% rep_abusive_annotation', '# total_rep_abusive_annotation',
               'rep_abusive_annotation_majority_cnt']] = df_merged.apply(
        lambda x: get_majority_vote_binary(x['rep_abusive1'], x['rep_abusive2'], x['rep_abusive3'], x['rep_abusive4'],
                                    x['rep_abusive5'], x['rep_abusive6'], x['rep_abusive7'], 'yes', 'no'), axis=1)
    df_merged[['rep_category_annotation', '% rep_category_annotation', 'rep_category_annotation_majority_cnt']] = df_merged.apply(
        lambda x: get_majority_category(x['rep_category1'], x['rep_category2'], x['rep_category3'], x['rep_category4'],
                                           x['rep_category5'], x['rep_category6'], x['rep_category7'], x['# of labels']), axis=1)
    df_merged[['rep_support_annotation', '% rep_support_annotation', '# total_rep_support_annotation',
               'rep_support_annotation_majority_cnt']] = df_merged.apply(
        lambda x: get_majority_vote_binary(x['rep_support1'], x['rep_support2'], x['rep_support3'], x['rep_support4'],
                                           x['rep_support5'], x['rep_support6'], x['rep_support7'], 'yes', 'no'),
        axis=1)

    # check if annotation matches with gold
    df_merged['rep_category_matched_gold'] = df_merged.apply(
        lambda x: same_with_gold(x['rep_category_annotation'], x['which_category_best_describes_the_reply_gold']), axis=1)
    df_merged['rep_abusive_matched_gold'] = df_merged.apply(
        lambda x: same_with_gold(x['rep_abusive_annotation'], x['is_the_reply_abusive_gold']), axis=1)
    df_merged['rep_support_matched_gold'] = df_merged.apply(
        lambda x: same_with_gold(x['rep_support_annotation'], x['is_the_reply_supporting_the_football_player_gold']), axis=1)

    # time control, get breakdown of time spent by each annotator and output it to a separate column
    df_merged['start_time_seconds1'] = pd.to_datetime(df_merged['start_time_second1'], format='%m/%d/%Y %H:%M:%S')
    df_merged['start_time_seconds2'] = pd.to_datetime(df_merged['start_time_second2'], format='%m/%d/%Y %H:%M:%S')
    df_merged['start_time_seconds3'] = pd.to_datetime(df_merged['start_time_second3'], format='%m/%d/%Y %H:%M:%S')
    df_merged['start_time_seconds4'] = pd.to_datetime(df_merged['start_time_second4'], format='%m/%d/%Y %H:%M:%S',errors='coerce')
    df_merged['start_time_seconds5'] = pd.to_datetime(df_merged['start_time_second5'], format='%m/%d/%Y %H:%M:%S',errors='coerce')
    df_merged['start_time_seconds6'] = pd.to_datetime(df_merged['start_time_second6'], format='%m/%d/%Y %H:%M:%S',errors='coerce')
    df_merged['start_time_seconds7'] = pd.to_datetime(df_merged['start_time_second7'], format='%m/%d/%Y %H:%M:%S',errors='coerce')
    df_merged['end_time_seconds1'] = pd.to_datetime(df_merged['end_time_second1'], format='%m/%d/%Y %H:%M:%S')
    df_merged['end_time_seconds2'] = pd.to_datetime(df_merged['end_time_second2'], format='%m/%d/%Y %H:%M:%S')
    df_merged['end_time_seconds3'] = pd.to_datetime(df_merged['end_time_second3'], format='%m/%d/%Y %H:%M:%S')
    df_merged['end_time_seconds4'] = pd.to_datetime(df_merged['end_time_second4'], format='%m/%d/%Y %H:%M:%S',errors='coerce')
    df_merged['end_time_seconds5'] = pd.to_datetime(df_merged['end_time_second5'], format='%m/%d/%Y %H:%M:%S',errors='coerce')
    df_merged['end_time_seconds6'] = pd.to_datetime(df_merged['end_time_second6'], format='%m/%d/%Y %H:%M:%S',errors='coerce')
    df_merged['end_time_seconds7'] = pd.to_datetime(df_merged['end_time_second7'], format='%m/%d/%Y %H:%M:%S',errors='coerce')
    df_merged['time_delta_seconds1'] = round(
        ((df_merged.end_time_seconds1 - df_merged.start_time_seconds1).dt.seconds) / 5, 2)
    df_merged['time_delta_seconds2'] = round(
        ((df_merged.end_time_seconds2 - df_merged.start_time_seconds2).dt.seconds) / 5, 2)
    df_merged['time_delta_seconds3'] = round(
        ((df_merged.end_time_seconds3 - df_merged.start_time_seconds3).dt.seconds) / 5, 2)
    df_merged['time_delta_seconds4'] = round(
        ((df_merged.end_time_seconds4 - df_merged.start_time_seconds4).dt.seconds) / 5, 2)
    df_merged['time_delta_seconds5'] = round(
        ((df_merged.end_time_seconds5 - df_merged.start_time_seconds5).dt.seconds) / 5, 2)
    df_merged['time_delta_seconds6'] = round(
        ((df_merged.end_time_seconds6 - df_merged.start_time_seconds6).dt.seconds) / 5, 2)
    df_merged['time_delta_seconds7'] = round(
        ((df_merged.end_time_seconds7 - df_merged.start_time_seconds7).dt.seconds) / 5, 2)
    df_merged[['average_time_seconds', 'average_time_minutes']] = df_merged.apply(
        lambda x: compute_avg_time(x['rep_abusive1'], x['rep_abusive2'],
                                   x['rep_abusive3'], x['rep_abusive4'], x['rep_abusive5'],
                                   x['rep_abusive6'], x['rep_abusive7'],
                                   x['time_delta_seconds1'], x['time_delta_seconds2'],
                                   x['time_delta_seconds3'], x['time_delta_seconds4'], x['time_delta_seconds5'],
                                   x['time_delta_seconds6'], x['time_delta_seconds7'],
                                   x['# total_rep_abusive_annotation']), axis=1)

    df_merged['average_time_interval_seconds'] = df_merged.apply(lambda x: time_group(x['average_time_seconds']),
                                                                 axis=1)
    df_merged['time_interval (seconds)1'] = df_merged.apply(lambda x: mark_annotator_time(x['time_delta_seconds1']),
                                                            axis=1)
    df_merged['time_interval (seconds)2'] = df_merged.apply(lambda x: mark_annotator_time(x['time_delta_seconds2']),
                                                            axis=1)
    df_merged['time_interval (seconds)3'] = df_merged.apply(lambda x: mark_annotator_time(x['time_delta_seconds3']),
                                                            axis=1)
    df_merged['time_interval (seconds)4'] = df_merged.apply(lambda x: mark_annotator_time(x['time_delta_seconds4']),
                                                            axis=1)
    df_merged['time_interval (seconds)5'] = df_merged.apply(lambda x: mark_annotator_time(x['time_delta_seconds5']),
                                                            axis=1)
    df_merged['time_interval (seconds)6'] = df_merged.apply(lambda x: mark_annotator_time(x['time_delta_seconds6']),
                                                            axis=1)
    df_merged['time_interval (seconds)7'] = df_merged.apply(lambda x: mark_annotator_time(x['time_delta_seconds7']),
                                                            axis=1)

    # flag rows for extra review
    df_merged[['CHECK for rep_category', 'CHECK for rep_abusive', 'CHECK for rep_support', 'CHECK for Time', 'expert_review']] = df_merged.apply(
        lambda x: flag_extra_review(x['Rep_ID'], x['# of labels'], x['# total_rep_abusive_annotation'],
                                    x['# total_rep_support_annotation'], x['% rep_category_annotation'],
                                    x['% rep_abusive_annotation'], x['% rep_support_annotation'], x['average_time_seconds'],
                                    reply_category_bar, reply_abusive_bar, reply_support_bar, time_bar,
                                    flag_reply_category, flag_reply_abusive, flag_reply_support, flag_time), axis=1)

    # Analyse gold questions
    df_gold_used = df_merged[df_merged['_golden'] == "TRUE"]
    print("\n--Analysis of Gold Questions--")
    print("How many gold standard is used in this job: ", len(df_gold_used))
    print("\n--gold rep_category overview: --")
    print(df_gold_used.groupby([column_reply_category+"_gold", column_reply_support+"_gold", column_reply_abusive+"_gold"]).size().sort_values(
        ascending=False))
    print(df_gold_used.groupby(
        ['rep_category_annotation', 'rep_support_annotation', "rep_abusive_annotation"]).size().sort_values(
        ascending=False))
    print("\n--accuracy overview: --")
    print(df_gold_used.groupby(["rep_category_matched_gold", "rep_support_matched_gold", "rep_abusive_matched_gold"]).size().sort_values(ascending=False))

    print("avg % of annotators agree on the majority vote for rep_category: ",
          round(df_gold_used["% rep_category_annotation"].mean(), 3))
    print("avg % of annotators agree on the majority vote for rep_support: ",
          round(df_gold_used["% rep_support_annotation"].mean(), 3))
    print("avg % of annotators agree on the majority vote for rep_abusive: ",
          round(df_gold_used["% rep_abusive_annotation"].mean(), 3))
    print("avg time spent (second): ", round(df_gold_used["average_time_seconds"].mean(), 3))
    print("avg time spent (minute): ", round(df_gold_used["average_time_minutes"].mean(), 3))

    # analyze work/quiz questions
    df_quiz = df_merged[df_merged['_golden'] == "FALSE"]
    print("\n--Analysis of Quiz Questions--")
    print("How many rows are there in the quiz mode: ", len(df_quiz))

    # check overview of source data
    # repeated_list = mark_replicate(df_quiz)
    list_id = df_quiz['Rep_ID'].tolist()

    uniq_id, size_id = len(set(list_id)), len(list_id)
    print(f'In quiz questions, there are in total {size_id} Rep_ID and {uniq_id} uniq Rep_ID.')
    # print(f'The list of repeated rows (none means no repetition): {repeated_list}')

    print("rep_category distribution: \n",
          df_quiz.rep_category_annotation.value_counts())
    print("rep_abusive distribution:  \n",
          df_quiz.rep_abusive_annotation.value_counts())
    print("rep_support distribution:  \n",
          df_quiz.rep_support_annotation.value_counts())
    print("\n--annotation overview: --")
    print(df_quiz.groupby(
        ["rep_category_annotation", "rep_support_annotation", "rep_abusive_annotation"]).size().sort_values(
        ascending=False))

    print("\n--rep_category label check: --")
    print('number of entries with each level of agreement:')
    print(df_quiz.groupby(
        ['# of labels', 'rep_category_annotation_majority_cnt',
         "% rep_category_annotation", 'rep_category_annotation']).size().sort_values(
        ascending=False))
    df_quiz.groupby(
        ["% rep_category_annotation"]).size().sort_values(
        ascending=False).to_csv(f'{data_dir}/2.Appen_export_rangled/{batch}_rep_category_agreement.csv')

    print("\n--rep_support label check: --")
    print('number of entries with each level of agreement:')
    print(df_quiz.groupby(
        ['# total_rep_support_annotation', 'rep_support_annotation_majority_cnt',
         "% rep_support_annotation", 'rep_support_annotation']).size().sort_values(
        ascending=False))
    df_quiz.groupby(
        ["% rep_support_annotation"]).size().sort_values(
        ascending=False).to_csv(f'{data_dir}/2.Appen_export_rangled/{batch}_rep_support_agreement.csv')

    print("\n--rep_abusive label check: --")
    print('number of entries with each level of agreement:')
    print(df_quiz.groupby(
        ['# total_rep_abusive_annotation', 'rep_abusive_annotation_majority_cnt',
         "% rep_abusive_annotation", 'rep_abusive_annotation']).size().sort_values(
        ascending=False))
    df_quiz.groupby(
        ["% rep_abusive_annotation"]).size().sort_values(
        ascending=False).to_csv(f'{data_dir}/2.Appen_export_rangled/{batch}_rep_abusive_agreement.csv')

    # print("\n--time control: --")
    # print(df_quiz.groupby(["average_time_interval_seconds"]).size().sort_values(ascending=False))
    # # df_quiz.groupby(["average_time_interval_seconds"]).size().sort_values(ascending=False).to_csv('.csv')
    # print(df_quiz.groupby(['time_interval (seconds)1']).size().sort_values(ascending=False))
    # print(df_quiz.groupby(['time_interval (seconds)2']).size().sort_values(ascending=False))
    # print(df_quiz.groupby(['time_interval (seconds)3']).size().sort_values(ascending=False))
    # print(df_quiz.groupby(['time_interval (seconds)4']).size().sort_values(ascending=False))
    # print(df_quiz.groupby(['time_interval (seconds)5']).size().sort_values(ascending=False))
    # print(df_quiz.groupby(['time_interval (seconds)6']).size().sort_values(ascending=False))
    # print(df_quiz.groupby(['time_interval (seconds)7']).size().sort_values(ascending=False))

    print("avg % of annotators agree on the majority vote for rep_category: ",
          round(df_quiz["% rep_category_annotation"].mean(), 3))
    print("avg % of annotators agree on the majority vote for rep_support: ",
          round(df_quiz["% rep_support_annotation"].mean(), 3))
    print("avg % of annotators agree on the majority vote for rep_abusive: ",
          round(df_quiz["% rep_abusive_annotation"].mean(), 3))
    print("avg time spent (second): ", round(df_quiz["average_time_seconds"].mean(), 3))
    print("avg time spent (minute): ", round(df_quiz["average_time_minutes"].mean(), 3))

    df_merged.to_csv(output_file, index=False)

    irr_analysis_multi(df_quiz, 'rep_category')
    irr_analysis_multi(df_quiz, 'rep_abusive')
    irr_analysis_multi(df_quiz, 'rep_support')

if __name__ == '__main__':
    args = parse_args()
    main(args.output_file, args.data_dir, args.annotation_filename, args.gold_filename, args.column_reply_category,
         args.column_reply_support, args.column_reply_abusive, args.flag_reply_category, args.flag_reply_support, args.flag_reply_abusive,
         args.flag_time, args.reply_category_bar, args.reply_support_bar, args.reply_abusive_bar, args.time_bar)

