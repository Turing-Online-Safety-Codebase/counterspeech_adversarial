import pandas
import argparse
import numpy as np


def parse_args():
    """Parses Command Line Args"""
    parser = argparse.ArgumentParser(description="Process labelled data for modeling")
    parser.add_argument('--input_path', default="data/counterspeech_plf/3.expert_reviewed/R1-R5_counter_speech_V2.xlsx", type=str, help='path to input data')
    parser.add_argument('--expert_reviewed_path', default="data/counterspeech_plf/3.expert_reviewed/expert_review_counter_speech_project_JB - R1-R5_final.csv", type=str, help='path to input data')
    # parser.add_argument('--output_reply_file', default="data/counterspeech_plf/pre-annotation/plf_replies_100_samples_v3_annotated_reply_cnt.csv", type=str, help='path to outputfile')
    args = parser.parse_args()

    print("the inputs are:")
    for arg in vars(args):
        print(f"{arg} is {getattr(args, arg)}")
    return args

def get_final_annotation(crowd, expert):
    if not pandas.isnull(expert):
        return expert
    return crowd

def get_text(df, ACT_text, Rep_text, column_name):
    """Given a unit_id from appen report, retrieve its text from a dataframe."""
    # (unit_id, df[df['_unit_id'] == int(unit_id)]["text"].iloc[0])[1]
    if Rep_text in df['rep_text'].tolist():
        # print(str(unit_id), unit_id)
        # print(df[df['Rep_ID'] == unit_id][column_name].iloc[0])
        return df[(df['ACT_text'] == ACT_text) & (df['rep_text'] == Rep_text)][column_name].iloc[0]
    return None


if __name__ == '__main__':
    args = parse_args()

    xls_1 = pandas.ExcelFile(args.input_path)
    df = pandas.read_excel(xls_1, 'R1-R5')
    df_expert = pandas.read_csv(args.expert_reviewed_path)
    df = df.astype({'Rep_ID': int}, {'ACT_ID': int})
    df_expert = df_expert.astype({'Rep_ID': int}, {'ACT_ID': int})
    print(len(df), len(df_expert))
    # print(df_expert['Rep_ID'])

    # Finalise reviewer2's data
    # add reviewer1's annotation to main dateframe df
    df['rep_category_annotation_JB'] = df.apply(lambda x: get_text(df_expert, x['ACT_text'], x['Rep_text'], 'JB reply category'), axis=1)
    df['rep_support_annotation_JB'] = df.apply(lambda x: get_text(df_expert, x['ACT_text'], x['Rep_text'], 'JB reply support'), axis=1)
    df['rep_abusive_annotation_JB'] = df.apply(lambda x: get_text(df_expert, x['ACT_text'], x['Rep_text'], 'JB reply abusive'), axis=1)
    df['ACT_text_notes_JB'] = df.apply(lambda x: get_text(df_expert, x['ACT_text'], x['Rep_text'], 'ACT_text_notes'), axis=1)
    df['rep_text_notes_JB'] = df.apply(lambda x: get_text(df_expert, x['ACT_text'], x['Rep_text'], 'rep_text_notes'), axis=1)
    # print(df["rep_category_annotation_JB"].value_counts())
    # print(df["rep_support_annotation_JB"].value_counts())
    # print(df["rep_abusive_annotation_JB"].value_counts())

    # Finalise data annotation
    # merge reviewer1's annotation
    df['rep_category_final'] = df.apply(
        lambda x: get_final_annotation(x['rep_category_annotation'], x['rep_category_annotation_YC']), axis=1)
    df['rep_support_final'] = df.apply(
        lambda x: get_final_annotation(x['rep_support_annotation'], x['rep_support_annotation_YC']), axis=1)
    df['rep_abusive_final'] = df.apply(
        lambda x: get_final_annotation(x['rep_abusive_annotation'], x['rep_abusive_annotation_YC']), axis=1)
    # merge reviewer2's annotation
    df['rep_category_final'] = df.apply(
        lambda x: get_final_annotation(x['rep_category_final'], x['rep_category_annotation_JB']), axis=1)
    df['rep_support_final'] = df.apply(
        lambda x: get_final_annotation(x['rep_support_final'], x['rep_support_annotation_JB']), axis=1)
    df['rep_abusive_final'] = df.apply(
        lambda x: get_final_annotation(x['rep_abusive_final'], x['rep_abusive_annotation_JB']), axis=1)

    print(df["rep_category_final"].value_counts())
    print(df["rep_support_final"].value_counts())
    print(df["rep_abusive_final"].value_counts())

    df = df[(df['ACT_text_normalised_notes'] != "not abusive") & (df['ACT_text_normalised_notes'] != "not so abusive") &
            (df['ACT_text_notes_JB'] != "Not abusive") & (df['rep_text_notes_JB'] != "reply not in english") &
            (df['rep_text_notes_JB'] != "reply not english")
            ]

    print(f'\n {len(df)}')
    print(df["rep_category_final"].value_counts())
    print(df["rep_support_final"].value_counts())
    print(df["rep_abusive_final"].value_counts())

    df.to_csv("data/counterspeech_plf/3.expert_reviewed/R1-R5_counter_speech_final.csv")