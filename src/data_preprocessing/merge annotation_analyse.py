import pandas
import argparse

"""
Merge expert annotation with crowd annotation.
Create final annotations for each instance.
"""

def parse_args():
    """Parses Command Line Args"""
    parser = argparse.ArgumentParser(description="Merge expert annotation with crowd annotation")
    parser.add_argument('--input_path', type=str, help='path to input data')
    parser.add_argument('--expert_reviewed_path', type=str, help='path to expert annotation')
    parser.add_argument('--output_path', type=str, help='path to outputfile')
    args = parser.parse_args()

    print("the inputs are:")
    for arg in vars(args):
        print(f"{arg} is {getattr(args, arg)}")
    return args

def get_final_annotation(crowd, expert):
    if not pandas.isnull(expert):
        return expert
    return crowd

def get_text(df1, ACT_text, Rep_text, column_name):
    """Given a unit_id from appen report, retrieve its text from a dataframe."""
    if Rep_text in df1['rep_text'].tolist():
        return df1[(df1['ACT_text'] == ACT_text) & (df1['rep_text'] == Rep_text)][column_name].iloc[0]
    return None


if __name__ == '__main__':
    args = parse_args()

    xls_1 = pandas.ExcelFile(args.input_path)
    df = pandas.read_excel(xls_1, 'R1-R5')
    df_expert = pandas.read_csv(args.expert_reviewed_path)
    df = df.astype({'Rep_ID': int}, {'ACT_ID': int})
    df_expert = df_expert.astype({'Rep_ID': int}, {'ACT_ID': int})
    print(len(df), len(df_expert))

    # Finalise reviewer2's data
    # add reviewer1's annotation to main dateframe df
    df['rep_category_annotation_JB'] = df.apply(lambda x: get_text(df_expert, x['ACT_text'], x['Rep_text'], 'JB reply category'), axis=1)
    df['rep_support_annotation_JB'] = df.apply(lambda x: get_text(df_expert, x['ACT_text'], x['Rep_text'], 'JB reply support'), axis=1)
    df['rep_abusive_annotation_JB'] = df.apply(lambda x: get_text(df_expert, x['ACT_text'], x['Rep_text'], 'JB reply abusive'), axis=1)
    df['ACT_text_notes_JB'] = df.apply(lambda x: get_text(df_expert, x['ACT_text'], x['Rep_text'], 'ACT_text_notes'), axis=1)
    df['rep_text_notes_JB'] = df.apply(lambda x: get_text(df_expert, x['ACT_text'], x['Rep_text'], 'rep_text_notes'), axis=1)

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

    df.to_csv(args.output_path)
