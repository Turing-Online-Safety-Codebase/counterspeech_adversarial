import pandas
import argparse

"""This is the script for creating train/dev/test data for counter speech generation"""

def parse_args():
    """Parses Command Line Args"""
    parser = argparse.ArgumentParser(description="Create modeling data for counterspeech generation")
    parser.add_argument('--input_data_path', type=str, default='data/final_modeling_data', help='Path to data directory')
    parser.add_argument('--input_filename', type=str, default='val_labelled.csv', help='filename of input data')
    parser.add_argument('--output_filename', type=str, default='val.txt', help='filename of output data')
    args = parser.parse_args()

    print("the inputs are:")
    for arg in vars(args):
        print(f"{arg} is {getattr(args, arg)}")
    return args

def text_preprocess(text):
    text = text.replace("[USER]", "").replace("[PLAYER]", "").replace("[CLUB]", "").replace("[URL]", "")
    return " ".join(text.split())

def main(data_dir, input_file, output_file):

    df = pandas.read_csv(f"{data_dir}/{input_file}")
    # preprocess text
    df['abusive_speech'] = df.apply(lambda x: text_preprocess(x['abusive_speech']), axis=1)
    df['counter_speech'] = df.apply(lambda x: text_preprocess(x['counter_speech']), axis=1)

    with open(f"{data_dir}/{output_file}", "w", encoding='utf8') as f:  #force_ascii=False
        for index, row in df.iterrows():
            input = row['abusive_speech']
            output = row['counter_speech']
            f.write("<HS> " + input + " <CN> " + output + "\n")

if __name__ == '__main__':

    args = parse_args()
    main(args.input_data_path, args.input_filename, args.output_filename)
    