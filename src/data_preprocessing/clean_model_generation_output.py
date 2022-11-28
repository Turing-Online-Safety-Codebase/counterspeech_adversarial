import glob
import argparse
import json
import pandas

def parse_args():
    """Parses Command Line Args"""
    parser = argparse.ArgumentParser(description="Clean model output of counter speech generation")
    parser.add_argument('--input_data_path', type=str, default='data', help='Path to data directory')
    parser.add_argument('--input_filename', type=str, default='run1_3epochs_length50.txt', help='filename of input data')
    parser.add_argument('--output_filename', type=str, default='run1_3epochs_length50_clean', help='filename of output data (without file extension)')
    args = parser.parse_args()

    print("the inputs are:")
    for arg in vars(args):
        print(f"{arg} is {getattr(args, arg)}")
    return args

def main(dir, input_filename, output_filename):

    outputf = open(f'{dir}/{output_filename}.txt', "w")
    generation = []
    with open(f'{dir}/{input_filename}') as f:
        previous_line = "====== New Entry======"
        for line in f:
            if previous_line == "====== New Entry======" and line.startswith("<HS>"):
                resp = line.split("<CS>")[1]
                if "<" in resp:
                    resp = resp.split("<")[0]
                cs =  " ".join(resp.split())
                generation.append(cs)
                outputf.write(cs+"\n")
            previous_line = line.strip()
    outputf.close()

    output_file = f'{dir}/{output_filename}.json'
    data = {}
    data["values"] = generation
    data["language"] = "en"
    print(f'There are {len(generation)} generation.')
    with open(output_file, 'w') as f:
        json.dump(data, f)

    df = pandas.DataFrame(generation, columns=['generation'])
    df.to_csv(f'{dir}/{output_filename}.csv', index=False)

if __name__ == '__main__':
    args = parse_args()

    main(args.input_data_path, args.input_filename, args.output_filename)
