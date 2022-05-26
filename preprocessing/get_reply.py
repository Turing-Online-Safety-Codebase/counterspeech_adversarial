import glob
import os
import pandas
import json

def get_id_text(file):
    with open(file, 'r') as f:
        data = json.load(f)
    text = data["text_replaced_b"]
    id_str = data["id_str"]
    return text, id_str

def get_reply_text(file):
    with open(file, 'r') as f:
        data = json.load(f)
    text = data["text_replaced_b"]
    id_str = data["id_str"]
    created_at = data["created_at"]
    return text, id_str, created_at

def get_files(dir, outputfile, root_file):
    josnl_outfile = open(f'{outputfile}.jl', 'w')
    root_cnt = 1
    act_id = []
    act_text = []
    act_cnt = []
    rep_id = []
    rep_text = []
    rep_cnt = []

    # read through folders
    for path in glob.glob(f'{dir}/*'):
        data = {}
        reply = {}
        reply_cnt = 1
        root_list = []
        reply_list = []
        folder_name = os.path.basename(path)
        files = glob.glob(f'{dir}/{folder_name}/*.json')
        files.sort()

        # get id, root_tweet
        root_text, root_id_str = get_id_text(f'{path}/{root_file}')
        root_ind = 'ACT_' + str(root_cnt)
        data["id_str"] = root_id_str
        data["text"] = root_text
        root_list.append(root_id_str)
        root_list.append(root_text)
        root_cnt += 1

        for file in files:
            file_name = os.path.basename(file)

            # get replies to root_tweet if any
            # how to store the replies in order?
            if file_name.startswith("reply"):
                reply_ind = 'reply_' + str(reply_cnt)
                re_text, re_id_str, re_created_at = get_reply_text(f'{path}/{file_name}')

                reply[reply_ind] = {}
                reply[reply_ind]["text"] = re_text
                reply[reply_ind]["id_str"] = re_id_str
                reply[reply_ind]["created_at"] = re_created_at
                reply_cnt += 1
                reply_list.append(re_text)

                act_id.append(root_id_str)
                act_text.append(root_text)
                act_cnt.append(root_ind)
                rep_id.append(re_id_str)
                rep_text.append(re_text)
                rep_cnt.append(reply_ind)

        # save them in jsonline
        if len(reply) > 0:
            data["reply"] = reply
        json.dump(data, josnl_outfile)
        josnl_outfile.write('\n')

    josnl_outfile.close()

    df = pandas.DataFrame({
        'ACT_ID': act_id,
        'ACT_cnt': act_cnt,
        'ACT_text': act_text,
        'is_text_abusive[Y/N]': "",
        'is_text_pers_directed[Y/N]': "",
        'is_text_ident_directed[Y/N]': "",
        'is_text_notes': "",
        'what_identity': "",
        'type_abuse': "",
        'Rep_ID': rep_id,
        'Rep_cnt': rep_cnt,
        'Rep_text': rep_text,
        'is_reply_cs': "",
        'reply_strategy': "",
        'reply_tone': "",
        'reply_recipient': "",
        'reply_notes': "",
    })
    df.to_csv(f'{outputfile}.csv')

if __name__ == '__main__':
    root_dir = 'counterspeech_mps/data/mps_replies'
    get_files(root_dir, "counterspeech_mps/mps_replies", "root.json")

    # df = pandas.read_csv("counterspeech_mps/mp_replies.csv")
    #
    # df = df.sample(n=100)
    # df.to_csv("counterspeech_mps/100_samples_mp_replies.csv")


