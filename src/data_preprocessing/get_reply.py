import glob
import os
import pandas
import json

"""extract all replies for each root tweet and store them in a single csv in long format"""

def get_id_text_lable(file):
    with open(file, 'r') as f:
        data = json.load(f)
    text_replaced = data["text_replaced_b"]
    text = data["text"]
    id_str = data["id_str"]
    abuse_label = str(data["abuse_mar22"]["label"]).replace("1", "abusive").replace("0", "not_abusive")
    abuse_prob = round(data["abuse_mar22"]["probability"], 8)
    return text, text_replaced, id_str, abuse_label, abuse_prob

def get_reply_text(file):
    with open(file, 'r') as f:
        data = json.load(f)
    text_replaced = data["text_replaced_b"]
    text = data["text"]
    id_str = data["id_str"]
    created_at = data["created_at"]
    return text, text_replaced, id_str, created_at

def get_files(directory, outputfile, root_file):
    josnl_outfile = open(f'{outputfile}.jl', 'w')
    root_cnt = 1
    act_id = []
    act_text = []
    act_text_replaced = []
    act_text_abuse_label = []
    act_text_abuse_prob = []
    act_cnt = []
    rep_id = []
    rep_text = []
    rep_text_replaced = []
    rep_cnt = []

    # read through folders
    for path in glob.glob(f'{directory}/*'):
        data = {}
        reply = {}
        reply_cnt = 1
        root_list = []
        reply_list = []
        folder_name = os.path.basename(path)
        files = glob.glob(f'{directory}/{folder_name}/*.json')
        files.sort()

        # get id, root_tweet
        root_text, root_text_replaced, root_id_str, abuse_label, abuse_prob = get_id_text_lable(f'{path}/{root_file}')
        root_ind = 'ACT_' + str(root_cnt)
        data["id_str"] = root_id_str
        data["text"] = root_text
        data["text_replaced"] = root_text_replaced
        data["text_abuse_label"] = abuse_label
        data["text_abuse_prob"] = abuse_prob
        root_list.append(root_id_str)
        root_list.append(root_text)
        root_cnt += 1

        for file in files:
            file_name = os.path.basename(file)

            # get replies to root_tweet if any
            # how to store the replies in order?
            if file_name.startswith("reply"):
                reply_ind = 'reply_' + str(reply_cnt)
                re_text, re_text_replaced, re_id_str, re_created_at = get_reply_text(f'{path}/{file_name}')

                reply[reply_ind] = {}
                reply[reply_ind]["text"] = re_text
                reply[reply_ind]["text_replaced"] = re_text_replaced
                reply[reply_ind]["id_str"] = re_id_str
                reply[reply_ind]["created_at"] = re_created_at
                reply_cnt += 1
                reply_list.append(re_text)

                act_id.append(root_id_str)
                act_text.append(root_text)
                act_text_replaced.append(root_text_replaced)
                act_text_abuse_label.append(abuse_label)
                act_text_abuse_prob.append(abuse_prob)
                act_cnt.append(root_ind)
                rep_id.append(re_id_str)
                rep_text.append(re_text)
                rep_text_replaced.append(re_text_replaced)
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
        'ACT_text_replaced': act_text_replaced,
        'ACT_text_abuse_label': act_text_abuse_label,
        'ACT_text_abuse_prob': act_text_abuse_prob,
        'Rep_ID': rep_id,
        'Rep_cnt': rep_cnt,
        'Rep_text': rep_text,
        'Rep_text_replaced': rep_text_replaced,
    })
    df.to_csv(f'{outputfile}.csv')

if __name__ == '__main__':
    root_dir = 'counterspeech_adversarial/data/twitter_plf_data/twitter_plf_raw/plf_replies'
    get_files(root_dir, "counterspeech_adversarial/data/twitter_plf_data/twitter_plf_raw/plf_replies_v5", "root_tweet.json")
