from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import datetime
import pandas
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Process labelled data for modeling")
    parser.add_argument('--input_data_path', type=str, default='data/adversarial_data/context_for_adversarial.csv', help='name of file for collecting counterspeech')
    parser.add_argument('--output_dir', type=str, default='data/adversarial_data', help='path to output directory')
    parser.add_argument('--model_path', type=str, default='models/iter_0/run1_final_deberta', help='name of the model')
    parser.add_argument('--num_round', type=str, default='1', help='number of round')
    pars_args = parser.parse_args()

    print("the inputs are:")
    for arg in vars(pars_args):
        print(f"{arg} is {getattr(pars_args, arg)}")
    return pars_args

def predict(hs, response, gold_class, model, tokenizer):
    """Model prediction.

    Args:
        hs (str): a sequence of text
        response (str): a sequence of response to the hs
        gold_class (int): true label for the response
        model (obj:`Model`): A model to use for classification.
        tokenizer (obj:`Tokenizer`): A ``Tokenizer`` of the current model.

    Returns:
        model prediction [0,1,2] and the probability of the model prediction
    Examples:
        hs_3 = "[PLAYER] I'd delete your profile you utter scumbag mate."
        sequence_3 = "[USER] [PLAYER] yes, agree"
        predict(hs_3, sequence_3, 0)
    """
    label_mapping_dict = {0: 'not_counterspeech', 1: 'counterspeech', 2: 'other'}

    predict = tokenizer(hs, response, return_tensors="pt")
    
    logits = model(**predict).logits
    scores = torch.softmax(logits, dim=1)
    pred_prob_list = scores.tolist()[0]
    pred_indice = scores.argmax().item()

    if pred_indice == gold_class:
        print(f"You didn't fool the model. The model predicted {label_mapping_dict[pred_indice]}, and you say {label_mapping_dict[gold_class]}")
        print(f"Confidence counter speech is {int(round(pred_prob_list[1] * 100))}%.\nConfidence not_counter_speech is {int(round(pred_prob_list[0] * 100))}%\nConfidence other is {int(round(pred_prob_list[2] * 100))}%")
    else:
        print(f"You fool the model. The model predicted {label_mapping_dict[pred_indice]}, and you say {label_mapping_dict[gold_class]}")
        print(f"Confidence counter speech is {int(round(pred_prob_list[1] * 100))}%.\nConfidence not_counter_speech is {int(round(pred_prob_list[0] * 100))}%\nConfidence other is {int(round(pred_prob_list[2] * 100))}%")
    
    return pred_indice, pred_prob_list[pred_indice]

def update_results(idx, text, response, gold_class, pred, pred_probs):

    new_row = {'ID':idx, 
            'abusive_speech': text, 
            'response': response, 
            'gold_label':gold_class,
            'pred_label': pred, 
            'pred_prob': pred_probs,
    }
    return new_row

def save_results(output_dir, outpue_filename, new_row):  
    # Not sure about the best way to save data   
    if os.path.exists(f"{output_dir}/adversarial_data/{outpue_filename}"):
        df = pandas.read_csv(f'{output_dir}/{outpue_filename}')
        print(f'There are {len(df)} rows in .')
        df_updated = df.append(new_row, ignore_index=True)
    else:
        df_updated = pandas.DataFrame(new_row)
    print(f'There are {len(df)} rows created.')
    now = str(datetime.datetime.now())
    df_updated.to_csv(f'{output_dir}/{outpue_filename}/{now}.csv')

def load_samples(df, priority):
    """Loads first entry of dataset.

    Args:
        data_dir (str): Directory of dataset.
        priority (str): which priority category of entry to retrieved [high, low, medium].

    Returns:
        pd.DataFrame: Dataset of first row.
    """
    subset_df = df[df['adversarial_priority']==priority]
    # first_row = subset_df.iloc[0]
    text = subset_df.iloc[0]["abusive_speech"]
    idx = subset_df.iloc[0]["Rep_ID"]
    return text, idx

def main(input_data_path, output_dir, outpue_filename, model_path):
    # To update
    # Loads data based on priority if not empty
    df = pandas.read_csv(input_data_path)
    if len(df[df['adversarial_priority']=='high']>0):
        text, idx = load_samples(df, 'high')
    elif len(df[df['adversarial_priority']=='medium']>0):
        text, idx = load_samples(df, 'medium')
    else:
        text, idx = load_samples(df, 'low')

    # Loads model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # TODO: 
    # get inputs (counterspeech response, and gold class) from users given a text loaded above

    # Makes model prediction
    prediction, prediction_prob = predict(text, response, gold_class, model, tokenizer)

    # update collected examples
    new_row = update_results(idx, text, response, gold_class, prediction, prediction_prob)

    # save data
    save_results(output_dir, outpue_filename)



if __name__ == "__main__":
    args = parse_args()

    main(args.input_data_path, args.output_dir, args.outpue_filename, args.model_path)
