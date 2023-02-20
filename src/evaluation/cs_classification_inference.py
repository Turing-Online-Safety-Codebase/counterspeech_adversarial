from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import datetime
import pandas

def predict(hs, response, gold_class, model, tokenizer):
    """Model prediction.

    Args:
        hs (str): a sequence of text
        response (str): a sequence of response to the hs
        gold_class (int): true label for the response
        model (obj:`Model`): A model to use for classification.
        tokenizer (obj:`Tokenizer`): A ``Tokenizer`` of the current model.

    Returns:
        model prediction [0,1] and the probability of the model prediction
    Examples:
        hs_3 = "[PLAYER] I'd delete your profile you utter scumbag mate."
        sequence_3 = "[USER] [PLAYER] yes, agree"
        predict(hs_3, sequence_3, 0)
    """
    label_mapping_dict = {1: 'not_counterspeech', 0: 'counterspeech'}

    predict = tokenizer(hs, response, return_tensors="pt")
    
    logits = model(**predict).logits
    scores = torch.softmax(logits, dim=1)
    pred_prob_list = scores.tolist()[0]
    pred_indice = scores.argmax().item()

    if pred_indice == gold_class:
        print(f"You didn't fool the model. The model predicted {label_mapping_dict[pred_indice]}, and you say {label_mapping_dict[gold_class]}")
        print(f"Confidence counter speech is {int(round(pred_prob_list[0] * 100))}%.\nConfidence not_counter_speech is {int(round(pred_prob_list[1] * 100))}%\n")
    else:
        print(f"You fool the model. The model predicted {label_mapping_dict[pred_indice]}, and you say {label_mapping_dict[gold_class]}")
        print(f"Confidence counter speech is {int(round(pred_prob_list[0] * 100))}%.\nConfidence not_counter_speech is {int(round(pred_prob_list[1] * 100))}%\n")
    
    return pred_indice, pred_prob_list[pred_indice]