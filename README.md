# Counter speech classification using adversarial training

This work aims at developing counter speech classifiers for mitigating online hate targeting at premium league footballers. It employs human-and-model-in-the-loop data collection over multiple rounds to test and improve model capabilities. 

## Folder overview

    └─ collection           -> Scripts for data collection from Twitter
    │
    ├── data                -> Folder to store raw data, labelled data, modeling data, 
    │                            and adversarial data
    │
    ├── dynabench           -> Task configuration for Dynabench
    │
    ├── dynalab             -> Model handler for Dynalab
    │
    ├── experiments         -> Folder to store logs and models
    │
    ├── scripts             -> Bash scripts for training models
    │
    └── src                 -> Source codes for data preprocessing and model training
        │
        └── data_preprocessing    ->  Scripts for rangling/analysing raw and anntated data,
        │                             creating modeling data
        │
        └── train                 ->  Scripts for model training


## Collect abusive tweets and their replies
The scripts in the folder `collection` contain example codes we use to gather potential abusive tweets and their replies. The example tweets and replies are stored as json files in [the directory](https://github.com/Turing-Online-Safety-Codebase/counterspeech_adversarial/tree/main/data/twitter_plf_data/twitter_plf_raw/plf_replies).

## Data preprosessing
The tweets and replies should be parsed and put converted into a single csv file which will then be used for annotation.

### Transform the collected data into long format
```
python src/data_preprocessing/get_reply.py 
```

This will run through each abusive tweet and all their replies in [the directory](https://github.com/Turing-Online-Safety-Codebase/counterspeech_adversarial/tree/main/data/twitter_plf_data/twitter_plf_raw/plf_replies), and parse necessary info that will then be stored in long data format (i.e. [plf_replies_v5.csv](https://github.com/Turing-Online-Safety-Codebase/counterspeech_adversarial/blob/main/data/twitter_plf_data/twitter_plf_raw/plf_replies_v5.csv)).

## Model training and evaluation
Once the data is labelled, we can start to train counter speech classifiers and collect dynamic adversarial data over multiple iterations. To train a counter speech classifier, run

```
bash ./scripts/train_model.sh
```

You can specify various training parameters when calling the script.

## Update adversarial examples to training data
To add new adversarial examples to training data after each iteration, run:

```
python src/data_preprocessing/add_adversaril_data.py \
    --current_batch <current_iteration> \
    --adversarial_data <filename_of_adversarial_data>
```

## Running and deallocating on Azure VM

The `run_python_then_dealloc.sh` shell script provides a simple way to have an Azure VM automatically deallocate after a python script finishes.

Edit the name of the python script to be run, the name of the log file to write to, and the name of the VM to deallocate, to match your use case. 

Before running it for the first time, you must run the following command:

`chmod +x run_python_then_dealloc.sh`

Then, to run the script in the background (so it will continue to run after exiting the VM), run the following:

`nohup ./run_python_then_dealloc.sh &`
