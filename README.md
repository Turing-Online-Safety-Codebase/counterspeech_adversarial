# Counterspeech classification and generation using adversarial training

This work aims at developing counterspeech classifiers for mitigating online hate targeting at premium league footballers. It employs human-and-model-in-the-loop data collection over multiple rounds to test and improve model capabilities. 

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
        └── data_preprocessing    ->  Scripts for wrangling/analysing raw and annotated data,
        │                             creating modeling data
        │
        └── train                 ->  Scripts for model training

## Data creation
### Collect abusive tweets and their replies
The scripts in the folder `collection` contain example codes we use to gather potential abusive tweets and their replies.


## Counterspeech classification through adversarial training
### Model training and evaluation
Once the data is labelled, we can start to train counterspeech classifiers and collect dynamic adversarial data over multiple iterations. To train/evaluate a counterspeech classifier, run

```
python -m src.train.cs_classification \
    --report_to wandb \
    --model_name_or_path $MODEL_PATH \
    --train_file $TRAIN_FILE \
    --validation_file $VAL_FILE \
    --test_file $TEST_FILE \
    --run_name $RUN_NAME \
    --learning_rate 2e-5 \
    --num_train_epochs $NUM_EPOCHS \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 256 \
    --per_device_train_batch_size 32 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --logging_steps 100 \
    --logging_first_step True \
    --evaluation_strategy steps \
    --load_best_model_at_end True \
    --metric_for_best_model 'f1' \
    --output_dir $OUTPUT_DIR
```

You can specify various training parameters when calling the script.

### Evaluate models on a given dataset

```
python -m src.evaluation.cs_classification_predict.py \
    --model_name_or_path $MODEL_PATH \
    --test_file $TEST_FILE \
    --run_name $RUN_NAME \    
    --do_predict \
    --max_seq_length 256 \
    --pred_file $PRED_FILE \
    --output_dir $OUTPUT_DIR
```

## Counterspeech generation

### Model training and evaluation
To train/evaluate a counterspeech generator, run

```
python src/train/cs_generation.py \
    --report_to wandb \
    --model_name_or_path $MODEL_PATH \
    --train_file $TRAIN_FILE \
    --validation_file $VAL_FILE \
    --run_name $RUN_NAME \
    --learning_rate 2e-5 \
    --num_train_epochs $NUM_EPOCHS \
    --do_train \
    --per_device_train_batch_size 4 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --logging_steps 100 \
    --logging_first_step True \
    --save_strategy steps \
    --save_steps 2000 \
    --load_best_model_at_end True \
    --output_dir $OUTPUT_DIR
```


### Evaluate models on a given dataset

```
python src/evaluation/cs_generation_test.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL_PATH \
  --test_file $TEST_FILE \
  --out_file_path $OUTPUT_FILE_PATH \
  --length 50 \
  --p 0.9 \
  --num_return_sequences 1  
```


## Running and deallocating on Azure VM

The `run_python_then_dealloc.sh` shell script provides a simple way to have an Azure VM automatically deallocate after a python script finishes.

Edit the name of the python script to be run, the name of the log file to write to, and the name of the VM to deallocate, to match your use case. 

Before running it for the first time, you must run the following command:

`chmod +x run_python_then_dealloc.sh`

Then, to run the script in the background (so it will continue to run after exiting the VM), run the following:

`nohup ./run_python_then_dealloc.sh &`
