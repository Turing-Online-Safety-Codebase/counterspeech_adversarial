#!/bin/bash

export TASK_NAME=path-to-hans
export MODEL_PATH=bert-base-cased
export TRAIN_FILE=data/final_modeling_data/train_labelled.csv
export VAL_FILE=data/final_modeling_data/val_labelled.csv
export TEST_FILE=data/final_modeling_data/test_labelled.csv
export NUM_EPOCHS=3
export OUTPUT_DIR=experiments/models/iter_0/run1_bert-base-cased/
export RUN_NAME=iter0_run1_bert-base-cased_epoch3_2e-5

mkdir experiments/experiment_logs/$RUN_NAME

# train model
echo "training model"
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
    --per_device_train_batch_size 16 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --logging_steps 100 \
    --logging_first_step True \
    --evaluation_strategy steps \
    --load_best_model_at_end True \
    --metric_for_best_model 'f1' \
    --output_dir $OUTPUT_DIR