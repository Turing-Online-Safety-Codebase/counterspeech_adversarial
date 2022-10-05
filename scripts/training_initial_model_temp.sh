#!/bin/bash

export TASK_NAME=path-to-hans
export MODEL_PATH=microsoft/deberta-v3-small
export TRAIN_FILE=./toy_examples/train_data_copy.csv
export VAL_FILE=./toy_examples/val_data_copy.csv
export TEST_FILE=./toy_examples/test_data_copy.csv
export NUM_EPOCHS=3
export OUTPUT_DIR=./result_logs/iter_1/run1_deberta/
export RUN_NAME=run5_xlm-roberta-base_epoch10_2e-5_static_it

# training initial model
echo "training initial model"
python run_cs_classification_v2.py \
    --report_to none \
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
    --load_best_model_at_end True \
    --metric_for_best_model 'f1' \
    --output_dir $OUTPUT_DIR