#!/bin/bash

export TASK_NAME=path-to-hans
export MODEL_PATH=experiments/models/iter_0/run2_final_deberta
export PRED_FILE=run2_final_deberta_predict
export TEST_FILE=data/final_modeling_data/test_labelled.csv
export OUTPUT_DIR=experiments/models/iter_0/run2_final_deberta/
export RUN_NAME=run2_final_deberta_predict

# evaluate model
echo "evaluating model"
python src/evaluation/predict.py \
    --model_name_or_path $MODEL_PATH \
    --test_file $TEST_FILE \
    --do_predict \
    --max_seq_length 256 \
    --pred_file $PRED_FILE \
    --output_dir $OUTPUT_DIR