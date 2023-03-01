#!/bin/bash

export TASK_NAME=cs-generation
export MODEL_PATH=experiments/models/iter0/opt_125m_trained_on_hscs_all_sources
export RUN_NAME=evaluation_opt_125m_trained_on_hscs_all_sources_plf
export MODEL_TYPE=opt
export TEST_FILE=data/final_modeling_data/test_cs_not_abusive.txt
export OUTPUT_RAW_FILE_PATH=$MODEL_PATH/raw_generated_data.txt
export OUTPUT_CLEAN_FILE_PATH=$MODEL_PATH/clean_generated_data.txt

python src/evaluation/cs_generation_test.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL_PATH \
  --test_file $TEST_FILE \
  --output_raw_file_path $OUTPUT_RAW_FILE_PATH \
  --output_clean_file_path $OUTPUT_CLEAN_FILE_PATH \
  --length 50 \
  --p 0.9 \
  --num_return_sequences 1  