#!/bin/bash

export TASK_NAME=cs-generation 
export MODEL_PATH=google/flan-t5-base
export MODEL_TYPE=t5-base
export TRAIN_FILE=data/final_modeling_data/multisource_generation_train_final.csv
export VAL_FILE=data/final_modeling_data/multisource_generation_test_final.csv
export NUM_EPOCHS=3
export OUTPUT_DIR=experiments/models/cs_generation/flan-t5-base_trained_on_hscs_all_sources_plf_final
export RUN_NAME=flan-t5-base_trained_on_hscs_all_sources_plf_final
export TEST_FILE=data/final_modeling_data/raw_multisource_generation_test_final.txt
export OUTPUT_RAW_FILE_PATH=$OUTPUT_DIR/raw_generated_data.txt
export OUTPUT_CLEAN_FILE_PATH=$OUTPUT_DIR/clean_generated_data.txt

# # train model
# echo "Language model training for counter speech generation "
# python src/train/cs_generation_s2slm.py \
#     --report_to wandb \
#     --model_name_or_path $MODEL_PATH \
#     --train_file $TRAIN_FILE \
#     --validation_file $VAL_FILE \
#     --run_name $RUN_NAME \
#     --learning_rate 2e-5 \
#     --num_train_epochs $NUM_EPOCHS \
#     --do_train \
#     --per_device_train_batch_size 2 \
#     --evaluation_strategy epoch \
#     --save_strategy epoch \
#     --load_best_model_at_end True \
#     --output_dir $OUTPUT_DIR \
#     --seq2seqlm


python src/evaluation/cs_generation_test_s2slm.py \
  --model_type $MODEL_TYPE \
  --seq2seqlm \
  --model_name_or_path $OUTPUT_DIR \
  --test_file $TEST_FILE \
  --output_raw_file_path $OUTPUT_RAW_FILE_PATH \
  --output_clean_file_path $OUTPUT_CLEAN_FILE_PATH \
  --length 50 \
  --p 0.9 \
  --num_return_sequences 1  