#!/bin/bash

export TASK_NAME=cs-generation 
export MODEL_PATH=gpt2
export MODEL_TYPE=gpt2
export TRAIN_FILE=data/final_modeling_data/multisource_adversarial_generation_train_final.txt
export VAL_FILE=data/final_modeling_data/val_cs_not_abusive.txt
export NUM_EPOCHS=3
export OUTPUT_DIR=experiments/models/cs_generation/gpt2_trained_on_hscs_all_sources_plf_adversarial_final
export RUN_NAME=gpt2_trained_on_hscs_all_sources_plf_adversarial_final
export TEST_FILE=data/final_modeling_data/multisource_generation_test_final.txt
export OUTPUT_RAW_FILE_PATH=$OUTPUT_DIR/raw_generated_data.txt
export OUTPUT_CLEAN_FILE_PATH=$OUTPUT_DIR/clean_generated_data.txt

# train model
echo "pre-train model on general domain data"
python src/train/cs_generation.py \
    --report_to None \
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
    --evaluation_strategy steps \
    --load_best_model_at_end True \
    --output_dir $OUTPUT_DIR


python src/evaluation/cs_generation_test.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL_PATH \
  --test_file $TEST_FILE \
  --output_raw_file_path $OUTPUT_RAW_FILE_PATH \
  --output_clean_file_path $OUTPUT_CLEAN_FILE_PATH \
  --length 50 \
  --p 0.9 \
  --num_return_sequences 1