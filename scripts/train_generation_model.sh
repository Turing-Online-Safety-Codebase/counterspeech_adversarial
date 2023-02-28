#!/bin/bash

export TASK_NAME=cs-generation 
export MODEL_PATH=facebook/opt-125m
export TRAIN_FILE=data/final_modeling_data/pre_trained_hscs_all_sources.txt
export VAL_FILE=data/final_modeling_data/val_cs_not_abusive.txt
export NUM_EPOCHS=3
export OUTPUT_DIR=experiments/models/iter0/opt_125m_trained_on_hscs_all_sources
export RUN_NAME=opt_125m_trained_on_hscs_all_sources

# train model
echo "Language model training for counter speech generation "
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
    --evaluation_strategy steps \
    --load_best_model_at_end True \
    --output_dir $OUTPUT_DIR