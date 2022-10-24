%%bash 

export TASK_NAME=path-to-hans
export MODEL_PATH=microsoft/deberta-v3-small
export TRAIN_FILE=data/sample_modeling_data/train_labelled_clean.csv
export VAL_FILE=data/sample_modeling_data/val_labelled_clean.csv
export TEST_FILE=data/sample_modeling_data/test_labelled_clean.csv
export NUM_EPOCHS=3
export OUTPUT_DIR=experiments/models/iter_1/run4_parameter_tuning/
export RUN_NAME=run4_deberta-v3-small_parameter_tuning

# train model
echo "training model"
python src/train/cs_classification_parameter_tuning.py \
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
    --load_best_model_at_end True \
    --metric_for_best_model 'f1' \
    --output_dir $OUTPUT_DIR