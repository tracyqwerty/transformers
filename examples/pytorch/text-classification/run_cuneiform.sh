#!/bin/bash

# Activate the conda environment or virtual environment if needed
# source /path/to/your/env/bin/activate
# or
# conda activate your_env_name


export CUDA_VISIBLE_DEVICES=0 

# Set the paths to your dataset files
TRAIN_FILE="/trunk2/datasets/cuneiform_tracy/final/bert/train_data.jsonl"
VALID_FILE="/trunk2/datasets/cuneiform_tracy/final/bert/valid_data.jsonl"
TEST_FILE="/trunk2/datasets/cuneiform_tracy/final/bert/test_data.jsonl"

# Set the output directory where the model, predictions, and checkpoints will be saved
OUTPUT_DIR="/trunk2/tracytian/transformers/examples/pytorch/text-classification/run"

# Define the model to be used
MODEL_NAME_OR_PATH="google-bert/bert-base-uncased"

# Define batch size, number of epochs, etc.
BATCH_SIZE=16
EVAL_BATCH_SIZE=16
NUM_EPOCHS=3
SAVE_STEPS=1000
SEED=42

# Run the Python script with the specified arguments
python cuneiform.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --train_file $TRAIN_FILE \
    --validation_file $VALID_FILE \
    --test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR \
    --do_train \
    --do_eval \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --num_train_epochs $NUM_EPOCHS \
    --save_steps $SAVE_STEPS \
    --seed $SEED \
    --metric_name f1 \
    --evaluation_strategy steps
