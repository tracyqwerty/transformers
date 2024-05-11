#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

TRAIN_FILE="/trunk2/datasets/cuneiform_tracy/final/bert/train_data.jsonl"
VALID_FILE="/trunk2/datasets/cuneiform_tracy/final/bert/valid_data.jsonl"
TEST_FILE="/trunk2/datasets/cuneiform_tracy/final/bert/test_data.jsonl"

OUTPUT_DIR="/trunk2/tracytian/transformers/examples/pytorch/text-classification/run_cuneiform/"

MODEL_NAME_OR_PATH="google-bert/bert-base-uncased"

BATCH_SIZE=16
EVAL_BATCH_SIZE=16
NUM_EPOCHS=1
SAVE_STEPS=1000
SEED=42
LR=2e-5

# Run the Python script with the specified arguments
python cuneiform.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --train_file $TRAIN_FILE \
    --validation_file $VALID_FILE \
    --test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LR \
    --save_steps $SAVE_STEPS \
    --eval_strategy steps
