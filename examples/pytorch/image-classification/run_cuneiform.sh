#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

OUTPUT_DIR="/trunk2/tracytian/transformers/examples/pytorch/image-classification/run_collection/"
LR=2e-5
NUM_EPOCHS=3
BATCH_SIZE=32
EVAL_BATCH_SIZE=32

python cuneiform.py \
    --model_name_or_path "google/vit-base-patch16-224-in21k" \
    --train_dir "/trunk2/datasets/cuneiform_tracy/final/collection_pando/train/" \
    --validation_dir "/trunk2/datasets/cuneiform_tracy/final/collection_pando/valid/" \
    --test_dir "/trunk2/datasets/cuneiform_tracy/final/collection_pando/test/" \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --learning_rate $LR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end False\
    --seed 1337

# epoch or steps?