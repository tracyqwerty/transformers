#!/bin/bash


OUTPUT_DIR="/trunk2/tracytian/transformers/examples/pytorch/image-classification/run_cuneiform/"
LR=2e-5
NUM_EPOCHS=1
BATCH_SIZE=16
EVAL_BATCH_SIZE=16

python cuneiform.py \
    --model_name_or_path "google/vit-base-patch16-224-in21k" \
    --train_dir "/trunk2/datasets/cuneiform_tracy/final/iid/full/train/" \
    --validation_dir "/trunk2/datasets/cuneiform_tracy/final/iid/full/valid/" \
    --test_dir "/trunk2/datasets/cuneiform_tracy/final/iid/full/test/" \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --learning_rate $LR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --seed 1337

# epoch or steps?