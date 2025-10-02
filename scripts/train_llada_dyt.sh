#!/bin/bash

# Train LLaDA model with DynTanh normalization
# Usage: ./train_llada_dyt.sh [size] [batch_size] [block_size] [output_dir]

# Default values
SIZE=${1:-small}
BATCH_SIZE=${2:-8}
BLOCK_SIZE=${3:-2048}
OUTPUT_DIR=${4:-out_llada_dyt}
OUTPUT_ROOT=${OUTPUT_ROOT:-ouputs}
OUTPUT_NAME="$OUTPUT_DIR"
OUTPUT_DIR="$OUTPUT_ROOT/$OUTPUT_NAME"
mkdir -p "$OUTPUT_DIR"

# Mapping taille -> dimensions
case $SIZE in
    "small")
        N_LAYER=12
        N_HEAD=12
        N_EMBD=768
        ;;
    "medium")
        N_LAYER=24
        N_HEAD=16
        N_EMBD=1024
        ;;
    "large")
        N_LAYER=36
        N_HEAD=20
        N_EMBD=1280
        ;;
    "xl")
        N_LAYER=48
        N_HEAD=25
        N_EMBD=1600
        ;;
    *)
        echo "Invalid size. Choose from: small, medium, large, xl"
        exit 1
        ;;
esac

echo "Training LLaDA model with DynTanh normalization"
echo "Size: $SIZE (n_layer=$N_LAYER, n_head=$N_HEAD, n_embd=$N_EMBD)"
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Output directory: $OUTPUT_DIR"
echo "Using DynTanh normalization instead of RMSNorm"

# Create output directory

# Run training with DynTanh enabled
python run_train.py \
    --model_type llada \
    --n_layer $N_LAYER \
    --n_head $N_HEAD \
    --n_embd $N_EMBD \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --num_epochs 10 \
    --learning_rate 3e-4 \
    --min_lr 3e-5 \
    --weight_decay 0.1 \
    --warmup_iters 1000 \
    --lr_decay_iters 10000 \
    --optimizer_type apollo-mini \
    --grad_clip 1.0 \
    --use_dyt \
    --dyt_alpha_init 0.5 \
    --num_experts 8 \
    --k 2 \
    --use_bd3_training \
    --bd3_block_length 512 \
    --compile \
    --use_gradient_checkpointing \
    --checkpoint_dir $OUTPUT_DIR \
    --log_dir $OUTPUT_DIR/logs \
    --eval_interval 1000 \
    --save_interval 5000 \
    --log_interval_steps 100 \
