#!/bin/bash

# Script to train Masked Diffusion Model (MDM)

# Default parameters
SIZE="${1:-small}"
BATCH_SIZE="${2:-8}"
BLOCK_SIZE="${3:-2048}"
OUTPUT_DIR="${4:-out_mdm}"
OUTPUT_ROOT=${OUTPUT_ROOT:-ouputs}
OUTPUT_NAME="$OUTPUT_DIR"
OUTPUT_DIR="$OUTPUT_ROOT/$OUTPUT_NAME"
mkdir -p "$OUTPUT_DIR"
USE_FP8="${5:-0}"

# Display configuration
echo "Training MDM model with:"
echo "  Size: $SIZE"
echo "  Batch size: $BATCH_SIZE"
echo "  Block size: $BLOCK_SIZE"
echo "  Output directory: $OUTPUT_DIR"
echo "  FP8 enabled: $USE_FP8"

# Setup environment
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Base command
CMD="python run_train.py \
    --model_type mdm \
    --size $SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --num_workers 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-4 \
    --weight_decay 0.1 \
    --max_iters 100000 \
    --warmup_iters 2000 \
    --lr_decay_iters 100000 \
    --grad_clip 1.0 \
    --eval_interval_steps 1000 \
    --log_interval_steps 10 \
    --precision bf16-mixed \
    --compile"

# Add FP8 flag if requested
if [ "$USE_FP8" -eq 1 ]; then
    CMD="$CMD --use_fp8"
fi

# Execute training
echo "Executing: $CMD"
$CMD
