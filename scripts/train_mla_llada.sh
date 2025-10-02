#!/bin/bash

# Training script for MLA-LLaDA model
# Combines Multi-head Latent Attention with LLaDA diffusion-based generation

# Default parameters
MODEL_SIZE=${1:-small}
BATCH_SIZE=${2:-8}
BLOCK_SIZE=${3:-2048}
OUTPUT_DIR=${4:-out_mla_llada}
OUTPUT_ROOT=${OUTPUT_ROOT:-ouputs}
OUTPUT_NAME="$OUTPUT_DIR"
OUTPUT_DIR="$OUTPUT_ROOT/$OUTPUT_NAME"
mkdir -p "$OUTPUT_DIR"
USE_FP8=${5:-0}
USE_DYT=${6:-1}  # Dynamic Tanh enabled by default

# Create output directory

# Export environment variables
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=${HF_TOKEN:-""}

# Build command
CMD="python run_train.py \
    --model_type mla_llada \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --learning_rate 3e-5 \
    --weight_decay 0.1 \
    --warmup_iters 2000 \
    --lr_decay_iters 600000 \
    --min_lr 3e-5 \
    --max_iters 600000 \
    --eval_interval_steps 5000 \
    --log_interval_steps 10 \
    --grad_clip 0 \
    --gradient_accumulation_steps 4 \
    --num_workers 4 \
    --dropout 0.1 \
    --precision bf16-mixed \
    --attention_backend flash \
    --compile \
    --optimize_attention \
    --optimizer_type adamw"

# Add FP8 flags if enabled
if [ "$USE_FP8" -eq 1 ]; then
    echo "Enabling FP8 precision..."
    CMD="$CMD --use_fp8"
fi

# Add DynamicTanh if enabled
if [ "$USE_DYT" -eq 1 ]; then
    echo "Enabling DynamicTanh normalization..."
    CMD="$CMD --use_dyt --dyt_alpha_init 0.5"
fi

# MLA-LLaDA specific parameters
CMD="$CMD \
    --remasking_strategy low_confidence \
    --mask_ratio_min 0.15 \
    --mask_ratio_max 0.85"

# Print configuration
echo "=================================="
echo "MLA-LLaDA Training Configuration"
echo "=================================="
echo "Model Size: $MODEL_SIZE"
echo "Batch Size: $BATCH_SIZE"
echo "Block Size: $BLOCK_SIZE"
echo "Output Directory: $OUTPUT_DIR"
echo "FP8 Enabled: $USE_FP8"
echo "DynamicTanh Enabled: $USE_DYT"
echo "=================================="

# Execute training
echo "Starting training..."
$CMD
