#!/bin/bash

# Simple training script for MLA-LLaDA without masking (standard LM training)
# To test if the model can learn at all

# Default parameters
MODEL_SIZE=${1:-small}
BATCH_SIZE=${2:-8}  # Larger batch size
BLOCK_SIZE=${3:-512}  # Shorter sequences
OUTPUT_DIR=${4:-out_mla_llada_simple}
OUTPUT_ROOT=${OUTPUT_ROOT:-ouputs}
OUTPUT_NAME="$OUTPUT_DIR"
OUTPUT_DIR="$OUTPUT_ROOT/$OUTPUT_NAME"
mkdir -p "$OUTPUT_DIR"

# Create output directory

# Export environment variables
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=${HF_TOKEN:-""}

# Build command for simple training
CMD="python run_train.py \
    --model_type mla_llada \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --learning_rate 5e-4 \
    --weight_decay 0.1 \
    --warmup_iters 500 \
    --lr_decay_iters 50000 \
    --min_lr 5e-5 \
    --max_iters 10000 \
    --eval_interval_steps 500 \
    --log_interval_steps 10 \
    --grad_clip 1.0 \
    --gradient_accumulation_steps 1 \
    --num_workers 2 \
    --dropout 0.1 \
    --precision bf16-mixed \
    --optimizer_type adamw"

# MLA-LLaDA specific - disable masking for simple LM training
CMD="$CMD \
    --mask_ratio_min 0.0 \
    --mask_ratio_max 0.0 \
    --num_diffusion_steps 1"

# Print configuration
echo "=========================================="
echo "MLA-LLaDA Simple Training (No Masking)"
echo "=========================================="
echo "Model Size: $MODEL_SIZE"
echo "Batch Size: $BATCH_SIZE"
echo "Block Size: $BLOCK_SIZE"
echo "Output Directory: $OUTPUT_DIR"
echo "Learning Rate: 5e-4"
echo "Gradient Clipping: 1.0"
echo "Dropout: 0.1"
echo "Masking: DISABLED"
echo "=========================================="

# Execute training
echo "Starting simple training without masking..."
$CMD
