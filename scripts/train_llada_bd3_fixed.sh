#!/bin/bash

# Train LLaDA model with BD3 configuration (fixed)
# This script uses proper optimizer and parameters for BD3 training

MODEL_SIZE=${1:-medium}
BATCH_SIZE=${2:-4}  # Réduit pour plus de stabilité
BLOCK_SIZE=${3:-2048}
OUTPUT_DIR=${4:-out_llada_bd3}
OUTPUT_ROOT=${OUTPUT_ROOT:-ouputs}
OUTPUT_NAME="$OUTPUT_DIR"
OUTPUT_DIR="$OUTPUT_ROOT/$OUTPUT_NAME"
mkdir -p "$OUTPUT_DIR"
RESUME=${5:-false}

echo "Training LLaDA with BD3 configuration (fixed)..."
echo "Model size: $MODEL_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Output dir: $OUTPUT_DIR"
echo "Resume: $RESUME"

# Set resume options
RESUME_ARGS=""
if [ "$RESUME" = "true" ] || [ "$RESUME" = "1" ]; then
    RESUME_ARGS="--init_from resume"
    echo "Will attempt to resume from last checkpoint in $OUTPUT_DIR"
fi

# Run training with BD3-optimized settings
python run_train.py \
    --model_type llada \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --precision bf16-mixed \
    --optimize_attention \
    --preallocate_memory \
    --grad_clip 1.0 \
    --use_dyt \
    --use_fp8 \
    --compile \
    --learning_rate 5e-5 \
    --optimizer_type lion \
    --weight_decay 0.1 \
    --warmup_iters 500 \
    --max_iters 10000000 \
    --eval_interval_steps 1000 \
    --log_interval_steps 10 \
    --gradient_accumulation_steps 2 \
    --use_bd3_training \
    --bd3_block_length 64 \
    --bd3_beta 0.3 \
    --bd3_omega 0.8 \
    --disable_entropy_regularization \
    $RESUME_ARGS

# Notes:
# - Using AdamW instead of Lion for stability with diffusion models
# - Increased warmup for stable training
# - Added gradient accumulation for effective larger batch
# - BD3 specific parameters for noise schedule
# - Disabled entropy regularization to reduce instability
