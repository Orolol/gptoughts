#!/bin/bash

# Train MLA model with different optimizer choices
# This script demonstrates how to use different optimizers with MLA models

MODEL_SIZE=${1:-small}
BATCH_SIZE=${2:-8}
BLOCK_SIZE=${3:-2048}
OPTIMIZER=${4:-adamw}  # Options: adamw, lion, apollo, apollo-mini, galore, galore-8bit
OUTPUT_DIR=${5:-out_mla_${OPTIMIZER}}
OUTPUT_ROOT=${OUTPUT_ROOT:-ouputs}
OUTPUT_NAME="$OUTPUT_DIR"
OUTPUT_DIR="$OUTPUT_ROOT/$OUTPUT_NAME"
mkdir -p "$OUTPUT_DIR"
RESUME=${6:-false}

echo "Training MLA model with ${OPTIMIZER} optimizer..."
echo "Model size: $MODEL_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Optimizer: $OPTIMIZER"
echo "Output dir: $OUTPUT_DIR"
echo "Resume: $RESUME"

# Set resume options
RESUME_ARGS=""
if [ "$RESUME" = "true" ] || [ "$RESUME" = "1" ]; then
    RESUME_ARGS="--init_from resume"
    echo "Will attempt to resume from last checkpoint in $OUTPUT_DIR"
fi

# Run training with specified optimizer
python run_train.py \
    --model_type mla \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --precision bf16-mixed \
    --optimizer_type $OPTIMIZER \
    --optimize_attention \
    --preallocate_memory \
    --grad_clip 1.0 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --warmup_iters 200 \
    --max_iters 10000 \
    --eval_interval_steps 500 \
    --log_interval_steps 10 \
    --gradient_accumulation_steps 1 \
    $RESUME_ARGS

# Usage examples:
# ./train_mla_with_optimizer.sh small 8 2048 adamw       # Use AdamW (default, most stable)
# ./train_mla_with_optimizer.sh small 8 2048 lion        # Use Lion optimizer (faster convergence)
# ./train_mla_with_optimizer.sh small 8 2048 apollo      # Use APOLLO (memory efficient)
# ./train_mla_with_optimizer.sh small 8 2048 apollo-mini # Use APOLLO-mini (most memory efficient)
# ./train_mla_with_optimizer.sh small 8 2048 galore      # Use GaLore (gradient low-rank projection)
# ./train_mla_with_optimizer.sh small 8 2048 galore-8bit # Use 8-bit GaLore (very memory efficient)
