#!/bin/bash

# Train MLA model with Dynamic Tanh (DyT) normalization
# DyT replaces RMSNorm for ~8% speedup in training and inference

MODEL_SIZE=${1:-small}
BATCH_SIZE=${2:-8}
BLOCK_SIZE=${3:-2048}
OUTPUT_DIR=${4:-out_mla_dyt}
USE_FP8=${5:-0}
RESUME=${6:-false}

echo "Training MLA model with Dynamic Tanh (DyT) normalization..."
echo "Model size: $MODEL_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Output dir: $OUTPUT_DIR"
echo "Use FP8: $USE_FP8"
echo "Resume: $RESUME"

# Set resume options
RESUME_ARGS=""
if [ "$RESUME" = "true" ] || [ "$RESUME" = "1" ]; then
    RESUME_ARGS="--init_from resume"
    echo "Will attempt to resume from last checkpoint in $OUTPUT_DIR"
fi

# Base training command with DyT enabled
TRAIN_CMD="python run_train.py \
    --model_type mla \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --precision bf16-mixed \
    --use_dyt \
    --dyt_alpha_init 0.5 \
    --optimize_attention \
    --preallocate_memory \
    --grad_clip 1.0 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --warmup_iters 200 \
    --max_iters 10000000 \
    --eval_interval_steps 1000 \
    --log_interval_steps 10 \
    --gradient_accumulation_steps 1 \
    --optimizer_type lion \
    $RESUME_ARGS"

# Add FP8 flag if requested
if [ "$USE_FP8" -eq 1 ]; then
    TRAIN_CMD="$TRAIN_CMD --use_fp8"
fi

# Execute training
eval $TRAIN_CMD