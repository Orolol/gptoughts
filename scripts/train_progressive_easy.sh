#!/bin/bash
# Easy Progressive Training Script for GPToughts
# 
# Usage: ./train_progressive_easy.sh [model_type] [size] [batch_size]
#
# Examples:
#   ./train_progressive_easy.sh mla small 8
#   ./train_progressive_easy.sh gpt medium 4
#   ./train_progressive_easy.sh llada small 16

set -e  # Exit on error

# Default values
MODEL_TYPE=${1:-"mla"}
SIZE=${2:-"small"}  
BATCH_SIZE=${3:-"8"}
DATASET=${4:-"apollo-mini"}
OUT_DIR=${5:-"out_progressive_${MODEL_TYPE}_${SIZE}"}

# Progressive training configuration
PROGRESSIVE_BLOCK_SIZES="256,512,1024,2048"
EPOCHS_PER_STAGE="3"
MAX_EPOCHS="24"  # 6 stages * 4 epochs each

# Learning configuration
LEARNING_RATE="3e-4"
LR_SCALE="0.8"

echo "=== Easy Progressive Training ==="
echo "Model: ${MODEL_TYPE} (${SIZE})"
echo "Batch size: ${BATCH_SIZE}"
echo "Block sizes: ${PROGRESSIVE_BLOCK_SIZES}"
echo "Epochs per stage: ${EPOCHS_PER_STAGE}"
echo "Max epochs: ${MAX_EPOCHS}"
echo "Dataset: ${DATASET}"
echo "Output: ${OUT_DIR}"
echo "================================="

# Detect if we have a GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected, using bf16-mixed precision"
    PRECISION="bf16-mixed"
    USE_COMPILE="--compile"
else
    echo "No GPU detected, using 32-bit precision"
    PRECISION="32"
    USE_COMPILE=""
fi

# Run progressive training
python train_progressive.py \
    --model_type "${MODEL_TYPE}" \
    --size "${SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --progressive_block_sizes "${PROGRESSIVE_BLOCK_SIZES}" \
    --progressive_epochs_per_stage "${EPOCHS_PER_STAGE}" \
    --progressive_lr_scale "${LR_SCALE}" \
    --use_position_interpolation \
    --learning_rate "${LEARNING_RATE}" \
    --max_epochs "${MAX_EPOCHS}" \
    --warmup_iters 1000 \
    --lr_decay_iters 200000 \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --dropout 0.1 \
    --decay_lr \
    --dataset "${DATASET}" \
    --dataloader_type "dynamic" \
    --out_dir "${OUT_DIR}" \
    --precision "${PRECISION}" \
    --accumulate_grad_batches 1 \
    --num_workers 4 \
    --early_stopping_patience 8 \
    --save_top_k 2 \
    --use_wandb \
    --wandb_project "gptoughts-progressive" \
    ${USE_COMPILE} \
    "$@"  # Pass any additional arguments

echo ""
echo "=== Progressive Training Completed ==="
echo "Results saved in: ${OUT_DIR}"
echo "Final model: ${OUT_DIR}/final_progressive_model.ckpt"
echo "Checkpoints: ${OUT_DIR}/checkpoints/"
echo "Stage checkpoints: ${OUT_DIR}/progressive_checkpoints/"
