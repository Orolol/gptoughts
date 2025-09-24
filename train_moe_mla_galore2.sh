#!/bin/bash

# Train MOE-MLA model with GaLore2 optimizer
# Combines Mixture of Experts with Multi-head Latent Attention
# Features: Shared expert weights (75%), FP8 training, DyT normalization, GaLore2 optimizer

MODEL_SIZE=${1:-small}
BATCH_SIZE=${2:-8}
BLOCK_SIZE=${3:-2048}
NUM_EXPERTS=${4:-8}
EXPERTS_PER_TOKEN=${5:-2}
SHARED_RATIO=${6:-0.75}
GALORE_RANK=${7:-128}
UPDATE_GAP=${8:-200}
OUTPUT_DIR=${9:-out_moe_mla_galore2}
RESUME=${10:-false}

echo "Training MOE-MLA model with GaLore2 optimizer..."
echo "Model size: $MODEL_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Number of experts: $NUM_EXPERTS"
echo "Experts per token: $EXPERTS_PER_TOKEN"
echo "Shared weight ratio: $SHARED_RATIO"
echo "GaLore rank: $GALORE_RANK"
echo "Update gap: $UPDATE_GAP"
echo "Output dir: $OUTPUT_DIR"
echo "Resume: $RESUME"

# Set resume options
RESUME_ARGS=""
if [ "$RESUME" = "true" ] || [ "$RESUME" = "1" ]; then
    RESUME_ARGS="--init_from resume"
    echo "Will attempt to resume from last checkpoint in $OUTPUT_DIR"
fi

# Run training with MOE-MLA + GaLore2
python run_train.py \
    --model_type moe_mla \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --num_experts $NUM_EXPERTS \
    --experts_per_token $EXPERTS_PER_TOKEN \
    --shared_weight_ratio $SHARED_RATIO \
    --output_dir $OUTPUT_DIR \
    --precision bf16-mixed \
    --optimizer_type adamw \
    --optimize_attention \
    --preallocate_memory \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --warmup_iters 200 \
    --max_iters 1000000 \
    --eval_interval_steps 1000 \
    --log_interval_steps 10 \
    --gradient_accumulation_steps 1 \
    --compile \
    --use_dyt \
    --use_fp8 \
    $RESUME_ARGS

# Usage examples:
# ./train_moe_mla_galore2.sh small 16 2048                              # Small model, default settings
# ./train_moe_mla_galore2.sh medium 8 2048 16 4                         # Medium model, 16 experts, top-4
# ./train_moe_mla_galore2.sh large 4 2048 32 4 0.8                      # Large model, 32 experts, 80% shared
# ./train_moe_mla_galore2.sh small 16 2048 8 2 0.75 64                  # Small model, rank 64 GaLore
# ./train_moe_mla_galore2.sh xl 2 4096 64 8 0.9 256 150                 # XL model, 64 experts, 90% shared

# Key features:
# - Mixture of Experts (MoE) with configurable number of experts
# - Multi-head Latent Attention (MLA) for efficient attention computation
# - Shared expert weights to reduce parameter count (default 75% shared)
# - FP8 training support for H100/H200 GPUs
# - Dynamic Tanh (DyT) normalization for ~8% speedup
# - GaLore2 optimizer with fast randomized SVD
# - Automatic mixed precision training (BF16)
# - Gradient checkpointing for memory efficiency