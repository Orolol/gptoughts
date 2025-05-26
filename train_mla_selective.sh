#!/bin/bash

# Train MLA model with Selective Attention
# Selective attention dynamically selects which tokens to attend to based on importance scores

MODEL_SIZE=${1:-small}
BATCH_SIZE=${2:-8}
BLOCK_SIZE=${3:-2048}
SELECTION_RATIO=${4:-0.5}  # Ratio of tokens to select (0.0 to 1.0)
SELECTION_METHOD=${5:-top_k}  # Method: top_k, threshold, or gumbel
OUTPUT_DIR=${6:-out_mla_selective}

echo "Training MLA model with Selective Attention..."
echo "Model size: $MODEL_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Selection ratio: $SELECTION_RATIO"
echo "Selection method: $SELECTION_METHOD"
echo "Output dir: $OUTPUT_DIR"

# Run training with Selective Attention
python run_train.py \
    --model_type mla_selective \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --precision bf16-mixed \
    --selection_ratio $SELECTION_RATIO \
    --selection_method $SELECTION_METHOD \
    --selection_temperature 1.0 \
    --optimize_attention \
    --preallocate_memory \
    --optimizer_type galore-8bit \
    --galore_rank 128 \
    --galore_update_proj_gap 200 \
    --galore_scale 0.25 \
    --galore_proj_type std \
    --grad_clip 0.0 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --warmup_iters 200 \
    --max_iters 10000 \
    --eval_interval_steps 1000 \
    --log_interval_steps 10 \
    --gradient_accumulation_steps 1 \
    --use_dyt \
    --use_fp8

# Usage examples:
# ./train_mla_selective.sh small 16 2048 0.5 top_k      # Small model, select top 50% tokens
# ./train_mla_selective.sh medium 8 2048 0.3 top_k      # Medium model, select top 30% tokens (more sparse)
# ./train_mla_selective.sh large 4 2048 0.7 threshold   # Large model, threshold-based selection
# ./train_mla_selective.sh small 16 2048 0.5 gumbel     # Small model, differentiable selection

# Notes:
# - Selective attention reduces computational cost by attending to fewer tokens
# - Lower selection ratio = more aggressive selection, faster but potentially less accurate
# - top_k: Select the k most important tokens (deterministic)
# - threshold: Select tokens above a dynamic threshold
# - gumbel: Use Gumbel-Softmax for differentiable selection
# - Recommended for long sequences or when computational efficiency is critical