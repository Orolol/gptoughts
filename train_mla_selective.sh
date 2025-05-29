#!/bin/bash

# Train MLA model with Selective Attention
# Based on "Selective Attention Improves Transformer" (arXiv:2410.02703)

MODEL_SIZE=${1:-small}
BATCH_SIZE=${2:-8}
BLOCK_SIZE=${3:-2048}
OUTPUT_DIR=${4:-out_mla_selective}
USE_FP8=${5:-0}
RESUME=${6:-false}

echo "Training MLA model with Selective Attention..."
echo "Model size: $MODEL_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Output dir: $OUTPUT_DIR"
echo "Use FP8: $USE_FP8"
echo "Resume: $RESUME"
echo ""
echo "Note: Selective attention is enabled by default in MLASelective model"
echo "This provides 16-47x memory reduction with optimal performance"
echo "Uses PyTorch FlexAttention for efficient custom attention patterns"

# Set resume options
RESUME_ARGS=""
if [ "$RESUME" = "true" ] || [ "$RESUME" = "1" ]; then
    RESUME_ARGS="--init_from resume"
    echo "Will attempt to resume from last checkpoint in $OUTPUT_DIR"
fi

# Set FP8 flag
FP8_FLAG=""
if [ "$USE_FP8" -eq 1 ]; then
    FP8_FLAG="--use_fp8"
fi

# Run training with Selective Attention
python run_train.py \
    --model_type mla_selective \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --precision bf16-mixed \
    --optimize_attention \
    --preallocate_memory \
    --optimizer_type lion \
    --grad_clip 1.0 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --warmup_iters 200 \
    --max_iters 10000000 \
    --eval_interval_steps 1000 \
    --log_interval_steps 10 \
    --gradient_accumulation_steps 1 \
    --use_dyt \
    --compile \
    $FP8_FLAG \
    $RESUME_ARGS

# Usage examples:
# ./train_mla_selective.sh small 16 2048 out_mla_selective 0    # Small model without FP8
# ./train_mla_selective.sh medium 8 2048 out_mla_selective 0     # Medium model without FP8
# ./train_mla_selective.sh large 4 2048 out_mla_selective 1      # Large model with FP8

# Notes:
# - Selective attention is always enabled for MLASelective model
# - Uses one attention head's output as selection function (no extra parameters)
# - Provides 16-47x memory reduction in attention computation
# - Now works WITH SDPA/Flash Attention (minimal speed overhead)
# - Recommended for very long sequences where memory is critical