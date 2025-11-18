#!/bin/bash

# Train Adaptive MoE model (MoE on Attention) with optimized defaults

MODEL_SIZE=${1:-medium}
BATCH_SIZE=${2:-8}
BLOCK_SIZE=${3:-2048}
OUTPUT_DIR=${4:-out_adaptive_moe}
OUTPUT_ROOT=${OUTPUT_ROOT:-ouputs}
OUTPUT_NAME="$OUTPUT_DIR"
OUTPUT_DIR="$OUTPUT_ROOT/$OUTPUT_NAME"
mkdir -p "$OUTPUT_DIR"
RESUME=${5:-false}

echo "Training Adaptive MoE Model (MoE on Attention)..."
echo "Model size: $MODEL_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Output dir: $OUTPUT_DIR"
echo "Resume: $RESUME"

RESUME_ARGS=()
if [ "$RESUME" = "true" ] || [ "$RESUME" = "1" ]; then
    RESUME_ARGS=(--init_from resume)
    echo "Will attempt to resume from last checkpoint in $OUTPUT_DIR"
fi

# Adjust k values based on model size
if [ "$MODEL_SIZE" = "small" ]; then
    K_PERIPHERAL="32"
    K_FOCAL="64"
    K_REFLECTIVE="128"
elif [ "$MODEL_SIZE" = "medium" ]; then
    K_PERIPHERAL="48"
    K_FOCAL="96"
    K_REFLECTIVE="192"
elif [ "$MODEL_SIZE" = "large" ]; then
    K_PERIPHERAL="64"
    K_FOCAL="128"
    K_REFLECTIVE="256"
elif [ "$MODEL_SIZE" = "xl" ]; then
    K_PERIPHERAL="80"
    K_FOCAL="160"
    K_REFLECTIVE="320"
fi

echo "K values: Peripheral=$K_PERIPHERAL, Focal=$K_FOCAL, Reflective=$K_REFLECTIVE"

python run_train.py \
    --model_type adaptive_moe \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --precision bf16-mixed \
    --optimize_attention \
    --preallocate_memory \
    --tokenizer_name "openai-community/gpt2" \
    --attention_backend sdpa \
    --grad_clip 1.0 \
    --learning_rate 6e-5 \
    --optimizer_type lion \
    --weight_decay 0.1 \
    --warmup_iters 400 \
    --max_iters 10000000 \
    --eval_interval_steps 1000 \
    --log_interval_steps 10 \
    --gradient_accumulation_steps 1 \
    --num_workers 8 \
    --k_peripheral $K_PERIPHERAL \
    --k_focal $K_FOCAL \
    --k_reflective $K_REFLECTIVE \
    --router_temperature 1.0 \
    "${RESUME_ARGS[@]}"

exit 0