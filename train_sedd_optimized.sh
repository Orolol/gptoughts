#!/bin/bash

# Train SEDD model with optimized memory settings
# Similar to train_llada_optimized.sh but configured for SEDD

MODEL_SIZE=${1:-small}
BATCH_SIZE=${2:-8}
BLOCK_SIZE=${3:-2048}
OUTPUT_DIR=${4:-out_sedd_optimized}
RESUME=${5:-false}

echo "Training SEDD model with optimized memory settings..."
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

# Run training with specific optimizations
python run_train.py \
    --model_type sedd \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --precision bf16-mixed \
    --optimize_attention \
    --preallocate_memory \
    --grad_clip 1.0 \
    --learning_rate 5e-5 \
    --optimizer_type lion \
    --weight_decay 0.1 \
    --warmup_iters 200 \
    --max_iters 10000000 \
    --eval_interval_steps 1000 \
    --log_interval_steps 10 \
    --gradient_accumulation_steps 1 \
    --compile \
    $RESUME_ARGS

# Note: SEDD doesn't use BD3-specific flags. The model trains with its Score Entropy loss.

