#!/bin/bash

# Train MLA model with optimized memory settings
# This script demonstrates how to train without gradient checkpointing
# for more stable VRAM usage (at the cost of using more memory)

MODEL_SIZE=${1:-small}
BATCH_SIZE=${2:-8}
BLOCK_SIZE=${3:-2048}
OUTPUT_DIR=${4:-out_mla_optimized}

echo "Training MLA model with optimized memory settings..."
echo "Model size: $MODEL_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Output dir: $OUTPUT_DIR"

# Run training with specific optimizations
python run_train.py \
    --model_type mla \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --precision bf16-mixed \
    --optimize_attention \
    --preallocate_memory \
    --grad_clip 1.0 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --warmup_iters 200 \
    --max_iters 100000 \
    --eval_interval_steps 1000 \
    --log_interval_steps 10 \
    --gradient_accumulation_steps 1

# Note: To disable gradient checkpointing, you would need to modify the model config
# in _create_mla_config to set use_gradient_checkpointing=False