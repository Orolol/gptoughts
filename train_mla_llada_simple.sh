#!/bin/bash

# Simple training script for MLA-LLaDA without compilation
# Usage: ./train_mla_llada_simple.sh [size] [batch_size] [block_size] [output_dir]

MODEL_SIZE=${1:-small}
BATCH_SIZE=${2:-4}
BLOCK_SIZE=${3:-512}
OUTPUT_DIR=${4:-out_mla_llada_simple}

# Create output directory
mkdir -p $OUTPUT_DIR

# Basic command without compilation
python run_train.py \
    --model_type mla_llada \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --learning_rate 3e-4 \
    --weight_decay 0.1 \
    --warmup_iters 1000 \
    --max_iters 10000 \
    --eval_interval_steps 500 \
    --log_interval_steps 10 \
    --grad_clip 1.0 \
    --gradient_accumulation_steps 4 \
    --dropout 0.1 \
    --precision bf16-mixed \
    --use_dyt \
    --remasking_strategy low_confidence \
    --optimizer_type adamw

echo "Training started with simple configuration:"
echo "- Model: MLA-LLaDA $MODEL_SIZE"
echo "- Batch Size: $BATCH_SIZE"
echo "- Block Size: $BLOCK_SIZE"
echo "- Output: $OUTPUT_DIR"