#!/bin/bash

# Training script for NSA (Native Sparse Attention) model
# Usage: ./train_nsa.sh [size] [batch_size] [block_size] [output_dir]

# Default parameters
SIZE=${1:-"medium"}
BATCH_SIZE=${2:-8}
BLOCK_SIZE=${3:-4096}
OUTPUT_DIR=${4:-"out_nsa"}
OUTPUT_ROOT=${OUTPUT_ROOT:-ouputs}
OUTPUT_NAME="$OUTPUT_DIR"
OUTPUT_DIR="$OUTPUT_ROOT/$OUTPUT_NAME"
mkdir -p "$OUTPUT_DIR"

# Optional parameters
LEARNING_RATE=${5:-5e-4}
WARMUP_ITERS=${6:-2000}
MAX_ITERS=${7:-50000}
GRADIENT_ACCUMULATION=${8:-4}

# Check if HuggingFace token is set
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. Some tokenizers may not work."
    echo "Set it with: export HF_TOKEN=your_token"
fi

echo "============================================"
echo "Training NSA Model"
echo "============================================"
echo "Model size: $SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Output directory: $OUTPUT_DIR"
echo "Learning rate: $LEARNING_RATE"
echo "Warmup iterations: $WARMUP_ITERS"
echo "Max iterations: $MAX_ITERS"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "============================================"

# Create output directory

# Build the command
CMD="python run_train.py \
    --model_type nsa \
    --size $SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --num_workers 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --learning_rate $LEARNING_RATE \
    --warmup_iters $WARMUP_ITERS \
    --lr_decay_iters 20000 \
    --max_iters $MAX_ITERS \
    --eval_interval_steps 2000 \
    --log_interval_steps 10 \
    --grad_clip 1.0 \
    --weight_decay 0.1 \
    --beta1 0.9 \
    --beta2 0.95 \
    --min_lr 3e-5 \
    --dropout 0.0 \
    --precision bf16-mixed \
    --keep_checkpoints 3"

# Add optional optimizations
echo ""
echo "Additional options you can add to the command:"
echo "  --compile                  # Enable torch.compile for ~20% speedup"
echo "  --use_fp8                  # Enable FP8 precision (requires compatible GPU)"
echo "  --use_dyt                  # Use Dynamic Tanh normalization"
echo "  --wandb_project PROJECT    # Enable WandB logging"
echo "  --devices N                # Use N GPUs (default: all available)"
echo ""

# Ask if user wants to add compile
read -p "Enable torch.compile optimization? (y/n) [n]: " enable_compile
if [[ "$enable_compile" =~ ^[Yy]$ ]]; then
    CMD="$CMD --compile"
    echo "✓ torch.compile enabled"
fi

# Ask if user wants to use FP8
read -p "Enable FP8 precision? (requires H100/H200/Blackwell) (y/n) [n]: " enable_fp8
if [[ "$enable_fp8" =~ ^[Yy]$ ]]; then
    CMD="$CMD --use_fp8"
    echo "✓ FP8 precision enabled"
fi

# Ask if user wants DyT
read -p "Use Dynamic Tanh normalization? (y/n) [n]: " enable_dyt
if [[ "$enable_dyt" =~ ^[Yy]$ ]]; then
    CMD="$CMD --use_dyt --dyt_alpha_init 0.5"
    echo "✓ Dynamic Tanh enabled"
fi

# Ask about WandB
read -p "Enable WandB logging? (y/n) [n]: " enable_wandb
if [[ "$enable_wandb" =~ ^[Yy]$ ]]; then
    read -p "Enter WandB project name: " wandb_project
    if [ -n "$wandb_project" ]; then
        CMD="$CMD --wandb_project $wandb_project"
        echo "✓ WandB logging enabled for project: $wandb_project"
    fi
fi

# Log the final command
echo ""
echo "Running command:"
echo "$CMD"
echo ""

# Execute training
exec $CMD
