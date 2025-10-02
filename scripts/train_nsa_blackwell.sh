#!/bin/bash

# Training script for NSA model with Blackwell optimizations
# Usage: ./train_nsa_blackwell.sh [size] [batch_size] [block_size] [output_dir] [use_blackwell] [profile] [resume]

# Default parameters
SIZE=${1:-"medium"}
BATCH_SIZE=${2:-8}
BLOCK_SIZE=${3:-4096}
OUTPUT_DIR=${4:-"out_nsa_blackwell"}
OUTPUT_ROOT=${OUTPUT_ROOT:-ouputs}
OUTPUT_NAME="$OUTPUT_DIR"
OUTPUT_DIR="$OUTPUT_ROOT/$OUTPUT_NAME"
mkdir -p "$OUTPUT_DIR"
USE_BLACKWELL=${5:-1}  # 1 to enable Blackwell optimizations, 0 to disable
PROFILE=${6:-0}  # 1 to enable profiling, 0 to disable
RESUME=${7:-false}

# Set environment variables for optimal performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="9.0;10.0"  # Hopper and Blackwell

echo "============================================"
echo "Training NSA Model with Blackwell Optimizations"
echo "============================================"
echo "Model size: $SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Output directory: $OUTPUT_DIR"
echo "Blackwell optimizations: $USE_BLACKWELL"
echo "Resume: $RESUME"
echo "============================================"

# Check if we're on a Blackwell GPU
python -c "
import torch
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    print(f'GPU: {name} (Compute Capability {cc[0]}.{cc[1]})')
    if cc[0] >= 10:
        print('✓ Blackwell architecture detected')
    elif cc[0] >= 9:
        print('✓ Hopper architecture detected')
    else:
        print('⚠ Older GPU - some optimizations may not be available')
"

# Set resume options
RESUME_ARGS=""
if [ "$RESUME" = "true" ] || [ "$RESUME" = "1" ]; then
    RESUME_ARGS="--init_from resume"
    echo "Will attempt to resume from last checkpoint in $OUTPUT_DIR"
fi

# Build the command
CMD="python run_train.py \
    --model_type nsa \
    --size $SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --num_workers 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-4 \
    --min_lr 3e-5 \
    --warmup_iters 2000 \
    --lr_decay_iters 20000 \
    --max_iters 50000 \
    --eval_interval_steps 2000 \
    --log_interval_steps 10 \
    --grad_clip 0 \
    --weight_decay 0.1 \
    $RESUME_ARGS"

# Add Blackwell-specific optimizations if enabled
if [ "$USE_BLACKWELL" -eq 1 ]; then
    CMD="$CMD \
        --use_fp8 \
        --compile \
        --precision bf16-mixed"
fi

# Create output directory

# Add profiling options if enabled
if [ "$PROFILE" = "1" ]; then
    CMD="$CMD \
    --profile \
    --profile_interval 50"
fi

# Log the command
echo ""
echo "Running command:"
echo "$CMD"
echo ""

# Execute training
exec $CMD
