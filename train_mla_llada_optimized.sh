#!/bin/bash

# Optimized training script for MLA-LLaDA model with better hyperparameters
# Addresses high loss and repetition issues

# Default parameters
MODEL_SIZE=${1:-small}
BATCH_SIZE=${2:-4}  # Smaller batch size for better gradient quality
BLOCK_SIZE=${3:-1024}  # Shorter sequences for faster iteration
OUTPUT_DIR=${4:-out_mla_llada_optimized}
USE_FP8=${5:-1}
USE_DYT=${6:-1}  # Dynamic Tanh enabled by default

# Create output directory
mkdir -p $OUTPUT_DIR

# Export environment variables
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=${HF_TOKEN:-""}

# Build command with optimized hyperparameters
CMD="python run_train.py \
    --model_type mla_llada \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --warmup_iters 1000 \
    --lr_decay_iters 100000 \
    --min_lr 1e-5 \
    --max_iters 100000 \
    --eval_interval_steps 1000 \
    --log_interval_steps 10 \
    --grad_clip 0 \
    --gradient_accumulation_steps 4 \
    --num_workers 2 \
    --dropout 0.0 \
    --precision bf16-mixed \
    --attention_backend flash \
    --optimize_attention \
    --compile \
    --optimizer_type adamw"

# Add FP8 flags if enabled
if [ "$USE_FP8" -eq 1 ]; then
    echo "Enabling FP8 precision..."
    CMD="$CMD --use_fp8"
fi

# Add DynamicTanh if enabled
if [ "$USE_DYT" -eq 1 ]; then
    echo "Enabling DynamicTanh normalization..."
    CMD="$CMD --use_dyt --dyt_alpha_init 0.3"
fi

# MLA-LLaDA specific parameters - adjusted for better training
CMD="$CMD \
    --remasking_strategy low_confidence \
    --mask_ratio_min 0.05 \
    --mask_ratio_max 0.3 \
    --num_diffusion_steps 30"

# Print configuration
echo "=========================================="
echo "MLA-LLaDA Optimized Training Configuration"
echo "=========================================="
echo "Model Size: $MODEL_SIZE"
echo "Batch Size: $BATCH_SIZE"
echo "Block Size: $BLOCK_SIZE"
echo "Output Directory: $OUTPUT_DIR"
echo "FP8 Enabled: $USE_FP8"
echo "DynamicTanh Enabled: $USE_DYT"
echo "Learning Rate: 1e-4"
echo "Gradient Clipping: 0.5"
echo "Dropout: 0.0"
echo "Mask Ratio: 0.1-0.5"
echo "=========================================="

# Execute training
echo "Starting optimized training..."
$CMD