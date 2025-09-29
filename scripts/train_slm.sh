#!/bin/bash

# SLM-MoE-MLA Training Script
# Small Language Model with Mixture of Experts and Multi-head Latent Attention
# Optimized for ~200M parameters with ultra-high weight sharing (90%)

set -e  # Exit on any error

# Parse command line arguments
SIZE=${1:-small}
BATCH_SIZE=${2:-16}
BLOCK_SIZE=${3:-2048}
OUTPUT_DIR=${4:-out_slm}
OUTPUT_ROOT=${OUTPUT_ROOT:-ouputs}
OUTPUT_NAME="$OUTPUT_DIR"
OUTPUT_DIR="$OUTPUT_ROOT/$OUTPUT_NAME"
mkdir -p "$OUTPUT_DIR"
USE_FP8=${5:-0}
NUM_EXPERTS=${6:-8}
EXPERTS_PER_TOKEN=${7:-1}

# Configuration
MODEL_TYPE="slm"
LEARNING_RATE="3e-4"
MIN_LR="3e-6"
MAX_ITERS="5000000"
WARMUP_ITERS="2000"
LR_DECAY_ITERS="40000"
EVAL_INTERVAL="500"
LOG_INTERVAL="10"
GRAD_CLIP="1.0"
WEIGHT_DECAY="0.1"
BETA1="0.9"
BETA2="0.95"

# Data configuration
DATALOADER_TYPE="packed"

# Advanced SLM configuration
SHARED_WEIGHT_RATIO="0.90"
USE_DYT=true
DYT_ALPHA_INIT="0.5"
ROUTER_Z_LOSS_COEF="0.001"

# Compute configuration
if [ "$USE_FP8" -eq 1 ]; then
    PRECISION="16-mixed"  # FP16 mixed precision
    USE_FP8_FLAG="--use_fp8"
    echo "Using FP8 precision for memory efficiency"
else
    PRECISION="bf16-mixed"  # BFloat16 mixed precision
    USE_FP8_FLAG=""
    echo "Using BFloat16 mixed precision"
fi

# Memory and performance optimizations
GRADIENT_ACCUMULATION_STEPS="1"
USE_COMPILE=true
USE_OPTIMIZE_ATTENTION=true

# Environment setup
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Check if pyenv is activated
if ! command -v python3 &> /dev/null || [[ "$(python3 -c 'import sys; print(sys.prefix)')" == "/usr" ]]; then
    echo "Warning: Consider activating pyenv environment (pyenv activate 5090) for optimal dependencies"
    echo "Continuing with system Python..."
fi

# Create output directory

# Print configuration
echo "=============================================="
echo "SLM-MoE-MLA Training Configuration"
echo "=============================================="
echo "Model Type: $MODEL_TYPE"
echo "Model Size: $SIZE"
echo "Batch Size: $BATCH_SIZE"
echo "Block Size: $BLOCK_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Max Iterations: $MAX_ITERS"
echo "Output Directory: $OUTPUT_DIR"
echo "Precision: $PRECISION"
echo "Number of Experts: $NUM_EXPERTS"
echo "Experts per Token: $EXPERTS_PER_TOKEN"
echo "Shared Weight Ratio: $SHARED_WEIGHT_RATIO"
echo "Use Dynamic Tanh: $USE_DYT"
echo "Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "=============================================="

# Check for CUDA availability
if ! python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')" 2>/dev/null; then
    echo "Error: Unable to detect PyTorch with CUDA support"
    echo "Please ensure PyTorch is installed with CUDA support"
    exit 1
fi

# Run training
echo "Starting SLM-MoE-MLA training..."
echo "Command: python3 run_train.py with the following parameters:"

# Build command dynamically
CMD="python3 run_train.py"
CMD="$CMD --model_type=$MODEL_TYPE"
CMD="$CMD --size=$SIZE"
CMD="$CMD --batch_size=$BATCH_SIZE"
CMD="$CMD --block_size=$BLOCK_SIZE"
CMD="$CMD --learning_rate=$LEARNING_RATE"
CMD="$CMD --min_lr=$MIN_LR"
CMD="$CMD --max_iters=$MAX_ITERS"
CMD="$CMD --warmup_iters=$WARMUP_ITERS"
CMD="$CMD --lr_decay_iters=$LR_DECAY_ITERS"
CMD="$CMD --eval_interval_steps=$EVAL_INTERVAL"
CMD="$CMD --log_interval_steps=$LOG_INTERVAL"
CMD="$CMD --grad_clip=$GRAD_CLIP"
CMD="$CMD --weight_decay=$WEIGHT_DECAY"
CMD="$CMD --beta1=$BETA1"
CMD="$CMD --beta2=$BETA2"
CMD="$CMD --output_dir=$OUTPUT_DIR"
CMD="$CMD --dataloader_type=$DATALOADER_TYPE"
CMD="$CMD --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
CMD="$CMD --precision=$PRECISION"
CMD="$CMD --num_experts=$NUM_EXPERTS"
CMD="$CMD --experts_per_token=$EXPERTS_PER_TOKEN"
CMD="$CMD --shared_weight_ratio=$SHARED_WEIGHT_RATIO"
CMD="$CMD --dyt_alpha_init=$DYT_ALPHA_INIT"
CMD="$CMD --router_z_loss_coef=$ROUTER_Z_LOSS_COEF"

# Add boolean flags conditionally
if [ "$USE_COMPILE" = true ]; then
    CMD="$CMD --compile"
fi

if [ "$USE_OPTIMIZE_ATTENTION" = true ]; then
    CMD="$CMD --optimize_attention"
fi

if [ "$USE_DYT" = true ]; then
    CMD="$CMD --use_dyt"
fi

# Add FP8 flag if enabled
if [ "$USE_FP8" -eq 1 ]; then
    CMD="$CMD --use_fp8"
fi

# Execute the command
echo "Executing: $CMD"
eval $CMD

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "=============================================="
    echo "SLM-MoE-MLA training completed successfully!"
    echo "Model saved to: $OUTPUT_DIR"
    echo "=============================================="
    
    # Print parameter count and model info
    echo "Getting model information..."
    python3 -c "
import torch
import sys
import os
sys.path.append('$PWD')

# Try to load and analyze the trained model
try:
    from models.models.slm_moe_mla import create_slm_model
    
    model = create_slm_model('$SIZE', 
        num_experts=$NUM_EXPERTS,
        experts_per_token=$EXPERTS_PER_TOKEN,
        shared_weight_ratio=$SHARED_WEIGHT_RATIO,
        vocab_size=$VOCAB_SIZE,
        block_size=$BLOCK_SIZE
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'Total parameters: {total_params/1e6:.2f}M')
    print(f'Trainable parameters: {trainable_params/1e6:.2f}M')
    
    # Calculate effective parameters (accounting for MoE routing)
    config = model.config
    effective_params = total_params
    
    # Estimate actual compute parameters based on expert usage
    expert_params = 0
    shared_params = 0
    
    for name, param in model.named_parameters():
        if 'expert_up_proj' in name or 'expert_down_proj' in name:
            expert_params += param.numel()
        else:
            shared_params += param.numel()
    
    # Effective params = shared + (expert_params / num_experts * experts_per_token)
    effective_expert_params = expert_params / config.num_experts * config.experts_per_token
    effective_total = shared_params + effective_expert_params
    
    print(f'Shared parameters: {shared_params/1e6:.2f}M ({shared_params/total_params*100:.1f}%)')
    print(f'Expert parameters: {expert_params/1e6:.2f}M ({expert_params/total_params*100:.1f}%)')
    print(f'Effective parameters: {effective_total/1e6:.2f}M')
    print(f'Parameter efficiency: {effective_total/total_params*100:.1f}%')
    
except Exception as e:
    print(f'Error analyzing model: {e}')
    "
    
else
    echo "=============================================="
    echo "Training failed with exit code: $?"
    echo "Check the logs above for error details"
    echo "=============================================="
    exit 1
fi
