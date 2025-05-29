#!/bin/bash

# Training script for MLA with DeepSeek-V3 FP8 optimizations
# This script demonstrates how to use the FP8 implementation

# Default values
MODEL_SIZE="${1:-small}"
BATCH_SIZE="${2:-8}"
BLOCK_SIZE="${3:-2048}"
OUTPUT_DIR="${4:-out_mla_fp8_deepseek}"
USE_FP8="${5:-1}"
GPU_ARCH="${6:-auto}"  # auto, hopper, ada, ampere

# Detect GPU architecture if auto
if [ "$GPU_ARCH" = "auto" ]; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    if [[ "$GPU_NAME" == *"H100"* ]] || [[ "$GPU_NAME" == *"H200"* ]]; then
        GPU_ARCH="hopper"
    elif [[ "$GPU_NAME" == *"4090"* ]] || [[ "$GPU_NAME" == *"6000 Ada"* ]]; then
        GPU_ARCH="ada"
    else
        GPU_ARCH="ampere"
    fi
    echo "Detected GPU architecture: $GPU_ARCH"
fi

# Set FP8 flags based on architecture
if [ "$USE_FP8" = "1" ]; then
    case "$GPU_ARCH" in
        "hopper")
            FP8_FLAGS="--use_fp8 --fp8_mla_params"
            echo "Using full FP8 optimizations for Hopper architecture"
            ;;
        "ada")
            FP8_FLAGS="--use_fp8"
            echo "Using FP8 with MLA params in FP16 for Ada architecture"
            ;;
        *)
            FP8_FLAGS=""
            echo "FP8 not supported on $GPU_ARCH, using BF16"
            ;;
    esac
else
    FP8_FLAGS=""
    echo "FP8 disabled, using BF16"
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# Log configuration
echo "Training MLA with DeepSeek FP8 optimizations"
echo "Model size: $MODEL_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Output directory: $OUTPUT_DIR"
echo "GPU architecture: $GPU_ARCH"
echo "FP8 flags: $FP8_FLAGS"

# Run training
python run_train.py \
    --model_type mla \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 4 \
    --block_size $BLOCK_SIZE \
    --n_layer 12 \
    --n_head 12 \
    --learning_rate 6e-4 \
    --min_lr 6e-5 \
    --warmup_iters 100 \
    --weight_decay 0.1 \
    --beta2 0.95 \
    --max_iters 50000 \
    --val_interval 1000 \
    --val_iters 100 \
    --sample_interval 1000 \
    --ckpt_interval 5000 \
    --dtype bfloat16 \
    --grad_clip 1.0 \
    --output_dir $OUTPUT_DIR \
    --log_dir $OUTPUT_DIR/logs \
    --wandb_project gptoughts-mla-fp8-deepseek \
    --torch_compile 1 \
    --flash_attention 1 \
    --q_lora_rank 0 \
    --kv_lora_rank 512 \
    --qk_nope_head_dim 128 \
    --qk_rope_head_dim 64 \
    --v_head_dim 128 \
    --mscale 1.0 \
    --rope_factor 1.0 \
    --dropout 0.0 \
    --seed 42 \
    --memory_checkpointing 1 \
    --deterministic 0 \
    --zero_stage 2 \
    --optim_bits 32 \
    --sharding_strategy FULL_SHARD \
    --mixed_precision bf16 \
    --max_grad_norm 1.0 \
    --offload_to_cpu 0 \
    --fp8_tile_size 128 \
    $FP8_FLAGS

echo "Training completed!"