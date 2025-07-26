#!/bin/bash

# Train MLA model with GaLore optimizer (memory-efficient gradient low-rank projection)
# GaLore enables training larger models on consumer GPUs

MODEL_SIZE=${1:-small}
BATCH_SIZE=${2:-8}
BLOCK_SIZE=${3:-2048}
USE_8BIT=${4:-1}  # 0 for standard GaLore, 1 for 8-bit GaLore (default)
GALORE_RANK=${5:-128}  # Low-rank dimension (lower = more memory efficient, but less expressive)
OUTPUT_DIR=${6:-out_mla_galore}
RESUME=${7:-false}

# Set optimizer type based on 8-bit flag
if [ "$USE_8BIT" = "1" ]; then
    OPTIMIZER="galore-8bit"
    echo "Using 8-bit GaLore optimizer (most memory efficient)"
else
    OPTIMIZER="galore"
    echo "Using standard GaLore optimizer"
fi

echo "Training MLA model with GaLore optimizer..."
echo "Model size: $MODEL_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "GaLore rank: $GALORE_RANK"
echo "Output dir: $OUTPUT_DIR"
echo "Resume: $RESUME"

# Set resume options
RESUME_ARGS=""
if [ "$RESUME" = "true" ] || [ "$RESUME" = "1" ]; then
    RESUME_ARGS="--init_from resume"
    echo "Will attempt to resume from last checkpoint in $OUTPUT_DIR"
fi

# Run training with GaLore
python run_train.py \
    --model_type mla \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --precision bf16-mixed \
    --optimizer_type $OPTIMIZER \
    --galore_rank $GALORE_RANK \
    --galore_update_proj_gap 200 \
    --galore_scale 0.25 \
    --galore_proj_type std \
    --optimize_attention \
    --preallocate_memory \
    --grad_clip 1.0 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --warmup_iters 200 \
    --max_iters 1000000 \
    --eval_interval_steps 1000 \
    --log_interval_steps 10 \
    --gradient_accumulation_steps 1 \
    --use_dyt \
    --use_fp8 \
    $RESUME_ARGS

# Usage examples:
# ./train_mla_galore.sh small 16 2048 1 128    # Small model, 8-bit GaLore, rank 128
# ./train_mla_galore.sh medium 8 2048 1 64     # Medium model, 8-bit GaLore, rank 64 (more aggressive compression)
# ./train_mla_galore.sh large 4 2048 1 256     # Large model, 8-bit GaLore, rank 256 (less compression)
# ./train_mla_galore.sh small 16 2048 0 128    # Small model, standard GaLore, rank 128

# Notes:
# - GaLore applies gradient low-rank projection to reduce optimizer memory
# - Lower rank = more memory savings but potentially slower convergence
# - 8-bit GaLore provides additional memory savings through quantization
# - Recommended for training on consumer GPUs (e.g., RTX 4090)
# - Works best with larger batch sizes when memory permits