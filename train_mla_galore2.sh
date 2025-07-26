#!/bin/bash

# Train MLA model with GaLore2 optimizer (fast randomized SVD + improved memory efficiency)
# GaLore2 features: 15x faster subspace updates, FSDP support, quantized projections

MODEL_SIZE=${1:-small}
BATCH_SIZE=${2:-8}
BLOCK_SIZE=${3:-2048}
GALORE_RANK=${4:-128}  # Low-rank dimension (lower = more memory efficient)
UPDATE_GAP=${5:-200}  # Frequency of projection updates (T in paper)
SCALE=${6:-0.25}  # Scaling factor (alpha in paper)
PROJ_TYPE=${7:-std}  # Projection type: std (SVD), random, 1bit, 2bit
QUANTIZE_PROJ=${8:-0}  # 0=none, 1=1-bit, 2=2-bit quantization
OUTPUT_DIR=${9:-out_mla_galore2}
RESUME=${10:-false}

echo "Training MLA model with GaLore2 optimizer (fast randomized SVD)..."
echo "Model size: $MODEL_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "GaLore rank: $GALORE_RANK"
echo "Update gap: $UPDATE_GAP"
echo "Scale: $SCALE"
echo "Projection type: $PROJ_TYPE"
echo "Quantize projection: $QUANTIZE_PROJ"
echo "Output dir: $OUTPUT_DIR"
echo "Resume: $RESUME"

# Set resume options
RESUME_ARGS=""
if [ "$RESUME" = "true" ] || [ "$RESUME" = "1" ]; then
    RESUME_ARGS="--init_from resume"
    echo "Will attempt to resume from last checkpoint in $OUTPUT_DIR"
fi

# Set quantization args if specified
QUANTIZE_ARGS=""
if [ "$QUANTIZE_PROJ" != "0" ]; then
    QUANTIZE_ARGS="--galore_quantize_proj $QUANTIZE_PROJ"
    echo "Using ${QUANTIZE_PROJ}-bit projection quantization"
fi

# Run training with GaLore2
python run_train.py \
    --model_type mla \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --precision bf16-mixed \
    --optimizer_type galore2 \
    --galore_rank $GALORE_RANK \
    --galore_update_proj_gap $UPDATE_GAP \
    --galore_scale $SCALE \
    --galore_proj_type $PROJ_TYPE \
    $QUANTIZE_ARGS \
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
    --compile \
    --use_dyt \
    --use_fp8 \
    $RESUME_ARGS

# Usage examples:
# ./train_mla_galore2.sh small 16 2048 128                    # Small model, rank 128, standard SVD
# ./train_mla_galore2.sh medium 8 2048 64 200 0.25 std       # Medium model, rank 64, standard SVD
# ./train_mla_galore2.sh large 4 2048 256 150 0.3 std        # Large model, rank 256, faster updates
# ./train_mla_galore2.sh small 16 2048 128 200 0.25 random   # Random projection (no SVD)
# ./train_mla_galore2.sh small 16 2048 128 200 0.25 std 1    # 1-bit quantized projections
# ./train_mla_galore2.sh small 16 2048 128 200 0.25 std 2    # 2-bit quantized projections

# GaLore2 improvements:
# - Fast randomized SVD: 15x faster than full SVD in GaLore1
# - Quantized projections: Further memory savings with 1-bit or 2-bit projections
# - Better FSDP integration: Works efficiently with distributed training
# - Lower memory usage: Improved memory efficiency through optimized updates