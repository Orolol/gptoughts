#!/bin/bash

# Train SWAN-GPT (hybrid NoPE + SWA-RoPE) with optimized defaults
# - Interleaves 1 global NoPE layer with 3 sliding-window RoPE layers per cycle
# - Applies logarithmic attention scaling for robust length extrapolation
# - Enables CUDA optimizations and bf16 mixed precision by default

MODEL_SIZE=${1:-medium}
BATCH_SIZE=${2:-8}
BLOCK_SIZE=${3:-2048}
OUTPUT_DIR=${4:-out_swan_optimized}
OUTPUT_ROOT=${OUTPUT_ROOT:-ouputs}
OUTPUT_NAME="$OUTPUT_DIR"
OUTPUT_DIR="$OUTPUT_ROOT/$OUTPUT_NAME"
mkdir -p "$OUTPUT_DIR"
RESUME=${5:-false}

echo "Training SWAN-GPT with optimized settings..."
echo "Model size: $MODEL_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Output dir: $OUTPUT_DIR"
echo "Resume: $RESUME"

# Resume handling
RESUME_ARGS=""
if [ "$RESUME" = "true" ] || [ "$RESUME" = "1" ]; then
    RESUME_ARGS="--init_from resume"
    echo "Will attempt to resume from last checkpoint in $OUTPUT_DIR"
fi

# Size presets (matches SWAN paper defaults)
case "$MODEL_SIZE" in
    small)
        N_LAYER=${N_LAYER_OVERRIDE:-16}
        N_HEAD=${N_HEAD_OVERRIDE:-16}
        N_EMBD=${N_EMBD_OVERRIDE:-1024}
        ;;
    medium|base|1b)
        N_LAYER=${N_LAYER_OVERRIDE:-24}
        N_HEAD=${N_HEAD_OVERRIDE:-16}
        N_EMBD=${N_EMBD_OVERRIDE:-1536}
        ;;
    large)
        N_LAYER=${N_LAYER_OVERRIDE:-28}
        N_HEAD=${N_HEAD_OVERRIDE:-24}
        N_EMBD=${N_EMBD_OVERRIDE:-2048}
        ;;
    xl|8b)
        N_LAYER=${N_LAYER_OVERRIDE:-32}
        N_HEAD=${N_HEAD_OVERRIDE:-32}
        N_EMBD=${N_EMBD_OVERRIDE:-4096}
        ;;
    *)
        echo "Unknown MODEL_SIZE '$MODEL_SIZE'. Expected one of: small, medium, large, xl." >&2
        exit 1
        ;;
esac

echo "Resolved architecture -> layers: $N_LAYER, heads: $N_HEAD, hidden: $N_EMBD"

# SWAN-specific knobs (overridable via environment)
GLOBAL_LAYERS_PER_CYCLE=${GLOBAL_LAYERS_PER_CYCLE:-1}
LOCAL_LAYERS_PER_CYCLE=${LOCAL_LAYERS_PER_CYCLE:-3}
SWA_WINDOW=${SWA_WINDOW:-512}
LOGIT_SCALE_BASE=${LOGIT_SCALE_BASE:-128.0}
LOGIT_SCALE_WINDOW=${LOGIT_SCALE_WINDOW:-128}
LOGIT_SCALE_OFFSET=${LOGIT_SCALE_OFFSET:-0}
LOGIT_SCALE_MIN=${LOGIT_SCALE_MIN:-1.0}
LOGIT_SCALE_MAX=${LOGIT_SCALE_MAX:-}
APPLY_LOGIT_SCALE_DURING_TRAINING=${APPLY_LOGIT_SCALE_DURING_TRAINING:-false}

LOGIT_ARGS=(
    --logit_scale_base "$LOGIT_SCALE_BASE"
    --logit_scale_window "$LOGIT_SCALE_WINDOW"
    --logit_scale_offset "$LOGIT_SCALE_OFFSET"
    --logit_scale_min "$LOGIT_SCALE_MIN"
)
if [ -n "$LOGIT_SCALE_MAX" ]; then
    LOGIT_ARGS+=(--logit_scale_max "$LOGIT_SCALE_MAX")
fi
if [ "$APPLY_LOGIT_SCALE_DURING_TRAINING" = "true" ] || [ "$APPLY_LOGIT_SCALE_DURING_TRAINING" = "1" ]; then
    LOGIT_ARGS+=(--apply_logit_scale_during_training)
fi

# Launch training
python run_train.py \
    --model_type swan \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --precision bf16-mixed \
    --optimize_attention \
    --preallocate_memory \
    --attention_backend sdpa \
    --grad_clip 1.0 \
    --learning_rate 3e-5 \
    --optimizer_type lion \
    --weight_decay 0.1 \
    --warmup_iters 400 \
    --max_iters 10000000 \
    --eval_interval_steps 1000 \
    --log_interval_steps 10 \
    --gradient_accumulation_steps 4 \
    --compile \
    --n_layer $N_LAYER \
    --n_head $N_HEAD \
    --n_embd $N_EMBD \
    --global_layers_per_cycle $GLOBAL_LAYERS_PER_CYCLE \
    --local_layers_per_cycle $LOCAL_LAYERS_PER_CYCLE \
    --swa_window $SWA_WINDOW \
    --ratio_kv 1 \
    "${LOGIT_ARGS[@]}" \
    $RESUME_ARGS

exit 0
