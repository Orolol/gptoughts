#!/bin/bash
# Training script for ParScale-MLA model using run_train.py

# Default values
SIZE=${1:-small}
BATCH_SIZE=${2:-8}
BLOCK_SIZE=${3:-2048}
OUTPUT_DIR=${4:-out_parscale}
OUTPUT_ROOT=${OUTPUT_ROOT:-ouputs}
OUTPUT_NAME="$OUTPUT_DIR"
OUTPUT_DIR="$OUTPUT_ROOT/$OUTPUT_NAME"
mkdir -p "$OUTPUT_DIR"
PARALLEL_STREAMS=${5:-8}
STAGE=${6:-1}  # 1 for base training, 2 for ParScale training
BASE_CHECKPOINT=${7:-""}  # Path to base model checkpoint for stage 2
RESUME=${8:-false}  # Resume from last checkpoint

echo "Training ParScale-MLA model"
echo "Size: $SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Output dir: $OUTPUT_DIR"
echo "Parallel streams: $PARALLEL_STREAMS"
echo "Training stage: $STAGE"
echo "Resume: $RESUME"

# Determine model type based on stage
if [ "$STAGE" -eq 1 ]; then
    MODEL_TYPE="mla"
    echo "Stage 1: Training base MLA model"
else
    MODEL_TYPE="parscale_mla"
    echo "Stage 2: Training ParScale components with frozen base model"
fi

# Common parameters
COMMON_PARAMS="
    --model_type $MODEL_TYPE
    --size $SIZE
    --batch_size $BATCH_SIZE
    --block_size $BLOCK_SIZE
    --gradient_accumulation_steps 8
    --learning_rate 3e-4
    --min_lr 3e-5
    --max_iters 10000
    --weight_decay 0.1
    --beta1 0.9
    --beta2 0.95
    --grad_clip 1.0
    --decay_lr
    --warmup_iters 1000
    --log_interval_steps 10
    --eval_interval_steps 500
    --output_dir $OUTPUT_DIR
    --precision bf16-mixed
    --use_lightning
"

# ParScale specific parameters (only for stage 2)
if [ "$STAGE" -eq 2 ]; then
    if [ -z "$BASE_CHECKPOINT" ]; then
        echo "Error: BASE_CHECKPOINT must be provided for stage 2 training"
        exit 1
    fi
    
    PARSCALE_PARAMS="
        --parallel_streams $PARALLEL_STREAMS
        --prefix_length 48
        --latent_prefix_length 16
        --aggregator_epsilon 0.1
        --diversity_weight 0.1
        --use_dynamic_inference
        --complexity_threshold 0.5
        --training_stage 2
        --base_checkpoint $BASE_CHECKPOINT
        --freeze_base_in_stage2
        --max_iters 2000
        --init_from resume
        --resume_ckpt_path $BASE_CHECKPOINT
    "
else
    # Stage 1 - check for resume
    if [ "$RESUME" = "true" ] || [ "$RESUME" = "1" ]; then
        PARSCALE_PARAMS="
            --init_from resume
        "
        echo "Will attempt to resume from last checkpoint in $OUTPUT_DIR"
    else
        PARSCALE_PARAMS="
            --init_from scratch
        "
    fi
fi

# Create output directory

# Run training
python run_train.py \
    $COMMON_PARAMS \
    $PARSCALE_PARAMS \
    2>&1 | tee $OUTPUT_DIR/training.log

echo "Training completed. Results saved to $OUTPUT_DIR"
