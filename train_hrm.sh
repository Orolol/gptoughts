#!/bin/bash

# Training script for HRM model
# Usage: ./train_hrm.sh [size] [batch_size] [block_size] [output_dir] [compile] [resume]

# Default parameters
SIZE=${1:-"medium"}
BATCH_SIZE=${2:-8}
BLOCK_SIZE=${3:-4096}
OUTPUT_DIR=${4:-"out_hrm"}
COMPILE=${5:-1}  # 1 to enable torch.compile, 0 to disable
RESUME=${6:-false}

echo "============================================"
echo "Training HRM Model"
echo "============================================"
echo "Model size: $SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Output directory: $OUTPUT_DIR"
echo "Compile: $COMPILE"
echo "Resume: $RESUME"
echo "============================================"

# Detect GPU
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    print(f'GPU: {name} (Compute Capability {cc[0]}.{cc[1]})')
"

# Resume option
RESUME_ARGS=""
if [ "$RESUME" = "true" ] || [ "$RESUME" = "1" ]; then
    RESUME_ARGS="--init_from resume"
    echo "Will attempt to resume from last checkpoint in $OUTPUT_DIR"
fi

# Build command
CMD="python run_train.py \
    --model_type hrm \
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
    --label_smoothing 0.0 \
    --ponder_loss_weight 0.01 \
    --halt_bias_init -2.0 \
    --hrm_deq_one_step \
    --hrm_use_deep_supervision \
    --hrm_n_supervision_segments 3 \
    $RESUME_ARGS"

if [ "$COMPILE" -eq 1 ]; then
    CMD="$CMD \
        --compile \
        --precision bf16-mixed"
fi

# Create output directory
mkdir -p $OUTPUT_DIR

echo ""
echo "Running command:"
echo "$CMD"
echo ""

exec $CMD

