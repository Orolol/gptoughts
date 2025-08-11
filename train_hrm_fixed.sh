#!/bin/bash

# Fixed training script for HRM model with better hyperparameters
# Usage: ./train_hrm_fixed.sh [size] [batch_size] [block_size] [output_dir] [compile]

# Default parameters
SIZE=${1:-"medium"}
BATCH_SIZE=${2:-8}
BLOCK_SIZE=${3:-1024}
OUTPUT_DIR=${4:-"out_hrm_fixed"}
COMPILE=${5:-0}  # 0 by default for debugging

echo "============================================"
echo "Training HRM Model (Fixed Version)"
echo "============================================"
echo "Model size: $SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Output directory: $OUTPUT_DIR"
echo "Compile: $COMPILE"
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

# Build command with improved hyperparameters
CMD="python run_train.py \
    --model_type hrm \
    --size $SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --num_workers 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-3 \
    --min_lr 1e-5 \
    --warmup_iters 500 \
    --lr_decay_iters 20000 \
    --max_iters 50000 \
    --eval_interval_steps 500 \
    --log_interval_steps 10 \
    --grad_clip 1.0 \
    --weight_decay 0.01 \
    --label_smoothing 0.0 \
    --ponder_loss_weight 0.001 \
    --halt_bias_init -3.0 \
    --hrm_gradient_steps -1 \
    --hrm_use_act false \
    --hrm_use_deep_supervision false \
    --hrm_cycles_per_segment 1 \
    --hrm_steps_per_cycle 2 \
    --hrm_max_segments 1"

# Key changes:
# 1. Removed --hrm_deq_one_step (disables 1-step gradient approximation)
# 2. Added --hrm_gradient_steps -1 (use gradients for all steps)
# 3. Disabled ACT with --hrm_use_act false (simplify for debugging)
# 4. Increased learning rate to 1e-3 (was 5e-4)
# 5. Added gradient clipping at 1.0
# 6. Reduced weight decay to 0.01
# 7. Reduced ponder loss weight
# 8. Simplified recurrence (1 segment, fewer cycles)
# 9. Disabled deep supervision initially

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