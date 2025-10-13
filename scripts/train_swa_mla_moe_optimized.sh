#!/bin/bash

# Train SWA-MLA-MOE hybrid model with Mixture of Experts

MODEL_SIZE=${1:-small}
BATCH_SIZE=${2:-8}
BLOCK_SIZE=${3:-2048}
OUTPUT_DIR=${4:-out_swa_mla_moe}
OUTPUT_ROOT=${OUTPUT_ROOT:-ouputs}
OUTPUT_NAME="$OUTPUT_DIR"
OUTPUT_DIR="$OUTPUT_ROOT/$OUTPUT_NAME"
mkdir -p "$OUTPUT_DIR"
RESUME=${5:-false}
OPTIMIZER=${6:-lion}
STRATEGY=${7:-ddp_find_unused_parameters_false}
NUM_EXPERTS=${8:-8}
EXPERTS_PER_TOKEN=${9:-1}

echo "Training SWA-MLA-MOE hybrid model with Mixture of Experts..."
echo "Model size: $MODEL_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Output dir: $OUTPUT_DIR"
echo "Resume: $RESUME"
echo "Optimizer: $OPTIMIZER"
echo "Strategy: $STRATEGY"
echo "MoE: $NUM_EXPERTS experts, top-$EXPERTS_PER_TOKEN routing"
echo ""
if [ "$STRATEGY" = "fsdp" ]; then
    echo "Using FSDP (Fully Sharded Data Parallel):"
    echo "  - Shards model parameters across GPUs"
    echo "  - Eliminates VRAM imbalance between ranks"
    echo "  - Better memory efficiency than DDP"
else
    echo "Multi-GPU DDP optimizations active:"
    echo "  - Model initialized on CPU to prevent rank 1 duplication"
    echo "  - broadcast_buffers=False (prevents buffer duplication)"
    echo "  - bucket_cap_mb=10 (reduced gradient buckets)"
    echo "  - gradient_as_bucket_view=True (memory optimization)"
fi
echo ""

RESUME_ARGS=()
if [ "$RESUME" = "true" ] || [ "$RESUME" = "1" ]; then
    RESUME_ARGS=(--init_from resume)
    echo "Will attempt to resume from last checkpoint in $OUTPUT_DIR"
fi

python run_train.py \
    --model_type swa_mla_moe \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --precision bf16-mixed \
    --optimize_attention \
    --preallocate_memory \
    --attention_backend sdpa \
    --tokenizer_name "openai-community/gpt2" \
    --grad_clip 1.0 \
    --learning_rate 6e-5 \
    --optimizer_type $OPTIMIZER \
    --weight_decay 0.1 \
    --warmup_iters 400 \
    --max_iters 10000000 \
    --eval_interval_steps 1000 \
    --log_interval_steps 10 \
    --gradient_accumulation_steps 1 \
    --num_workers 8 \
    --compile \
    --swa_layers_per_cycle 2 \
    --mla_layers_per_cycle 1 \
    --swa_window 256 \
    --swa_sink_size 4 \
    --mla_q_lora_rank 0 \
    --mla_kv_lora_rank 256 \
    --mla_qk_nope_head_dim 128 \
    --mla_qk_rope_head_dim 64 \
    --mla_v_head_dim 128 \
    --num_experts $NUM_EXPERTS \
    --experts_per_token $EXPERTS_PER_TOKEN \
    --strategy $STRATEGY \
    "${RESUME_ARGS[@]}"

# Note: --compile removed for better DDP compatibility
# torch.compile can cause issues with DDP synchronization

# Usage examples:
# ./train_swa_mla_moe_optimized.sh small 16 2048                               # Default: 8 experts, top-2, AdamW + DDP
# ./train_swa_mla_moe_optimized.sh medium 16 2048 out false adamw ddp 16 2    # 16 experts, top-2
# ./train_swa_mla_moe_optimized.sh medium 16 2048 out true                    # Resume training
# ./train_swa_mla_moe_optimized.sh large 8 4096 out false lion ddp 8 2        # Large model with Lion optimizer

exit 0
