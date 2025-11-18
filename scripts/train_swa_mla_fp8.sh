#!/bin/bash

# Train SWA+MLA with FP8 optimizations based on DeepSeek-V3 approach
# This script enables FP8 mixed precision training for memory efficiency and speed

MODEL_SIZE=${1:-small}
BATCH_SIZE=${2:-8}
BLOCK_SIZE=${3:-2048}
OUTPUT_DIR=${4:-out_swa_mla_fp8}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs}
OUTPUT_NAME="$OUTPUT_DIR"
OUTPUT_DIR="$OUTPUT_ROOT/$OUTPUT_NAME"
mkdir -p "$OUTPUT_DIR"
RESUME=${5:-false}
OPTIMIZER=${6:-adamw}
STRATEGY=${7:-ddp_find_unused_parameters_false}

echo "========================================="
echo "Training SWA+MLA with FP8 Optimizations"
echo "========================================="
echo ""
echo "Configuration:"
echo "  Model size: $MODEL_SIZE"
echo "  Batch size: $BATCH_SIZE"
echo "  Block size: $BLOCK_SIZE"
echo "  Output dir: $OUTPUT_DIR"
echo "  Resume: $RESUME"
if [ "$OPTIMIZER" = "lion" ]; then
    echo "  Optimizer: $OPTIMIZER (will use FP8Lion with BF16 momentum - ~50% less memory than AdamW)"
else
    echo "  Optimizer: $OPTIMIZER (will use FP8AdamW with BF16 moments)"
fi
echo "  Strategy: $STRATEGY"
echo ""
echo "FP8 Mixed Precision Strategy (following DeepSeek-V3):"
echo "  🔥 FP8 for:"
echo "    - MLA linear layers (wq, wkv_a, wkv_b, wo)"
echo "    - MLP linear layers (gate_up_proj, down_proj)"
echo "    - Activations during computation"
echo "    - Optimizer moments (BF16 instead of FP32)"
echo "    - KV cache during inference"
echo ""
echo "  ⚡ High Precision (BF16/FP32) for:"
echo "    - Embeddings"
echo "    - Layer Normalization (RMSNorm/DynamicTanh)"
echo "    - RoPE (positional encoding)"
echo "    - Attention computation (scaled dot-product)"
echo "    - Loss computation (cross entropy)"
echo "    - Master weights (optimizer)"
echo "    - Gradients (optimizer)"
echo ""
echo "Expected Benefits:"
echo "  💾 Memory: ~40-45% reduction (activations + optimizer states)"
echo "  ⚡ Speed: +30-50% tokens/s on H100/H200/RTX 4090/5090"
echo "  🎯 Quality: <0.25% loss difference vs BF16"
echo ""
echo "Hardware Requirements:"
echo "  ✅ Optimal: H100, H200 (native FP8 Tensor Cores)"
echo "  ✅ Good: RTX 4090, RTX 5090 (Ada architecture)"
echo "  ⚠️  Fallback: Other GPUs (emulated FP8, slower)"
echo ""

if [ "$STRATEGY" = "fsdp" ]; then
    echo "⚠️  Warning: FSDP is not compatible with SWA-MLA"
    echo "   Falling back to DDP with optimizations"
    STRATEGY="ddp_find_unused_parameters_false"
fi

echo "Multi-GPU DDP optimizations:"
echo "  - Model initialized on CPU to prevent rank 1 duplication"
echo "  - broadcast_buffers=False (prevents buffer duplication)"
echo "  - bucket_cap_mb=10 (reduced gradient buckets)"
echo "  - gradient_as_bucket_view=True (memory optimization)"
echo ""

RESUME_ARGS=()
if [ "$RESUME" = "true" ] || [ "$RESUME" = "1" ]; then
    RESUME_ARGS=(--init_from resume)
    echo "Will attempt to resume from last checkpoint in $OUTPUT_DIR"
fi

echo "Starting training..."
echo ""

python run_train.py \
    --model_type swa_mla \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --precision bf16-mixed \
    --use_fp8 \
    --fp8_mla_params \
    --fp8_tile_size 128 \
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
    --swa_layers_per_cycle 2 \
    --mla_layers_per_cycle 1 \
    --swa_window 256 \
    --swa_sink_size 4 \
    --mla_q_lora_rank 0 \
    --mla_kv_lora_rank 256 \
    --mla_qk_nope_head_dim 128 \
    --mla_qk_rope_head_dim 64 \
    --mla_v_head_dim 128 \
    --compile \
    --strategy $STRATEGY \
    "${RESUME_ARGS[@]}"

echo ""
echo "Training completed!"
echo ""
echo "Usage examples:"
echo ""
echo "# With AdamW (2 moments: momentum + variance)"
echo "./scripts/train_swa_mla_fp8.sh small 16 2048 out_fp8_adamw false adamw"
echo ""
echo "# With Lion (1 moment: momentum only - ~50% less optimizer memory)"
echo "./scripts/train_swa_mla_fp8.sh small 16 2048 out_fp8_lion false lion"
echo ""
echo "# Compare with standard BF16 training (no FP8)"
echo "./scripts/train_swa_mla_optimized_gpt_tok.sh small 16 2048 out_bf16"
echo ""
echo "Expected memory savings with FP8:"
echo "  - Activations: ~50% reduction"
echo "  - Optimizer states (AdamW): ~50% reduction (BF16 vs FP32)"
echo "  - Optimizer states (Lion): ~50% less than AdamW + 50% reduction (BF16 vs FP32)"
echo "  - Total (AdamW): ~40-45% overall memory reduction"
echo "  - Total (Lion): ~50-55% overall memory reduction"

exit 0
