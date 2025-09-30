#!/bin/bash

# Train SWA+MLA hybrid model with optimized defaults

MODEL_SIZE=${1:-medium}
BATCH_SIZE=${2:-8}
BLOCK_SIZE=${3:-2048}
OUTPUT_DIR=${4:-out_swa_mla}
OUTPUT_ROOT=${OUTPUT_ROOT:-ouputs}
OUTPUT_NAME="$OUTPUT_DIR"
OUTPUT_DIR="$OUTPUT_ROOT/$OUTPUT_NAME"
mkdir -p "$OUTPUT_DIR"
RESUME=${5:-false}

echo "Training SWA+MLA hybrid model..."
echo "Model size: $MODEL_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Output dir: $OUTPUT_DIR"
echo "Resume: $RESUME"

RESUME_ARGS=()
if [ "$RESUME" = "true" ] || [ "$RESUME" = "1" ]; then
    RESUME_ARGS=(--init_from resume)
    echo "Will attempt to resume from last checkpoint in $OUTPUT_DIR"
fi

python run_train.py \
    --model_type swa_mla \
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
    --optimizer_type lion \
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
    --strategy ddp_find_unused_parameters_false \
    --compile \
    "${RESUME_ARGS[@]}"

# To use MLA Selective instead of standard MLA, add these flags:
# --use_mla_selective \
# --mla_selection_head_idx 0 \

exit 0
