#!/bin/bash

# Train HSE model (Hierarchical Sparse Experts) with optimized settings
# - Uses standard attention backend (sdpa/flash/xformers auto)
# - Keeps MoE top-2 experts and logs router loss
# - Enables CUDA and attention optimizations + optional FP8/compile

MODEL_SIZE=${1:-small}
BATCH_SIZE=${2:-8}
BLOCK_SIZE=${3:-2048}
OUTPUT_DIR=${4:-out_hse_optimized}
RESUME=${5:-false}

echo "Training HSE model with optimized settings..."
echo "Model size: $MODEL_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Block size: $BLOCK_SIZE"
echo "Output dir: $OUTPUT_DIR"
echo "Resume: $RESUME"

# Set resume options
RESUME_ARGS=""
if [ "$RESUME" = "true" ] || [ "$RESUME" = "1" ]; then
    RESUME_ARGS="--init_from resume"
    echo "Will attempt to resume from last checkpoint in $OUTPUT_DIR"
fi

# Recommended defaults for HSE
NUM_EXPERTS=${NUM_EXPERTS:-8}
EXPERTS_PER_TOKEN=${EXPERTS_PER_TOKEN:-1}
SCRIBE_CHUNK_SIZE=${SCRIBE_CHUNK_SIZE:-2048}
SCRIBE_SUMMARY_LEN=${SCRIBE_SUMMARY_LEN:-128}
QAP_PER_STEP=${QAP_PER_STEP:-12}
QAP_PER_EXPERT=${QAP_PER_EXPERT:-6}
QAP_MAX_QUERIES=${QAP_MAX_QUERIES:-20}

# Run training
python run_train.py \
    --model_type hse \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --precision bf16-mixed \
    --optimize_attention \
    --preallocate_memory \
    --attention_backend sdpa \
    --grad_clip 1.0 \
    --learning_rate 5e-5 \
    --optimizer_type lion \
    --weight_decay 0.1 \
    --warmup_iters 200 \
    --max_iters 10000000 \
    --eval_interval_steps 1000 \
    --log_interval_steps 10 \
    --gradient_accumulation_steps 4 \
    --use_dyt \
    --compile \
    --num_experts $NUM_EXPERTS \
    --experts_per_token $EXPERTS_PER_TOKEN \
    --scribe_chunk_size $SCRIBE_CHUNK_SIZE \
    --scribe_summary_len $SCRIBE_SUMMARY_LEN \
    --qap_per_step $QAP_PER_STEP \
    --qap_per_expert $QAP_PER_EXPERT \
    --qap_max_queries $QAP_MAX_QUERIES \
    $RESUME_ARGS

exit 0

