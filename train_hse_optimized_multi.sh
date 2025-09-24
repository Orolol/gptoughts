#!/bin/bash

# Optimized training launcher for the HSE model on 2x H100 GPUs.
# Defaults assume two Hopper 80GB devices and PyTorch Lightning backend.


MODEL_SIZE=${1:-small}                # Model config preset defined in lightning module
BATCH_SIZE_PER_GPU=${2:-8}             # Per-device batch size
BLOCK_SIZE=${3:-2048}                  # Sequence length
OUTPUT_DIR=${4:-out_hse_optimized_2xh100}
RESUME=${5:-false}

# Tunables exposed via environment variables (override when sourcing the script)
DEVICES=${DEVICES:-2}
GRAD_ACCUM=${GRAD_ACCUM:-2}
PRECISION=${PRECISION:-bf16-mixed}
USE_FP8=${USE_FP8:-0}
NUM_WORKERS=${NUM_WORKERS:-4}
DDP_STRATEGY=${DDP_STRATEGY:-ddp_find_unused_parameters_true}

# Environment tailored for Hopper class GPUs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"9.0"}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True,max_split_size_mb:512"}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONUNBUFFERED=1

printf '\n=== HSE 2xH100 training ===\n'
printf 'Model size ............... %s\n' "$MODEL_SIZE"
printf 'Block size ............... %s\n' "$BLOCK_SIZE"
printf 'Per-GPU batch size ....... %s\n' "$BATCH_SIZE_PER_GPU"
printf 'Gradient accumulation .... %s\n' "$GRAD_ACCUM"
printf 'Requested devices ........ %s\n' "$DEVICES"
printf 'DDP strategy ............. %s\n' "$DDP_STRATEGY"
printf 'Precision mode ........... %s\n' "$PRECISION"
printf 'Output directory ......... %s\n' "$OUTPUT_DIR"
printf 'Resume from checkpoint ... %s\n' "$RESUME"
printf 'FP8 enabled .............. %s\n\n' "$USE_FP8"

# Quick sanity check on visible GPUs
python - <<'PY'
import torch, sys
if not torch.cuda.is_available():
    sys.exit('CUDA not available - aborting.')
device_count = torch.cuda.device_count()
print(f"Detected {device_count} CUDA device(s):")
for idx in range(device_count):
    props = torch.cuda.get_device_properties(idx)
    print(f"  [{idx}] {props.name} (SM {props.major}.{props.minor}) - {props.total_memory / (1024**3):.1f} GB")
if device_count < 2:
    sys.exit('Need at least 2 GPUs for this script.')
PY

# Allow resume behaviour to be toggled from CLI or env
RESUME_ARGS=""
if [[ "$RESUME" == "true" || "$RESUME" == "1" ]]; then
    RESUME_ARGS="--init_from resume"
fi

# Recommended HSE defaults (override via env when needed)
NUM_EXPERTS=${NUM_EXPERTS:-8}
EXPERTS_PER_TOKEN=${EXPERTS_PER_TOKEN:-2}
SCRIBE_CHUNK_SIZE=${SCRIBE_CHUNK_SIZE:-2048}
SCRIBE_SUMMARY_LEN=${SCRIBE_SUMMARY_LEN:-128}
QAP_PER_STEP=${QAP_PER_STEP:-12}
QAP_PER_EXPERT=${QAP_PER_EXPERT:-6}
QAP_MAX_QUERIES=${QAP_MAX_QUERIES:-20}

mkdir -p "$OUTPUT_DIR"

# Build command
CMD=(python run_train.py
    --model_type hse
    --size "$MODEL_SIZE"
    --batch_size "$BATCH_SIZE_PER_GPU"
    --block_size "$BLOCK_SIZE"
    --output_dir "$OUTPUT_DIR"
    --precision "$PRECISION"
    --devices "$DEVICES"
    --strategy "$DDP_STRATEGY"
    --num_workers "$NUM_WORKERS"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --learning_rate 5e-5
    --min_lr 3e-6
    --warmup_iters 200
    --max_iters 10000000
    --eval_interval_steps 1000
    --log_interval_steps 10
    --grad_clip 1.0
    --weight_decay 0.1
    --optimizer_type lion
    --optimize_attention
    --preallocate_memory
    --attention_backend sdpa
    --use_dyt
    --compile
    --num_experts "$NUM_EXPERTS"
    --experts_per_token "$EXPERTS_PER_TOKEN"
    --scribe_chunk_size "$SCRIBE_CHUNK_SIZE"
    --scribe_summary_len "$SCRIBE_SUMMARY_LEN"
    --qap_per_step "$QAP_PER_STEP"
    --qap_per_expert "$QAP_PER_EXPERT"
    --qap_max_queries "$QAP_MAX_QUERIES"
)

if [[ "$USE_FP8" == "1" || "$USE_FP8" == "true" ]]; then
    CMD+=(--use_fp8)
fi

if [[ -n "$RESUME_ARGS" ]]; then
    CMD+=($RESUME_ARGS)
fi

printf 'Launching command:\n%s\n' "${CMD[*]}"

"${CMD[@]}"
