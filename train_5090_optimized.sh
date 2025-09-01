#!/bin/bash

# Script optimisé pour RTX 5090 (32GB VRAM)
# Utilise les meilleures pratiques pour maximiser les performances

MODEL_TYPE=${1:-"mla"}
SIZE=${2:-"large"}
BATCH_SIZE=${3:-32}  # Augmenté pour RTX 5090
BLOCK_SIZE=${4:-4096}  # Séquences plus longues possibles
OUTPUT_DIR=${5:-"out_5090_optimized"}

# Configuration GPU RTX 5090
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"

# Optimisations CUDA pour Ada Lovelace
export TORCH_CUDA_ARCH_LIST="8.9"  # RTX 5090 est Ada Lovelace
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"

# Optimisations PyTorch 2.x
export TORCH_COMPILE_CACHE_SIZE=256
export TORCH_COMPILE_MODE="max-autotune"
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1
export TORCHINDUCTOR_TRITON_UNIQUE_KERNEL_NAMES=1

# Optimisations réseau pour les gros modèles
export NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_LEVEL=2

# Configuration de l'entraînement
GRAD_ACCUM=2  # Réduit car batch_size déjà élevé
EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM))

echo "🚀 Training optimized for RTX 5090"
echo "Model: $MODEL_TYPE ($SIZE)"
echo "Batch Size: $BATCH_SIZE (Effective: $EFFECTIVE_BATCH)"
echo "Sequence Length: $BLOCK_SIZE"
echo "Output: $OUTPUT_DIR"
echo "----------------------------------------"

python run_train.py \
    --model_type "$MODEL_TYPE" \
    --size "$SIZE" \
    --batch_size "$BATCH_SIZE" \
    --block_size "$BLOCK_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --output_dir "$OUTPUT_DIR" \
    --max_iters 50000 \
    --learning_rate 6e-4 \
    --min_lr 6e-5 \
    --warmup_iters 2000 \
    --weight_decay 0.1 \
    --grad_clip 1.0 \
    --eval_interval 500 \
    --log_interval 10 \
    --compile \
    --use_fp8 \
    --fp8_tile_size 256 \
    --optimize_attention \
    --preallocate_memory \
    --attention_backend "flash" \
    --beta1 0.9 \
    --beta2 0.95 \
