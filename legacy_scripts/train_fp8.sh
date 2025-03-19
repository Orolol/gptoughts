#!/bin/bash

# train_fp8.sh - Script pour l'entraînement avec précision FP8 (nécessite NVIDIA H100/H200)
# Ce script permet d'entraîner des modèles en utilisant la précision FP8 pour une vitesse maximale

# Vérification de la présence d'un GPU compatible FP8 (H100, H200)
CUDA_VISIBLE_DEVICES=0 python -c "
import torch
import sys
try:
    import transformer_engine as te
    if torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).major >= 9 or (torch.cuda.get_device_properties(0).major == 8 and torch.cuda.get_device_properties(0).minor >= 6):
            print('FP8-compatible GPU detected')
            sys.exit(0)
    print('No FP8-compatible GPU found. Requires NVIDIA H100, H200 or newer.')
    sys.exit(1)
except ImportError:
    print('transformer-engine package not found. Please install: pip install -r requirements-fp8.txt')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "Exiting: FP8 training requirements not met."
    exit 1
fi

# Configuration par défaut (peut être modifiée par l'utilisateur)
MODEL_TYPE=${1:-"llada"}
SIZE=${2:-"medium"}
BATCH_SIZE=${3:-32}
BLOCK_SIZE=${4:-4096}
GRADIENT_ACCUM=${5:-1}
LEARNING_RATE=${6:-1e-5}
OUTPUT_DIR=${7:-"out/fp8_training"}

echo "Starting FP8 training with model: $MODEL_TYPE ($SIZE)"
echo "Batch size: $BATCH_SIZE, Block size: $BLOCK_SIZE, Gradient accumulation: $GRADIENT_ACCUM"

# Configuration des optimisations CUDA
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_AUTO_BOOST=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Exécution de l'entraînement avec FP8
python run_train.py \
    --model_type $MODEL_TYPE \
    --size $SIZE \
    --use_fp8 \
    --batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUM \
    --learning_rate $LEARNING_RATE \
    --output_dir $OUTPUT_DIR \
    --decay_lr \
    --optimizer_type apollo-mini \
    --warmup_iters 100 \
    --always_save_checkpoint

echo "FP8 training completed!" 