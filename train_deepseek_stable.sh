#!/bin/bash

# Script optimisé pour l'entraînement du modèle DeepSeek
# Avec des paramètres adaptés pour éviter les problèmes de VRAM et de NaN

# Paramètres de base
MODEL="deepseek"
SIZE="small"
BATCH_SIZE=2
BLOCK_SIZE=16  # Augmenté de 1 à 16 pour plus de stabilité
GRAD_CLIP=0.1  # Réduit pour éviter l'explosion des gradients
LEARNING_RATE=1e-6  # Fortement réduit pour plus de stabilité
WARMUP_ITERS=1000  # Augmenté pour un warmup plus progressif
DROPOUT=0.2  # Augmenté pour une meilleure régularisation

# Paramètres d'optimisation mémoire
GRADIENT_ACCUMULATION=8  # Accumulation de gradient pour simuler un batch plus grand
OPTIMIZER="adamw"  # AdamW est plus stable que Lion pour ce cas
WEIGHT_DECAY=0.01  # Réduit pour éviter l'explosion des gradients

# Paramètres de sauvegarde et d'évaluation
EVAL_INTERVAL=500
LOG_INTERVAL=10
OUTPUT_DIR="checkpoints/deepseek_stable"

# Exécution de l'entraînement avec les paramètres optimisés
python run_train.py \
  --model_type=$MODEL \
  --size=$SIZE \
  --batch_size=$BATCH_SIZE \
  --block_size=$BLOCK_SIZE \
  --grad_clip=$GRAD_CLIP \
  --learning_rate=$LEARNING_RATE \
  --warmup_iters=$WARMUP_ITERS \
  --dropout=$DROPOUT \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION \
  --optimizer_type=$OPTIMIZER \
  --weight_decay=$WEIGHT_DECAY \
  --output_dir=$OUTPUT_DIR \
  --eval_interval=$EVAL_INTERVAL \
  --log_interval=$LOG_INTERVAL \
  --beta1=0.9 \
  --beta2=0.999 \
  --min_lr=1e-7