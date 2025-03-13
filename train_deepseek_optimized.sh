#!/bin/bash

# Script optimisé pour l'entraînement du modèle DeepSeek
# Avec des paramètres adaptés pour éviter les problèmes de VRAM et de NaN

# Paramètres de base
MODEL="deepseek"
SIZE="small"
BATCH_SIZE=2
BLOCK_SIZE=16  # Augmenté de 1 à 16 pour plus de stabilité
GRAD_CLIP=0.5  # Réduit pour éviter l'explosion des gradients
LEARNING_RATE=1e-5  # Réduit pour plus de stabilité
WARMUP_ITERS=500  # Augmenté pour un warmup plus progressif
DROPOUT=0.1  # Ajout d'un peu de dropout pour la régularisation

# Paramètres d'optimisation mémoire
GRADIENT_ACCUMULATION=4  # Accumulation de gradient pour simuler un batch plus grand
OPTIMIZER="adamw"  # AdamW est plus stable que Lion pour ce cas

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
  --output_dir="checkpoints/deepseek_optimized" \
  --eval_interval=500 \
  --log_interval=10