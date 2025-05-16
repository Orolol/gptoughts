#!/bin/bash
# Script pour lancer l'entraînement distribué du modèle DeepSeek avec MTP (Multi-Token Prediction)

# Paramètres MTP
NUM_MTP_MODULES=3       # Plus de modules pour un modèle plus grand
LAYERS_PER_MTP=2        # Plus de couches par module pour mieux exploiter MTP
MTP_LOSS_FACTOR=0.15    # Facteur de perte légèrement augmenté

# Paramètres de base
MODEL_SIZE="medium"     # medium pour un modèle plus grand
BATCH_SIZE=4            # Batch size réduit pour tenir en mémoire sur chaque GPU
CONTEXT_SIZE=2048       # Contexte plus grand 
GRAD_ACCUM=4            # Accumulation de gradient pour compenser le batch size réduit
MAX_ITERS=150000        # Plus d'iterations pour un modèle plus grand
OUTPUT_DIR="out/deepseek_mtp_distributed"

# Paramètres d'optimisation
OPTIMIZER="lion"        # lion est efficace pour les grands modèles
LEARNING_RATE=2e-5      # Learning rate réduit pour un modèle plus grand
WEIGHT_DECAY=0.1

# Optimisations mémoire (cruciales pour les grands modèles)
USE_GRAD_CHECKPOINT="--use_gradient_checkpointing"
STABILIZE_GRADS="--stabilize_gradients"
OPTIMIZE_ATTENTION="--optimize_attention"
USE_PREALLOCATE="--preallocate_memory"

# Support pour BF16 si disponible
USE_BF16="--dtype bfloat16"

# Vérifier si le répertoire de sortie existe
mkdir -p $OUTPUT_DIR

echo "Lancement de l'entraînement distribué DeepSeek avec MTP..."
echo "Configuration MTP: $NUM_MTP_MODULES modules, $LAYERS_PER_MTP couches/module, facteur de perte $MTP_LOSS_FACTOR"
echo "Utilisation du mode distribué sur tous les GPUs disponibles"

python run_train.py \
    --model_type deepseek \
    --size $MODEL_SIZE \
    --use_mtp \
    --num_mtp_modules $NUM_MTP_MODULES \
    --layers_per_mtp $LAYERS_PER_MTP \
    --mtp_loss_factor $MTP_LOSS_FACTOR \
    --batch_size $BATCH_SIZE \
    --block_size $CONTEXT_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_iters $MAX_ITERS \
    --output_dir $OUTPUT_DIR \
    --optimizer_type $OPTIMIZER \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    $USE_GRAD_CHECKPOINT \
    $STABILIZE_GRADS \
    $OPTIMIZE_ATTENTION \
    $USE_PREALLOCATE \
    $USE_BF16 \
    --eval_interval 500 \
    --log_interval 10 \
    --keep_checkpoints 5 \
    --distributed