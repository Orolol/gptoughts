#!/bin/bash
# Script pour lancer l'entraînement du modèle DeepSeek avec MTP (Multi-Token Prediction)

# Paramètres MTP
NUM_MTP_MODULES=2       # Nombre de modules MTP (2-4 recommandé)
LAYERS_PER_MTP=1        # Nombre de couches par module MTP
MTP_LOSS_FACTOR=0.1     # Facteur de pondération pour la perte MTP

# Paramètres de base
MODEL_SIZE="small"      # small, medium, large
BATCH_SIZE=8            # Taille du batch par GPU
CONTEXT_SIZE=1024       # Taille du contexte (en tokens)
GRAD_ACCUM=4            # Étapes d'accumulation de gradient
MAX_ITERS=100000        # Nombre maximal d'iterations
OUTPUT_DIR="out/deepseek_mtp"

# Paramètres d'optimisation
OPTIMIZER="lion"        # adamw, lion, apollo, apollo-mini
LEARNING_RATE=3e-5
WEIGHT_DECAY=0.1

# Optimisations mémoire
USE_GRAD_CHECKPOINT="--use_gradient_checkpointing"
STABILIZE_GRADS="--stabilize_gradients"
OPTIMIZE_ATTENTION="--optimize_attention"

# Vérifier si le répertoire de sortie existe
mkdir -p $OUTPUT_DIR

echo "Lancement de l'entraînement DeepSeek avec MTP..."
echo "Configuration MTP: $NUM_MTP_MODULES modules, $LAYERS_PER_MTP couches/module, facteur de perte $MTP_LOSS_FACTOR"

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
    --eval_interval 500 \
    --log_interval 10 \
    --keep_checkpoints 5