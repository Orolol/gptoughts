#!/bin/bash

# Script pour lancer l'entraînement avec optimisations GPU avancées
# Ce script configure automatiquement les paramètres optimaux pour maximiser l'utilisation du GPU

# Vérifier si CUDA est disponible
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "CUDA détecté, configuration des optimisations GPU..."
else
    echo "CUDA non détecté, l'entraînement sera exécuté sur CPU."
fi

# Paramètres par défaut
MODEL_TYPE="llada"  # Options: deepseek, llada, gpt
SIZE="medium"          # Options: small, medium, large
BATCH_SIZE=0           # 0 pour auto-détection
GRAD_ACCUM=0           # 0 pour auto-détection
BLOCK_SIZE=2048        # Taille du contexte
COMPILE=true           # Utiliser torch.compile
PREALLOCATE=true       # Préallouer la mémoire CUDA
ASYNC_LOADING=true     # Chargement de données asynchrone
OPTIMIZE_ATTENTION=true # Optimiser les opérations d'attention

# Traiter les arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --size)
            SIZE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad_accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        --block_size)
            BLOCK_SIZE="$2"
            shift 2
            ;;
        --no_compile)
            COMPILE=false
            shift
            ;;
        --no_preallocate)
            PREALLOCATE=false
            shift
            ;;
        --no_async)
            ASYNC_LOADING=false
            shift
            ;;
        --no_attention_opt)
            OPTIMIZE_ATTENTION=false
            shift
            ;;
        *)
            echo "Option inconnue: $1"
            exit 1
            ;;
    esac
done

# Construire la commande
CMD="python run_train_enhanced.py --model_type $MODEL_TYPE --size $SIZE --block_size $BLOCK_SIZE"

# Ajouter les options conditionnelles
if [ "$BATCH_SIZE" -ne 0 ]; then
    CMD="$CMD --batch_size $BATCH_SIZE --batch_size_auto_optimize False"
fi

if [ "$GRAD_ACCUM" -ne 0 ]; then
    CMD="$CMD --gradient_accumulation_steps $GRAD_ACCUM --grad_accum_auto_optimize False"
fi

if [ "$COMPILE" = true ]; then
    CMD="$CMD --compile"
fi

if [ "$PREALLOCATE" = true ]; then
    CMD="$CMD --preallocate_memory"
fi

if [ "$ASYNC_LOADING" = true ]; then
    CMD="$CMD --async_data_loading"
fi

if [ "$OPTIMIZE_ATTENTION" = true ]; then
    CMD="$CMD --optimize_attention"
fi

# Ajouter des options supplémentaires pour maximiser l'utilisation du GPU
CMD="$CMD --decay_lr"

# Afficher la commande
echo "Exécution de la commande: $CMD"

# Exécuter la commande
eval $CMD