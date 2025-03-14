#!/bin/bash

# Script pour relancer l'entraînement à partir d'un checkpoint avec des paramètres optimisés
# pour éviter les problèmes de NaN loss

# Créer le répertoire de sortie pour cette reprise
OUTPUT_DIR="out/llada_resumed_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Paramètres de base
MODEL_TYPE="llada"
SIZE="small"
BLOCK_SIZE=512
BATCH_SIZE=16  # Réduit par rapport à 20 pour plus de stabilité
GRAD_ACCUM=2

# Créer un script Python pour l'entraînement optimisé
TRAIN_SCRIPT=$(mktemp)
cat > $TRAIN_SCRIPT << 'EOF'
import os
import sys
import torch
import argparse

# Ajouter le répertoire courant au path
sys.path.append('.')

# Fix for PyTorch serialization security with transformers
try:
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
    import torch.serialization
    torch.serialization.add_safe_globals([PreTrainedTokenizerFast])
    print("Successfully registered PreTrainedTokenizerFast with PyTorch's safe globals")
except (ImportError, AttributeError) as e:
    print(f"Note: Could not register tokenizer with safe globals: {e}")

# Importer les optimisations avancées
try:
    import gc
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()
    
    from gpu_optimization_enhanced import setup_enhanced_cuda_optimizations
    setup_enhanced_cuda_optimizations()
except ImportError:
    print("Module gpu_optimization_enhanced non trouvé, aucune optimisation appliquée")

# Importer le module d'entraînement
try:
    from run_train import main as run_train_main
    print("Utilisation du module run_train standard")
except ImportError:
    print("Erreur: Impossible de trouver le module d'entraînement")
    sys.exit(1)

# Exécuter l'entraînement avec les arguments de ligne de commande
if __name__ == "__main__":
    # Transmettre tous les arguments à run_train_main
    run_train_main()
EOF

# Déterminer le dernier checkpoint stable
LAST_STABLE="./checkpoints/ckpt_iter_928.pt"

echo "Utilisation du checkpoint: $LAST_STABLE"

# Construire la commande avec les paramètres optimisés
CMD="python $TRAIN_SCRIPT --model_type $MODEL_TYPE --size $SIZE --block_size $BLOCK_SIZE --batch_size $BATCH_SIZE --gradient_accumulation_steps $GRAD_ACCUM --init_from resume --output_dir $OUTPUT_DIR"

# Ajouter des paramètres pour éviter les NaN
CMD="$CMD --grad_clip 1.0 --weight_decay 0.05 --learning_rate 3e-5 --min_lr 1e-6 --router_z_loss_coef 0.0001"

# Utiliser bfloat16 si disponible pour une meilleure stabilité numérique
if python -c "import torch; print(torch.cuda.is_bf16_supported())" | grep -q "True"; then
    CMD="$CMD --dtype bfloat16"
    echo "BFloat16 détecté, utilisation pour une meilleure stabilité numérique"
fi

# Ajouter des options supplémentaires pour la stabilité
CMD="$CMD --eval_interval 100 --log_interval 20 --decay_lr --max_iters 30000"

# Copier le dernier checkpoint stable dans le nouveau répertoire de sortie
cp "$LAST_STABLE" "$OUTPUT_DIR/"

# Afficher la commande
echo "Exécution de la commande: $CMD"

# Exécuter la commande
eval $CMD

# Supprimer le script temporaire
rm $TRAIN_SCRIPT

echo "Entraînement de récupération terminé" 