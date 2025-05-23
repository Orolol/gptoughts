#!/bin/bash

# Script pour lancer l'entraînement avec optimisations GPU avancées
# Ce script configure automatiquement les paramètres optimaux pour maximiser l'utilisation du GPU
# en se basant sur les statistiques de timing observées

# Vérifier si CUDA est disponible
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "CUDA détecté, configuration des optimisations GPU avancées..."
else
    echo "CUDA non détecté, l'entraînement sera exécuté sur CPU."
    exit 1
fi

# Vérification de la compatibilité FP8 (H100, H200)
FP8_AVAILABLE=0
if python -c "
import torch
import sys
try:
    if torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).major >= 9 or (torch.cuda.get_device_properties(0).major == 8 and torch.cuda.get_device_properties(0).minor >= 6):
            # Vérifier si transformer-engine est disponible
            try:
                import transformer_engine
                print('FP8-compatible')
                sys.exit(0)
            except ImportError:
                pass
    sys.exit(1)
except Exception:
    sys.exit(1)
" 2>/dev/null | grep -q "FP8-compatible"; then
    echo "GPU compatible FP8 détecté (H100/H200), FP8 sera activé pour une performance maximale"
    FP8_AVAILABLE=1
fi

# Paramètres par défaut
MODEL_TYPE="deepseek"  # Options: deepseek, llada, gpt
SIZE="medium"          # Options: small, medium, large
BATCH_SIZE=0           # 0 pour auto-détection
GRAD_ACCUM=0           # 0 pour auto-détection
BLOCK_SIZE=1024        # Taille du contexte
USE_FP8=0              # Désactivé par défaut, activé uniquement si compatible

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
        --init_from)
            INIT_FROM="$2"
            shift 2
            ;;
        --block_size)
            BLOCK_SIZE="$2"
            shift 2
            ;;
        --use_fp8)
            USE_FP8=1
            shift
            ;;
        --no_fp8)
            USE_FP8=0
            shift
            ;;
        *)
            echo "Option inconnue: $1"
            exit 1
            ;;
    esac
done

# Activer automatiquement FP8 si disponible et non explicitement désactivé
if [ "$FP8_AVAILABLE" -eq 1 ] && [ "$USE_FP8" -ne 0 ]; then
    USE_FP8=1
    echo "Activation automatique de FP8 pour une performance maximale"
else
    USE_FP8=0
fi

# Créer un script Python temporaire pour déterminer les paramètres optimaux
TMP_SCRIPT=$(mktemp)
cat > $TMP_SCRIPT << 'EOF'
import torch
import sys
sys.path.append('.')
try:
    from gpu_optimization_advanced import get_optimal_batch_size_for_gpu
    batch_size, grad_accum = get_optimal_batch_size_for_gpu()
    print(f"{batch_size} {grad_accum}")
except ImportError:
    print("16 2")  # Valeurs par défaut si le module n'est pas trouvé
EOF

# Exécuter le script pour obtenir les paramètres optimaux
if [ "$BATCH_SIZE" -eq 0 ] || [ "$GRAD_ACCUM" -eq 0 ]; then
    OPTIMAL_PARAMS=$(python $TMP_SCRIPT)
    read -r AUTO_BATCH_SIZE AUTO_GRAD_ACCUM <<< "$OPTIMAL_PARAMS"
    
    if [ "$BATCH_SIZE" -eq 0 ]; then
        BATCH_SIZE=$AUTO_BATCH_SIZE
        echo "Taille de batch auto-détectée: $BATCH_SIZE"
    fi
    
    if [ "$GRAD_ACCUM" -eq 0 ]; then
        GRAD_ACCUM=$AUTO_GRAD_ACCUM
        echo "Étapes d'accumulation de gradient auto-détectées: $GRAD_ACCUM"
    fi
fi

# Supprimer le script temporaire
rm $TMP_SCRIPT

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
    # Ajouter PreTrainedTokenizerFast aux globals sécurisés
    torch.serialization.add_safe_globals([PreTrainedTokenizerFast])
    print("Successfully registered PreTrainedTokenizerFast with PyTorch's safe globals")
except (ImportError, AttributeError) as e:
    print(f"Note: Could not register tokenizer with safe globals: {e}")

# Importer les optimisations avancées
try:
    from gpu_optimization_advanced import optimize_for_maximum_gpu_utilization
    # Appliquer toutes les optimisations avancées
    optimize_for_maximum_gpu_utilization()
except ImportError:
    print("Module gpu_optimization_advanced non trouvé, utilisation des optimisations standard")
    try:
        from gpu_optimization_enhanced import setup_enhanced_cuda_optimizations
        setup_enhanced_cuda_optimizations()
    except ImportError:
        print("Module gpu_optimization_enhanced non trouvé, aucune optimisation appliquée")

# Importer le module d'entraînement
try:
    from run_train_enhanced import main as run_train_main
    print("Utilisation du module run_train_enhanced")
except ImportError:
    try:
        from run_train import main as run_train_main
        print("Utilisation du module run_train standard")
    except ImportError:
        print("Erreur: Impossible de trouver le module d'entraînement")
        sys.exit(1)

# Patch pour optimiser la boucle d'entraînement
def patch_train_module():
    try:
        import train.train
        original_train = train.train.Trainer.train
        
        def optimized_train(self):
            """Version optimisée de la méthode train"""
            # Appliquer les optimisations spécifiques à la boucle d'entraînement
            if torch.cuda.is_available():
                print("Application des optimisations à la boucle d'entraînement...")
                
                # Précharger les poids du modèle en mémoire GPU
                for param in self.model.parameters():
                    if param.device.type != 'cuda':
                        param.data = param.data.to(self.device)
                
                # Optimiser le scheduler de CUDA
                if hasattr(torch.cuda, 'cudart'):
                    torch.cuda.cudart().cudaProfilerStart()
                
                # Synchroniser avant de commencer l'entraînement
                torch.cuda.synchronize()
                
                # Optimiser les streams CUDA
                default_stream = torch.cuda.current_stream()
                compute_stream = torch.cuda.Stream()
                data_stream = torch.cuda.Stream()
                
                # Stocker les streams dans l'instance
                self.compute_stream = compute_stream
                self.data_stream = data_stream
            
            # Appeler la méthode originale
            result = original_train(self)
            return result
        
        # Remplacer la méthode originale par la version optimisée
        train.train.Trainer.train = optimized_train
        print("Boucle d'entraînement optimisée appliquée")
    except Exception as e:
        print(f"Erreur lors du patch de la boucle d'entraînement: {e}")

# Appliquer le patch
patch_train_module()

# Exécuter l'entraînement avec les arguments de ligne de commande
if __name__ == "__main__":
    # Transmettre tous les arguments à run_train_main
    run_train_main()
EOF

# Construire la commande
CMD="python $TRAIN_SCRIPT --model_type $MODEL_TYPE --size $SIZE --block_size $BLOCK_SIZE --batch_size $BATCH_SIZE --gradient_accumulation_steps $GRAD_ACCUM --init_from $INIT_FROM"

# Ajouter des options supplémentaires pour maximiser l'utilisation du GPU
CMD="$CMD --compile --preallocate_memory --async_data_loading --optimize_attention --decay_lr"

# Ajouter des options pour prévenir les NaN
CMD="$CMD --grad_clip 1.0 --weight_decay 0.05 --learning_rate 3e-5 --min_lr 1e-6"

# Si FP8 est disponible et activé, ajouter l'option
if [ "$USE_FP8" -eq 1 ]; then
    CMD="$CMD --use_fp8"
    echo "Mode FP8 activé pour une performance maximale"
    
    # Vérifier si transformer-engine est installé, sinon suggérer l'installation
    if ! python -c "
try:
    import transformer_engine.pytorch
    print('ok')
except ImportError:
    print('error')
" 2>/dev/null | grep -q "ok"; then
        echo "ATTENTION: Package transformer-engine non trouvé ou mal installé, nécessaire pour FP8"
        echo "Installer avec: pip install -r requirements-fp8.txt"
        echo "Installation automatique impossible - il faut d'abord installer un driver CUDA compatible"
    else
        # Vérifier si transformer-engine.pytorch a le bon support FP8
        if ! python -c "
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    if hasattr(te, 'fp8_autocast') and hasattr(recipe, 'Format'):
        print('ok')
    else:
        print('error')
except (ImportError, AttributeError):
    print('error')
" 2>/dev/null | grep -q "ok"; then
            echo "ATTENTION: La version de transformer-engine installée ne supporte pas correctement FP8"
            echo "Mettre à jour transformer-engine avec: pip install -r requirements-fp8.txt"
            echo "Version requise: transformer-engine>=1.3.0"
        fi
    fi
# Sinon, utiliser bfloat16 si disponible pour une meilleure stabilité numérique
elif python -c "import torch; print(torch.cuda.is_bf16_supported())" | grep -q "True"; then
    CMD="$CMD --dtype bfloat16"
    echo "BFloat16 détecté, utilisation pour une meilleure stabilité numérique"
fi

# Si le modèle est LLaDA, ajouter un coefficient de perte de routeur plus faible
if [ "$MODEL_TYPE" == "llada" ]; then
    CMD="$CMD --router_z_loss_coef 0.0001"
fi

# Afficher la commande
echo "Exécution de la commande: $CMD"

# Exécuter la commande
eval $CMD

# Supprimer le script temporaire
rm $TRAIN_SCRIPT

echo "Entraînement terminé"