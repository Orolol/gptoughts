#!/bin/bash

# Script d'installation du support FP8 pour l'entraînement des modèles LLM
# Ce script aide à installer et configurer transformer-engine avec le support FP8
# pour les GPU NVIDIA H100/H200 ou compatibles FP8

echo "==========================================================="
echo "Installation du support FP8 pour l'entraînement des modèles"
echo "==========================================================="

# Vérifier si CUDA est disponible
if ! python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "ERREUR: CUDA n'est pas disponible. FP8 nécessite un GPU NVIDIA avec CUDA."
    exit 1
fi

# Vérifier la version de CUDA
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda if torch.version.cuda else 'Unknown')")
echo "Version CUDA détectée: $CUDA_VERSION"

if [[ "$CUDA_VERSION" == "Unknown" ]]; then
    echo "ERREUR: Version CUDA non détectée correctement."
    exit 1
fi

# Vérifier si le GPU est compatible FP8 (H100, H200 ou autre avec SM86+)
if ! python -c "
import torch
import sys
try:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        if props.major >= 9 or (props.major == 8 and props.minor >= 6):
            print('FP8-compatible')
            sys.exit(0)
    sys.exit(1)
except Exception:
    sys.exit(1)
" 2>/dev/null | grep -q "FP8-compatible"; then
    echo "ATTENTION: Votre GPU n'est pas compatible avec FP8."
    echo "FP8 nécessite un GPU NVIDIA H100, H200 ou compatible (architecture SM86+)."
    echo "L'installation va continuer, mais FP8 ne sera pas fonctionnel."
    read -p "Voulez-vous continuer quand même? (o/n): " CONTINUE
    if [[ "$CONTINUE" != "o" && "$CONTINUE" != "O" ]]; then
        echo "Installation annulée."
        exit 1
    fi
else
    echo "GPU compatible FP8 détecté!"
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "GPU: $GPU_NAME"
fi

# Vérifier la version de PyTorch
TORCH_VERSION=$(python -c "import torch; print(torch.version.__version__)")
echo "Version PyTorch: $TORCH_VERSION"

# Vérifier si pip est disponible
if ! command -v pip &> /dev/null; then
    echo "ERREUR: pip n'est pas installé. Installation impossible."
    exit 1
fi

echo "Installation des dépendances pour le support FP8..."

# Vérifier si un environnement virtuel est activé
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "ATTENTION: Aucun environnement virtuel détecté."
    echo "Il est recommandé d'installer transformer-engine dans un environnement virtuel."
    read -p "Voulez-vous continuer l'installation globalement? (o/n): " GLOBAL_INSTALL
    if [[ "$GLOBAL_INSTALL" != "o" && "$GLOBAL_INSTALL" != "O" ]]; then
        echo "Installation annulée."
        exit 1
    fi
fi

# Installer les dépendances
echo "Installation des dépendances à partir de requirements-fp8.txt..."
pip install -r requirements-fp8.txt

# Vérifier l'installation de transformer-engine
if ! python -c "import transformer_engine" &>/dev/null; then
    echo "ERREUR: L'installation de transformer-engine a échoué."
    echo "Veuillez vérifier les erreurs ci-dessus et réessayer."
    exit 1
fi

# Vérifier si transformer-engine a le support FP8
if ! python -c "
import transformer_engine
import sys
if hasattr(transformer_engine, 'fp8'):
    print('FP8 support OK')
    sys.exit(0)
sys.exit(1)
" 2>/dev/null | grep -q "FP8 support OK"; then
    echo "ATTENTION: La version de transformer-engine installée ne semble pas avoir le support FP8."
    echo "Cela peut être dû à une installation incorrecte ou à une version incompatible."
else
    echo "Vérification réussie: transformer-engine avec support FP8 est correctement installé!"
fi

# Vérifier les modules CUDA nécessaires
echo "Vérification des modules CUDA..."
python -c "
import sys
try:
    import nvidia.cublas
    import nvidia.cudnn
    print('Modules CUDA OK')
except ImportError as e:
    print(f'Module manquant: {e}')
    sys.exit(1)
"

echo ""
echo "==========================================================="
echo "Installation terminée!"
echo "Pour utiliser FP8 dans l'entraînement, exécutez:"
echo "./train_max_gpu.sh --use_fp8"
echo "===========================================================" 