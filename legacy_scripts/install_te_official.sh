#!/bin/bash
# Script d'installation officiel pour transformer-engine suivant les recommandations NVIDIA

set -e  # Arrêt à la première erreur

echo "=== Installation de transformer-engine pour RTX 6000 Ada ==="
echo "Basé sur les instructions officielles: https://github.com/NVIDIA/TransformerEngine"

# Vérifier les prérequis
echo -e "\n=== Vérification des prérequis ==="
if ! command -v nvcc &> /dev/null; then
    echo "⚠️ NVCC non trouvé. Assurez-vous que CUDA 12.1+ est installé et dans votre PATH."
else
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "✓ CUDA détecté: $CUDA_VERSION"
    if [[ $(echo "$CUDA_VERSION" | cut -d. -f1) -lt 12 ]] || [[ $(echo "$CUDA_VERSION" | cut -d. -f1) -eq 12 && $(echo "$CUDA_VERSION" | cut -d. -f2) -lt 1 ]]; then
        echo "⚠️ CUDA 12.1+ est requis. Version actuelle: $CUDA_VERSION"
    fi
fi

# Nettoyer l'installation précédente
echo -e "\n=== Nettoyage de l'installation précédente ==="
pip uninstall -y transformer_engine
python -c "import shutil; shutil.rmtree('TransformerEngine', ignore_errors=True)"

# Installer depuis le repo GitHub - version stable
echo -e "\n=== Installation depuis la branche stable officielle ==="
echo "Option 1: Installation directe depuis GitHub (recommandée)"
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable

# Vérifier l'installation
echo -e "\n=== Vérification de l'installation ==="
if python -c "import transformer_engine; print(f'Transformer Engine version: {transformer_engine.__version__}')" &> /dev/null; then
    echo "✓ Transformer Engine importé avec succès."
    
    # Vérifier PyTorch
    if python -c "import transformer_engine.pytorch; print('Module PyTorch disponible')" &> /dev/null; then
        echo "✓ Module transformer_engine.pytorch disponible"
    else
        echo "❌ Module transformer_engine.pytorch manquant"
        echo -e "\n=== Installation alternative depuis PyPI ==="
        echo "Essai d'installation avec spécification explicite de PyTorch..."
        pip uninstall -y transformer_engine
        pip install -U "transformer_engine[pytorch]"
    fi
else
    echo "❌ Échec de l'importation de transformer_engine"
    
    echo -e "\n=== Installation depuis la source ==="
    echo "Essai d'installation depuis la source..."
    
    # Cloner le dépôt
    git clone https://github.com/NVIDIA/TransformerEngine.git
    cd TransformerEngine
    
    # Installer en spécifiant le framework
    NVTE_FRAMEWORK=pytorch pip install .
    
    cd ..
fi

# Vérification finale
echo -e "\n=== Vérification finale ==="
python - << 'END'
try:
    import sys
    import torch
    import transformer_engine
    print(f"Transformer Engine version: {transformer_engine.__version__}")
    
    # Vérifier la présence du module pytorch
    try:
        import transformer_engine.pytorch as te
        print("✓ Module transformer_engine.pytorch disponible")
        
        # Vérifier fp8_autocast
        if hasattr(te, 'fp8_autocast'):
            print("✓ API fp8_autocast disponible")
            
            # Vérifier si le GPU est Ada Lovelace
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                if props.major >= 8 and props.minor >= 9:
                    print(f"✓ GPU Ada Lovelace détecté ({torch.cuda.get_device_name(0)})")
                    print("Le support FP8 est disponible sur ce GPU")
                else:
                    print(f"GPU détecté: {torch.cuda.get_device_name(0)} - SM {props.major}.{props.minor}")
        else:
            print("❌ API fp8_autocast non disponible")
            
    except ImportError as e:
        print(f"❌ Module transformer_engine.pytorch non disponible: {e}")
        
except ImportError as e:
    print(f"❌ Erreur d'importation: {e}")
    sys.exit(1)
END

echo -e "\n=== Exemple d'utilisation ==="
echo "Pour utiliser Transformer Engine avec PyTorch:"
echo "
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# Créer une recette FP8
fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)

# Activer l'autocast pour le forward pass
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    # Votre code ici
    pass
"

echo -e "\n=== Installation terminée ===" 