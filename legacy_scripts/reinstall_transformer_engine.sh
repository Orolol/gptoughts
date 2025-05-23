#!/bin/bash
# Script pour réinstaller transformer-engine pour le support FP8

echo "=== Réinstallation de transformer-engine pour le support FP8 ==="

# Fonction pour vérifier les erreurs
check_error() {
    if [ $? -ne 0 ]; then
        echo "❌ Erreur: $1"
        exit 1
    fi
}

# Vérifier la version de CUDA avec PyTorch
python -c "import torch; print(f'Version CUDA détectée: {torch.version.cuda}')" || echo "⚠️ PyTorch non installé ou CUDA non disponible"

# Désinstaller transformer-engine existant
echo -e "\n=== Désinstallation de transformer-engine existant ==="
pip uninstall -y transformer-engine
check_error "Impossible de désinstaller transformer-engine"

# Nettoyer le cache pip pour éviter les problèmes
echo -e "\n=== Nettoyage du cache pip ==="
pip cache purge
check_error "Impossible de nettoyer le cache pip"

# Installer une version récente de PyTorch avec CUDA 12.1
echo -e "\n=== Installation de PyTorch avec support CUDA 12.1 ==="
pip install --upgrade "torch>=2.1.0" --extra-index-url https://download.pytorch.org/whl/cu121
check_error "Impossible d'installer PyTorch"

# Installer transformer-engine
echo -e "\n=== Installation de transformer-engine >=1.3.0 ==="
pip install "transformer-engine>=1.3.0"
check_error "Impossible d'installer transformer-engine"

# Vérifier l'installation
echo -e "\n=== Vérification de l'installation ==="
python -c "import transformer_engine.pytorch as te; print(f'transformer-engine installé avec succès, API fp8_autocast: {hasattr(te, \"fp8_autocast\")}')"
check_error "Impossible d'importer transformer_engine.pytorch"

echo -e "\n=== Installation terminée avec succès ==="
echo "Exécutez ./check_transformer_engine.py pour vérifier l'installation complète"

# Option avancée: installation depuis la source si nécessaire
echo -e "\n=== Installation depuis la source (optionnel) ==="
echo "Si l'installation depuis PyPI ne fonctionne pas, vous pouvez installer depuis la source avec:"
echo "1. git clone https://github.com/NVIDIA/TransformerEngine.git"
echo "2. cd TransformerEngine"
echo "3. python setup.py install" 