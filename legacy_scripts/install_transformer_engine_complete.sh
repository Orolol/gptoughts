#!/bin/bash
# Script complet pour installer transformer-engine et ses dépendances pour le support FP8

set -e  # Arrêt à la première erreur

echo "=== Installation complète de transformer-engine pour le support FP8 ==="

# Vérifier si l'utilisateur est root ou peut utiliser sudo
check_sudo() {
    if [ "$(id -u)" -eq 0 ]; then
        echo "Exécution en tant que root"
        SUDO=""
    elif command -v sudo &> /dev/null; then
        echo "Utilisation de sudo pour les commandes système"
        SUDO="sudo"
    else
        echo "⚠️ Ni root ni sudo disponible. Certaines étapes pourraient échouer."
        SUDO=""
    fi
}

# Vérifier la version CUDA
check_cuda() {
    echo -e "\n=== Vérification de CUDA ==="
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        echo "✓ CUDA détecté: $CUDA_VERSION"
    else
        echo "❌ NVCC non trouvé. CUDA n'est peut-être pas installé ou n'est pas dans le PATH."
        if command -v nvidia-smi &> /dev/null; then
            DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
            echo "✓ Driver NVIDIA détecté: $DRIVER_VERSION"
        else
            echo "❌ NVIDIA driver non détecté."
            echo "⚠️ Installation de CUDA nécessaire avant de continuer."
            exit 1
        fi
    fi
}

# Vérifier l'environnement Python et pip
check_python() {
    echo -e "\n=== Vérification de l'environnement Python ==="
    PYTHON_VERSION=$(python --version 2>&1)
    echo "Python version: $PYTHON_VERSION"
    
    if ! command -v pip &> /dev/null; then
        echo "❌ pip non trouvé. Installation..."
        if [ -n "$SUDO" ]; then
            $SUDO apt-get update
            $SUDO apt-get install -y python3-pip
        else
            echo "❌ Impossible d'installer pip sans privilèges root."
            exit 1
        fi
    fi
    
    echo "✓ pip disponible"
    
    # Vérifier si c'est un environnement virtuel
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "⚠️ Pas d'environnement virtuel détecté. L'installation sera globale."
        read -p "Continuer l'installation globale? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Installation annulée. Créez un environnement virtuel avec:"
            echo "python -m venv env && source env/bin/activate"
            exit 1
        fi
    else
        echo "✓ Environnement virtuel détecté: $VIRTUAL_ENV"
    fi
}

# Installer les dépendances système nécessaires
install_system_deps() {
    echo -e "\n=== Installation des dépendances système ==="
    if [ -n "$SUDO" ]; then
        $SUDO apt-get update
        $SUDO apt-get install -y build-essential cmake git
        echo "✓ Dépendances système installées"
    else
        echo "⚠️ Impossible d'installer les dépendances système sans privilèges root."
        echo "Dépendances requises: build-essential, cmake, git"
    fi
}

# Installer PyTorch avec CUDA
install_pytorch() {
    echo -e "\n=== Installation de PyTorch avec CUDA ==="
    pip install --upgrade pip
    
    if pip list | grep -q torch; then
        echo "PyTorch déjà installé. Vérification de la version CUDA..."
        python -c "import torch; print(f'PyTorch {torch.__version__} avec CUDA {torch.version.cuda if torch.cuda.is_available() else \"non disponible\"}')"
        
        if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA non disponible'"; then
            echo "❌ PyTorch installé mais CUDA non disponible. Réinstallation..."
            pip uninstall -y torch
            pip install --upgrade "torch>=2.1.0" --extra-index-url https://download.pytorch.org/whl/cu121
        fi
    else
        echo "Installation de PyTorch avec CUDA 12.1..."
        pip install --upgrade "torch>=2.1.0" --extra-index-url https://download.pytorch.org/whl/cu121
    fi
    
    # Vérifier l'installation de PyTorch
    if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA non disponible'"; then
        echo "❌ PyTorch installé mais CUDA toujours non disponible. Vérifiez votre installation CUDA."
        echo "⚠️ Continuons mais transformer-engine pourrait ne pas fonctionner correctement."
    else
        echo "✓ PyTorch installé avec support CUDA"
    fi
}

# Installer transformer-engine
install_transformer_engine() {
    echo -e "\n=== Installation de transformer-engine ==="
    
    # Désinstaller les versions existantes
    pip uninstall -y transformer-engine
    
    # Nettoyer le cache pip
    pip cache purge
    
    # D'abord essayer l'installation via pip
    echo "Installation via pip..."
    pip install "transformer-engine>=1.3.0"
    
    # Vérifier l'installation
    if python -c "import transformer_engine.pytorch as te; assert hasattr(te, 'fp8_autocast'), 'fp8_autocast non disponible'" 2>/dev/null; then
        echo "✓ transformer-engine installé avec succès via pip"
        return 0
    else
        echo "❌ Installation via pip échouée. Tentative d'installation depuis la source..."
    fi
    
    # Si l'installation pip échoue, essayer depuis la source
    echo -e "\n=== Installation depuis la source ==="
    
    # Cloner le dépôt
    TMP_DIR=$(mktemp -d)
    cd "$TMP_DIR"
    
    git clone https://github.com/NVIDIA/TransformerEngine.git
    cd TransformerEngine
    
    # Installer depuis la source
    python setup.py install
    
    # Vérifier l'installation
    if python -c "import transformer_engine.pytorch as te; assert hasattr(te, 'fp8_autocast'), 'fp8_autocast non disponible'" 2>/dev/null; then
        echo "✓ transformer-engine installé avec succès depuis la source"
        cd -
        rm -rf "$TMP_DIR"
        return 0
    else
        echo "❌ Installation depuis la source échouée."
        cd -
        rm -rf "$TMP_DIR"
        
        echo "⚠️ Échec de l'installation de transformer-engine."
        echo "Causes possibles:"
        echo "1. Version de CUDA incompatible"
        echo "2. Matériel non compatible avec FP8 (requis: H100, H200 ou SM86+)"
        echo "3. Problèmes de dépendances"
        
        echo -e "\nContinuons avec un fallback vers BF16. L'entraînement utilisera BF16 au lieu de FP8."
        return 1
    fi
}

# Vérification finale
verify_installation() {
    echo -e "\n=== Vérification finale ==="
    
    # Vérifier PyTorch
    if python -c "import torch; print(f'PyTorch {torch.__version__} avec CUDA {torch.version.cuda if torch.cuda.is_available() else \"non disponible\"}')"; then
        echo "✓ PyTorch correctement installé"
    else
        echo "❌ Problème avec PyTorch"
    fi
    
    # Vérifier transformer-engine
    if python -c "import transformer_engine; print(f'transformer-engine version: {transformer_engine.__version__ if hasattr(transformer_engine, \"__version__\") else \"inconnue\"}')"; then
        echo "✓ transformer-engine importé avec succès"
        
        # Vérifier fp8_autocast
        if python -c "import transformer_engine.pytorch as te; print(f'fp8_autocast disponible: {hasattr(te, \"fp8_autocast\")}')"; then
            echo "✓ API fp8_autocast disponible - FP8 est supporté!"
        else
            echo "❌ API fp8_autocast non disponible - FP8 non supporté"
        fi
    else
        echo "❌ Impossible d'importer transformer-engine"
    fi
    
    echo -e "\n=== Notes importantes ==="
    echo "1. Si tout est marqué ✓, vous pouvez utiliser FP8 en exécutant vos commandes avec --use_fp8"
    echo "2. Si des erreurs sont présentes, l'entraînement utilisera automatiquement BF16 comme fallback"
    echo "3. Pour les GPU non H100/H200, le support FP8 peut être limité ou non disponible"
}

# Fonction principale
main() {
    check_sudo
    check_cuda
    check_python
    install_system_deps
    install_pytorch
    install_transformer_engine
    verify_installation
    
    echo -e "\n=== Installation terminée ==="
    echo "Exécutez le script d'entraînement avec --use_fp8 pour utiliser FP8 si disponible"
    echo "En cas d'erreur, l'entraînement utilisera automatiquement BF16 comme fallback"
}

# Exécuter le script
main 