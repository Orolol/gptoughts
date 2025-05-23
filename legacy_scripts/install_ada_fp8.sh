#!/bin/bash
# Script d'installation pour transformer-engine avec support FP8 partiel pour RTX 6000 Ada

set -e  # Arrêt à la première erreur

echo "=== Installation de transformer-engine pour RTX 6000 Ada avec support FP8 partiel ==="
echo "Ce script est optimisé pour les GPUs Ada Lovelace comme la RTX 6000 Ada"
echo "Pour les H100/H200, utilisez plutôt install_transformer_engine_complete.sh"

# Vérifier le GPU
check_gpu() {
    echo -e "\n=== Vérification du GPU ==="
    if ! command -v nvidia-smi &> /dev/null; then
        echo "❌ nvidia-smi non trouvé. GPU NVIDIA non détecté ou pilote non installé."
        return 1
    fi
    
    # Vérifier le modèle de GPU
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    echo "GPU détecté: $GPU_NAME"
    
    if [[ "$GPU_NAME" == *"RTX"* ]] || [[ "$GPU_NAME" == *"Ada"* ]] || [[ "$GPU_NAME" == *"6000"* ]]; then
        echo "✓ GPU compatible détecté"
    else
        echo "⚠️ Ce GPU pourrait ne pas être pleinement compatible avec FP8"
        echo "Les GPUs Ada Lovelace (RTX 4000 series, RTX 6000 Ada) ont un support partiel de FP8"
        echo "Voulez-vous continuer l'installation? (y/n)"
        read -r response
        if [[ "$response" != "y" ]]; then
            echo "Installation annulée."
            return 1
        fi
    fi
    
    return 0
}

# Installer PyTorch et Transformer Engine
install_packages() {
    echo -e "\n=== Installation des packages ===\n"
    
    # Désinstaller transformer-engine s'il est déjà installé
    pip uninstall -y transformer-engine
    
    # Installer depuis requirements-fp8-ada.txt
    if [ -f "requirements-fp8-ada.txt" ]; then
        echo "Installation depuis requirements-fp8-ada.txt..."
        pip install -r requirements-fp8-ada.txt
    else
        echo "Fichier requirements-fp8-ada.txt non trouvé, installation manuelle..."
        
        # Installation de PyTorch avec CUDA 12.1
        pip install torch>=2.1.0 --extra-index-url https://download.pytorch.org/whl/cu121
        
        # Installation de transformer-engine version compatible Ada
        pip install transformer-engine==0.10.0
        
        # Installation de Flash Attention
        pip install flash-attn>=2.3.0
    fi
    
    return 0
}

# Vérifier l'installation
verify_installation() {
    echo -e "\n=== Vérification de l'installation ===\n"
    
    # Créer un script Python temporaire
    TMP_SCRIPT=$(mktemp)
    cat > "$TMP_SCRIPT" << 'EOF'
import torch
import sys

try:
    import transformer_engine
    print(f"transformer-engine version: {getattr(transformer_engine, '__version__', 'inconnue')}")
    
    if not hasattr(transformer_engine, 'pytorch'):
        print("❌ Module transformer_engine.pytorch manquant")
        sys.exit(1)
        
    import transformer_engine.pytorch as te
    
    if hasattr(te, 'fp8_autocast'):
        print("✓ API fp8_autocast disponible")
    else:
        print("❌ API fp8_autocast non disponible")
        
    # Vérifier si recipe est disponible (versions récentes)
    if hasattr(te, 'common') and hasattr(te.common, 'recipe'):
        print("✓ API recipe disponible (support FP8 complet)")
    else:
        print("⚠️ API recipe non disponible (support FP8 limité)")
        
    # Vérifier BF16 comme fallback
    if torch.cuda.is_bf16_supported():
        print("✓ BF16 supporté (bon fallback si FP8 échoue)")
    else:
        print("⚠️ BF16 non supporté (FP16 sera utilisé comme fallback)")
        
    print("\nInstallation réussie!")
    
except ImportError as e:
    print(f"❌ Erreur d'importation: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Erreur: {e}")
    sys.exit(1)
EOF

    # Exécuter le script de vérification
    python "$TMP_SCRIPT"
    RESULT=$?
    
    # Nettoyer
    rm "$TMP_SCRIPT"
    
    # Vérifier si tout a fonctionné
    if [ $RESULT -eq 0 ]; then
        echo -e "\n✅ L'installation est terminée avec succès!"
        echo "Utilisez --use_fp8 lors de l'entraînement pour activer FP8"
        echo "Si FP8 n'est pas disponible, le système basculera automatiquement vers BF16"
        
        # Créer un lien vers ada_fp8_utils.py si nécessaire
        if [ -f "ada_fp8_utils.py" ]; then
            echo -e "\nVous pouvez également utiliser les fonctions de ada_fp8_utils.py:"
            echo "  - ada_fp8_autocast(): contexte optimisé pour FP8 sur Ada"
            echo "  - check_fp8_support(): vérifier le support FP8 détaillé"
            echo "  - get_best_precision_for_ada(): obtenir la meilleure précision disponible"
        fi
    else
        echo -e "\n❌ L'installation a échoué."
        echo "Essayez d'installer une version moins récente de transformer-engine:"
        echo "pip install transformer-engine==0.8.0"
    fi
}

# Fonction principale
main() {
    echo "Installation de transformer-engine avec support FP8 partiel pour RTX 6000 Ada"
    
    if ! check_gpu; then
        echo "Installation annulée: GPU incompatible"
        exit 1
    fi
    
    if ! install_packages; then
        echo "Installation des packages échouée"
        exit 1
    fi
    
    verify_installation
}

# Exécuter le script
main 