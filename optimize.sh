#!/bin/bash

# optimize.sh - Script principal pour configurer les optimisations GPU
# Ce script permet de sélectionner différentes optimisations pour l'entraînement

# Couleurs pour une meilleure lisibilité
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Fonction d'affichage
print_header() {
    echo -e "\n${BLUE}====== $1 ======${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Variables par défaut
USE_FP8=false
MEMORY_OPTIM=false
CUDA_OPTIM=false
FLASH_ATTENTION=false
PREALLOC_MEMORY=false
ALL_OPTIM=false
SHOW_GPU_STATS=false
ADVANCED_MODE=false
MODEL="llada"
SIZE="medium"
OUTPUT_DIR="out"
TRAIN_AFTER_OPTIM=false
OPTIMIZE_ATTENTION=false
AUTO_BATCH=false
COMPILE=false
SETUP_ONLY=false
RECOVER_NAN=false

# Fonction d'affichage d'aide
show_help() {
    echo -e "${BLUE}Optimize.sh - Utilitaire de configuration des optimisations GPU${NC}"
    echo ""
    echo "Options:"
    echo "  --fp8                  Active les optimisations FP8 (H100, H200, Ada Lovelace)"
    echo "  --memory               Active les optimisations mémoire"
    echo "  --cuda                 Active les optimisations CUDA générales"
    echo "  --flash-attention      Active Flash Attention si disponible"
    echo "  --attention            Active les optimisations d'attention avancées"
    echo "  --prealloc             Préalloue la mémoire GPU pour éviter la fragmentation"
    echo "  --auto-batch           Calcule automatiquement la taille de batch optimale"
    echo "  --compile              Active torch.compile pour le modèle (PyTorch 2.0+)"
    echo "  --recover-nan          Active la récupération automatique des NaN"
    echo "  --all                  Active toutes les optimisations disponibles"
    echo "  --stats                Affiche les statistiques GPU"
    echo "  --setup-only           Uniquement configurer l'environnement, sans exécuter"
    echo "  --advanced             Mode avancé avec plus d'options"
    echo "  --model MODEL          Spécifie le modèle à utiliser (défaut: llada)"
    echo "  --size SIZE            Spécifie la taille du modèle (défaut: medium)"
    echo "  --output-dir DIR       Spécifie le répertoire de sortie (défaut: out)"
    echo "  --train                Lance l'entraînement après configuration"
    echo "  --help                 Affiche cette aide"
    echo ""
    echo "Exemples:"
    echo "  ./optimize.sh --all"
    echo "  ./optimize.sh --fp8 --memory --train"
    echo "  ./optimize.sh --memory --cuda --auto-batch --model llada --size small"
}

# Analyser les arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --fp8) USE_FP8=true ;;
        --memory) MEMORY_OPTIM=true ;;
        --cuda) CUDA_OPTIM=true ;;
        --flash-attention) FLASH_ATTENTION=true ;;
        --attention) OPTIMIZE_ATTENTION=true ;;
        --prealloc) PREALLOC_MEMORY=true ;;
        --auto-batch) AUTO_BATCH=true ;;
        --compile) COMPILE=true ;;
        --recover-nan) RECOVER_NAN=true ;;
        --all) ALL_OPTIM=true ;;
        --stats) SHOW_GPU_STATS=true ;;
        --setup-only) SETUP_ONLY=true ;;
        --advanced) ADVANCED_MODE=true ;;
        --model) MODEL="$2"; shift ;;
        --size) SIZE="$2"; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift ;;
        --train) TRAIN_AFTER_OPTIM=true ;;
        --help) show_help; exit 0 ;;
        *) echo "Option inconnue: $1"; show_help; exit 1 ;;
    esac
    shift
done

# Si aucune option n'est spécifiée, afficher l'aide
if [[ "$USE_FP8" == "false" && "$MEMORY_OPTIM" == "false" && "$CUDA_OPTIM" == "false" && 
      "$FLASH_ATTENTION" == "false" && "$ALL_OPTIM" == "false" && "$SHOW_GPU_STATS" == "false" && 
      "$ADVANCED_MODE" == "false" && "$TRAIN_AFTER_OPTIM" == "false" && "$OPTIMIZE_ATTENTION" == "false" &&
      "$AUTO_BATCH" == "false" && "$COMPILE" == "false" && "$SETUP_ONLY" == "false" && "$RECOVER_NAN" == "false" ]]; then
    show_help
    exit 0
fi

# Si --all est spécifié, activer toutes les optimisations
if [[ "$ALL_OPTIM" == "true" ]]; then
    USE_FP8=true
    MEMORY_OPTIM=true
    CUDA_OPTIM=true
    FLASH_ATTENTION=true
    OPTIMIZE_ATTENTION=true
    PREALLOC_MEMORY=true
    AUTO_BATCH=true
    COMPILE=true
    RECOVER_NAN=true
    print_success "Toutes les optimisations activées"
fi

# Afficher les statistiques GPU si demandé
if [[ "$SHOW_GPU_STATS" == "true" || "$ALL_OPTIM" == "true" ]]; then
    print_header "STATISTIQUES GPU"
    python -c "from optimization.cuda_optim import print_gpu_stats; print_gpu_stats()"
fi

# Initialiser les variables pour stocker les résultats
BATCH_SIZE=""
GRAD_ACCUM=""

# Construire la chaîne de commande Python pour les optimisations
PYTHON_CMD="import sys;"

# Ajouter les importations nécessaires
if [[ "$SETUP_ONLY" == "true" ]]; then
    PYTHON_CMD+="from optimization import autoconfigure_environment; autoconfigure_environment();"
    python -c "$PYTHON_CMD"
    print_success "Environnement configuré avec succès"
    exit 0
fi

# Pour l'auto-batch, utiliser configure_all_optimizations
if [[ "$AUTO_BATCH" == "true" ]]; then
    print_header "CALCUL DE LA TAILLE DE BATCH OPTIMALE"
    
    # Construire les arguments pour configure_all_optimizations
    CONFIG_ARGS="use_fp8=$([[ $USE_FP8 == true ]] && echo 'True' || echo 'False'), "
    CONFIG_ARGS+="prealloc_memory=$([[ $PREALLOC_MEMORY == true ]] && echo 'True' || echo 'False'), "
    CONFIG_ARGS+="enable_flash_attn=$([[ $FLASH_ATTENTION == true ]] && echo 'True' || echo 'False'), "
    CONFIG_ARGS+="optimize_attention=$([[ $OPTIMIZE_ATTENTION == true ]] && echo 'True' || echo 'False'), "
    CONFIG_ARGS+="optimize_batch=True"
    
    # Exécuter et capturer le résultat
    RESULT=$(python -c "
import json
from optimization import configure_all_optimizations
result = configure_all_optimizations($CONFIG_ARGS)
print(json.dumps(result))
")
    
    # Extraire les résultats
    BATCH_SIZE=$(echo $RESULT | python -c "import sys, json; print(json.loads(sys.stdin.read()).get('batch_size', ''))")
    GRAD_ACCUM=$(echo $RESULT | python -c "import sys, json; print(json.loads(sys.stdin.read()).get('gradient_accumulation_steps', ''))")
    
    print_success "Taille de batch optimale: $BATCH_SIZE, étapes d'accumulation: $GRAD_ACCUM"
else
    # Importations standard (comme avant)
    if [[ "$USE_FP8" == "true" ]]; then
        PYTHON_CMD+="from optimization.fp8_optim import check_fp8_support, optimize_fp8_settings;"
    fi

    if [[ "$MEMORY_OPTIM" == "true" ]]; then
        PYTHON_CMD+="from optimization.memory_optim import optimize_memory_for_device, set_memory_optim_env_vars;"
        if [[ "$PREALLOC_MEMORY" == "true" ]]; then
            PYTHON_CMD+="from optimization.memory_optim import preallocate_cuda_memory;"
        fi
    fi

    if [[ "$CUDA_OPTIM" == "true" ]]; then
        PYTHON_CMD+="from optimization.cuda_optim import setup_cuda_optimizations;"
    fi

    if [[ "$FLASH_ATTENTION" == "true" ]]; then
        PYTHON_CMD+="from optimization.cuda_optim import enable_flash_attention;"
    fi
    
    if [[ "$OPTIMIZE_ATTENTION" == "true" ]]; then
        PYTHON_CMD+="from optimization.training_optim import optimize_attention_operations;"
    fi

    # Ajouter les optimisations à exécuter
    if [[ "$USE_FP8" == "true" ]]; then
        print_header "OPTIMISATIONS FP8"
        PYTHON_CMD+="print('Vérification du support FP8...'); fp8_supported = check_fp8_support(); optimize_fp8_settings();"
    fi

    if [[ "$MEMORY_OPTIM" == "true" ]]; then
        print_header "OPTIMISATIONS MÉMOIRE"
        PYTHON_CMD+="print('Configuration des optimisations mémoire...'); optimize_memory_for_device(); set_memory_optim_env_vars();"
        if [[ "$PREALLOC_MEMORY" == "true" ]]; then
            PYTHON_CMD+="preallocate_cuda_memory();"
        fi
    fi

    if [[ "$CUDA_OPTIM" == "true" ]]; then
        print_header "OPTIMISATIONS CUDA"
        PYTHON_CMD+="print('Configuration des optimisations CUDA...'); setup_cuda_optimizations();"
    fi

    if [[ "$FLASH_ATTENTION" == "true" ]]; then
        print_header "FLASH ATTENTION"
        PYTHON_CMD+="print('Vérification du support Flash Attention...'); enable_flash_attention();"
    fi
    
    if [[ "$OPTIMIZE_ATTENTION" == "true" ]]; then
        print_header "OPTIMISATIONS D'ATTENTION"
        PYTHON_CMD+="print('Optimisation des opérations d\'attention...'); optimize_attention_operations();"
    fi

    # Exécuter les optimisations Python
    python -c "$PYTHON_CMD"
fi

# Si l'option --train est activée, lancer l'entraînement
if [[ "$TRAIN_AFTER_OPTIM" == "true" ]]; then
    print_header "LANCEMENT DE L'ENTRAÎNEMENT"
    
    # Déterminer quel script d'entraînement utiliser
    TRAIN_SCRIPT="run_train.py"
    if [ -f "run_train_enhanced.py" ]; then
        TRAIN_SCRIPT="run_train_enhanced.py"
        print_success "Utilisation du script d'entraînement amélioré"
    fi
    
    # Construire la commande d'entraînement
    TRAIN_CMD="python $TRAIN_SCRIPT --model_type $MODEL --size $SIZE --output_dir $OUTPUT_DIR"
    
    # Ajouter les options d'optimisation
    if [[ "$USE_FP8" == "true" ]]; then
        TRAIN_CMD+=" --use_fp8"
    fi
    
    if [[ "$COMPILE" == "true" ]]; then
        TRAIN_CMD+=" --compile"
    fi
    
    if [[ "$AUTO_BATCH" == "true" && "$BATCH_SIZE" != "" ]]; then
        TRAIN_CMD+=" --batch_size $BATCH_SIZE"
        if [[ "$GRAD_ACCUM" != "" ]]; then
            TRAIN_CMD+=" --gradient_accumulation_steps $GRAD_ACCUM"
        fi
    fi
    
    # Ajouter des options pour éviter les NaN si demandé
    if [[ "$RECOVER_NAN" == "true" ]]; then
        TRAIN_CMD+=" --grad_clip 1.0 --weight_decay 0.05"
        if [[ "$MODEL" == "llada" ]]; then
            TRAIN_CMD+=" --router_z_loss_coef 0.0001"
        fi
    fi
    
    print_success "Commande: $TRAIN_CMD"
    
    # Lancer l'entraînement
    eval $TRAIN_CMD
fi

print_header "TERMINÉ"
print_success "Les optimisations ont été configurées avec succès!"
if [[ "$TRAIN_AFTER_OPTIM" == "false" ]]; then
    print_success "Pour lancer l'entraînement avec ces optimisations: ./optimize.sh --all --train"
fi 