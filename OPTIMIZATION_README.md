# Migration des optimisations GPU

Ce document explique la nouvelle organisation des scripts d'optimisation GPU.

## Nouvelle structure

Les scripts d'optimisation ont été réorganisés comme suit :

1. Un nouveau dossier `optimization/` a été créé pour regrouper toutes les fonctions d'optimisation
2. Les fonctions ont été regroupées par catégorie :
   - `fp8_optim.py` : Optimisations FP8
   - `memory_optim.py` : Optimisations mémoire
   - `cuda_optim.py` : Optimisations CUDA générales
   - `training_optim.py` : Optimisations spécifiques à l'entraînement
3. Un script unique `optimize.sh` à la racine permet de sélectionner les optimisations

## Scripts obsolètes

Les scripts suivants sont désormais considérés comme obsolètes :

- `ada_fp8_utils.py` → remplacé par `optimization/fp8_optim.py`
- `install_ada_fp8.sh` → fonctionnalité intégrée dans `optimize.sh --fp8`
- `install_fp8_support.sh` → fonctionnalité intégrée dans `optimize.sh --fp8`
- `gpu_optimization.py` → séparé entre `memory_optim.py` et `cuda_optim.py`
- `gpu_optimization_advanced.py` → fonctionnalités intégrées dans le module `optimization`
- `train_fp8.sh` → remplacé par `optimize.sh --fp8 --train`
- `train_max_gpu.sh` → remplacé par `optimize.sh --all --train`
- `train_optimized.sh` → remplacé par `optimize.sh` avec les options appropriées
- `recover_from_nan.sh` → fonctionnalité intégrée dans `optimize.sh --recover-nan`

## Comment utiliser le nouveau système

Le nouveau script `optimize.sh` est conçu pour être simple à utiliser :

```bash
# Activer toutes les optimisations
./optimize.sh --all

# Activer uniquement certaines optimisations
./optimize.sh --memory --cuda

# Activer les optimisations et lancer l'entraînement
./optimize.sh --all --train --model llada --size medium

# Calculer automatiquement la taille de batch optimale
./optimize.sh --auto-batch

# Activer la récupération des NaN et la compilation du modèle
./optimize.sh --recover-nan --compile --train
```

## Nouvelles fonctionnalités

Le système d'optimisation offre maintenant plus de fonctionnalités :

1. **Détection automatique de la taille de batch optimale** (`--auto-batch`)
   - Calcule la taille de batch et le nombre d'étapes d'accumulation de gradient optimaux
   - Prend en compte le type de modèle et la mémoire GPU disponible

2. **Optimisations d'attention avancées** (`--attention`)
   - Active Flash Attention quand disponible
   - Optimise les opérations de Scaled Dot-Product Attention (SDPA)

3. **Compilation du modèle** (`--compile`)
   - Utilise `torch.compile` pour accélérer l'exécution du modèle
   - Nécessite PyTorch 2.0 ou supérieur

4. **Récupération automatique des NaN** (`--recover-nan`)
   - Détecte et corrige automatiquement les problèmes de NaN pendant l'entraînement
   - Ajuste les hyperparamètres pour améliorer la stabilité numérique

5. **Chargeur de données asynchrone**
   - Précharge les données sur GPU en parallèle pour réduire les temps d'attente

Pour plus de détails sur les options disponibles :

```bash
./optimize.sh --help
```

## Utilisation dans le code Python

Pour utiliser les optimisations dans votre code Python, vous pouvez importer le module d'optimisation :

```python
# Importer toutes les optimisations
from optimization import configure_all_optimizations

# Configurer toutes les optimisations en une fois et récupérer les paramètres optimaux
params = configure_all_optimizations(use_fp8=True, prealloc_memory=True, optimize_batch=True)
batch_size = params.get('batch_size')
grad_accum = params.get('gradient_accumulation_steps')

# Ou importer et utiliser des optimisations spécifiques
from optimization import (
    setup_cuda_optimizations,
    optimize_memory_for_device,
    get_optimal_batch_size_for_gpu,
    enable_torch_compile,
    recover_from_nan
)

# Exemples d'utilisation
setup_cuda_optimizations()
optimize_memory_for_device()
batch_size, grad_accum = get_optimal_batch_size_for_gpu(model_type="llada", size="medium")
model = enable_torch_compile(model)
```

## Documentation détaillée

Pour plus d'informations sur l'utilisation du module d'optimisation, consultez le fichier `optimization/README.md`. 