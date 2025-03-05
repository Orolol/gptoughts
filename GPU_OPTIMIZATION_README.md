# Optimisations GPU pour l'entraînement des modèles LLM

Ce document explique les optimisations GPU implémentées pour maximiser l'utilisation du GPU pendant l'entraînement des modèles LLM.

## Problème initial

Le GPU n'était pas pleinement utilisé pendant l'entraînement :
- La VRAM était presque pleine
- L'utilisation du GPU (GPU-util) était seulement autour de 50%
- La consommation d'énergie était de 230W sur un maximum de 450W

## Solutions implémentées

### 1. Optimisations CUDA avancées

Le fichier `gpu_optimization_enhanced.py` étend les optimisations CUDA existantes avec :

- **Optimisations spécifiques à l'architecture** : Configurations optimisées pour les GPUs Hopper, Ampere et grand public
- **Optimisations de mémoire** : Meilleure gestion de la fragmentation et préallocation de mémoire
- **Optimisations d'attention** : Activation de Flash Attention et autres optimisations d'attention si disponibles
- **Optimisations de calcul** : Configuration optimale pour les opérations de matmul et les Tensor Cores

### 2. Chargement de données asynchrone

- Implémentation d'un `AsyncDataLoader` qui précharge les batches en parallèle
- Réduction des temps d'attente du GPU pendant le chargement des données
- Transfert asynchrone des données vers le GPU avec `non_blocking=True`

### 3. Optimisation automatique des paramètres d'entraînement

- Détection automatique de la taille de batch optimale en fonction de la mémoire GPU disponible
- Ajustement automatique des étapes d'accumulation de gradient
- Sélection du type de données optimal (bfloat16, float16, float32) en fonction du GPU

### 4. Optimisations de la boucle d'entraînement

- Préchargement des batches pour masquer la latence de chargement
- Synchronisation CUDA stratégique pour maximiser l'utilisation du GPU
- Nettoyage de mémoire optimisé pour réduire la fragmentation

### 5. Compilation et parallélisme

- Activation de `torch.compile` pour les GPUs compatibles
- Optimisations pour l'entraînement distribué avec NCCL

## Comment utiliser ces optimisations

### Option 1 : Script d'entraînement optimisé

Utilisez le script `train_optimized.sh` qui configure automatiquement les paramètres optimaux :

```bash
# Entraînement avec paramètres par défaut
./train_optimized.sh

# Personnaliser le modèle et la taille
./train_optimized.sh --model deepseek --size medium

# Spécifier manuellement la taille du batch et l'accumulation de gradient
./train_optimized.sh --batch_size 16 --grad_accum 2

# Désactiver certaines optimisations
./train_optimized.sh --no_compile --no_preallocate
```

### Option 2 : Utiliser directement run_train_enhanced.py

```bash
python run_train_enhanced.py --model_type deepseek --size medium --compile --preallocate_memory --async_data_loading --optimize_attention
```

## Paramètres d'optimisation

| Paramètre | Description | Valeur par défaut |
|-----------|-------------|------------------|
| `--batch_size_auto_optimize` | Optimiser automatiquement la taille du batch | `True` |
| `--grad_accum_auto_optimize` | Optimiser automatiquement les étapes d'accumulation de gradient | `True` |
| `--preallocate_memory` | Préallouer la mémoire CUDA pour réduire la fragmentation | `True` |
| `--async_data_loading` | Utiliser le chargement de données asynchrone | `True` |
| `--optimize_attention` | Optimiser les opérations d'attention | `True` |
| `--dtype` | Type de données à utiliser (float32, float16, bfloat16) | Auto-détecté |
| `--compile` | Compiler le modèle avec torch.compile | `False` |

## Surveillance des performances

Pour surveiller l'utilisation du GPU pendant l'entraînement, vous pouvez utiliser :

```bash
# Dans un autre terminal
watch -n 1 nvidia-smi
```

Les statistiques d'utilisation du GPU sont également affichées périodiquement pendant l'entraînement.

## Dépannage

Si vous rencontrez des problèmes avec les optimisations :

1. **Erreurs de mémoire** : Réduisez la taille du batch ou augmentez les étapes d'accumulation de gradient
   ```bash
   ./train_optimized.sh --batch_size 8 --grad_accum 4
   ```

2. **Problèmes avec torch.compile** : Désactivez la compilation
   ```bash
   ./train_optimized.sh --no_compile
   ```

3. **Problèmes avec le chargement de données asynchrone** : Désactivez cette fonctionnalité
   ```bash
   ./train_optimized.sh --no_async
   ```

4. **Utilisation du GPU toujours faible** : Essayez d'ajuster manuellement les paramètres CUDA
   ```bash
   PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512" ./train_optimized.sh
   ```

## Optimisations futures possibles

- Implémentation de techniques de parallélisme de modèle (ZeRO, FSDP)
- Optimisations spécifiques aux opérateurs pour les modèles MoE
- Support pour les opérations en FP8 sur les GPUs Hopper
- Intégration avec CUDA Graphs pour les opérations répétitives