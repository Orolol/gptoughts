# Module d'Optimisations GPU

Ce module regroupe les différentes optimisations GPU pour l'entraînement des modèles LLM.

## Structure

Le module d'optimisation est organisé comme suit :

- `fp8_optim.py` : Optimisations liées à la précision FP8
- `memory_optim.py` : Optimisations pour la gestion de la mémoire GPU
- `cuda_optim.py` : Optimisations générales CUDA et GPU
- `__init__.py` : Importations et fonction de configuration globale

## Utilisation

Le moyen le plus simple d'utiliser ces optimisations est via le script `optimize.sh` situé à la racine du projet.

### Utilisation du script optimize.sh

```bash
# Afficher l'aide et les options disponibles
./optimize.sh --help

# Activer toutes les optimisations
./optimize.sh --all

# Activer uniquement certaines optimisations
./optimize.sh --memory --cuda

# Activer les optimisations et lancer l'entraînement
./optimize.sh --all --train --model llada --size medium
```

### Options disponibles

- `--fp8` : Active les optimisations FP8 (pour GPUs H100, H200, Ada Lovelace)
- `--memory` : Active les optimisations de gestion mémoire
- `--cuda` : Active les optimisations CUDA générales
- `--flash-attention` : Active Flash Attention si disponible
- `--prealloc` : Préalloue la mémoire GPU pour éviter la fragmentation
- `--all` : Active toutes les optimisations disponibles
- `--stats` : Affiche les statistiques détaillées du GPU
- `--train` : Lance l'entraînement après la configuration

## Utilisation via Python

Vous pouvez également utiliser les optimisations directement dans votre code Python :

```python
# Importer toutes les optimisations
from optimization import configure_all_optimizations

# Configurer toutes les optimisations en une fois
configure_all_optimizations(use_fp8=True, prealloc_memory=True)

# Ou importer et utiliser des optimisations spécifiques
from optimization import setup_cuda_optimizations, optimize_memory_for_device

setup_cuda_optimizations()
optimize_memory_for_device()
```

## Notes

- Les optimisations FP8 nécessitent un GPU compatible (H100, H200, ou RTX Ada Lovelace)
- Les optimisations sont ajustées automatiquement en fonction du matériel détecté
- Pour les configurations avancées, consultez le code source des fichiers d'optimisation 