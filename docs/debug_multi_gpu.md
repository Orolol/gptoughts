# Debugging Multi-GPU Training Issues

## Problème : VRAM fluctuante sur le GPU secondaire

Si vous observez que le GPU 1 a sa VRAM qui se vide et se remplit constamment pendant que le GPU 0 reste stable, cela indique un problème de **distribution de données inégale** ou de **synchronisation**.

## Modifications apportées

### 1. DataLoader conscient du DDP

**Fichier : `data/datasets.py`**
- Changé `batch_size=1` → `batch_size=None` pour le PackedFinewebDataset
- Ajouté `persistent_workers=False`

### 2. Dataset avec sharding DDP

**Fichier : `data/data_loader_packed.py`**
- Détection automatique du rang DDP et world_size
- Chaque rank skip des exemples différents pour éviter les overlaps
- Sharding au niveau de l'itération : `if example_count % world_size != rank: continue`

### 3. Configuration DDP optimisée

**Fichier : `run_train.py`**
- `find_unused_parameters=False` pour réduire l'overhead
- `gradient_as_bucket_view=True` pour optimiser la mémoire
- `static_graph=True` pour un graph de calcul fixe

### 4. Compilation désactivée

**Fichier : `scripts/train_swa_mla_optimized_gpt_tok.sh`**
- `--compile` retiré temporairement
- torch.compile peut causer des problèmes de synchronisation avec DDP

## Vérifications à faire

### 1. Vérifier que DDP est correctement détecté

Au démarrage de l'entraînement, vous devriez voir dans les logs :

```
[PackedDataset] DDP detected: rank=0, world_size=2
[PackedDataset] DDP detected: rank=1, world_size=2
[PackedDataset Rank 0] Will process every 2th example starting from offset 0
[PackedDataset Rank 1] Will process every 2th example starting from offset 1000
```

Si vous ne voyez PAS ces messages, le DDP n'est pas initialisé correctement.

### 2. Monitorer l'utilisation GPU en temps réel

```bash
# Terminal 1
watch -n 0.5 nvidia-smi

# Terminal 2 - Pour voir les logs détaillés
python run_train.py ... 2>&1 | tee training.log
```

### 3. Vérifier la distribution des batches

Ajoutez ce code dans `train/lightning_module.py` dans la méthode `training_step` :

```python
def training_step(self, batch, batch_idx):
    # Debugging: log batch info every 100 steps
    if self.global_step % 100 == 0 and self.trainer.world_size > 1:
        input_ids, targets = self._unpack_batch(batch)
        print(f"[Rank {self.global_rank}] Step {self.global_step}: "
              f"batch_shape={input_ids.shape}, "
              f"device={input_ids.device}, "
              f"non_pad_tokens={((targets != -100).sum().item())}")

    # ... rest of the method
```

**Attendu** : Chaque rank devrait afficher des informations similaires avec des shapes identiques.

## Solutions alternatives si le problème persiste

### Solution 1 : Utiliser FSDP au lieu de DDP

```bash
python run_train.py \
    ... \
    --strategy fsdp \
    ...
```

FSDP (Fully Sharded Data Parallel) distribue le modèle de manière plus équitable.

### Solution 2 : Forcer le dataloader classique avec DistributedSampler

Modifier `data/datasets.py` pour utiliser un dataset non-iterable avec DistributedSampler :

```python
from torch.utils.data.distributed import DistributedSampler

# Create dataset
train_dataset = NonIterableDataset(...)

# Create sampler for DDP
train_sampler = DistributedSampler(
    train_dataset,
    shuffle=True,
    drop_last=True  # Important for consistent batch sizes
)

# Create loader
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True
)
```

### Solution 3 : Réduire la complexité du modèle

Si le déséquilibre persiste, cela peut indiquer que certaines couches du modèle ont des charges de calcul très différentes.

Test : entraîner un modèle GPT simple pour voir si le problème existe :

```bash
python run_train.py \
    --model_type gpt \
    --size small \
    --batch_size 16 \
    --block_size 1024 \
    --strategy ddp_find_unused_parameters_false \
    --devices 2
```

Si le GPT simple fonctionne bien mais SWA-MLA non, le problème vient de l'architecture hybride.

### Solution 4 : Désactiver le gradient checkpointing

Le gradient checkpointing peut causer des patterns de mémoire irréguliers en DDP.

Dans les configs du modèle, chercher et mettre à `False` :
- `use_gradient_checkpointing=False`

## Diagnostic avancé avec PyTorch Profiler

Créer un script de profiling :

```python
# profile_ddp.py
import torch
from torch.profiler import profile, ProfilerActivity
import sys

# Import your training setup
from run_train import main

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # Run a few training steps
    main()  # Or call your training function

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Export for visualization
prof.export_chrome_trace("trace_ddp.json")
```

Puis visualiser avec Chrome : chrome://tracing

## Commandes de debugging NCCL

Si le problème vient de la communication inter-GPU (NCCL) :

```bash
# Activer les logs NCCL détaillés
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Vérifier la topologie NCCL
python -c "import torch; print(torch.cuda.nccl.version())"

# Test de bandwidth entre GPUs
nvidia-smi nvlink -s
```

## Checklist finale

- [ ] Les deux GPUs sont détectés : `nvidia-smi`
- [ ] DDP est initialisé : voir les logs `[PackedDataset] DDP detected`
- [ ] Chaque rank traite des données différentes : voir les logs de sharding
- [ ] Les shapes de batch sont identiques sur les deux ranks
- [ ] `gradient_as_bucket_view=True` est activé
- [ ] `find_unused_parameters=False` est activé
- [ ] torch.compile est désactivé temporairement
- [ ] Les GPUs ont des températures et utilisation similaires après 2-3 minutes

## Si rien ne fonctionne

Dernier recours : entraîner sur un seul GPU pour valider que le problème vient bien du multi-GPU :

```bash
python run_train.py \
    --model_type swa_mla \
    --size medium \
    --batch_size 32 \  # Double batch size puisqu'un seul GPU
    --block_size 2048 \
    --devices 1 \
    ...
```

Puis comparer les performances avec 2 GPUs. Si 2 GPUs donnent moins de 1.5x le throughput d'1 GPU, il y a définitivement un problème de scalabilité.

## Contact et ressources

- PyTorch DDP troubleshooting : https://pytorch.org/docs/stable/notes/ddp.html
- Lightning multi-GPU guide : https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html
- NCCL documentation : https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/