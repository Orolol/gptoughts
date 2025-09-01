# Corrections apportées au modèle LLaDA/BD3

## Problèmes identifiés

1. **Optimizer incompatible** : Lion optimizer causait des problèmes de double backward avec LLaDA
2. **Loss instable** : La régularisation d'entropie (coefficient 0.01) causait de l'instabilité
3. **Noise schedule non optimal** : Les paramètres beta/omega n'étaient pas configurables
4. **Calcul de loss BD3 incorrect** : Variables omega/beta non définies dans le contexte

## Corrections apportées

### 1. Fichiers modifiés

- `models/llada/model.py` : 
  - Ajout de la possibilité de désactiver la régularisation d'entropie
  - Paramètres BD3 (beta, omega) configurables via config
  - Correction du calcul de loss BD3
  
- `train/lightning_module.py` :
  - Passage des paramètres BD3 à la configuration du modèle
  
- `run_train.py` :
  - Ajout des arguments pour contrôler BD3 training

### 2. Nouveau script d'entraînement

Créé `train_llada_bd3_fixed.sh` avec :
- AdamW au lieu de Lion (plus stable pour diffusion)
- Learning rate réduit à 1e-4
- Warmup augmenté à 500 steps
- Batch size réduit à 4 avec gradient accumulation = 2
- Paramètres BD3 optimaux : beta=0.3, omega=0.8
- Désactivation de la régularisation d'entropie

## Utilisation recommandée

```bash
# Pour entraîner avec les corrections
./train_llada_bd3_fixed.sh medium 4 2048 out_llada_bd3

# Pour reprendre l'entraînement
./train_llada_bd3_fixed.sh medium 4 2048 out_llada_bd3 true
```

## Paramètres clés pour BD3

- `--use_bd3_training` : Active le mode BD3
- `--bd3_block_length 128` : Taille des blocs pour BD3
- `--bd3_beta 0.3` : Taux de masquage minimum
- `--bd3_omega 0.8` : Taux de masquage maximum  
- `--disable_entropy_regularization` : Désactive la régularisation d'entropie

## Monitoring

Surveillez ces métriques :
- `train/loss` : Devrait descendre progressivement (cible < 5)
- `train/grad_norm` : Devrait rester stable (< 10)
- Les textes générés devraient devenir cohérents après ~2000 steps