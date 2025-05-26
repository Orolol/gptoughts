# GPToughts - Framework d'entraînement LLM optimisé pour GPU

GPToughts est un framework de pointe pour l'entraînement et le fine-tuning de Large Language Models (LLMs), avec un focus particulier sur les optimisations GPU pour l'entraînement à échelle hobby/recherche.

## 🚀 Caractéristiques principales

### Architectures de modèles supportées

- **GPT-style autoregressive models** : Modèles de langage classiques avec attention causale
- **DeepSeek models** : Implémentation complète avec adapters et variantes MTP (Multi-Token Prediction)
- **MLA (Multi-head Latent Attention)** : Architecture d'attention optimisée avec compression latente
- **LLaDA (Large Language Diffusion with mAsking)** : Approche innovante basée sur la diffusion
- **MoE (Mixture of Experts)** : Support pour les modèles à experts multiples

### Optimisations GPU avancées

- **Support FP8** : Précision réduite pour GPUs H100/H200 avec stabilité numérique
- **GaLore optimizer** : Optimiseur à faible rang pour réduire l'utilisation mémoire
- **AdEMAMix optimizer** : Optimiseur adaptatif avec momentum mixte
- **Gradient checkpointing** : Économie de mémoire via recomputation
- **Flash Attention** : Implémentation optimisée de l'attention
- **Optimisations CUDA spécifiques** : Kernels personnalisés pour DeepSeek

### Infrastructure d'entraînement

- **PyTorch Lightning** : Framework d'entraînement moderne et scalable
- **Data loaders optimisés** : Multiples implémentations (packed, concatenated, legacy)
- **Monitoring avancé** : Intégration Weights & Biases, métriques détaillées
- **Checkpointing intelligent** : Sauvegarde automatique et reprise d'entraînement

## 📦 Installation

```bash
# Cloner le repository
git clone https://github.com/yourusername/gptoughts.git
cd gptoughts

# Installer les dépendances
pip install -r requirements.txt

# Configuration de l'environnement (recommandé)
pyenv activate 5090
```

## 🏃 Utilisation rapide

### Entraînement basique

```bash
# Entraîner un modèle GPT small
python run_train.py --model_type gpt --size small --batch_size 8 --block_size 2048

# Entraîner un modèle MLA avec optimisations
./train_mla_optimized.sh

# Entraîner avec GaLore (économie de mémoire)
./train_mla_galore.sh
```

### Scripts d'entraînement spécialisés

- `train_mla_optimized.sh` : MLA avec toutes les optimisations
- `train_mla_galore.sh` : MLA avec optimiseur GaLore
- `train_mla_selective.sh` : MLA avec attention sélective
- `train_parscale_mla.sh` : MLA avec ParScale (normalisation avancée)
- `train_deepseek_mtp.sh` : DeepSeek avec Multi-Token Prediction

### Optimisations GPU

```bash
# Activer toutes les optimisations
./optimize.sh --all

# Optimisations spécifiques
./optimize.sh --memory --cuda --fp8

# Récupération d'urgence en cas d'instabilité
./emergency_training.sh [checkpoint_dir]
```

## 🏗️ Architecture du projet

```
gptoughts/
├── models/               # Architectures de modèles
│   ├── blocks/          # Blocs de base (attention, MLP, etc.)
│   ├── deepseek/        # Variantes DeepSeek
│   ├── llada/           # Implémentation LLaDA
│   └── models/          # Classes de modèles haut niveau
├── train/               # Utilitaires d'entraînement
├── data/                # Loaders de données
├── optimization/        # Modules d'optimisation GPU
└── docs/                # Documentation détaillée
```

## 🔧 Configurations avancées

### MLA (Multi-head Latent Attention)

```python
# Configuration recommandée pour MLA
config = {
    "model_type": "mla",
    "size": "medium",
    "grad_clip": 1.0,
    "learning_rate": 3e-4,
    "use_fp8": False,  # Activer sur H100/H200
    "optimizer": "galore",  # ou "ademamix"
}
```

### Stabilité numérique

Pour éviter les problèmes de NaN/Inf :
- Utiliser `--grad_clip 1.0`
- Commencer avec BF16 avant FP8
- Activer les normalisations ParScale
- Voir `docs/numerical_stability.md`

## 📊 Performances

### Benchmarks typiques (RTX 4090)

| Modèle | Taille | Batch Size | Seq Length | Tokens/sec |
|--------|--------|------------|------------|------------|
| GPT    | Small  | 8          | 2048       | ~15K       |
| MLA    | Small  | 8          | 2048       | ~18K       |
| MLA+FP8| Small  | 16         | 2048       | ~32K       |

### Utilisation mémoire

- **Sans optimisations** : ~20GB pour modèle medium
- **Avec GaLore** : ~12GB pour modèle medium
- **Avec FP8** : ~10GB pour modèle medium

## 🛠️ Développement

### Ajouter une nouvelle architecture

1. Créer les blocs dans `models/blocks/`
2. Implémenter le modèle dans `models/models/`
3. Ajouter la configuration dans `models/config.py`
4. Créer un script d'entraînement

### Tests

```bash
# Validation rapide
python run_train.py --model_type your_model --size small --batch_size 1 --block_size 128 --max_steps 10

# Test de stabilité numérique
python test_dyt.py
```

## 📚 Documentation

- `docs/mla_doc.md` : Architecture MLA détaillée
- `docs/llada.md` : Approche diffusion LLaDA
- `docs/numerical_stability.md` : Guide de stabilité
- `docs/parscale_mla.md` : Normalisation ParScale

## 🤝 Contribution

Les contributions sont les bienvenues ! Points d'intérêt actuels :
- Optimisations supplémentaires pour GPUs consumer
- Support de nouvelles architectures (Mamba, RWKV)
- Amélioration de la stabilité FP8
- Documentation et tutoriels

## 📄 Licence

[À définir]

## 🙏 Remerciements

- PyTorch et PyTorch Lightning teams
- Auteurs des papers MLA, DeepSeek, et LLaDA
- Communauté open-source ML

---

*GPToughts - Entraînez vos LLMs efficacement, même sur du hardware consumer !*