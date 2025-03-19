# GPToughts - Framework personnel d'entraînement de LLMs

Un framework léger mais puissant pour l'entraînement et le fine-tuning de modèles de langage (LLMs) comme hobby, avec un accent particulier sur l'optimisation des performances GPU.

## 🌟 Vue d'ensemble

GPToughts est un projet personnel visant à faciliter l'entraînement et l'expérimentation avec des modèles de langage. Ce framework a été conçu pour :

- Servir de plateforme d'apprentissage et d'expérimentation pour les modèles LLM
- Maximiser l'utilisation des ressources GPU disponibles
- Offrir une flexibilité dans le choix des modèles et des datasets
- Permettre d'implémenter et tester facilement des optimisations de performance

## 🧩 Caractéristiques principales

- **Entraînement hautement optimisé** : Multiples couches d'optimisations GPU pour maximiser l'utilisation du matériel
- **Architecture modulaire** : Séparation claire entre les modules de données, d'entraînement et d'optimisation
- **Support de divers modèles** : Compatible avec les modèles de la bibliothèque Hugging Face Transformers
- **Chargement de données efficace** : Implémentation de chargeurs de données asynchrones et optimisés
- **Optimisations spécifiques au matériel** : Configurations adaptées à différentes architectures GPU (Hopper, Ampere, etc.)
- **Entraînement distribué** : Support pour l'entraînement sur plusieurs GPUs

## 🛠️ Structure du projet

```
├── run_train.py              # Point d'entrée principal pour l'entraînement
├── run_train_enhanced.py     # Version améliorée avec optimisations avancées
├── gpu_optimization.py       # Optimisations GPU de base
├── gpu_optimization_advanced.py  # Optimisations GPU avancées
├── gpu_optimization_enhanced.py  # Optimisations GPU encore plus poussées
├── data/                     # Modules de chargement de données
├── train/                    # Utilitaires et modules d'entraînement
├── models/                   # Configurations et définitions de modèles
├── checkpoints/              # Sauvegarde des points de contrôle d'entraînement
├── out/                      # Sorties et logs d'entraînement
├── scripts d'entraînement    # Scripts pour lancer différentes configurations
│   ├── train_optimized.sh
│   ├── train_max_gpu.sh
│   ├── train_deepseek_optimized.sh
│   └── train_deepseek_stable.sh
└── documentation             # Documentation détaillée des fonctionnalités
    ├── GPU_OPTIMIZATION_README.md
    ├── ADVANCED_GPU_OPTIMIZATION.md
    └── llada.md
```

## �� Installation

Pour installer et configurer ce projet :

```bash
# Cloner le dépôt
git clone https://github.com/Orolol/gptoughts
cd gptoughts

# Utiliser la branche llada pour les dernières fonctionnalités
git checkout llada

# Installer les dépendances
pip install -r requirements.txt

# Configurer le token Hugging Face (nécessaire pour accéder à certains modèles)
export HF_TOKEN=your_huggingface_token
```

## 🚀 Utilisation

### Entraînement de base

```bash
python run_train.py --model_name huggyllama/llama-7b \
                   --block_size 2048 \
                   --batch_size 2 \
                   --learning_rate 1e-5
```

### Entraînement avec précision FP8 (requiert GPU H100/H200)

```bash
python run_train.py --model_type llada \
                   --size medium \
                   --use_fp8 \
                   --batch_size 32 \
                   --block_size 4096 \
                   --learning_rate 1e-5
```

### Entraînement optimisé

```bash
./train_optimized.sh
```

### Entraînement avec optimisations maximales

```bash
./train_max_gpu.sh
```

## 🔧 Optimisations GPU

Ce projet inclut plusieurs optimisations GPU pour améliorer les performances d'entraînement des modèles LLM. Pour utiliser ces optimisations, exécutez :

```bash
# Voir toutes les options disponibles
./optimize.sh --help

# Activer toutes les optimisations
./optimize.sh --all

# Activer uniquement certaines optimisations
./optimize.sh --memory --cuda

# Activer les optimisations et lancer l'entraînement
./optimize.sh --all --train --model llada --size medium
```

Pour plus de détails sur les optimisations disponibles, consultez le fichier [OPTIMIZATION_README.md](OPTIMIZATION_README.md).

## 🧪 Modèles supportés

- Modèles Llama (Llama 2, Llama 3)
- Modèles DeepSeek
- Autres modèles compatibles avec l'interface HuggingFace Transformers

## 📊 Suivi des expériences

Le framework intègre optionnellement Weights & Biases (wandb) pour le suivi des expérimentations :
- Métriques d'entraînement en temps réel
- Suivi de l'utilisation des ressources GPU
- Comparaison des différentes configurations d'entraînement

## 🔮 Projets futurs

- Implémentation de techniques d'optimisation supplémentaires
- Support pour PEFT (Parameter-Efficient Fine-Tuning)
- Intégration de techniques d'évaluation automatique de modèles
- Support pour l'entraînement hybride CPU/GPU pour les grands modèles

## 📝 Licence

Ce projet est un projet personnel développé comme hobby. Merci de respecter la propriété intellectuelle.

---

*GPToughts - Parce que l'entraînement des LLMs devrait être accessible à tous les passionnés d'IA.* 