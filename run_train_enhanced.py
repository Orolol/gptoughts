"""
Point d'entrée amélioré pour l'entraînement des modèles LLM avec optimisations GPU avancées.
Ce script étend run_train.py avec des optimisations pour maximiser l'utilisation du GPU.
"""

import os
import sys
import argparse
import torch
import subprocess
from transformers import AutoTokenizer

# Importer les optimisations GPU avancées
from gpu_optimization_enhanced import (
    setup_enhanced_cuda_optimizations, 
    optimize_training_parameters,
    AsyncDataLoader,
    optimize_memory_allocation,
    optimize_attention_operations,
    print_enhanced_gpu_stats
)
from train.train_utils import get_gpu_count

def get_enhanced_datasets(block_size, batch_size, device, tokenizer=None, async_loading=True):
    """
    Retourne les datasets d'entraînement et de validation avec chargement asynchrone.
    
    Args:
        block_size: Taille du contexte
        batch_size: Taille du batch
        device: Appareil d'entraînement
        tokenizer: Tokenizer à utiliser
        async_loading: Activer le chargement asynchrone
    
    Returns:
        Tuple de (train_dataset, val_dataset)
    """
    try:
        # Essayer d'abord d'importer FinewebDataset
        from data.data_loader_original import FinewebDataset
        
        print("Utilisation de FinewebDataset pour l'entraînement")
        train_dataset = FinewebDataset(
            split='train',
            max_length=block_size,
            buffer_size=batch_size * 8,  # Augmenter la taille du buffer
            shuffle=True,
            tokenizer=tokenizer,
            batch_size=batch_size
        )
        
        val_dataset = FinewebDataset(
            split='train',
            max_length=block_size,
            buffer_size=batch_size * 4,  # Augmenter la taille du buffer
            shuffle=False,
            tokenizer=tokenizer,
            batch_size=batch_size
        )
        
    except ImportError:
        try:
            # Essayer ensuite StreamingDataset
            from data.data_loader_legacy import StreamingDataset
            
            print("Utilisation de StreamingDataset pour l'entraînement")
            train_dataset = StreamingDataset(
                split='train',
                block_size=block_size,
                batch_size=batch_size,
                device=device,
                tokenizer=tokenizer
            )
            
            val_dataset = StreamingDataset(
                split='validation',
                block_size=block_size,
                batch_size=batch_size,
                device=device,
                tokenizer=tokenizer
            )
            
        except ImportError:
            # Fallback sur un dataset simple
            print("Attention: Impossible d'importer les datasets spécialisés, utilisation d'un dataset simple")
            from torch.utils.data import TensorDataset
            
            # Créer des données aléatoires pour le test
            train_data = torch.randint(0, 1000, (100, block_size))
            val_data = torch.randint(0, 1000, (20, block_size))
            
            train_dataset = TensorDataset(train_data, train_data)
            val_dataset = TensorDataset(val_data, val_data)
    
    # Wrapper les datasets avec AsyncDataLoader si demandé
    if async_loading and torch.cuda.is_available():
        print("Activation du chargement de données asynchrone")
        train_dataset = AsyncDataLoader(train_dataset, buffer_size=3)
        val_dataset = AsyncDataLoader(val_dataset, buffer_size=2)
    
    return train_dataset, val_dataset

def launch_distributed_training(args, world_size):
    """
    Lance l'entraînement distribué avec le nombre spécifié de GPUs.
    Optimisé pour de meilleures performances.
    """
    print(f"Détection de {world_size} GPUs. Lancement de l'entraînement distribué...")
    
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = "localhost"
    current_env["MASTER_PORT"] = "29500"
    current_env["WORLD_SIZE"] = str(world_size)
    
    # Optimisations NCCL pour l'entraînement distribué
    current_env["NCCL_DEBUG"] = "INFO"
    current_env["NCCL_IB_DISABLE"] = "0"
    current_env["NCCL_NET_GDR_LEVEL"] = "2"
    current_env["NCCL_P2P_DISABLE"] = "0"
    current_env["NCCL_MIN_NCHANNELS"] = "4"
    current_env["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    current_env["NCCL_BUFFSIZE"] = "4194304"  # 4MB
    current_env["NCCL_ALGO"] = "Ring"
    current_env["NCCL_PROTO"] = "Simple"
    current_env["NCCL_NSOCKS_PERTHREAD"] = "4"
    current_env["NCCL_SOCKET_NTHREADS"] = "4"
    
    # Construire la commande avec tous les arguments
    cmd = [sys.executable, "train/train.py"]
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None:
            if isinstance(arg_value, bool):
                if arg_value:
                    cmd.append(f"--{arg_name}")
            else:
                cmd.append(f"--{arg_name}")
                cmd.append(str(arg_value))
    
    processes = []
    for rank in range(world_size):
        current_env["RANK"] = str(rank)
        current_env["LOCAL_RANK"] = str(rank)
        
        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)
    
    return processes

def monitor_processes(processes):
    """
    Surveille les processus d'entraînement et gère les échecs.
    """
    for process in processes:
        process.wait()
        if process.returncode != 0:
            print(f"L'entraînement a échoué avec le code de retour {process.returncode}")
            # Tuer tous les processus restants
            for p in processes:
                if p.poll() is None:  # Si le processus est toujours en cours
                    p.kill()
            return False
    return True

def main():
    """Point d'entrée principal avec optimisations GPU avancées"""
    parser = argparse.ArgumentParser(description='Entraîner des modèles LLM avec optimisations GPU avancées')
    
    # Paramètres du modèle
    parser.add_argument('--model_type', type=str, choices=['deepseek', 'llada', 'gpt'], default='gpt',
                       help='Type de modèle à entraîner')
    parser.add_argument('--size', type=str, choices=['small', 'medium', 'large'], default='small',
                       help='Taille du modèle')
    
    # Paramètres d'E/S
    parser.add_argument('--output_dir', type=str, default='out',
                       help='Répertoire de sortie pour les checkpoints')
    parser.add_argument('--eval_interval', type=int, default=1000,
                       help='Intervalle d\'évaluation')
    parser.add_argument('--log_interval', type=int, default=1,
                       help='Intervalle de journalisation')
    parser.add_argument('--eval_iters', type=int, default=100,
                       help='Nombre d\'itérations d\'évaluation')
    parser.add_argument('--eval_only', action='store_true',
                       help='Exécuter uniquement l\'évaluation')
    parser.add_argument('--always_save_checkpoint', action='store_true',
                       help='Toujours sauvegarder un checkpoint après chaque évaluation')
    parser.add_argument('--init_from', type=str, default='scratch', choices=['scratch', 'resume'],
                       help='Initialiser depuis zéro ou reprendre l\'entraînement')
    parser.add_argument('--keep_checkpoints', type=int, default=3,
                       help='Nombre de checkpoints à conserver')
    
    # Paramètres de données
    parser.add_argument('--batch_size', type=int, default=12,
                       help='Taille du batch')
    parser.add_argument('--block_size', type=int, default=512,
                       help='Taille du contexte')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Nombre d\'étapes d\'accumulation de gradient')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Taux de dropout')
    parser.add_argument('--bias', action='store_true',
                       help='Utiliser des biais dans les couches linéaires')
    parser.add_argument('--attention_backend', type=str, default=None,
                       help='Backend d\'attention à utiliser')
        # Paramètres d'optimisation
    parser.add_argument('--optimizer_type', type=str, default=None,
                       choices=['adamw', 'lion', 'apollo', 'apollo-mini'],
                       help='Type d\'optimiseur à utiliser (par défaut: adamw pour GPT, lion pour MoE/LLaDA)')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Taux d\'apprentissage')
    parser.add_argument('--max_iters', type=int, default=100000,
                       help='Nombre maximal d\'itérations')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                       help='Décroissance des poids')
    parser.add_argument('--beta1', type=float, default=0.9,
                       help='Beta1 pour Adam')
    parser.add_argument('--beta2', type=float, default=0.95,
                       help='Beta2 pour Adam')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Clip de gradient')
    parser.add_argument('--decay_lr', action='store_true',
                       help='Décroissance du taux d\'apprentissage')
    parser.add_argument('--warmup_iters', type=int, default=200,
                       help='Nombre d\'itérations de warmup')
    parser.add_argument('--lr_decay_iters', type=int, default=600000,
                       help='Nombre d\'itérations pour la décroissance du taux d\'apprentissage')
    parser.add_argument('--min_lr', type=float, default=3e-6,
                       help='Taux d\'apprentissage minimal')
    
    # Paramètres DDP
    parser.add_argument('--backend', type=str, default='nccl',
                       help='Backend pour DDP')
    parser.add_argument('--compile', action='store_true',
                       help='Compiler le modèle avec torch.compile')
    parser.add_argument('--distributed', action='store_true',
                       help='Utiliser l\'entraînement distribué')
    
    # Paramètres MoE
    parser.add_argument('--router_z_loss_coef', type=float, default=0.001,
                       help='Coefficient pour la perte du routeur')
    
    # Paramètres de tokenizer
    parser.add_argument('--tokenizer_name', type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                       help='Nom du tokenizer à utiliser')
    
    # Paramètres d'optimisation GPU avancés
    parser.add_argument('--batch_size_auto_optimize', action='store_true', default=True,
                       help='Optimiser automatiquement la taille du batch')
    parser.add_argument('--grad_accum_auto_optimize', action='store_true', default=True,
                       help='Optimiser automatiquement les étapes d\'accumulation de gradient')
    parser.add_argument('--preallocate_memory', action='store_true', default=True,
                       help='Préallouer la mémoire CUDA pour réduire la fragmentation')
    parser.add_argument('--async_data_loading', action='store_true', default=True,
                       help='Utiliser le chargement de données asynchrone')
    parser.add_argument('--optimize_attention', action='store_true', default=True,
                       help='Optimiser les opérations d\'attention')
    parser.add_argument('--dtype', type=str, choices=['float32', 'float16', 'bfloat16'], default='bfloat16',
                       help='Type de données à utiliser pour l\'entraînement')
    
    args = parser.parse_args()
    
    # Initialiser le tokenizer
    try:
        access_token = os.getenv('HF_TOKEN')
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True, access_token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        args.vocab_size = len(tokenizer)
        args.tokenizer = tokenizer
        print(f"Tokenizer initialisé: {args.tokenizer_name} avec taille de vocabulaire {args.vocab_size}")
    except Exception as e:
        print(f"Échec du chargement du tokenizer: {e}")
        args.vocab_size = 32000  # Valeur par défaut
        args.tokenizer = None
    
    # Initialiser les optimisations CUDA avancées
    if torch.cuda.is_available():
        setup_enhanced_cuda_optimizations()
        print_enhanced_gpu_stats()
        
        # Optimiser les opérations d'attention si demandé
        if args.optimize_attention:
            optimize_attention_operations()
        
        # Optimiser les paramètres d'entraînement
        # args = optimize_training_parameters(args)
        
        # Optimiser l'allocation de mémoire
        if args.preallocate_memory:
            optimize_memory_allocation()
    
    # Obtenir le nombre de GPUs disponibles
    world_size = get_gpu_count()
    
    # Remplacer la fonction get_datasets par get_enhanced_datasets
    from run_train import get_datasets
    get_datasets = lambda block_size, batch_size, device, tokenizer=None: get_enhanced_datasets(
        block_size, batch_size, device, tokenizer, async_loading=args.async_data_loading
    )
    
    # Injecter la fonction get_datasets améliorée dans le module run_train
    import run_train
    run_train.get_datasets = get_datasets
    
    if args.distributed and world_size > 1:
        # Lancer l'entraînement distribué
        processes = launch_distributed_training(args, world_size)
        
        # Surveiller les processus
        success = monitor_processes(processes)
        if not success:
            sys.exit(1)
    else:
        # Entraînement sur un seul GPU ou CPU
        print("Exécution sur un seul appareil")
        from train.train import Trainer
        
        trainer = Trainer(args)
        trainer.train()
    
    print("Entraînement terminé avec succès")

if __name__ == "__main__":
    main()