"""
Point d'entrée pour l'entraînement des modèles LLM.
Ce script permet de configurer et lancer l'entraînement de différents types de modèles.
"""

import os
import sys
import argparse
import torch
import subprocess
from transformers import AutoTokenizer

from gpu_optimization import setup_cuda_optimizations, print_gpu_stats
from train.train_utils import get_gpu_count

def get_datasets(block_size, batch_size, device, tokenizer=None):
    """
    Retourne les datasets d'entraînement et de validation.
    """
    try:
        # Essayer d'abord d'importer FinewebDataset
        from data.data_loader_original import FinewebDataset
        
        print("Using FinewebDataset for training")
        train_dataset = FinewebDataset(
            split='train',
            max_length=block_size,
            buffer_size=batch_size * 4,
            shuffle=True,
            tokenizer=tokenizer,
            batch_size=batch_size
        )
        
        val_dataset = FinewebDataset(
            split='train',
            max_length=block_size,
            buffer_size=batch_size * 2,
            shuffle=False,
            tokenizer=tokenizer,
            batch_size=batch_size
        )
        
    except ImportError:
        try:
            # Essayer ensuite StreamingDataset
            from data.data_loader_legacy import StreamingDataset
            
            print("Using StreamingDataset for training")
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
            print("Warning: Could not import specialized datasets, using simple dataset")
            from torch.utils.data import TensorDataset
            
            # Créer des données aléatoires pour le test
            train_data = torch.randint(0, 1000, (100, block_size))
            val_data = torch.randint(0, 1000, (20, block_size))
            
            train_dataset = TensorDataset(train_data, train_data)
            val_dataset = TensorDataset(val_data, val_data)
    
    return train_dataset, val_dataset

def launch_distributed_training(args, world_size):
    """
    Lance l'entraînement distribué avec le nombre spécifié de GPUs.
    """
    print(f"Found {world_size} GPUs. Launching distributed training...")
    
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = "localhost"
    current_env["MASTER_PORT"] = "29500"
    current_env["WORLD_SIZE"] = str(world_size)
    
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
            print(f"Training failed with return code {process.returncode}")
            # Tuer tous les processus restants
            for p in processes:
                if p.poll() is None:  # Si le processus est toujours en cours
                    p.kill()
            return False
    return True

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description='Train LLM models')
    
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
    parser.add_argument('--grad_clip', type=float, default=0.5,
                       help='Clip de gradient')
    parser.add_argument('--decay_lr', action='store_true',
                       help='Décroissance du taux d\'apprentissage')
    parser.add_argument('--warmup_iters', type=int, default=200,
                       help='Nombre d\'itérations de warmup')
    parser.add_argument('--lr_decay_iters', type=int, default=600000,
                       help='Nombre d\'itérations pour la décroissance du taux d\'apprentissage')
    parser.add_argument('--min_lr', type=float, default=3e-6,
                       help='Taux d\'apprentissage minimal')
    
    # Paramètres de précision
    parser.add_argument('--dtype', type=str, default=None,
                       choices=['float32', 'float16', 'bfloat16', 'fp8'],
                       help='Type de données à utiliser pour l\'entraînement')
    parser.add_argument('--use_fp8', action='store_true',
                       help='Utiliser la précision FP8 pour l\'entraînement (nécessite un GPU compatible comme H100)')
    
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
    
    args = parser.parse_args()
    
    # Initialiser le tokenizer
    try:
        access_token = os.getenv('HF_TOKEN')
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True, access_token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        args.vocab_size = len(tokenizer)
        args.tokenizer = tokenizer
        print(f"Initialized tokenizer: {args.tokenizer_name} with vocab size {args.vocab_size}")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        args.vocab_size = 32000  # Valeur par défaut
        args.tokenizer = None
    
    # Initialiser les optimisations CUDA
    if torch.cuda.is_available():
        setup_cuda_optimizations()
        print_gpu_stats()
    
    # Obtenir le nombre de GPUs disponibles
    world_size = get_gpu_count()
    
    if args.distributed and world_size > 1:
        # Lancer l'entraînement distribué
        processes = launch_distributed_training(args, world_size)
        
        # Surveiller les processus
        success = monitor_processes(processes)
        if not success:
            sys.exit(1)
    else:
        # Entraînement sur un seul GPU ou CPU
        print("Running on single device")
        from train.train import Trainer
        
        trainer = Trainer(args)
        trainer.train()
    
    print("Training completed successfully")

if __name__ == "__main__":
    main()