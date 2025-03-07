"""
Optimisations GPU avancées pour l'entraînement des modèles LLM.
Ce module étend les optimisations existantes pour maximiser l'utilisation du GPU.
"""

import os
import gc
import time
import torch
import random
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor

# Importer les fonctions d'optimisation existantes
from gpu_optimization import (
    setup_cuda_optimizations, cleanup_memory, print_memory_stats, 
    print_gpu_stats, preallocate_cuda_memory
)

def setup_enhanced_cuda_optimizations():
    """Configure des optimisations CUDA avancées pour maximiser l'utilisation du GPU."""
    # Appliquer d'abord les optimisations de base
    setup_cuda_optimizations()
    
    if not torch.cuda.is_available():
        print("CUDA n'est pas disponible, aucune optimisation avancée appliquée")
        return
    
    # Détecter le type de GPU pour des optimisations spécifiques
    gpu_name = torch.cuda.get_device_name().lower()
    detected_arch = "unknown"
    
    # Détection d'architecture
    if any(x in gpu_name for x in ['h100', 'h800']):
        detected_arch = "hopper"
    elif any(x in gpu_name for x in ['a100', 'a6000', 'a5000', 'a4000']):
        detected_arch = "ampere"
    elif any(x in gpu_name for x in ['4090', '4080', '4070', '4060', '3090', '3080', '3070', '2070']):
        detected_arch = "consumer"
    
    print(f"Configuration d'optimisations avancées pour GPU: {torch.cuda.get_device_name()} (architecture: {detected_arch})")
    
    # Optimisations communes avancées
    
    # Augmenter la taille du cache CUDA pour réduire les réallocations
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # Optimiser le backend cuDNN
    torch.backends.cudnn.benchmark = True
    
    # Optimisations spécifiques à l'architecture
    if detected_arch == "hopper":
        # H100/H800 - GPUs HPC haut de gamme
        os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"
        os.environ["NCCL_SOCKET_NTHREADS"] = "4"
        os.environ["NCCL_BUFFSIZE"] = "4194304"  # 4MB
        
        # Optimisations pour les opérations de calcul
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_tf32_reduced_precision_reduction = True
        
        # Optimisations de mémoire pour Hopper
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.8"
        
    elif detected_arch == "ampere":
        # A100 et autres GPUs datacenter Ampere
        os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"
        os.environ["NCCL_SOCKET_NTHREADS"] = "4"
        os.environ["NCCL_BUFFSIZE"] = "2097152"  # 2MB
        
        # Optimisations pour les opérations de calcul
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        
        # Optimisations de mémoire pour Ampere
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.8"
        
    elif detected_arch == "consumer":
        # GPUs grand public (séries RTX 3000/4000)
        os.environ["NCCL_NSOCKS_PERTHREAD"] = "2"
        os.environ["NCCL_SOCKET_NTHREADS"] = "2"
        
        # Optimisations pour les opérations de calcul
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        
        # Optimisations de mémoire pour GPUs grand public
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.7"
    
    # Optimisations pour les opérations de calcul
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, 'enable_math_sdp'):
        torch.backends.cuda.enable_math_sdp(True)
    
    # Optimisations pour les opérations de convolution
    if hasattr(torch.backends.cudnn, 'allow_tensor_core'):
        torch.backends.cudnn.allow_tensor_core = True
    
    print("Configuration d'optimisations CUDA avancées terminée")

def optimize_training_parameters(args):
    """
    Optimise les paramètres d'entraînement pour maximiser l'utilisation du GPU.
    
    Args:
        args: Arguments d'entraînement à optimiser
    
    Returns:
        args: Arguments optimisés
    """
    # Vérifier si CUDA est disponible
    if not torch.cuda.is_available():
        return args
    
    # Obtenir les propriétés du GPU
    device_props = torch.cuda.get_device_properties(0)
    total_memory = device_props.total_memory
    compute_capability = device_props.major + device_props.minor / 10
    
    # Calculer la mémoire disponible (en Go)
    total_memory_gb = (total_memory / (1024**3))
    print(f"Mémoire GPU totale: {total_memory_gb:.2f} Go")
    
    # Optimiser la taille du batch en fonction de la mémoire disponible
    # et du modèle utilisé
    model_type = args.model_type.lower()
    model_size = args.size.lower()
    
    # Facteurs de taille de modèle approximatifs (en paramètres)
    model_size_factors = {
        'deepseek': {
            'small': 1.0,
            'medium': 1.5,
            'large': 2.0
        },
        'llada': {
            'small': 0.5,
            'medium': 1.2,
            'large': 1.5
        },
        'gpt': {
            'small': 0.3,
            'medium': 0.7,
            'large': 1.5
        }
    }
    
    # Obtenir le facteur de taille du modèle
    size_factor = model_size_factors.get(model_type, {}).get(model_size, 1.0)
    
    # Calculer la taille de batch optimale
    # Formule heuristique basée sur la mémoire disponible et la taille du modèle
    optimal_batch_size = max(1, int((total_memory_gb * 0.7) / size_factor))
    print(f"Taille du batch optimale: {optimal_batch_size}")
    # Limiter la taille du batch à une valeur raisonnable
    optimal_batch_size = min(optimal_batch_size, 64)
    print(f"Taille du batch optimale limitée: {optimal_batch_size}")
    
    # Calculer les étapes d'accumulation de gradient optimales
    # Pour les petits batches, augmenter l'accumulation de gradient
    if optimal_batch_size < 4:
        optimal_grad_accum = 8
    elif optimal_batch_size < 8:
        optimal_grad_accum = 4
    elif optimal_batch_size < 16:
        optimal_grad_accum = 2
    else:
        optimal_grad_accum = 1
    
    # Ne pas modifier les paramètres si l'utilisateur les a explicitement définis
    # et qu'ils sont raisonnables
    if not hasattr(args, 'batch_size_auto_optimize') or args.batch_size_auto_optimize:
        if args.batch_size < optimal_batch_size // 2 or args.batch_size > optimal_batch_size * 2:
            print(f"Optimisation de la taille du batch: {args.batch_size} -> {optimal_batch_size}")
            args.batch_size = optimal_batch_size
    
    if not hasattr(args, 'grad_accum_auto_optimize') or args.grad_accum_auto_optimize:
        if args.gradient_accumulation_steps < optimal_grad_accum // 2 or args.gradient_accumulation_steps > optimal_grad_accum * 2:
            print(f"Optimisation des étapes d'accumulation de gradient: {args.gradient_accumulation_steps} -> {optimal_grad_accum}")
            args.gradient_accumulation_steps = optimal_grad_accum
    
    # Activer la compilation si disponible et si le GPU le supporte
    if compute_capability >= 7.0 and not args.compile:
        print("Activation de torch.compile pour accélérer l'entraînement")
        args.compile = True
    
    # Optimiser le type de données en fonction du GPU
    if not hasattr(args, 'dtype') or args.dtype is None:
        if compute_capability >= 8.0:
            # Ampere et plus récent - utiliser bfloat16
            args.dtype = 'bfloat16'
            print("Utilisation de bfloat16 pour l'entraînement")
        elif compute_capability >= 7.0:
            # Volta/Turing - utiliser float16
            args.dtype = 'float16'
            print("Utilisation de float16 pour l'entraînement")
        else:
            # GPUs plus anciens - utiliser float32
            args.dtype = 'float32'
            print("Utilisation de float32 pour l'entraînement")
    
    # Activer la préallocation de mémoire pour réduire la fragmentation
    if not hasattr(args, 'preallocate_memory'):
        args.preallocate_memory = True
        print("Activation de la préallocation de mémoire")
    
    return args

class AsyncDataLoader:
    """
    Chargeur de données asynchrone qui précharge les batches en parallèle
    pour éviter les goulots d'étranglement de chargement de données.
    """
    def __init__(self, dataset, buffer_size=3):
        """
        Initialise le chargeur de données asynchrone.
        
        Args:
            dataset: Dataset à charger
            buffer_size: Nombre de batches à précharger
        """
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.buffer = []
        self.iterator = iter(dataset)
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # Démarrer le thread de préchargement
        self.preload_thread = threading.Thread(target=self._preload_data)
        self.preload_thread.daemon = True
        self.preload_thread.start()
    
    def _preload_data(self):
        """Thread de préchargement des données."""
        while not self.stop_event.is_set():
            if len(self.buffer) < self.buffer_size:
                try:
                    # Obtenir le prochain batch
                    batch = next(self.iterator)
                    
                    # Ajouter au buffer
                    with self.lock:
                        self.buffer.append(batch)
                except StopIteration:
                    # Redémarrer l'itérateur si on atteint la fin
                    self.iterator = iter(self.dataset)
            else:
                # Attendre si le buffer est plein
                time.sleep(0.001)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """Retourne le prochain batch du buffer."""
        # Attendre qu'un batch soit disponible
        while len(self.buffer) == 0:
            time.sleep(0.001)
            
        # Retirer un batch du buffer
        with self.lock:
            batch = self.buffer.pop(0)
        
        return batch
    
    def close(self):
        """Ferme le chargeur de données et arrête le thread de préchargement."""
        self.stop_event.set()
        self.preload_thread.join(timeout=1.0)
        self.executor.shutdown(wait=False)

def optimize_memory_allocation():
    """
    Optimise l'allocation de mémoire CUDA pour réduire la fragmentation
    et améliorer les performances.
    """
    if not torch.cuda.is_available():
        return
    
    # Vider le cache CUDA
    torch.cuda.empty_cache()
    
    # Forcer une synchronisation CUDA
    torch.cuda.synchronize()
    
    # Collecter les objets Python non utilisés
    gc.collect()
    
    # Préallouer la mémoire CUDA
    preallocate_cuda_memory()
    
    # Optimiser l'allocateur CUDA
    if hasattr(torch.cuda, 'memory_stats'):
        # Configurer l'allocateur pour libérer la mémoire plus agressivement
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "") + ",garbage_collection_threshold:0.6"
    
    print("Optimisation de l'allocation de mémoire terminée")

def optimize_attention_operations():
    """
    Optimise les opérations d'attention pour améliorer les performances.
    """
    if not torch.cuda.is_available():
        return
    
    # Activer les optimisations d'attention Flash si disponibles
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'cuda'):
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
            print("Flash Attention activée")
        
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            print("Memory-efficient Attention activée")
    
    # Optimiser les opérations de matmul pour l'attention
    torch.backends.cuda.matmul.allow_tf32 = True
    
    print("Optimisation des opérations d'attention terminée")

def optimize_dataloader(dataloader, num_workers=2, pin_memory=True, prefetch_factor=2):
    """
    Optimise un DataLoader PyTorch pour améliorer les performances.
    
    Args:
        dataloader: DataLoader à optimiser
        num_workers: Nombre de workers pour le chargement parallèle
        pin_memory: Utiliser la mémoire épinglée pour les transferts CPU->GPU
        prefetch_factor: Facteur de préchargement par worker
    
    Returns:
        DataLoader optimisé
    """
    if not isinstance(dataloader, torch.utils.data.DataLoader):
        # Si ce n'est pas un DataLoader standard, retourner tel quel
        return dataloader
    
    # Créer un nouveau DataLoader avec des paramètres optimisés
    optimized_dataloader = torch.utils.data.DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        shuffle=isinstance(dataloader.sampler, torch.utils.data.RandomSampler),
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )
    
    return optimized_dataloader

def print_enhanced_gpu_stats():
    """Affiche des statistiques GPU détaillées avec des informations d'utilisation."""
    if not torch.cuda.is_available():
        print("CUDA n'est pas disponible")
        return
    
    # Afficher les statistiques de base
    print_gpu_stats()
    
    # Afficher des informations supplémentaires si disponibles
    try:
        # Essayer d'obtenir des statistiques d'utilisation via nvidia-smi
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit', '--format=csv,noheader,nounits'], 
                               stdout=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            stats = result.stdout.strip().split(',')
            if len(stats) >= 5:
                gpu_util = stats[0].strip()
                mem_util = stats[1].strip()
                temp = stats[2].strip()
                power_draw = stats[3].strip()
                power_limit = stats[4].strip()
                
                print("\n=== Statistiques d'utilisation GPU ===")
                print(f"Utilisation GPU: {gpu_util}%")
                print(f"Utilisation mémoire: {mem_util}%")
                print(f"Température: {temp}°C")
                print(f"Consommation: {power_draw}W / {power_limit}W ({float(power_draw)/float(power_limit)*100:.1f}%)")
                print("====================================\n")
    except Exception as e:
        print(f"Impossible d'obtenir des statistiques d'utilisation GPU détaillées: {e}")