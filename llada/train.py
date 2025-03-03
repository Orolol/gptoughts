import os
import math
import argparse
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler
from contextlib import nullcontext
import gc
import glob
import random
import threading
from collections import defaultdict
from contextlib import contextmanager
import sys
import traceback
# Add parent directory to path to import from root project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utilities
from train_utils import (
    get_gpu_count, setup_distributed, reduce_metrics, calculate_perplexity,
    get_lr, cleanup_old_checkpoints, cleanup_memory, ensure_model_dtype,
    print_memory_stats, setup_cuda_optimizations, save_checkpoint,
    load_checkpoint, find_latest_checkpoint, get_context_manager
)

# Import data loaders
from data_loader import FinewebDataset
from data_loader_llada import create_llm_data_loader

from llada import LLaDAConfig, LLaDAModel
from llada.llada import generate_example_wrapper, generate_and_display

# CUDA optimizations - optimized configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"

# Configure PyTorch threading for better performance
torch.set_num_threads(min(8, os.cpu_count()))
torch.set_num_interop_threads(min(8, os.cpu_count()))

# Configure inductor and dynamo for better compiled graph performance
try:
    import torch._inductor.config
    import torch._dynamo.config
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.cache_size_limit = 16384
    torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = False
    torch._inductor.config.debug = False
except ImportError:
    pass

class AveragedTimingStats:
    def __init__(self, print_interval=100):
        self.timings = defaultdict(list)
        self.print_interval = print_interval
        self.current_step = 0
        self.current_context = []
        
    @contextmanager
    def track(self, name):
        self.current_context.append(name)
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            # Only track the first level and its immediate children
            if len(self.current_context) <= 2:
                full_name = '/'.join(self.current_context)
                self.timings[full_name].append(duration)
            self.current_context.pop()
    
    def step(self):
        self.current_step += 1
    
    def should_print(self):
        return self.current_step % self.print_interval == 0
    
    def get_averaged_stats(self):
        if not self.timings:
            return {}, {}
        
        # Calculate averages
        avg_timings = {}
        for name, times in self.timings.items():
            avg_timings[name] = sum(times) / len(times)
        
        # Get total time from root operations
        total_time = sum(avg_timings[name] for name in avg_timings if '/' not in name)
        
        # Calculate percentages
        percentages = {}
        for name, time in avg_timings.items():
            percentages[name] = (time / total_time) * 100 if total_time > 0 else 0
        
        # Clear timings for next window
        self.timings.clear()
        
        return avg_timings, percentages
    
    def print_stats(self):
        timings, percentages = self.get_averaged_stats()
        if not timings:
            return
        
        print(f"\nTiming breakdown over last {self.print_interval} iterations:")
        
        # Sort by time spent (descending)
        sorted_timings = sorted(
            timings.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Print root operations first
        print("\nMain operations:")
        for name, time in sorted_timings:
            if '/' not in name:
                print(f"{name}: {time*1000:.1f}ms ({percentages[name]:.1f}%)")
        
        # Print sub-operations
        print("\nDetailed breakdown:")
        for name, time in sorted_timings:
            if '/' in name:
                print(f"{name}: {time*1000:.1f}ms ({percentages[name]:.1f}%)")

class TimingStats:
    """Track timing information for model training"""
    def __init__(self):
        self.timings = {}
        self.context_timers = {}
        self.counts = {}
        self.current_context = None
        
    def start_timer(self, name):
        if name not in self.timings:
            self.timings[name] = 0.0
            self.counts[name] = 0
        self.context_timers[name] = time.time()
        
    def end_timer(self, name):
        if name in self.context_timers:
            self.timings[name] += time.time() - self.context_timers[name]
            self.counts[name] += 1
            del self.context_timers[name]
            
    def track(self, name):
        class TimerContext:
            def __init__(self, stats, name):
                self.stats = stats
                self.name = name
                
            def __enter__(self):
                self.stats.start_timer(self.name)
                self.stats.current_context = self.name
                return self
                
            def __exit__(self, *args):
                self.stats.end_timer(self.name)
                self.stats.current_context = None
                
        return TimerContext(self, name)
        
    def summary(self):
        result = []
        total_time = sum(self.timings.values())
        for name, timing in self.timings.items():
            avg_time = timing / max(1, self.counts[name])
            percentage = 100 * timing / max(1e-6, total_time)
            result.append(f"{name}: {timing:.4f}s total, {avg_time:.6f}s avg, {percentage:.2f}%")
        return "\n".join(result)

class TextDataset(Dataset):
    """Simple text dataset for demonstration"""
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
        
    def __len__(self):
        return len(self.data) - self.block_size
        
    def __getitem__(self, idx):
        # Get chunk of data of block_size
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning with prompt-response pairs"""
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # Get prompt-response pair
        prompt, response = self.data[idx]
        
        # Ensure total length doesn't exceed block_size
        total_length = len(prompt) + len(response)
        if total_length > self.block_size:
            # Truncate from the right (response end)
            response = response[:self.block_size - len(prompt)]
            
        # Combine prompt and response
        combined = prompt + response
        x = torch.tensor(combined, dtype=torch.long)
        prompt_length = torch.tensor(len(prompt), dtype=torch.long)
        
        return {
            'input_ids': x,
            'prompt_lengths': prompt_length
        }

def preallocate_cuda_memory():
    """Préalloue de la mémoire CUDA pour éviter la fragmentation et améliorer les performances"""
    print("Préallocation de mémoire CUDA...")
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        try:
            # Calcule environ 70% de la mémoire disponible
            free_mem, total_mem = torch.cuda.mem_get_info(device)
            # Utilise 70% de la mémoire disponible
            mem_to_allocate = int(free_mem * 0.8)
            # Alloue un grand bloc continu
            prealloc = torch.empty(mem_to_allocate // 4, dtype=torch.float, device='cuda')
            # Libère immédiatement, mais laisse le bloc réservé dans le cache CUDA
            del prealloc
            torch.cuda.empty_cache()
            print(f"Préallocation terminée, {mem_to_allocate / (1024**3):.2f} GB réservés")
        except Exception as e:
            # Fallback si mem_get_info n'est pas disponible
            print(f"Préallocation avec méthode alternative: {e}")
            # Alloue ~70% de la mémoire totale si on ne peut pas obtenir l'info précise
            total_mem = torch.cuda.get_device_properties(device).total_memory
            mem_to_allocate = int(total_mem * 0.7)
            try:
                prealloc = torch.empty(mem_to_allocate // 4, dtype=torch.float, device='cuda')
                del prealloc
                torch.cuda.empty_cache()
                print(f"Préallocation terminée, ~{mem_to_allocate / (1024**3):.2f} GB réservés")
            except RuntimeError:
                print("Préallocation échouée - mémoire insuffisante. Continuons sans préallocation.")

def setup_cuda_optimizations():
    """Configure CUDA optimizations based on the detected GPU hardware."""
    if not torch.cuda.is_available():
        print("CUDA n'est pas disponible, aucune optimisation appliquée")
        return
    
    # Optimisations TF32 pour accélérer les calculs matriciels
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    
    # Detect GPU type for further optimizations
    gpu_name = torch.cuda.get_device_name().lower()
    detected_arch = "unknown"
    
    # Détection de l'architecture
    if any(x in gpu_name for x in ['h100', 'h800']):
        detected_arch = "hopper"
    elif any(x in gpu_name for x in ['a100', 'a6000', 'a5000', 'a4000']):
        detected_arch = "ampere"
    elif any(x in gpu_name for x in ['4090', '4080', '4070', '4060', '3090', '3080', '3070', '2070']):
        detected_arch = "consumer"
    
    print(f"Détecté GPU: {torch.cuda.get_device_name()} (architecture: {detected_arch})")
    
    # Optimisations communes
    torch.backends.cudnn.benchmark = True
    
    # Optimisations spécifiques à l'architecture
    if detected_arch == "hopper":
        # H100/H800 - GPUs HPC haut de gamme
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        torch.cuda.set_per_process_memory_fraction(0.98)
        # Désactiver le GC Python pour les cas temps-réel
        gc.disable()
        print("Optimisations Hopper appliquées: précision réduite pour BF16/FP16, connexions GPU optimisées")
        
    elif detected_arch == "ampere":
        # A100 et autres GPUs datacenter Ampere
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        torch.cuda.set_per_process_memory_fraction(0.95)
        print("Optimisations Ampere appliquées: précision réduite pour BF16/FP16, fraction mémoire 95%")
        
    elif detected_arch == "consumer":
        # GPUs grand public (RTX série 3000/4000)
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "8"
        torch.cuda.set_per_process_memory_fraction(0.85)
        print("Optimisations GPU Consumer appliquées: connexions multiples, fraction mémoire 85%")
        
    else:
        # GPU inconnu ou plus ancien - optimisations génériques
        print("Architecture GPU non reconnue, application d'optimisations génériques")
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
        torch.cuda.set_per_process_memory_fraction(0.8)
    
    # Optimisations de cuDNN
    if hasattr(torch.backends.cudnn, "allow_tensor_core"):
        torch.backends.cudnn.allow_tensor_core = True
        
    # Autocast settings
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        # Patch pour assurer la compatibilité avec les instructions personnalisées
        original_autocast = torch.cuda.amp.autocast
        def patched_autocast(*args, **kwargs):
            if not kwargs.get("enabled", True):
                return nullcontext()
            return torch.amp.autocast(device_type='cuda', *args, **kwargs)
        torch.cuda.amp.autocast = patched_autocast
        
    print("Configuration des optimisations CUDA terminée")

def print_gpu_stats():
    """Affiche des statistiques détaillées sur l'utilisation du GPU."""
    if not torch.cuda.is_available():
        return
        
    # Obtenir les propriétés du GPU
    device = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device)
    gpu_name = gpu_properties.name
    
    # Obtenir les informations de mémoire
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    max_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    
    # Calcul de la fragmentation (approximation)
    fragmentation = 0
    if reserved > 0:
        fragmentation = (reserved - allocated) / reserved * 100
    
    # Affichage
    print(f"\n===== GPU Stats ({gpu_name}) =====")
    print(f"Mémoire réservée: {reserved:.1f}MB (max: {max_reserved:.1f}MB)")
    print(f"Mémoire allouée: {allocated:.1f}MB (max: {max_allocated:.1f}MB)")
    print(f"Fragmentation estimée: {fragmentation:.1f}%")
    
    # Essayer d'obtenir des statistiques supplémentaires si nvidia-smi est disponible
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,temperature.gpu', 
                                 '--format=csv,noheader,nounits'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True)
        
        if result.returncode == 0:
            gpu_util, mem_util, temp = result.stdout.strip().split(',')
            print(f"Utilisation GPU: {gpu_util.strip()}%, Utilisation Mémoire: {mem_util.strip()}%, Température: {temp.strip()}°C")
    except:
        pass
    
    print("==============================")

def train_llada(args):
    """Train a LLaDA model with diffusion-based learning"""
    # DDP init
    ddp = args.distributed
    if ddp:
        init_distributed(args.backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        # otherwise we use the single-process version
        master_process = True
        device = args.device
    
    print(f"Starting training on device: {device}")
    
    # Création des dossiers de sauvegarde
    if master_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Application des optimisations CUDA
    if torch.cuda.is_available():
        # Configuration des optimisations CUDA de base
        setup_cuda_optimizations()
        
        # Préallocation de la mémoire si demandée
        if args.preallocate_memory:
            print("Préallocation de la mémoire CUDA...")
            preallocate_cuda_memory()
            
        # Affichage des statistiques GPU initiales
        if args.gpu_monitor_interval > 0 and master_process:
            print_gpu_stats()
    
    # dtype init
    dtype_dict = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    dtype = dtype_dict[args.dtype]
    
    # model init
    model_args = dict(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd, 
        block_size=args.block_size,
        dropout=args.dropout,
        vocab_size=args.vocab_size,
        mask_token_id=args.mask_token_id,
        num_experts=args.num_experts,
        k=args.k,
        bias=args.bias,
    )
    
    # Set random seed
    torch.manual_seed(1337)
    
    # Set up precision context
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    if args.dtype == 'float16':
        ctx = torch.amp.autocast(enabled=True, device_type='cuda')
        scaler = GradScaler()
    elif args.dtype == 'bfloat16' and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        ctx = torch.amp.autocast(enabled=True, device_type='cuda')
        scaler = GradScaler()
    else:
        ctx = nullcontext()
        scaler = None
    
    # Set up model
    print(f"Initializing LLaDA model with {args.n_layer} layers, {args.n_embd} dimensions")
    print(f"Use gradient checkpointing: {args.gradient_checkpointing}")
    config = LLaDAConfig(**model_args)
    
    # Resume from checkpoint if specified
    if args.init_from == 'resume':
        print(f"Attempting to resume training from {args.output_dir}")
        ckpt_path = find_latest_checkpoint(args.output_dir)
        if not ckpt_path:
            print("No checkpoints found, initializing from scratch instead")
            args.init_from = 'scratch'
        else:
            print(f"Loading checkpoint: {ckpt_path}")
            model = LLaDAModel(config)
            model, optimizer, iter_num, best_val_loss, _, _ = load_checkpoint(
                ckpt_path, model, map_location='cpu'
            )
            
            if optimizer is None:
                optimizer = model.configure_optimizers(
                    weight_decay=args.weight_decay,
                    learning_rate=args.learning_rate,
                    betas=(args.beta1, args.beta2),
                    device_type=device_type
                )
            
            # Move model to device and ensure correct dtype
            model = model.to(device)
            ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
            model = ensure_model_dtype(model, ptdtype)
            
            # Reset scaler
            scaler = GradScaler(enabled=(args.dtype == 'bfloat16' or args.dtype == 'float16'))
            
            # Initialize learning rate scheduler as None since we're using manual LR updates
            lr_scheduler = None
            
            cleanup_memory()
    
    # Initialize from scratch
    if args.init_from == 'scratch':
        print("Initializing a new LLaDA model from scratch")
        model = LLaDAModel(config)
        model = model.to(device)
        optimizer = model.configure_optimizers(
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            betas=(args.beta1, args.beta2),
            device_type=device_type
        )
        iter_num = 0
        best_val_loss = float('inf')
        
        # Initialize learning rate scheduler as None since we're using manual LR updates
        lr_scheduler = None
    
    # Print model size
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Set up timing stats
    timing_stats = AveragedTimingStats(print_interval=10)
    model.set_timing_stats(timing_stats)
    
    # Initialize DDP if needed
    if args.distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(
            model,
            device_ids=[ddp_local_rank],
            output_device=ddp_local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True
        )
        ddp = True
    else:
        ddp = False
    
    # Enable gradient checkpointing for memory efficiency if requested
    if args.gradient_checkpointing:
        if ddp:
            model.module.set_gradient_checkpointing(True)
        else:
            model.set_gradient_checkpointing(True)
    
    # Compile model if requested
    if args.compile and hasattr(torch, "compile"):
        print(f"Compilation du modèle optimisée avec max-autotune...")
        if hasattr(torch._inductor, "config"):
            # Configurations avancées pour l'inductor
            torch._inductor.config.conv_1x1_as_mm = True
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.epilogue_fusion = True
            torch._inductor.config.coordinate_descent_check_all_directions = True
            
            # Mode pour CPU ou GPU
            if device.startswith('cuda'):
                torch._inductor.config.triton.unique_kernel_names = True
                torch._inductor.config.triton.persistent_reductions = True
                
        compile_mode = "max-autotune" if "max-autotune" in torch.compile.__code__.co_varnames else "default"
        model = torch.compile(model, mode=compile_mode, dynamic=True, fullgraph=False)
        print("Modèle compilé avec succès")
    
    # Use streaming dataset for better performance
    print("Setting up data loaders...")
    
    # Helper function to get a new loader instance
    def get_train_loader():
        if args.use_streaming_dataset:
            # Determine recommended number of workers based on CPUs
            cpu_count = os.cpu_count() or 4
            num_workers = min(4, max(1, cpu_count // 2))
            
            # Create optimized data loader
            return create_llm_data_loader(
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_length=args.block_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                buffer_size=max(8, args.buffer_size),
                num_workers=num_workers,
                report_interval=100
            )
        else:
            # Use the simple datasets for demonstration
            if args.mode == 'pretrain':
                # Random data for pre-training
                # Use a slightly smaller upper bound to avoid edge cases
                safe_vocab_size = min(args.vocab_size, args.vocab_size - 100)
                data = torch.randint(0, safe_vocab_size, (10000,)).tolist()
                dataset = TextDataset(data, args.block_size)
                return DataLoader(
                    dataset, 
                    batch_size=args.batch_size, 
                    shuffle=True,
                    pin_memory=True,
                    num_workers=2
                )
            else:
                # Simple supervised fine-tuning data
                # Create prompt-response pairs
                data = [
                    ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]),  # prompt, response
                    ([11, 12, 13], [14, 15, 16, 17, 18, 19, 20]),
                    ([21, 22, 23, 24], [25, 26, 27, 28, 29])
                ]
                dataset = SFTDataset(data, args.block_size)
                return DataLoader(
                    dataset, 
                    batch_size=args.batch_size, 
                    shuffle=True,
                    pin_memory=True,
                    num_workers=2
                )
    
    if args.use_streaming_dataset:
        print("Using streaming dataset")
        from transformers import AutoTokenizer
        
        # Initialize tokenizer
        access_token = os.getenv('HF_TOKEN')
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct", 
            use_fast=True, 
            access_token=access_token
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # Determine recommended number of workers based on CPUs
        cpu_count = os.cpu_count() or 4
        num_workers = min(4, max(1, cpu_count // 2))
        
        print(f"Using optimized LLMDataLoader with {num_workers} workers")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    
    # Get initial train loader
    train_loader = get_train_loader()
    
    # Initialize timing
    t0 = time.time()
    local_iter_num = 0
    train_start_time = time.time()
    tokens_per_iter = args.batch_size * args.block_size * args.gradient_accumulation_steps
    total_tokens = 0
    tokens_window = []  # For calculating a moving average of tokens/s
    window_size = 10    # Size of the window for the moving average
    
    print(f"Beginning {args.mode} training")
    print(f"Model size: {config.n_embd}d, {config.n_layer}l, {config.n_head}h")
    print(f"Batch size: {args.batch_size}, Block size: {args.block_size}")
    print(f"Using {args.num_experts} experts with top-{args.k} routing")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print_memory_stats("Initial")
    
    # Training loop
    global_step = 0
    total_tokens = 0
    
    # Initialisation des variables de tracking de performances
    batch_start_time = time.time()
    last_log_time = time.time()
    
    # BOUCLE PRINCIPALE
    try:
        while True:
            # Temps de début pour le calcul des tokens/seconde
            batch_start_time = time.time()
            
            # determine and set the learning rate for this iteration
            with timing_stats.track("lr_update"):
                lr = get_lr(iter_num, args.warmup_iters, args.lr_decay_iters, args.learning_rate, args.min_lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            # Generate text occasionally to show progress
            if iter_num % 200 == 0 and args.mode == 'pretrain' and master_process:
                try:
                    model.eval()
                    with torch.no_grad():
                        with timing_stats.track("generation"):
                            # Simple prompt (just 3 tokens for testing)
                            prompt = [1, 2, 3]  # Example prompt
                            prompt_tokens = torch.tensor([prompt], dtype=torch.long, device=device)
                            
                            # Check if we have the new wrapper function
                            if args.use_streaming_dataset and 'tokenizer' in locals():
                                # If we have a streaming dataset, we should have a tokenizer
                                output_tokens, output_text = generate_example_wrapper(
                                    model.module if ddp else model,
                                    prompt_tokens,
                                    tokenizer=tokenizer,
                                    steps=args.gen_steps,
                                    gen_length=args.gen_length,
                                    temperature=args.temperature
                                )
                            else:
                                # Fall back to the old method
                                generated_text = generate_example(
                                    model.module if ddp else model, 
                                    prompt, 
                                    device, 
                                    args
                                )
                                print("\nGenerated example:")
                                print(generated_text)
                                print("\n")
                    model.train()
                except Exception as e:
                    print(f"Generation during training failed: {e}")
                    print("Continuing with training...")
                    model.train()  # Ensure we go back to training mode
            
            # forward backward update, with gradient accumulation
            with timing_stats.track("optimization"):
                # Réinitialiser le timer pour calculer le débit en tokens/sec
                batch_start_time = time.time()
                total_tokens = 0
                
                # Réinitialiser les gradients
                optimizer.zero_grad(set_to_none=True)
                
                # Pertes cumulatives pour le reporting
                loss_accumulator = 0.0
                router_loss_accumulator = 0.0
                
                # Réinitialiser DDP sync si nécessaire
                if ddp:
                    model.require_backward_grad_sync = (args.gradient_accumulation_steps == 1)
                
                # Configuration pour CUDA Graphs
                use_cuda_graph = (args.use_cuda_graphs and 
                                  torch.cuda.is_available() and 
                                  hasattr(torch.cuda, 'CUDAGraph') and 
                                  not ddp and 
                                  device_type == 'cuda')
                
                # Importante optimisation: ne pas utiliser CUDA Graphs si accumulation > 4
                # car cela peut causer des ralentissements importants
                if args.gradient_accumulation_steps > 4:
                    use_cuda_graph = False
                    if args.use_cuda_graphs and iter_num == 0:
                        print("CUDA Graphs désactivés car gradient_accumulation_steps > 4")
                elif use_cuda_graph and iter_num == 0:
                    print("CUDA Graphs activés - premier iter servira de warmup")
                
                # Utilisation du monitoring GPU si demandé
                if args.gpu_monitor_interval > 0 and iter_num % args.gpu_monitor_interval == 0 and master_process:
                    print_gpu_stats()
                
                # Limiter le nombre de graphes pour éviter explosion mémoire
                max_graphs_per_shape = 2
                static_input_shapes = {}
                cuda_graphs = {}
                
                # Pré-chargement des lots pour tous les micro-steps
                batches = []
                try:
                    with timing_stats.track("data_preloading"):
                        # Vérifier si le data loader a une méthode pour récupérer plusieurs lots en une fois
                        data_loading_start = time.time()
                        
                        if args.use_streaming_dataset and hasattr(train_loader, 'get_accumulation_batches'):
                            # Méthode optimisée pour récupérer tous les lots d'un coup
                            batches = train_loader.get_accumulation_batches(args.gradient_accumulation_steps)
                            if master_process and iter_num % args.log_interval == 0:
                                data_loading_time = time.time() - data_loading_start
                                print(f"🔄 Data loading time (optimized): {data_loading_time*1000:.1f}ms for {args.gradient_accumulation_steps} batches")
                                if len(batches) != args.gradient_accumulation_steps:
                                    print(f"⚠️ Warning: Expected {args.gradient_accumulation_steps} batches, got {len(batches)}")
                        else:
                            # Fallback: récupérer les lots un par un
                            for i in range(args.gradient_accumulation_steps):
                                load_start = time.time()
                                if args.use_streaming_dataset:
                                    # Nouveau data loader optimisé
                                    batch = next(train_loader)
                                    batches.append(batch)
                                elif args.mode == 'pretrain':
                                    if isinstance(train_loader, DataLoader):
                                        x, y = next(iter(train_loader))
                                        batches.append((x, y))
                                    else:
                                        x, y = next(train_loader)
                                        batches.append((x, y))
                                else:
                                    # SFT batch
                                    if isinstance(train_loader, DataLoader):
                                        batch = next(iter(train_loader))
                                        batches.append(batch)
                                    else:
                                        batch = next(train_loader)
                                        batches.append(batch)
                                
                                # Log détaillé du temps de chargement
                                if master_process and iter_num % args.log_interval == 0:
                                    load_time = time.time() - load_start
                                    print(f"🔄 Data loading time (batch {i+1}/{args.gradient_accumulation_steps}): {load_time*1000:.1f}ms")
                except StopIteration:
                    if args.use_streaming_dataset and master_process:
                        print("Fin de l'itération du dataloader, réinitialisation...")
                    train_loader = get_train_loader()
                    # On saute cette itération
                    continue
                
                # Boucle pour l'accumulation de gradients
                for micro_step in range(args.gradient_accumulation_steps):
                    # Activer la synchronisation de gradient DDP uniquement pour le dernier micro_step
                    if ddp and micro_step == args.gradient_accumulation_steps - 1:
                        model.require_backward_grad_sync = True
                    
                        # Récupération du batch pré-chargé
                        with timing_stats.track(f"data_loading_{micro_step}"):
                            if args.use_streaming_dataset:
                                batch = batches[micro_step]
                                # Transférer les données en mode non-bloquant et asynchrone
                                with torch.cuda.stream(torch.cuda.Stream()) if not args.disable_non_blocking else nullcontext():
                                    x = batch['input_ids'].to(device, non_blocking=not args.disable_non_blocking)
                                    y = batch['labels'].to(device, non_blocking=not args.disable_non_blocking)
                            elif args.mode == 'pretrain':
                                x, y = batches[micro_step]
                                x = x.to(device, non_blocking=not args.disable_non_blocking)
                                y = y.to(device, non_blocking=not args.disable_non_blocking)
                            else:
                                # SFT batch
                                batch = batches[micro_step]
                                batch = {k: v.to(device, non_blocking=not args.disable_non_blocking) for k, v in batch.items()}
                        
                        # Création d'une clé unique pour chaque taille d'entrée
                        input_shape_key = None
                        if args.use_streaming_dataset or args.mode == 'pretrain':
                            input_shape_key = tuple(x.shape)
                        else:
                            input_shape_key = tuple(batch['input_ids'].shape)
                            
                        # Clé unique pour ce micro-step et cette forme d'entrée
                        cuda_graph_key = (input_shape_key, micro_step % max_graphs_per_shape) if input_shape_key is not None else None
                        use_graph_for_step = use_cuda_graph and cuda_graph_key is not None
                        
                        # TRAITEMENT SELON LA STRATÉGIE: CUDA GRAPH RÉUTILISÉ, CAPTURE, OU EXÉCUTION NORMALE
                        if use_graph_for_step and cuda_graph_key in cuda_graphs and iter_num > 1:
                            # CAS 1: RÉUTILISATION D'UN GRAPHE CUDA EXISTANT
                            with timing_stats.track(f"cuda_graph_replay_{micro_step}"):
                                # Récupérer les entrées statiques précédemment créées
                                static_inputs = static_input_shapes[cuda_graph_key]
                                
                                # Copier les nouvelles données dans les tenseurs statiques
                                with torch.no_grad():
                                    if args.use_streaming_dataset or args.mode == 'pretrain':
                                        static_inputs['x'].copy_(x)
                                        static_inputs['y'].copy_(y)
                                    else:
                                        for k, v in batch.items():
                                            static_inputs[k].copy_(v)
                                
                                # Exécuter le graphe CUDA préalablement capturé
                                cuda_graphs[cuda_graph_key].replay()
                                

                                # Récupérer les résultats depuis les tenseurs statiques
                                loss_value = static_inputs['loss'].item()
                                router_loss_value = static_inputs.get('router_loss', torch.tensor(0.0)).item()
                                
                                    # Mise à jour des statistiques
                                loss_accumulator += loss_value
                                router_loss_accumulator += router_loss_value
                                
                                # Comptabiliser les tokens traités
                                batch_size = x.size(0) if args.use_streaming_dataset or args.mode == 'pretrain' else batch['input_ids'].size(0)
                                batch_tokens = batch_size * args.block_size
                                total_tokens += batch_tokens
                                
                            if iter_num % 100 == 0 and micro_step == 0:
                                print(f"Réutilisation du graphe CUDA pour micro_step {micro_step}")
                                
                        elif use_graph_for_step and iter_num == 1 and micro_step < max_graphs_per_shape:
                            # CAS 2: CAPTURE D'UN NOUVEAU GRAPHE CUDA
                            with timing_stats.track(f"cuda_graph_capture_{micro_step}"):
                                # Créer des copies des entrées
                                static_inputs = {}
                                if args.use_streaming_dataset or args.mode == 'pretrain':
                                    static_inputs['x'] = x.clone()
                                    static_inputs['y'] = y.clone()
                                else:
                                    static_inputs = {k: v.clone() for k, v in batch.items()}
                                
                                # Préparer les tenseurs de sortie
                                static_inputs['loss'] = torch.zeros(1, device=device)
                                if args.mode == 'pretrain':
                                    static_inputs['router_loss'] = torch.zeros(1, device=device)
                                
                                # Synchroniser et préparer le graphe
                                torch.cuda.synchronize()
                                cuda_graphs[cuda_graph_key] = torch.cuda.CUDAGraph()
                                
                                # Capturer le graphe complet (forward + backward)
                                with torch.cuda.graph(cuda_graphs[cuda_graph_key]):
                                    # Forward
                                    with torch.amp.autocast(enabled=True, device_type='cuda') if ctx else nullcontext():
                                        if args.use_streaming_dataset or args.mode == 'pretrain':
                                            outputs = model(static_inputs['x'], targets=static_inputs['y'])
                                        else:
                                            outputs = model(**{k: static_inputs[k] for k in batch.keys()})
                                        
                                        # Traitement de la loss
                                        loss = outputs['loss']
                                        static_inputs['loss'].copy_(loss.detach())
                                        
                                        # Router loss si présent
                                        if args.mode == 'pretrain' and 'router_loss' in outputs:
                                            router_loss = outputs['router_loss'] 
                                            static_inputs['router_loss'].copy_(router_loss.detach())
                                            # Combiner les pertes pour le backward
                                            loss = loss + router_loss * args.router_loss_weight
                                        
                                        # Backward
                                        micro_loss = loss / args.gradient_accumulation_steps
                                        if scaler is not None:
                                            scaler.scale(micro_loss).backward()
                                        else:
                                            micro_loss.backward()
                                
                                # Stocker le graphe et les entrées pour réutilisation
                                static_input_shapes[cuda_graph_key] = static_inputs
                                
                                # Mise à jour des statistiques après la capture
                                loss_value = static_inputs['loss'].item()
                                router_loss_value = static_inputs.get('router_loss', torch.tensor(0.0)).item()
                                loss_accumulator += loss_value
                                router_loss_accumulator += router_loss_value
                                
                                # Comptabiliser les tokens traités
                                batch_size = x.size(0) if args.use_streaming_dataset or args.mode == 'pretrain' else batch['input_ids'].size(0)
                                batch_tokens = batch_size * args.block_size
                                total_tokens += batch_tokens
                                
                                print(f"Nouveau graphe CUDA capturé pour micro_step {micro_step}")
                        else:
                            # CAS 3: EXÉCUTION NORMALE SANS CUDA GRAPH
                            with timing_stats.track(f"forward_{micro_step}"):
                                # Forward pass normal
                                with torch.amp.autocast(enabled=True, device_type='cuda') if ctx else nullcontext():
                                    if args.use_streaming_dataset or args.mode == 'pretrain':
                                        outputs = model(x, targets=y)
                                    else:
                                        outputs = model(**batch)
                                    
                                    # Traitement de la loss selon le format des outputs
                                    if isinstance(outputs, dict):
                                        loss = outputs['loss']
                                        router_loss = outputs.get('router_loss', None)
                                    elif isinstance(outputs, tuple) and len(outputs) >= 2:
                                        # Tuple format (logits, loss, [router_loss])
                                        _, loss = outputs[:2]
                                        router_loss = outputs[2] if len(outputs) > 2 else None
                                    else:
                                        # Fallback 
                                        loss = outputs
                                        router_loss = None
                                    
                                    # Router loss si présent
                                    if router_loss is not None and args.mode == 'pretrain':
                                        router_loss = router_loss * args.router_loss_weight
                                        loss = loss + router_loss
                                        router_loss_value = router_loss.item()
                                    else:
                                        router_loss_value = 0.0
                                    
                                    # Mise à jour des statistiques
                                    loss_accumulator += loss.item()
                                    router_loss_accumulator += router_loss_value
                                    
                                    # Comptabiliser les tokens traités
                                    batch_size = x.size(0) if args.use_streaming_dataset or args.mode == 'pretrain' else batch['input_ids'].size(0)
                                    batch_tokens = batch_size * args.block_size
                                    total_tokens += batch_tokens
                                    
                                    # Backward pass
                                    micro_loss = loss / args.gradient_accumulation_steps
                                    if scaler is not None:
                                        scaler.scale(micro_loss).backward()
                                    else:
                                        micro_loss.backward()
                
                # Calculate tokens per second
                tokens_per_sec = total_tokens / (time.time() - batch_start_time) if total_tokens > 0 else 0
                
                # Étape d'optimisation
                with timing_stats.track("optimizer_step"):
                    # Gradient clipping si activé
                    if args.grad_clip != 0.0 and scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters() if not ddp else model.module.parameters(),
                            args.grad_clip
                        )
                        
                    # Mise à jour des paramètres
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    # Mise à jour du learning rate si nécessaire
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    
                    # Tracker les tokens traités pour les statistiques
                    tokens_window.append((time.time(), total_tokens))
                    if len(tokens_window) > window_size:
                        tokens_window.pop(0)
                
                # timing and logging
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                
                # Progress reporting
                if iter_num % args.log_interval == 0 and master_process:
                    # Convert tensor losses to scalars for reporting
                    avg_loss = loss_accumulator / args.gradient_accumulation_steps
                    avg_router_loss = router_loss_accumulator / args.gradient_accumulation_steps
                    
                    # Calculate tokens per second (moving average)
                    if len(tokens_window) > 1:
                        window_time = tokens_window[-1][0] - tokens_window[0][0]
                        window_tokens = sum(tokens for _, tokens in tokens_window)
                        current_tokens_per_sec = window_tokens / window_time if window_time > 0 else 0
                    else:
                        current_tokens_per_sec = 0
                    
                    # Calculate overall tokens per second
                    elapsed = time.time() - train_start_time
                    overall_tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                    
                    print(f"Iter {iter_num}: loss {avg_loss:.4f}, router_loss {avg_router_loss:.4f}, "
                          f"time {dt*1000:.2f}ms, lr {lr:.6f}, tokens {total_tokens:,}, "
                          f"tok/s {current_tokens_per_sec:.2f}, avg tok/s {overall_tokens_per_sec:.2f}")
                    
                    timing_stats.step()
                    
                    if timing_stats.should_print():
                        timing_stats.print_stats()
                
                # Save checkpoint
                if (iter_num > 0 and iter_num % args.save_interval == 0) and master_process:
                    with timing_stats.track("checkpointing"):
                        checkpoint_path = save_checkpoint(
                            model.module if ddp else model,
                            optimizer,
                            config.__dict__,
                            iter_num,
                            best_val_loss,
                            vars(args),
                            args.output_dir
                        )
                        print(f"Saved checkpoint to {checkpoint_path}")
                        
                        # Cleanup old checkpoints
                        cleanup_old_checkpoints(args.output_dir, keep_num=3)
                
                # Periodic memory cleanup
                if iter_num % 100 == 0:
                    cleanup_memory()
                
                iter_num += 1
                local_iter_num += 1
                
                # Check for max iterations or epochs
                if iter_num >= args.max_iters:
                    break
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    # Save final model
    if master_process:
        print("Training complete, saving final model")
        final_checkpoint_path = save_checkpoint(
            model.module if ddp else model,
            optimizer,
            config.__dict__,
            iter_num,
            best_val_loss,
            vars(args),
            args.output_dir,
            val_loss=None,
        )
        print(f"Saved final model to {final_checkpoint_path}")
    
    # Clean up distributed training if needed
    if args.distributed:
        from torch.distributed import destroy_process_group
        destroy_process_group()
    
    # Print final timing stats
    if master_process:
        print(f"\nFinal timing stats:\n{timing_stats.get_averaged_stats()}\n")
        # Calculate overall throughput
        elapsed = time.time() - train_start_time
        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
        print(f"Total tokens: {total_tokens:,}")
        print(f"Average throughput: {tokens_per_sec:.2f} tokens/s")
        print(f"Total training time: {elapsed:.2f}s")

def generate_example(model, prompt, device, args):
    """Generate text using the trained model"""
    model.eval()
    # Convert prompt to tensor with proper shape
    tokens = torch.tensor([prompt], dtype=torch.long, device=device)
    
    # Try to import tokenizer for text output
    try:
        from transformers import AutoTokenizer
        tokenizer = None
        try:
            # Initialize tokenizer if possible
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except:
            print("Could not load tokenizer, will show only token IDs")
    except ImportError:
        tokenizer = None
    
    # Safe generation with error handling
    try:
        with torch.no_grad():
            # Use the new detokenizing generate method
            if hasattr(model, 'generate') and tokenizer is not None:
                output_tokens, output_text = model.generate(
                    tokens,
                    steps=args.gen_steps,
                    gen_length=args.gen_length,
                    block_length=args.gen_block_length,
                    temperature=args.temperature,
                    remasking=args.remasking,
                    tokenizer=tokenizer
                )
                return f"Generated ids: {output_tokens[0].tolist()}\n\nGenerated text:\n{output_text}"
            else:
                # Fallback to the simpler generation approach
                generated = []
                current_tokens = tokens.clone()
                
                for _ in range(min(50, args.gen_length)):  # Limit generation length for safety
                    # Get predictions
                    with torch.amp.autocast(enabled=True, device_type='cuda'):
                        logits, _, _ = model(current_tokens)
                    
                    # Get next token (simple greedy decoding)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    current_tokens = torch.cat([current_tokens, next_token], dim=1)
                    generated.append(next_token.item())
                    
                    # Stop if we generate an EOS token (assuming 0 is EOS for simplicity)
                    if next_token.item() == 0:
                        break
                
                # If we have a tokenizer, try to decode the tokens
                if tokenizer is not None:
                    try:
                        full_sequence = prompt + generated
                        text = tokenizer.decode(full_sequence)
                        return f"Generated ids: {full_sequence}\n\nGenerated text:\n{text}"
                    except:
                        pass
                
                # Fallback to just showing token IDs
                return f"Generated ids: {prompt + generated}"
    except Exception as e:
        # Provide detailed error info for debugging
        print(traceback.format_exc())
        return f"Generation failed with error: {str(e)}"

def generate_example_wrapper(model, prompt_tokens, tokenizer=None, steps=32, gen_length=32, temperature=0.7):
    """
    Wrapper function for generate_example that displays both token IDs and text.
    To be used in training scripts.
    
    Example usage in training script:
    ```
    # Replace:
    output_tokens = generate_example(model, prompt_tokens)
    print(f"Generated ids: {output_tokens.tolist()}")
    
    # With:
    from llada.llada import generate_example_wrapper
    generate_example_wrapper(model, prompt_tokens, tokenizer)
    ```
    """
    # Import HF tokenizer if needed and not provided
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            print("Using default GPT-2 tokenizer")
        except ImportError:
            print("Warning: transformers not installed. Text output will be unavailable.")
            tokenizer = None
    
    # Run generation with text output
    print("\nGenerating example...")
    
    try:
        with torch.no_grad():
            with torch.amp.autocast(enabled=True, device_type='cuda'):
                output_tokens, output_text = model.generate(
                    prompt_tokens,
                    steps=steps,
                    gen_length=gen_length,
                    temperature=temperature,
                    tokenizer=tokenizer
                )
        
        # Display token IDs
        print(f"Generated ids: {output_tokens[0].tolist()}")
        
        # Display text
        print("\nGenerated text:")
        print("-" * 40)
        print(output_text)
        print("-" * 40)
        
        return output_tokens, output_text
        
    except Exception as e:
        print(f"Generation failed with error: {str(e)}")
        return None, str(e)

def main():
    parser = argparse.ArgumentParser(description="Train or use a LLaDA model")
    
    # Model configuration
    parser.add_argument('--n_layer', type=int, default=12, help='Number of layers')
    parser.add_argument('--n_head', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--bias', action='store_true', help='Use bias in linear layers')
    parser.add_argument('--block_size', type=int, default=1024, help='Context window size')
    parser.add_argument('--vocab_size', type=int, default=50304, help='Vocabulary size')
    parser.add_argument('--mask_token_id', type=int, default=126336, help='ID for mask token')
    parser.add_argument('--num_experts', type=int, default=8, help='Number of experts in MoE')
    parser.add_argument('--k', type=int, default=2, help='Top-k experts to use per token')
    
    # Training parameters
    parser.add_argument('--mode', type=str, choices=['pretrain', 'finetune'], default='pretrain')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients')
    parser.add_argument('--learning_rate', type=float, default=4e-4, help='Max learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Min learning rate')
    parser.add_argument('--warmup_iters', type=int, default=2000, help='Warmup iterations')
    parser.add_argument('--lr_decay_iters', type=int, default=50000, help='LR decay iterations')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.95, help='Adam beta2')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--router_loss_weight', type=float, default=0.001, help='Router loss weight')
    
    # Training control
    parser.add_argument('--max_iters', type=int, default=100000, help='Max training iterations')
    parser.add_argument('--max_epochs', type=int, default=10, help='Max training epochs')
    parser.add_argument('--log_interval', type=int, default=1, help='Log every N iterations')
    parser.add_argument('--save_interval', type=int, default=1000, help='Save every N iterations')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Use gradient checkpointing')
    
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', type=str, choices=['float32', 'float16', 'bfloat16'], default='float16')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--backend', type=str, default='nccl', help='Distributed backend')
    parser.add_argument('--use_streaming_dataset', action='store_true', help='Use streaming dataset')
    parser.add_argument('--buffer_size', type=int, default=64, help='Buffer size for streaming dataset')
    parser.add_argument('--init_from', type=str, default='scratch', choices=['scratch', 'resume'], 
                        help='Initialize from scratch or resume training')
    
    # Generation parameters
    parser.add_argument('--gen_steps', type=int, default=128, help='Steps for generation')
    parser.add_argument('--gen_length', type=int, default=128, help='Length to generate')
    parser.add_argument('--gen_block_length', type=int, default=32, help='Block length for generation')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for sampling')
    parser.add_argument('--remasking', type=str, default='low_confidence', 
                       choices=['low_confidence', 'random'], help='Remasking strategy')
    
    # Nouvelles options d'optimisation
    parser.add_argument('--compile', action='store_true', help='Compile the model with torch.compile')
    parser.add_argument('--use_cuda_graphs', action='store_true', help='Use CUDA Graphs for optimization')
    parser.add_argument('--preallocate_memory', action='store_true', help='Preallocate CUDA memory to reduce fragmentation')
    parser.add_argument('--disable_non_blocking', action='store_true', help='Disable non-blocking data transfers')
    parser.add_argument('--gpu_monitor_interval', type=int, default=0, help='Interval for GPU statistics monitoring (0 to disable)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the model
    train_llada(args)

if __name__ == '__main__':
    main() 