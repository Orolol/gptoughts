import os
import math
import argparse
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from contextlib import nullcontext
import gc
import glob
import random
import threading
from collections import defaultdict
from contextlib import contextmanager
import sys
import traceback
from datasets import load_dataset
from data.data_loader import StreamingBatchLoader

# Add parent directory to path to import from root project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utilities
from train_utils import (
    get_lr, cleanup_old_checkpoints, cleanup_memory, ensure_model_dtype,
    print_memory_stats, save_checkpoint, load_checkpoint, find_latest_checkpoint
)

# Import data loaders
from llada.data_loadar_llada import create_llm_data_loader

from llada.llada import LLaDAConfig, LLaDAModel
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

def create_streaming_loader(dataset_name, dataset_split, tokenizer, batch_size, gradient_accumulation_steps, 
                           max_length=1024, text_column="text", num_workers=4):
    """Create a DataLoader using our StreamingBatchLoader for efficient multiprocessing."""
    dataset = StreamingBatchLoader(
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        tokenizer=tokenizer,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_length,
        text_column=text_column
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,  # We handle batching in the dataset
        num_workers=num_workers,
        pin_memory=True, 
        persistent_workers=True,
        prefetch_factor=4
    )
    
    return loader

def preallocate_cuda_memory():
    """Preallocate CUDA memory to avoid fragmentation and improve performance"""
    print("Preallocating CUDA memory...")
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        try:
            # Calculate available memory
            free_mem, total_mem = torch.cuda.mem_get_info(device)
            
            # Use a more conservative allocation to avoid OOM errors (65% of free memory)
            mem_to_allocate = int(free_mem * 0.65)
            
            # Clear existing allocations first
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Allocate in fewer, larger blocks to reduce fragmentation
            # Using 4 blocks instead of 8 for more efficient memory usage
            block_size = mem_to_allocate // 4
            allocations = []
            
            # Allocate blocks sequentially
            for i in range(4):
                try:
                    # Use float16 for more efficient allocation
                    allocations.append(torch.empty(block_size // 2, 
                                               dtype=torch.float16, 
                                               device='cuda'))
                    print(f"Allocated block {i+1}/4 ({block_size/(1024**3):.2f} GB)")
                except RuntimeError as e:
                    print(f"Stopped at block {i+1}/4: {e}")
                    break
            
            # Hold allocations for a moment to allow cache optimization
            torch.cuda.synchronize()
            time.sleep(0.5)
            
            # Free the allocations but keep them in CUDA cache
            for block in allocations:
                del block
            
            allocations = []
            torch.cuda.empty_cache()
            
            # Force a CUDA synchronization to complete pending operations
            torch.cuda.synchronize()
            
            # Calculate how much we've successfully preallocated
            _, new_free = torch.cuda.mem_get_info(device)
            preallocated = total_mem - new_free
            
            print(f"Preallocation complete, {preallocated/(1024**3):.2f} GB reserved")
            
        except Exception as e:
            # Fallback if mem_get_info isn't available
            print(f"Preallocation using alternate method: {e}")
            try:
                # Allocate ~70% of total memory in smaller chunks
                total_mem = torch.cuda.get_device_properties(device).total_memory
                mem_to_allocate = int(total_mem * 0.7)
                block_size = mem_to_allocate // 8
                
                allocations = []
                for i in range(8):
                    try:
                        allocations.append(torch.empty(block_size // 2, 
                                                     dtype=torch.float16, 
                                                     device='cuda'))
                    except RuntimeError:
                        break
                
                for block in allocations:
                    del block
                allocations = []
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                print(f"Preallocation complete (fallback method)")
            except RuntimeError:
                print("Preallocation failed - insufficient memory. Continuing without preallocation.")

def check_cuda_graph_compatibility():
    """Check if the current environment is compatible with CUDA graphs"""
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    try:
        # Check if CUDA graphs are supported
        if not hasattr(torch.cuda, 'CUDAGraph'):
            return False, "CUDA graphs not supported in this PyTorch version"
        
        # Check CUDA version
        cuda_version = torch.version.cuda
        if cuda_version is not None:
            major, minor = map(int, cuda_version.split('.')[:2])
            if major < 10 or (major == 10 and minor < 2):
                return False, f"CUDA version {cuda_version} is too old for reliable graph usage (need 10.2+)"
        
        # Check GPU compute capability
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        compute_cap = props.major + props.minor / 10
        if compute_cap < 7.0:
            return False, f"GPU compute capability {compute_cap} is too low for optimal CUDA graph support"
        
        # Check environment variables that might interfere
        problematic_vars = [
            "CUDA_LAUNCH_BLOCKING",
            "PYTORCH_NO_CUDA_MEMORY_CACHING",
        ]
        for var in problematic_vars:
            if os.environ.get(var, "0") == "1":
                return False, f"Environment variable {var}=1 may interfere with CUDA graphs"
        
        return True, "CUDA graphs compatible"
    except Exception as e:
        return False, f"Error checking CUDA graph compatibility: {e}"

def setup_cuda_optimizations():
    """Configure CUDA optimizations based on the detected GPU hardware."""
    if not torch.cuda.is_available():
        print("CUDA is not available, no optimizations applied")
        return
    
    # TF32 optimizations for accelerated matrix calculations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    
    # Detect GPU type for further optimizations
    gpu_name = torch.cuda.get_device_name().lower()
    detected_arch = "unknown"
    
    # Architecture detection
    if any(x in gpu_name for x in ['h100', 'h800']):
        detected_arch = "hopper"
    elif any(x in gpu_name for x in ['a100', 'a6000', 'a5000', 'a4000']):
        detected_arch = "ampere"
    elif any(x in gpu_name for x in ['4090', '4080', '4070', '4060', '3090', '3080', '3070', '2070']):
        detected_arch = "consumer"
    
    print(f"Detected GPU: {torch.cuda.get_device_name()} (architecture: {detected_arch})")
    
    # Common optimizations
    torch.backends.cudnn.benchmark = True
    
    gc.enable()
    
    torch.cuda.set_per_process_memory_fraction(0.95)
    
    # Architecture-specific optimizations
    if detected_arch == "hopper":
        # H100/H800 - High-end HPC GPUs
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        
        # Don't set a fixed memory fraction - let PyTorch manage dynamically
        # Instead set these memory-related environment variables
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.9"
        
        # Enable cudnn benchmark mode for optimizing convolution algorithms
        torch.backends.cudnn.benchmark = True
        
        # Don't disable Python's GC completely - can cause memory leaks
        # Instead, tune GC parameters for more aggressive collection
        gc.set_threshold(100, 5, 5)  # More frequent collection
        
        print("Hopper optimizations applied: reduced precision for BF16/FP16, optimized GPU connections")
        
    elif detected_arch == "ampere":
        # A100 and other Ampere datacenter GPUs
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        
        # Dynamic memory management instead of fixed percentage
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
        
        # Tune GC for large models
        gc.set_threshold(700, 10, 5)
        
        print("Ampere optimizations applied: reduced precision for BF16/FP16, dynamic memory management")
        
    elif detected_arch == "consumer":
        # Consumer GPUs (RTX 3000/4000 series)
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"  # Reduced from 8 to prevent fragmentation
        
        # More conservative memory settings for consumer GPUs
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"
        torch.cuda.empty_cache()  # Clear cache before starting
        
        # More aggressive GC settings for consumer GPUs
        gc.set_threshold(100, 5, 2)  # More frequent collection
        
        print("Consumer GPU optimizations applied: optimized connections and memory management")
        
    else:
        # Unknown or older GPU - generic optimizations
        print("Unrecognized GPU architecture, applying generic optimizations")
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32"
    
    # cuDNN optimizations
    if hasattr(torch.backends.cudnn, "allow_tensor_core"):
        torch.backends.cudnn.allow_tensor_core = True
    
    # Enable better memory defrag
    if hasattr(torch.cuda, "memory_stats"):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "") + ",garbage_collection_threshold:0.6"
    
    # Optimize memory allocator
    if hasattr(torch, "use_deterministic_algorithms"):
        # Only do this in non-deterministic mode
        try:
            torch.use_deterministic_algorithms(False)
        except:
            pass
    
    # Autocast settings
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        # Patch for compatibility with custom instructions
        original_autocast = torch.cuda.amp.autocast
        def patched_autocast(*args, **kwargs):
            if not kwargs.get("enabled", True):
                return nullcontext()
            return torch.amp.autocast(device_type='cuda', *args, **kwargs)
        torch.cuda.amp.autocast = patched_autocast
    
    # NCCL optimization for multi-GPU if available
    if hasattr(torch.distributed, "is_available") and torch.distributed.is_available():
        # Set NCCL environment variables for better performance
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    
    print("CUDA optimization configuration complete")

def print_gpu_stats():
    """Print detailed GPU usage statistics"""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(device)
    
    # Get memory information
    try:
        free_mem, total_mem = torch.cuda.mem_get_info(device)
        allocated_mem = torch.cuda.memory_allocated(device)
        reserved_mem = torch.cuda.memory_reserved(device)
        
        # Calculate percentages
        used_percent = 100 - (free_mem / total_mem * 100)
        allocated_percent = allocated_mem / total_mem * 100
        reserved_percent = reserved_mem / total_mem * 100
        
        # Get fragmentation info if available
        fragmentation = "N/A"
        if hasattr(torch.cuda, "memory_stats") and callable(getattr(torch.cuda, "memory_stats")):
            stats = torch.cuda.memory_stats(device)
            if "allocated_bytes.all.peak" in stats and "reserved_bytes.all.peak" in stats:
                peak_allocated = stats["allocated_bytes.all.peak"]
                peak_reserved = stats["reserved_bytes.all.peak"]
                if peak_reserved > 0:
                    fragmentation = f"{(1 - peak_allocated / peak_reserved) * 100:.1f}%"
        
        # Display memory information
        print("\n=== GPU Memory Status ===")
        print(f"Device: {device_properties.name}")
        print(f"Memory: {free_mem/(1024**3):.2f}GB free / {total_mem/(1024**3):.2f}GB total ({used_percent:.1f}% used)")
        print(f"Allocated: {allocated_mem/(1024**3):.2f}GB ({allocated_percent:.1f}%)")
        print(f"Reserved: {reserved_mem/(1024**3):.2f}GB ({reserved_percent:.1f}%)")
        print(f"Fragmentation: {fragmentation}")
        print(f"Memory states: {torch.cuda.memory_stats(device)['reserved_bytes.all.current'] / (1024**3):.2f}GB reserved currently")
        print("========================\n")
    except (RuntimeError, AttributeError) as e:
        print(f"Error getting memory info: {e}")
        
    # Print device info
    print(f"Device: {device_properties.name}")
    print(f"Compute capability: {device_properties.major}.{device_properties.minor}")


def init_distributed(backend):
    """Initialize PyTorch distributed training backend"""
    import torch.distributed as dist
    
    if not dist.is_available():
        raise RuntimeError("PyTorch distributed not available")
        
    # Initialize the process group
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group(backend=backend)
    
    print(f"Initialized distributed training with backend: {backend}")
    print(f"Rank: {os.environ['RANK']}, Local Rank: {os.environ['LOCAL_RANK']}, World Size: {os.environ['WORLD_SIZE']}")

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
        scaler = GradScaler(device==device_type)
    elif args.dtype == 'bfloat16' and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        ctx = torch.amp.autocast(enabled=True, device_type='cuda')
        scaler = GradScaler(device==device_type)
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
        iter_num = 1
        best_val_loss = float('inf')
        
        # Initialize learning rate scheduler as None since we're using manual LR updates
        lr_scheduler = None
    
    # Print model size
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Set up timing stats
    timing_stats = AveragedTimingStats(print_interval=50)
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

        num_workers = min(8, max(4, os.cpu_count() // 2))
        
        # Use our accelerated DataLoader with StreamingBatchLoader
        return create_streaming_loader(
            dataset_name=args.dataset_name if hasattr(args, 'dataset_name') else 'HuggingFaceFW/fineweb-edu',
            dataset_split=args.dataset_split if hasattr(args, 'dataset_split') else 'train',
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_length=args.block_size,
            text_column="text",
            num_workers=num_workers
        )


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
        
    
    # Get initial train loader
    train_loader = get_train_loader()
    # Create an iterator from the DataLoader
    train_iter = iter(train_loader)
    
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

                
                # GPU monitoring if requested
                if args.gpu_monitor_interval > 0 and iter_num % args.gpu_monitor_interval == 0 and master_process:
                    print_gpu_stats()
                    
                with timing_stats.track("data_preloading"):
                    # Check if data loader has a method to fetch multiple batches at once
                    data_loading_start = time.time()
                    
                    # For StreamingBatchLoader with DataLoader, we'll get batches for each gradient accumulation step
                    batches = []
                    for _ in range(args.gradient_accumulation_steps):
                        try:
                            batch = next(train_iter)
                            batches.append(batch)
                        except StopIteration:
                            # If we hit the end of the dataset, restart it
                            train_iter = iter(train_loader)
                            batch = next(train_iter)
                            batches.append(batch)
                    
                    if master_process and iter_num % args.log_interval == 0:
                        data_loading_time = time.time() - data_loading_start
                        # print(f"🔄 DataLoader loading time: {data_loading_time*1000:.1f}ms for {len(batches)} batches")
                

                # Gradient accumulation loop
                for micro_step in range(args.gradient_accumulation_steps):
                    # Enable DDP gradient sync only for the last micro_step
                    if ddp:
                        model.require_backward_grad_sync = (micro_step == args.gradient_accumulation_steps - 1)
                    
                    # Fetch data from the pre-loaded batches
                    with timing_stats.track(f"data_loading_{micro_step}"):
                        # Use the pre-chunked tensors with optimized transfer
                        # Always use non_blocking=True for better overlapping of data transfer and computation
                        x = batches[micro_step][0].to(device, non_blocking=True)
                        y = batches[micro_step][1].to(device, non_blocking=True)
                        
                        # Pin memory to speed up CPU->GPU transfers if not already done
                        torch.cuda.synchronize()  # Ensure data transfer is complete before computation
                       
                    with torch.amp.autocast(enabled=True, device_type='cuda') if ctx else nullcontext():
                        with timing_stats.track(f"forward_{micro_step}"):
                            # For stable training iterations, capture CUDA graph

                            outputs = model(x, targets=y)
                            
                            # Process loss based on output format
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
                                
                            # Router loss if present
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
                            batch_size = x.size(0)
                            batch_tokens = batch_size * args.block_size
                            total_tokens += batch_tokens
                        with timing_stats.track(f"backward_{micro_step}"):
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
                    
                    # Periodic memory management to prevent fragmentation
                    if iter_num % 10 == 0:
                        # Only run empty_cache occasionally to avoid overhead
                        # But frequently enough to reduce fragmentation
                        torch.cuda.empty_cache()
                
                # timing and logging
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                
                # Update timing stats
                timing_stats.step()
                
                
                # Print training progress
                if iter_num % args.log_interval == 0:
                    # Calculate loss across this interval
                    loss_avg = loss_accumulator / args.gradient_accumulation_steps
                    router_loss_avg = router_loss_accumulator / args.gradient_accumulation_steps if router_loss_accumulator > 0 else 0
                    
                    # Calculate tokens/s
                    tokens_per_sec_fmt = f"{tokens_per_sec:.0f}"
                    
                    # Print statistics
                    if args.mode == 'pretrain':
                        print(f"iter {iter_num}: loss {loss_avg:.4f}, router_loss {router_loss_avg:.6f}, {tokens_per_sec_fmt} tokens/s")
                    else:
                        print(f"iter {iter_num}: loss {loss_avg:.4f}, {tokens_per_sec_fmt} tokens/s")
                    
                    # Print timing statistics if available
                    if timing_stats.should_print():
                        timing_stats.print_stats()
                    
                    # Reset accumulators
                    loss_accumulator = 0.0
                    router_loss_accumulator = 0.0
                    
                    # Print GPU stats every 10 iterations to monitor memory usage
                    if iter_num % 10 == 0:
                        print_gpu_stats()

                # Set up next iteration
                t0 = t1
                
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
                

                
                if master_process and iter_num % 100 == 0:
                    # Synchronize and print memory stats periodically
                    torch.cuda.synchronize()
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / (1024**3)
                        reserved = torch.cuda.memory_reserved() / (1024**3)
                        print(f"Memory stats: Allocated {allocated:.2f} GB, Reserved {reserved:.2f} GB")
                
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
    parser.add_argument('--block_size', type=int, default=64, help='Context window size')
    parser.add_argument('--vocab_size', type=int, default=128001, help='Vocabulary size')
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
    parser.add_argument('--router_loss_weight', type=float, default=0.1, help='Router loss weight')
    
    # Training control
    parser.add_argument('--max_iters', type=int, default=100000, help='Max training iterations')
    parser.add_argument('--max_epochs', type=int, default=10, help='Max training epochs')
    parser.add_argument('--log_interval', type=int, default=1, help='Log every N iterations')
    parser.add_argument('--save_interval', type=int, default=1000, help='Save every N iterations')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Use gradient checkpointing')
    parser.add_argument('--memory_efficient', action='store_true', help='Use memory efficient implementation')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', type=str, choices=['float32', 'float16', 'bfloat16'], default='bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16')
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
    
    # Advanced optimization options
    parser.add_argument('--compile', action='store_true', help='Compile the model with torch.compile')
    parser.add_argument('--use_cuda_graphs', action='store_true', help='Use CUDA Graphs for optimization')
    parser.add_argument('--cuda_graphs_safe_mode', action='store_true', help='Safer but potentially slower CUDA Graphs mode')
    parser.add_argument('--preallocate_memory', action='store_true', help='Preallocate CUDA memory to reduce fragmentation')
    parser.add_argument('--disable_non_blocking', action='store_true', help='Disable non-blocking data transfers')
    parser.add_argument('--gpu_monitor_interval', type=int, default=0, help='Interval for GPU statistics monitoring (0 to disable)')
    parser.add_argument('--memory_optimization_level', type=int, default=1, choices=[0, 1, 2, 3], 
                       help='Memory optimization level (0=off, 1=basic, 2=aggressive, 3=maximum)')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Number of batches to prefetch in dataloader')
    parser.add_argument('--async_loading', action='store_true', help='Enable async data loading')
    parser.add_argument('--batch_fusion', action='store_true', help='Enable batch fusion for better GPU utilization')
    parser.add_argument('--amp_dtype', type=str, choices=['float16', 'bfloat16'], 
                       default='bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16',
                       help='Data type for AMP (automatic mixed precision)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the model
    train_llada(args)

if __name__ == '__main__':
    main() 
