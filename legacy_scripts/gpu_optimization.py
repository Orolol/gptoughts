"""
Optimisations GPU pour l'entraînement des modèles LLM.
Ce module contient des fonctions spécialisées pour optimiser les performances GPU.
"""

import os
import gc
import time
import torch
import random
import numpy as np

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
                from contextlib import nullcontext
                return nullcontext()
            return torch.amp.autocast(device_type='cuda', *args, **kwargs)
        torch.cuda.amp.autocast = patched_autocast
    
    # NCCL optimization for multi-GPU if available
    if hasattr(torch.distributed, "is_available") and torch.distributed.is_available():
        # Set NCCL environment variables for better performance
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    
    print("CUDA optimization configuration complete")

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

def cleanup_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if gc.isenabled():
            gc.collect()

def print_memory_stats(prefix=""):
    """Print GPU memory statistics for debugging"""
    if not torch.cuda.is_available():
        return
        
    print(f"{prefix} Memory stats:")
    print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Additional deterministic settings
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False