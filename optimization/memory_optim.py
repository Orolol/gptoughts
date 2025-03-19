"""
Optimisations mémoire pour l'entraînement des modèles LLM.
Ce module contient des fonctions pour optimiser l'utilisation de la mémoire GPU.
"""

import os
import gc
import time
import torch

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

def cleanup_memory():
    """Clean up GPU memory by forcing collection of unused tensors"""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def print_memory_stats(prefix=""):
    """Print memory statistics for the current device"""
    if torch.cuda.is_available():
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            used_mem = total_mem - free_mem
            print(f"{prefix}Memory: {used_mem/(1024**3):.2f}/{total_mem/(1024**3):.2f} GB ({100*used_mem/total_mem:.1f}% used)")
        except:
            # Fallback for older PyTorch versions
            alloc_mem = torch.cuda.memory_allocated()
            reserved_mem = torch.cuda.memory_reserved()
            print(f"{prefix}Memory: {alloc_mem/(1024**3):.2f} GB allocated, {reserved_mem/(1024**3):.2f} GB reserved")

def set_memory_optim_env_vars():
    """Configure environment variables for optimized memory management"""
    # Variables pour la gestion de la mémoire GPU
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "0"
    os.environ["NCCL_IB_DISABLE"] = "0"
    
    # Variables pour le débogage de la mémoire
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Mettre à 1 pour le débogage seulement
    
    # Variables pour l'optimisation des performances
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["NCCL_DEBUG"] = "WARN"
    
    print("Environment variables for memory optimization have been set")

def optimize_memory_for_device():
    """Configure optimal memory settings based on detected GPU"""
    if not torch.cuda.is_available():
        print("CUDA is not available, no memory optimizations applied")
        return
        
    # Detect GPU type for further optimizations
    gpu_name = torch.cuda.get_device_name().lower()
    detected_arch = "unknown"
    
    # Architecture detection
    if any(x in gpu_name for x in ['h100', 'h800']):
        detected_arch = "hopper"
    elif any(x in gpu_name for x in ['a100', 'a6000', 'a5000', 'a4000']):
        detected_arch = "ampere"
    elif any(x in gpu_name for x in ['4090', '4080', '4070', '4060', '3090', '3080', '3070', 'rtx 6000']):
        detected_arch = "consumer"
    
    print(f"Optimizing memory for {gpu_name} (architecture: {detected_arch})")
    
    # Architecture-specific optimizations
    if detected_arch == "hopper":
        # H100/H800 - High-end HPC GPUs
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.9"
        gc.set_threshold(100, 5, 5)  # More frequent collection
        
    elif detected_arch == "ampere":
        # A100 and other Ampere datacenter GPUs
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
        gc.set_threshold(700, 10, 5)
        
    elif detected_arch == "consumer":
        # Consumer GPUs (RTX 3000/4000/6000 series)
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"
        torch.cuda.empty_cache()
        gc.set_threshold(100, 5, 2)
        
    else:
        # Unknown or older GPU - generic optimizations
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32"
    
    # Enable better memory defrag
    if hasattr(torch.cuda, "memory_stats"):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "") + ",garbage_collection_threshold:0.6"
    
    print("Memory optimization configuration complete")

if __name__ == "__main__":
    print("Memory optimization tools available:")
    print("- preallocate_cuda_memory(): Pre-allocate memory to reduce fragmentation")
    print("- cleanup_memory(): Clean up unused GPU memory")
    print("- print_memory_stats(): Display current memory usage")
    print("- set_memory_optim_env_vars(): Configure environment variables")
    print("- optimize_memory_for_device(): Apply device-specific optimizations") 