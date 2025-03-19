"""
Optimisations CUDA pour l'entraînement des modèles LLM.
Ce module contient des fonctions pour optimiser les performances CUDA.
"""

import os
import gc
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
    
    # Common optimizations
    torch.backends.cudnn.benchmark = True
    
    # cuDNN optimizations
    if hasattr(torch.backends.cudnn, "allow_tensor_core"):
        torch.backends.cudnn.allow_tensor_core = True
    
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
    """Print detailed GPU statistics"""
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return
    
    try:
        device_count = torch.cuda.device_count()
        print(f"=== GPU Stats ({device_count} devices) ===")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            
            print(f"\nDevice {i}: {props.name}")
            print(f"  - Architecture: SM {props.major}.{props.minor}")
            print(f"  - Total Memory: {props.total_memory / (1024**3):.2f} GB")
            
            if hasattr(torch.cuda, 'mem_get_info'):
                free, total = torch.cuda.mem_get_info(i)
                used = total - free
                print(f"  - Free Memory: {free / (1024**3):.2f} GB")
                print(f"  - Used Memory: {used / (1024**3):.2f} GB ({100*used/total:.1f}%)")
            
            print(f"  - Clock Rate: {props.clock_rate / 1000} MHz")
            print(f"  - Multiprocessors: {props.multi_processor_count}")
            
            if props.major >= 7:
                print(f"  - Tensor Cores: Yes")
            else:
                print(f"  - Tensor Cores: No")
                
            if props.major >= 8:
                print(f"  - Ampere+ Architecture: Yes")
            else:
                print(f"  - Ampere+ Architecture: No")
                
            if props.major >= 9:
                print(f"  - FP8 Support: Yes")
            else:
                print(f"  - FP8 Support: No")
    
    except Exception as e:
        print(f"Error getting GPU stats: {e}")

def set_seed(seed=42):
    """Set all random seeds for deterministic results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Optional for further determinism - can impact performance
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print(f"Warning: Couldn't set deterministic algorithms: {e}")
    elif hasattr(torch, 'set_deterministic'):
        try:
            torch.set_deterministic(True)
        except Exception as e:
            print(f"Warning: Couldn't set deterministic mode: {e}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Make cuDNN deterministic if available
    # Note: this can reduce performance
    if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'deterministic'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def optimize_distributed_params():
    """Optimize parameters for distributed training"""
    # Optimisations de base pour torch.distributed
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    
    # Optimisations NCCL pour la communication inter-GPU
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_P2P_DISABLE"] = "0"
    os.environ["NCCL_BUFFSIZE"] = "4194304"  # 4MB pour optimiser les transferts
    
    # Amélioration des performances NCCL
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_ALGO"] = "Ring"
    os.environ["NCCL_MAX_NCHANNELS"] = "8"
    
    # Optimisations avancées
    os.environ["NCCL_PROTO"] = "Simple"
    os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"
    os.environ["NCCL_SOCKET_NTHREADS"] = "4"
    os.environ["NCCL_MIN_NCHANNELS"] = "4"
    
    print("Distributed training parameters optimized")

def enable_flash_attention():
    """Enable Flash Attention if available"""
    try:
        import transformers
        
        if hasattr(transformers, "modeling_utils") and hasattr(transformers.modeling_utils, "is_flash_attn_available"):
            flash_available = transformers.modeling_utils.is_flash_attn_available()
            if flash_available:
                print("Flash Attention is available and enabled")
                os.environ["TRANSFORMERS_ENABLE_FLASH_ATTN"] = "1"
                return True
            else:
                print("Flash Attention is not available on this system")
        else:
            try:
                import flash_attn
                print(f"Flash Attention library detected (version: {flash_attn.__version__})")
                os.environ["USE_FLASH_ATTENTION"] = "1"
                return True
            except ImportError:
                print("Flash Attention library not installed")
                
        return False
    except ImportError:
        print("Could not check for Flash Attention - transformers package not available")
        return False

if __name__ == "__main__":
    print("CUDA optimization tools available:")
    print("- setup_cuda_optimizations(): Configure CUDA optimizations")
    print("- check_cuda_graph_compatibility(): Check CUDA graph support")
    print("- print_gpu_stats(): Display detailed GPU information")
    print("- set_seed(): Set random seeds for deterministic results")
    print("- optimize_distributed_params(): Optimize distributed training")
    print("- enable_flash_attention(): Enable Flash Attention if available") 