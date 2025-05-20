"""
Fonctions utilitaires pour l'entraînement des modèles LLM.
"""

import os
import gc
import glob
import math
import time
import torch
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp
from contextlib import nullcontext, contextmanager
from collections import defaultdict

# Import LLaDAModel to check instance type
try:
    from models.llada.model import LLaDAModel
except ImportError:
    LLaDAModel = None # Define as None if import fails
# Removed duplicate defaultdict import

# Add torch.cuda.nvtx import for profiling FP8 operations (if available)
try:
    import torch.cuda.nvtx as nvtx
    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False

# Constantes pour le calcul des FLOPS
GPU_FLOPS = {
    (7, 0): 312e12,      # A100-40GB (312 TFLOPS en FP16)
    (8, 0): 312e12,      # A100-80GB (312 TFLOPS en FP16)
    (8, 6): 756e12,      # H100-SXM (756 TFLOPS en FP16)
    (9, 0): 495e12,      # H200 (495 TFLOPS en FP16)
    (8, 9): 82.6e12,     # RTX 4090 (82.6 TFLOPS en FP16/FP32 avec Tensor Cores)
    (9, 0, 'fp8'): 989e12,  # H200 en FP8 (989 TFLOPS)
    (8, 6, 'fp8'): 1513e12, # H100 en FP8 (1513 TFLOPS)
    (12, 0): 2000e12,     # Estimated RTX 5090/CC 12.0 (FP16)
    (12, 0, 'fp8'): 4000e12, # Estimated RTX 5090/CC 12.0 (FP8)
}
def get_gpu_count():
    """Return the number of available GPUs."""
    return torch.cuda.device_count()

def setup_distributed(backend='nccl'):
    """Setup distributed training environment with optimized settings."""
    # Optimizations for NCCL
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_NET_GDR_LEVEL"] = "2"
    os.environ["NCCL_P2P_DISABLE"] = "0"
    os.environ["NCCL_MIN_NCHANNELS"] = "4"
    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    os.environ["NCCL_BUFFSIZE"] = "2097152"
    
    # Reduce synchronization frequency
    os.environ["NCCL_ALGO"] = "Ring"
    os.environ["NCCL_PROTO"] = "Simple"
    
    # Async error handling
    os.environ["NCCL_BLOCKING_WAIT"] = "0"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    
    # Initialize process group
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    
    return ddp_rank, ddp_local_rank, ddp_world_size, device

def reduce_metrics(metrics, world_size):
    """
    Reduce metrics across all processes in distributed training.
    Args:
        metrics: Tensor or float to reduce
        world_size: Total number of processes
    Returns:
        Reduced metric value
    """
    if not isinstance(metrics, torch.Tensor):
        metrics = torch.tensor(metrics, device='cuda')
    all_reduce(metrics, op=ReduceOp.SUM)
    return metrics.item() / world_size

def calculate_perplexity(loss):
    """
    Calculate perplexity from cross-entropy loss
    Args:
        loss: Loss value (cross-entropy)
    Returns:
        float: Perplexity value
    """
    return torch.exp(torch.tensor(loss)).item()

def get_lr(current_iter, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """
    Learning rate scheduler with cosine decay and warmup
    Args:
        current_iter: Current training iteration
        warmup_iters: Number of warmup iterations
        lr_decay_iters: Number of iterations over which to decay LR
        learning_rate: Base learning rate
        min_lr: Minimum learning rate
    Returns:
        Current learning rate based on scheduler
    """
    if current_iter < warmup_iters:
        return learning_rate * current_iter / warmup_iters
    if current_iter > lr_decay_iters:
        return min_lr
    decay_ratio = (current_iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def cleanup_old_checkpoints(out_dir, keep_num=3):
    """
    Keep only the 'keep_num' most recent checkpoints.
    Args:
        out_dir: Directory containing checkpoints
        keep_num: Number of checkpoints to keep
    """
    checkpoints = glob.glob(os.path.join(out_dir, 'ckpt_iter_*.pt'))
    if len(checkpoints) <= keep_num:
        return
        
    # Sort checkpoints by iteration number
    def extract_iter_num(filename):
        try:
            return int(filename.split('iter_')[1].split('_loss')[0])
        except:
            return 0
            
    checkpoints.sort(key=extract_iter_num)
    
    # Delete oldest checkpoints
    for ckpt in checkpoints[:-keep_num]:
        try:
            os.remove(ckpt)
            print(f"Removed old checkpoint: {ckpt}")
        except Exception as e:
            print(f"Error removing checkpoint {ckpt}: {e}")

def ensure_model_dtype(model, dtype):
    """
    Ensure model parameters are in the correct dtype
    Args:
        model: PyTorch model
        dtype: Target dtype
    Returns:
        Model with parameters in the target dtype
    """
    for param in model.parameters():
        param.data = param.data.to(dtype=dtype)
    return model

def save_checkpoint(model, optimizer, model_args, iter_num, best_val_loss, config, out_dir, val_loss=None):
    """
    Save model checkpoint with optimized memory handling
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        model_args: Model architecture arguments
        iter_num: Current iteration number
        best_val_loss: Best validation loss so far
        config: Training configuration
        out_dir: Output directory
        val_loss: Current validation loss (optional)
    """
    # Save to CPU to avoid GPU memory duplication
    raw_model = model.module if hasattr(model, 'module') else model
    
    # Préparer le dictionnaire de checkpoint
    checkpoint = {
        'model': {k: v.cpu() for k, v in raw_model.state_dict().items()},
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    
    # Gérer différents types d'optimiseurs
    optimizer_state = optimizer.state_dict()
    
    # Vérifier si c'est un MultiOptimizer (qui a une structure différente)
    if 'opt_0' in optimizer_state:
        # C'est un MultiOptimizer, conserver sa structure spéciale
        checkpoint['optimizer'] = optimizer_state
    else:
        # C'est un optimiseur standard, traiter comme avant
        if 'state' in optimizer_state and 'param_groups' in optimizer_state:
            checkpoint['optimizer'] = {
                'state': {k: {sk: sv.cpu() if isinstance(sv, torch.Tensor) else sv
                            for sk, sv in v.items()}
                            for k, v in optimizer_state['state'].items()},
                'param_groups': optimizer_state['param_groups']
            }
        else:
            # Fallback pour d'autres structures d'optimiseurs
            checkpoint['optimizer'] = optimizer_state
    
    checkpoint_name = f'ckpt_iter_{iter_num}'
    if val_loss is not None:
        checkpoint_name += f'_loss_{val_loss:.4f}'
    checkpoint_path = os.path.join(out_dir, f'{checkpoint_name}.pt')
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path

def load_checkpoint(ckpt_path, model, optimizer=None, map_location='cpu'):
    """
    Load checkpoint for model and optimizer.
    
    Args:
        ckpt_path (str): Path to checkpoint
        model (torch.nn.Module): Model to load checkpoint into
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into
        map_location (str, optional): Device to map tensors to. Defaults to 'cpu'.
        
    Returns:
        tuple: (model, optimizer, iter_num, best_val_loss)
    """
    # Add PreTrainedTokenizerFast to safe globals for loading with weights_only=True
    try:
        from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
        import torch.serialization
        torch.serialization.add_safe_globals([PreTrainedTokenizerFast])
    except (ImportError, AttributeError):
        print("Warning: Could not add PreTrainedTokenizerFast to safe globals. Will attempt loading with weights_only=False.")
        checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    else:
        try:
            # Try loading with weights_only=True
            checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=True)
        except Exception as e:
            print(f"Warning: Failed to load with weights_only=True: {e}")
            print("Falling back to weights_only=False for backward compatibility")
            checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    
    # Clean state dict before loading
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    cleaned_state_dict = {}
    
    for k, v in state_dict.items():
        if k.startswith(unwanted_prefix):
            cleaned_state_dict[k[len(unwanted_prefix):]] = v
        else:
            cleaned_state_dict[k] = v
    
    # Load model state
    model.load_state_dict(cleaned_state_dict, strict=False)
    
    # Clean memory
    del state_dict
    del cleaned_state_dict
    torch.cuda.empty_cache()
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer_state = checkpoint['optimizer']
        
        # Vérifier si c'est un format MultiOptimizer
        if 'opt_0' in optimizer_state:
            # Format MultiOptimizer - vérifier si l'optimiseur actuel est aussi un MultiOptimizer
            if hasattr(optimizer, 'load_state_dict'):
                optimizer.load_state_dict(optimizer_state)
        else:
            # Format standard avec 'state' et 'param_groups'
            if 'state' in optimizer_state:
                # Clean optimizer states that might be in float32
                for state in optimizer_state['state'].values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(dtype=torch.get_default_dtype())
            
            # Charger l'état de l'optimiseur
            optimizer.load_state_dict(optimizer_state)
        
        del optimizer_state
        torch.cuda.empty_cache()
    
    iter_num = checkpoint.get('iter_num', 0) + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # Free memory
    model_args = checkpoint.get('model_args', None)
    config = checkpoint.get('config', None)
    del checkpoint
    torch.cuda.empty_cache()
    
    return model, optimizer, iter_num, best_val_loss, model_args, config

def find_latest_checkpoint(out_dir):
    """
    Find the latest checkpoint in a directory
    Args:
        out_dir: Directory to search
    Returns:
        str or None: Path to latest checkpoint or None if none found
    """
    checkpoints = glob.glob(os.path.join(out_dir, 'ckpt_iter_*.pt'))
    if not checkpoints:
        return None
        
    # Extract iteration numbers and sort
    def extract_iter_num(filename):
        try:
            return int(filename.split('iter_')[1].split('_loss')[0])
        except:
            return 0
        
    checkpoints.sort(key=extract_iter_num)
    return checkpoints[-1]

@contextmanager
def custom_fp8_autocast():
    """
    Custom context manager for FP8 precision on compatible hardware.
    This is required since PyTorch doesn't natively support FP8 through autocast yet.
    """
    # Mark the start of FP8 region for profiling
    try:
        if NVTX_AVAILABLE:
            nvtx.range_push("FP8_region")
        
        # Check if we can use ada_fp8_utils for better Ada Lovelace support
        try:
            from ada_fp8_utils import ada_fp8_autocast
            # Utiliser la version adaptée aux GPU Ada
            with ada_fp8_autocast():
                yield
            return
        except ImportError:
            # ada_fp8_utils non disponible, continuons avec la méthode standard
            pass
        
        # Import transformer-engine for FP8 support (if available)
        import transformer_engine.pytorch as te
        
        # Vérifier que fp8_autocast est disponible
        if not hasattr(te, "fp8_autocast"):
            print("WARNING: transformer_engine.pytorch n'a pas l'attribut fp8_autocast. Fallback vers bfloat16.")
            # Fallback vers bfloat16
            with torch.amp.autocast(enabled=True, device_type='cuda'):
                yield
            return
        
        # Check if we're using the newer version with recipe API
        if hasattr(te, "common") and hasattr(te.common, "recipe"):
            # Initialize FP8 meta tensors with proper recipe configuration
            # Format.HYBRID uses E4M3 for forward and E5M2 for backward
            from transformer_engine.common import recipe
            fp8_recipe = recipe.DelayedScaling(
                margin=0,
                interval=1,
                fp8_format=recipe.Format.HYBRID
            )
            
            # Use the correct te.fp8_autocast with the recipe
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                yield
        else:
            # Ancienne version de transformer-engine, fp8_autocast sans recipe
            # Cette version fonctionne mieux avec Ada Lovelace
            with te.fp8_autocast():
                yield
    except ImportError as e:
        print(f"WARNING: transformer_engine.pytorch n'est pas disponible: {e}. Fallback vers bfloat16.")
        # Fallback vers bfloat16
        with torch.amp.autocast(enabled=True, device_type='cuda'):
            yield
    except AttributeError as e:
        print(f"WARNING: erreur dans transformer_engine.pytorch: {e}. Fallback vers bfloat16.")
        # Fallback vers bfloat16
        with torch.amp.autocast(enabled=True, device_type='cuda'):
            yield
    except Exception as e:
        print(f"WARNING: erreur inattendue avec transformer_engine: {e}. Fallback vers bfloat16.")
        # Fallback vers bfloat16
        with torch.amp.autocast(enabled=True, device_type='cuda'):
            yield
    finally:
        # Mark the end of FP8 region for profiling
        if NVTX_AVAILABLE:
            nvtx.range_pop()

def get_context_manager(device_type, dtype):
    """
    Retourne le context manager approprié pour le type d'appareil et le dtype.
    
    Args:
        device_type (str): Type d'appareil ('cuda' ou 'cpu')
        dtype (str): Type de données ('float32', 'bfloat16', 'float16', ou 'fp8')
        
    Returns:
        context manager: Contexte pour l'autocast
    """
    if dtype == 'fp8':
        return custom_fp8_autocast()
    elif device_type == 'cuda':
        if dtype == 'bfloat16':
            # BFloat16 a une plage dynamique suffisante, pas besoin de GradScaler
            return torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        elif dtype == 'float16':
            # Float16 nécessite GradScaler pour stabilité
            return torch.amp.autocast(device_type=device_type, dtype=torch.float16)
        else:
            # Pas d'autocast pour float32
            return nullcontext()
    else:
        # CPU n'utilise pas d'autocast
        return nullcontext()

@torch.no_grad()
def estimate_loss(model, train_dataset, val_dataset, eval_iters, device, ddp=False, ddp_world_size=1):
    """
    Estime la perte sur les ensembles d'entraînement et de validation.
    
    Args:
        model: Le modèle à évaluer
        train_dataset: Dataset d'entraînement
        val_dataset: Dataset de validation
        eval_iters: Nombre d'itérations pour l'évaluation
        device: Appareil sur lequel effectuer l'évaluation
        ddp: Si True, le modèle est entraîné en DDP
        ddp_world_size: Nombre de processus DDP
    
    Returns:
        Un dictionnaire contenant les pertes moyennes
    """
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    out = {}
    model.eval()
    for split_name, dataset in [('train', train_dataset), ('val', val_dataset)]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            try:
                batch = next(iter(dataset))
            except StopIteration:
                # Réinitialiser l'itérateur si nécessaire
                iterator = iter(dataset)
                batch = next(iterator)
            
            # Handle different batch formats
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                targets = batch.get('labels', input_ids).to(device)
            elif isinstance(batch, tuple) and len(batch) >= 2:
                input_ids = batch[0].to(device)
                targets = batch[-1].to(device)
            else:
                input_ids = batch.to(device)
                targets = batch.to(device)
            
            with torch.amp.autocast(enabled=True, device_type=device_type):
                logits, loss = model(input_ids, targets=targets) if not hasattr(model, 'module') else model.module(input_ids, targets=targets)
            losses[k] = loss.item()
        
        # Reduce losses across ranks if using DDP
        if ddp:
            losses = reduce_metrics(losses, ddp_world_size)
        
        # Calculate mean loss and perplexity
        out[split_name] = losses.mean().item()
        out[f'{split_name}_ppl'] = calculate_perplexity(out[split_name])
    
    model.train()
    return out

def generate_text(model, encoder_input, max_new_tokens=50, temperature=0.8, top_k=40, tokenizer=None):
    """
    Generate text using the model
    Args:
        model: Model with generate method
        encoder_input: Input tensor [batch, seq_len]
        max_new_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        top_k: Top-k for sampling
        tokenizer: Tokenizer for decoding (optional)
    Returns:
        str: Generated text
    """
    # Ensure model is in eval mode for generation
    model.eval()

    output_tokens = None # Initialize
    try:
        with torch.no_grad(): # No need for gradients during generation
            # Check if the model has a generate method (standard for HF-like models)
            if hasattr(model, 'generate') and callable(getattr(model, 'generate')):
                # Explicitly pass input_ids and other relevant args
                # Remove tokenizer from this call as it's not a direct model.generate param for all models
                generate_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_k": top_k
                }
                # Some models might return more than just tokens (e.g. scores), capture them if so.
                # Assuming the first returned item is the sequence of tokens.
                returned_from_generate = model.generate(
                    input_ids=encoder_input, 
                    **generate_kwargs
                )
                if isinstance(returned_from_generate, tuple):
                    output_tokens = returned_from_generate[0]
                else:
                    output_tokens = returned_from_generate
            else:
                # Check if it's the LLaDA model and call the appropriate generate method
                if LLaDAModel is not None and isinstance(model.module if hasattr(model, 'module') else model, LLaDAModel):
                    # Call the original LLaDA generation method
                    # Ensure arguments match generate_original_llada signature
                    # Note: generate_original_llada might return (tokens, None) or (tokens, error)
                    output_tokens, error_info = model.generate_original_llada(
                        prompt=encoder_input,
                        steps=32,  # Keep original steps argument if needed by this method
                        gen_length=max_new_tokens,
                        block_length=max_new_tokens, # Or a different block length if appropriate
                        temperature=temperature,
                        tokenizer=tokenizer,
                        remasking='low_confidence' 
                    )
                    if error_info is not None: print(f"Original LLaDA generation returned error: {error_info}")
                    # output_text_from_generate remains None here as generate_original_llada doesn't return decoded text
                else:
                    # Standard generate call for other models (e.g., GPT) OR LLaDA if isinstance fails
                    # Assuming they have a generate method compatible with these args
                    # The new LLaDA generate returns (tokens, None)
                    output_tokens, _ = model.generate( # Use _ to ignore the second return value
                        prompt=encoder_input, # Use 'prompt' as it's expected by new LLaDA generate
                        gen_length=max_new_tokens, # Match new LLaDA generate signature
                        temperature=temperature,
                        top_k=top_k
                    )
    
    except Exception as e:
        print(f"Error during text generation: {e}")
        traceback.print_exc()
    
    # If output_text is already provided by the model, use it
    # This check is likely redundant now as neither path assigns to output_text_from_generate yet
    if output_tokens is not None:
        return output_tokens
    
    # Otherwise, decode the tokens if tokenizer is available
    if tokenizer:
        if output_tokens is None:
            print("Warning: output_tokens is None in generate_text, cannot decode.")
"""
Fonctions utilitaires pour l'entraînement des modèles LLM.
"""

import os
import gc
import glob
import math
import time
import torch
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp
from contextlib import nullcontext, contextmanager
from collections import defaultdict

# Import LLaDAModel to check instance type
try:
    from models.llada.model import LLaDAModel
except ImportError:
    LLaDAModel = None # Define as None if import fails
# Removed duplicate defaultdict import

# Add torch.cuda.nvtx import for profiling FP8 operations (if available)
try:
    import torch.cuda.nvtx as nvtx
    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False

# Constantes pour le calcul des FLOPS
GPU_FLOPS = {
    (7, 0): 312e12,      # A100-40GB (312 TFLOPS en FP16)
    (8, 0): 312e12,      # A100-80GB (312 TFLOPS en FP16)
    (8, 6): 756e12,      # H100-SXM (756 TFLOPS en FP16)
    (9, 0): 495e12,      # H200 (495 TFLOPS en FP16)
    (8, 9): 82.6e12,     # RTX 4090 (82.6 TFLOPS en FP16/FP32 avec Tensor Cores)
    (9, 0, 'fp8'): 989e12,  # H200 en FP8 (989 TFLOPS)
    (8, 6, 'fp8'): 1513e12, # H100 en FP8 (1513 TFLOPS)
    (12, 0): 2000e12,     # Estimated RTX 5090/CC 12.0 (FP16)
    (12, 0, 'fp8'): 4000e12, # Estimated RTX 5090/CC 12.0 (FP8)
}
def get_gpu_count():
    """Return the number of available GPUs."""
    return torch.cuda.device_count()

def setup_distributed(backend='nccl'):
    """Setup distributed training environment with optimized settings."""
    # Optimizations for NCCL
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_NET_GDR_LEVEL"] = "2"
    os.environ["NCCL_P2P_DISABLE"] = "0"
    os.environ["NCCL_MIN_NCHANNELS"] = "4"
    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    os.environ["NCCL_BUFFSIZE"] = "2097152"
    
    # Reduce synchronization frequency
    os.environ["NCCL_ALGO"] = "Ring"
    os.environ["NCCL_PROTO"] = "Simple"
    
    # Async error handling
    os.environ["NCCL_BLOCKING_WAIT"] = "0"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    
    # Initialize process group
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    
    return ddp_rank, ddp_local_rank, ddp_world_size, device

def reduce_metrics(metrics, world_size):
    """
    Reduce metrics across all processes in distributed training.
    Args:
        metrics: Tensor or float to reduce
        world_size: Total number of processes
    Returns:
        Reduced metric value
    """
    if not isinstance(metrics, torch.Tensor):
        metrics = torch.tensor(metrics, device='cuda')
    all_reduce(metrics, op=ReduceOp.SUM)
    return metrics.item() / world_size

def calculate_perplexity(loss):
    """
    Calculate perplexity from cross-entropy loss
    Args:
        loss: Loss value (cross-entropy)
    Returns:
        float: Perplexity value
    """
    return torch.exp(torch.tensor(loss)).item()

def get_lr(current_iter, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """
    Learning rate scheduler with cosine decay and warmup
    Args:
        current_iter: Current training iteration
        warmup_iters: Number of warmup iterations
        lr_decay_iters: Number of iterations over which to decay LR
        learning_rate: Base learning rate
        min_lr: Minimum learning rate
    Returns:
        Current learning rate based on scheduler
    """
    if current_iter < warmup_iters:
        return learning_rate * current_iter / warmup_iters
    if current_iter > lr_decay_iters:
        return min_lr
    decay_ratio = (current_iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def cleanup_old_checkpoints(out_dir, keep_num=3):
    """
    Keep only the 'keep_num' most recent checkpoints.
    Args:
        out_dir: Directory containing checkpoints
        keep_num: Number of checkpoints to keep
    """
    checkpoints = glob.glob(os.path.join(out_dir, 'ckpt_iter_*.pt'))
    if len(checkpoints) <= keep_num:
        return
        
    # Sort checkpoints by iteration number
    def extract_iter_num(filename):
        try:
            return int(filename.split('iter_')[1].split('_loss')[0])
        except:
            return 0
            
    checkpoints.sort(key=extract_iter_num)
    
    # Delete oldest checkpoints
    for ckpt in checkpoints[:-keep_num]:
        try:
            os.remove(ckpt)
            print(f"Removed old checkpoint: {ckpt}")
        except Exception as e:
            print(f"Error removing checkpoint {ckpt}: {e}")

def ensure_model_dtype(model, dtype):
    """
    Ensure model parameters are in the correct dtype
    Args:
        model: PyTorch model
        dtype: Target dtype
    Returns:
        Model with parameters in the target dtype
    """
    for param in model.parameters():
        param.data = param.data.to(dtype=dtype)
    return model

def save_checkpoint(model, optimizer, model_args, iter_num, best_val_loss, config, out_dir, val_loss=None):
    """
    Save model checkpoint with optimized memory handling
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        model_args: Model architecture arguments
        iter_num: Current iteration number
        best_val_loss: Best validation loss so far
        config: Training configuration
        out_dir: Output directory
        val_loss: Current validation loss (optional)
    """
    # Save to CPU to avoid GPU memory duplication
    raw_model = model.module if hasattr(model, 'module') else model
    
    # Préparer le dictionnaire de checkpoint
    checkpoint = {
        'model': {k: v.cpu() for k, v in raw_model.state_dict().items()},
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    
    # Gérer différents types d'optimiseurs
    optimizer_state = optimizer.state_dict()
    
    # Vérifier si c'est un MultiOptimizer (qui a une structure différente)
    if 'opt_0' in optimizer_state:
        # C'est un MultiOptimizer, conserver sa structure spéciale
        checkpoint['optimizer'] = optimizer_state
    else:
        # C'est un optimiseur standard, traiter comme avant
        if 'state' in optimizer_state and 'param_groups' in optimizer_state:
            checkpoint['optimizer'] = {
                'state': {k: {sk: sv.cpu() if isinstance(sv, torch.Tensor) else sv
                            for sk, sv in v.items()}
                            for k, v in optimizer_state['state'].items()},
                'param_groups': optimizer_state['param_groups']
            }
        else:
            # Fallback pour d'autres structures d'optimiseurs
            checkpoint['optimizer'] = optimizer_state
    
    checkpoint_name = f'ckpt_iter_{iter_num}'
    if val_loss is not None:
        checkpoint_name += f'_loss_{val_loss:.4f}'
    checkpoint_path = os.path.join(out_dir, f'{checkpoint_name}.pt')
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path

def load_checkpoint(ckpt_path, model, optimizer=None, map_location='cpu'):
    """
    Load checkpoint for model and optimizer.
    
    Args:
        ckpt_path (str): Path to checkpoint
        model (torch.nn.Module): Model to load checkpoint into
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into
        map_location (str, optional): Device to map tensors to. Defaults to 'cpu'.
        
    Returns:
        tuple: (model, optimizer, iter_num, best_val_loss)
    """
    # Add PreTrainedTokenizerFast to safe globals for loading with weights_only=True
    try:
        from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
        import torch.serialization
        torch.serialization.add_safe_globals([PreTrainedTokenizerFast])
    except (ImportError, AttributeError):
        print("Warning: Could not add PreTrainedTokenizerFast to safe globals. Will attempt loading with weights_only=False.")
        checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    else:
        try:
            # Try loading with weights_only=True
            checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=True)
        except Exception as e:
            print(f"Warning: Failed to load with weights_only=True: {e}")
            print("Falling back to weights_only=False for backward compatibility")
            checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    
    # Clean state dict before loading
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    cleaned_state_dict = {}
    
    for k, v in state_dict.items():
        if k.startswith(unwanted_prefix):
            cleaned_state_dict[k[len(unwanted_prefix):]] = v
        else:
            cleaned_state_dict[k] = v
    
    # Load model state
    model.load_state_dict(cleaned_state_dict, strict=False)
    
    # Clean memory
    del state_dict
    del cleaned_state_dict
    torch.cuda.empty_cache()
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer_state = checkpoint['optimizer']
        
        # Vérifier si c'est un format MultiOptimizer
        if 'opt_0' in optimizer_state:
            # Format MultiOptimizer - vérifier si l'optimiseur actuel est aussi un MultiOptimizer
            if hasattr(optimizer, 'load_state_dict'):
                optimizer.load_state_dict(optimizer_state)
        else:
            # Format standard avec 'state' et 'param_groups'
            if 'state' in optimizer_state:
                # Clean optimizer states that might be in float32
                for state in optimizer_state['state'].values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(dtype=torch.get_default_dtype())
            
            # Charger l'état de l'optimiseur
            optimizer.load_state_dict(optimizer_state)
        
        del optimizer_state
        torch.cuda.empty_cache()
    
    iter_num = checkpoint.get('iter_num', 0) + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # Free memory
    model_args = checkpoint.get('model_args', None)
    config = checkpoint.get('config', None)
    del checkpoint
    torch.cuda.empty_cache()
    
    return model, optimizer, iter_num, best_val_loss, model_args, config

def find_latest_checkpoint(out_dir):
    """
    Find the latest checkpoint in a directory
    Args:
        out_dir: Directory to search
    Returns:
        str or None: Path to latest checkpoint or None if none found
    """
    checkpoints = glob.glob(os.path.join(out_dir, 'ckpt_iter_*.pt'))
    if not checkpoints:
        return None
        
    # Extract iteration numbers and sort
    def extract_iter_num(filename):
        try:
            return int(filename.split('iter_')[1].split('_loss')[0])
        except:
            return 0
        
    checkpoints.sort(key=extract_iter_num)
    return checkpoints[-1]

@contextmanager
def custom_fp8_autocast():
    """
    Custom context manager for FP8 precision on compatible hardware.
    This is required since PyTorch doesn't natively support FP8 through autocast yet.
    """
    # Mark the start of FP8 region for profiling
    try:
        if NVTX_AVAILABLE:
            nvtx.range_push("FP8_region")
        
        # Check if we can use ada_fp8_utils for better Ada Lovelace support
        try:
            from ada_fp8_utils import ada_fp8_autocast
            # Utiliser la version adaptée aux GPU Ada
            with ada_fp8_autocast():
                yield
            return
        except ImportError:
            # ada_fp8_utils non disponible, continuons avec la méthode standard
            pass
        
        # Import transformer-engine for FP8 support (if available)
        import transformer_engine.pytorch as te
        
        # Vérifier que fp8_autocast est disponible
        if not hasattr(te, "fp8_autocast"):
            print("WARNING: transformer_engine.pytorch n'a pas l'attribut fp8_autocast. Fallback vers bfloat16.")
            # Fallback vers bfloat16
            with torch.amp.autocast(enabled=True, device_type='cuda'):
                yield
            return
        
        # Check if we're using the newer version with recipe API
        if hasattr(te, "common") and hasattr(te.common, "recipe"):
            # Initialize FP8 meta tensors with proper recipe configuration
            # Format.HYBRID uses E4M3 for forward and E5M2 for backward
            from transformer_engine.common import recipe
            fp8_recipe = recipe.DelayedScaling(
                margin=0,
                interval=1,
                fp8_format=recipe.Format.HYBRID
            )
            
            # Use the correct te.fp8_autocast with the recipe
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                yield
        else:
            # Ancienne version de transformer-engine, fp8_autocast sans recipe
            # Cette version fonctionne mieux avec Ada Lovelace
            with te.fp8_autocast():
                yield
    except ImportError as e:
        print(f"WARNING: transformer_engine.pytorch n'est pas disponible: {e}. Fallback vers bfloat16.")
        # Fallback vers bfloat16
        with torch.amp.autocast(enabled=True, device_type='cuda'):
            yield
    except AttributeError as e:
        print(f"WARNING: erreur dans transformer_engine.pytorch: {e}. Fallback vers bfloat16.")
        # Fallback vers bfloat16
        with torch.amp.autocast(enabled=True, device_type='cuda'):
            yield
    except Exception as e:
        print(f"WARNING: erreur inattendue avec transformer_engine: {e}. Fallback vers bfloat16.")
        # Fallback vers bfloat16
        with torch.amp.autocast(enabled=True, device_type='cuda'):
            yield
    finally:
        # Mark the end of FP8 region for profiling
        if NVTX_AVAILABLE:
            nvtx.range_pop()

def get_context_manager(device_type, dtype):
    """
    Retourne le context manager approprié pour le type d'appareil et le dtype.
    
    Args:
        device_type (str): Type d'appareil ('cuda' ou 'cpu')
        dtype (str): Type de données ('float32', 'bfloat16', 'float16', ou 'fp8')
        
    Returns:
        context manager: Contexte pour l'autocast
    """
    if dtype == 'fp8':
        return custom_fp8_autocast()
    elif device_type == 'cuda':
        if dtype == 'bfloat16':
            # BFloat16 a une plage dynamique suffisante, pas besoin de GradScaler
            return torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        elif dtype == 'float16':
            # Float16 nécessite GradScaler pour stabilité
            return torch.amp.autocast(device_type=device_type, dtype=torch.float16)
        else:
            # Pas d'autocast pour float32
            return nullcontext()
    else:
        # CPU n'utilise pas d'autocast
        return nullcontext()

@torch.no_grad()
def estimate_loss(model, train_dataset, val_dataset, eval_iters, device, ddp=False, ddp_world_size=1):
    """
    Estime la perte sur les ensembles d'entraînement et de validation.
    
    Args:
        model: Le modèle à évaluer
        train_dataset: Dataset d'entraînement
        val_dataset: Dataset de validation
        eval_iters: Nombre d'itérations pour l'évaluation
        device: Appareil sur lequel effectuer l'évaluation
        ddp: Si True, le modèle est entraîné en DDP
        ddp_world_size: Nombre de processus DDP
    
    Returns:
        Un dictionnaire contenant les pertes moyennes
    """
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    out = {}
    model.eval()
    for split_name, dataset in [('train', train_dataset), ('val', val_dataset)]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            try:
                batch = next(iter(dataset))
            except StopIteration:
                # Réinitialiser l'itérateur si nécessaire
                iterator = iter(dataset)
                batch = next(iterator)
            
            # Handle different batch formats
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                targets = batch.get('labels', input_ids).to(device)
            elif isinstance(batch, tuple) and len(batch) >= 2:
                input_ids = batch[0].to(device)
                targets = batch[-1].to(device)
            else:
                input_ids = batch.to(device)
                targets = batch.to(device)
            
            with torch.amp.autocast(enabled=True, device_type=device_type):
                logits, loss = model(input_ids, targets=targets) if not hasattr(model, 'module') else model.module(input_ids, targets=targets)
            losses[k] = loss.item()
        
        # Reduce losses across ranks if using DDP
        if ddp:
            losses = reduce_metrics(losses, ddp_world_size)
        
        # Calculate mean loss and perplexity
        out[split_name] = losses.mean().item()
        out[f'{split_name}_ppl'] = calculate_perplexity(out[split_name])
    
    model.train()
    return out

def generate_text(model, encoder_input, max_new_tokens=50, temperature=0.8, top_k=40, tokenizer=None):
    """
    Generate text using the model
    Args:
        model: Model with generate method
        encoder_input: Input tensor [batch, seq_len]
        max_new_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        top_k: Top-k for sampling
        tokenizer: Tokenizer for decoding (optional)
    Returns:
        str: Generated text
    """
    model.eval()
    
    with torch.no_grad():
        with torch.amp.autocast(enabled=True, device_type='cuda'):
            output_tokens = None
            output_text_from_generate = None # Store text if generate provides it directly

            # Check if it's the LLaDA model and call the appropriate generate method
            if LLaDAModel is not None and isinstance(model.module if hasattr(model, 'module') else model, LLaDAModel):
                # Call the original LLaDA generation method
                # Ensure arguments match generate_original_llada signature
                # Note: generate_original_llada might return (tokens, None) or (tokens, error)
                output_tokens, error_info = model.generate_original_llada(
                    prompt=encoder_input,
                    steps=32,  # Keep original steps argument if needed by this method
                    gen_length=max_new_tokens,
                    block_length=max_new_tokens, # Or a different block length if appropriate
                    temperature=temperature,
                    tokenizer=tokenizer,
                    remasking='low_confidence' 
                )
                if error_info is not None: print(f"Original LLaDA generation returned error: {error_info}")
                # output_text_from_generate remains None here as generate_original_llada doesn't return decoded text
            else:
                # Check which parameters the model's generate method accepts
                if model.__class__.__name__ == 'MLAModel':
                    # MLAModel expects 'idx' as the primary parameter
                    output_tokens, _ = model.generate(
                        idx=encoder_input,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k
                    )
                else:
                    # Standard generate call for other models (e.g., GPT) OR LLaDA if isinstance fails
                    # Assuming they have a generate method compatible with these args
                    # The new LLaDA generate returns (tokens, None)
                    output_tokens, _ = model.generate( # Use _ to ignore the second return value
                        prompt=encoder_input, # Use 'prompt' as it's expected by new LLaDA generate
                        gen_length=max_new_tokens, # Match new LLaDA generate signature
                        temperature=temperature,
                        top_k=top_k
                    )
    
    # If output_text is already provided by the model, use it
    # This check is likely redundant now as neither path assigns to output_text_from_generate yet
    if output_text_from_generate is not None:
        return output_text_from_generate
    
    # Otherwise, decode the tokens if tokenizer is available
    if tokenizer:
        if output_tokens is None:
            print("Warning: output_tokens is None in generate_text, cannot decode.")
            return ""
        # Ensure output_tokens is on CPU and is a LongTensor before converting to list
        if isinstance(output_tokens, torch.Tensor):
            output_tokens_cpu = output_tokens.detach().cpu()
            # Handle potential batch dimension (take the first sequence)
            if output_tokens_cpu.ndim > 1:
                token_list = output_tokens_cpu[0].tolist()
            else:
                token_list = output_tokens_cpu.tolist()
            
            # Decode the list of integers
            try:
                output_text = tokenizer.decode(token_list, skip_special_tokens=True)
            except Exception as e:
                print(f"Error during token decoding: {e}")
                print(f"Problematic token list (first 10): {token_list[:10]}")
                output_text = "[Decoding Error]"
        else:
            print(f"Warning: output_tokens is not a tensor ({type(output_tokens)}), cannot decode.")
            output_text = "[Token Type Error]"
            
        # Add basic cleanup for generated text
        output_text = output_text.strip()
        return output_text
    else:
        # If no tokenizer, return raw tokens (or error message)
        if output_tokens is not None:
            return f"[Raw Tokens: {output_tokens.tolist()}]" # Return list representation
        else:
            return "[No Tokens Generated]"

def evaluate_model(model, eval_dataset, ctx, num_examples=5, max_tokens=50, temperature=0.8, tokenizer=None):
    """
    Évalue le modèle en générant quelques exemples.
    
    Args:
        model: Le modèle à évaluer
        eval_dataset: Dataset d'évaluation pour obtenir des prompts
        ctx: Contexte pour l'autocast
        num_examples: Nombre d'exemples à générer
        max_tokens: Nombre maximum de tokens à générer par exemple
        temperature: Température pour l'échantillonnage
        tokenizer: Tokenizer pour décoder l'entrée et la sortie
        
    Returns:
        None: Affiche les exemples générés
    """
    model.eval()
    print("\n--- Generating Sample Text ---")
    
    try:
        eval_iterator = iter(eval_dataset)
        for i in range(num_examples):
            try:
                batch = next(eval_iterator)
            except StopIteration:
                print("Reached end of evaluation dataset.")
                break
                
            # Handle different batch formats
            if isinstance(batch, dict):
                input_ids = batch['input_ids']
            elif isinstance(batch, tuple) and len(batch) >= 1:
                input_ids = batch[0]
            else:
                input_ids = batch
                
            # Prendre seulement le premier exemple du batch pour le prompt
            start_ids = input_ids[0, :50].unsqueeze(0).to(model.device) # Prendre les 50 premiers tokens comme prompt
            
            # Décoder le prompt si possible
            if tokenizer:
                prompt_text = tokenizer.decode(start_ids[0].tolist(), skip_special_tokens=True)
                print(f"\nExample {i+1}:")
                print(f"Prompt: '{prompt_text}'")
            else:
                print(f"\nExample {i+1}:")
                print(f"Prompt Tokens: {start_ids[0].tolist()}")
            
            # Générer le texte
            generated_text = generate_text(
                model, 
                start_ids, 
                max_new_tokens=max_tokens, 
                temperature=temperature, 
                tokenizer=tokenizer
            )
            
            print(f"Generated: '{generated_text}'")
            
    except Exception as e:
        print(f"\nError during text generation: {e}")
        traceback.print_exc()
        
    print("--- End Sample Text Generation ---\n")
    model.train()

class AveragedTimingStats:
    """
    Classe pour calculer et afficher des statistiques de temps moyennées.
    """
    def __init__(self, print_interval=100):
        """
        Initialise les statistiques de temps.
        
        Args:
            print_interval (int): Intervalle d'impression des statistiques.
        """
        self.timings = defaultdict(lambda: {'total': 0.0, 'count': 0})
        self.start_time = time.time()
        self.last_print_time = time.time()
        self.print_interval = print_interval
        self.step_count = 0

    @contextmanager
    def track(self, name):
        """
        Context manager pour suivre le temps d'une section de code.
        
        Args:
            name (str): Nom de la section à suivre.
        """
        start = time.time()
        yield
        end = time.time()
        self.timings[name]['total'] += (end - start)
        self.timings[name]['count'] += 1

    def step(self):
        """Incrémente le compteur d'étapes."""
        self.step_count += 1

    def should_print(self):
        """Vérifie s'il est temps d'imprimer les statistiques."""
        return self.step_count % self.print_interval == 0

    def get_averaged_stats(self):
        """
        Calcule les statistiques de temps moyennées.
        
        Returns:
            dict: Dictionnaire contenant les temps moyens pour chaque section.
        """
        averaged = {}
        for name, data in self.timings.items():
            if data['count'] > 0:
                averaged[name] = (data['total'] / data['count']) * 1000 # Convertir en ms
        return averaged

    def print_stats(self):
        """Imprime les statistiques de temps moyennées."""
        if not self.should_print():
            return
            
        current_time = time.time()
        elapsed_since_last = current_time - self.last_print_time
        
        stats = self.get_averaged_stats()
        print(f"\n--- Timing Stats (Avg ms over last {self.print_interval} steps) ---")
        total_tracked_time = 0
        for name, avg_time in stats.items():
            print(f"{name:<20}: {avg_time:.2f} ms")
            total_tracked_time += avg_time
            
        # Réinitialiser pour la prochaine fenêtre
        self.timings = defaultdict(lambda: {'total': 0.0, 'count': 0})
        self.last_print_time = current_time
        
        print(f"{'Total Tracked':<20}: {total_tracked_time:.2f} ms")
        print(f"{'Interval Duration':<20}: {elapsed_since_last:.2f} s")
        print("------------------------------------------\n")


# --- Fonctions de calcul FLOPS/MFU ---

def calculate_model_flops(model, batch_size, seq_length, dtype=torch.float16):
    """
    Estime les FLOPS par itération pour un modèle Transformer standard.
    Formule approximative : 6 * N * B * S * H^2 * (1 + S / (6 * H))
    Simplifiée à : 6 * num_params * B * S
    
    Args:
        model: Le modèle PyTorch
        batch_size: Taille du batch
        seq_length: Longueur de la séquence
        dtype: Type de données utilisé (affecte les TFLOPS du GPU)
        
    Returns:
        float or None: FLOPS estimés par itération, ou None si non calculable
    """
    try:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Approximation simplifiée : 6 * N * B * S
        # N = nombre de paramètres
        # B = batch size
        # S = sequence length
        flops_per_token = 6 * num_params
        flops_per_iter = flops_per_token * batch_size * seq_length
        
        # Ajustement pour MoE (plus coûteux) - Facteur heuristique
        if hasattr(model, 'config') and hasattr(model.config, 'num_experts'):
            # Supposons que MoE ajoute environ 2x le coût d'un FFN dense
            # (1 expert actif + coût du routeur)
            ffn_params = 0
            # Estimer les paramètres FFN (souvent ~2/3 des paramètres totaux dans les grands modèles)
            # Ceci est une heuristique très approximative
            if hasattr(model, 'blocks'): 
                for block in model.blocks:
                    if hasattr(block, 'expert_group'): # LLaDA MoE
                         ffn_params += sum(p.numel() for p in block.expert_group.parameters())
                    elif hasattr(block, 'mlp'): # GPT MLP
                         ffn_params += sum(p.numel() for p in block.mlp.parameters())

            if ffn_params > 0:
                 # Supposons que le FFN représente ~2/3 des FLOPS (6*Nbs approx 4*Nbs pour attn + 2*Nbs pour FFN)
                 non_ffn_flops = flops_per_iter * (1 - (ffn_params / num_params)) 
                 # Coût MoE = Coût FFN dense * (num_experts_per_token / num_experts) * num_experts (simplifié à * k)
                 # Ou plus simplement, approx 2x le coût FFN dense si k=2
                 k = getattr(model.config, 'num_experts_per_token', 2) # Nombre d'experts actifs
                 moe_ffn_flops = (flops_per_iter * (ffn_params / num_params)) * k 
                 flops_per_iter = non_ffn_flops + moe_ffn_flops
                #  print(f"Adjusted FLOPS for MoE (k={k}): {flops_per_iter / 1e12:.2f} TFLOPS/iter (heuristic)")


        return flops_per_iter
        
    except Exception as e:
        print(f"Could not estimate FLOPS: {e}")
        return None

def estimate_mfu(model, batch_size, seq_length, dt, dtype=torch.float16):
    """
    Estime l'utilisation des FLOPS du modèle (MFU) en pourcentage.
    
    Args:
        model: Le modèle PyTorch
        batch_size: Taille du batch
        seq_length: Longueur de la séquence
        dt: Temps d'exécution de l'itération (en secondes)
        dtype: Type de données utilisé (affecte les TFLOPS du GPU)
        
    Returns:
        float: MFU estimée en pourcentage (0-100), ou 0.0 si non calculable
    """
    if dt == 0:
        return 0.0
        
    # Obtenir les FLOPS matériels
    try:
        dev_prop = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu_key = (dev_prop.major, dev_prop.minor)
        
        # Clé spécifique pour FP8 si nécessaire
        if dtype == torch.uint8 or dtype == torch.int8 or str(dtype) == 'fp8': # Handle string 'fp8' too
             gpu_key_fp8 = (dev_prop.major, dev_prop.minor, 'fp8')
             if gpu_key_fp8 in GPU_FLOPS:
                 gpu_key = gpu_key_fp8
             elif gpu_key not in GPU_FLOPS: # Fallback if FP8 specific not found and base not found
                 print(f"Warning: GPU compute capability {gpu_key} (or FP8 variant) not found in GPU_FLOPS map.")
                 return 0.0
        elif gpu_key not in GPU_FLOPS:
             print(f"Warning: GPU compute capability {gpu_key} not found in GPU_FLOPS map.")
             return 0.0
             
        peak_flops = GPU_FLOPS[gpu_key]
        
    except Exception as e:
        print(f"Could not get GPU peak FLOPS: {e}")
        return 0.0
        
    # Calculer les FLOPS du modèle
    model_flops = calculate_model_flops(model, batch_size, seq_length, dtype)
    if model_flops is None:
        return 0.0
        
    # Calculer MFU
    mfu = (model_flops / dt) / peak_flops
    
    return min(mfu * 100, 100.0) # Retourner en pourcentage, capé à 100%
