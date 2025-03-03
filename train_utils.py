import os
import gc
import glob
import math
import time
import torch
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp

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

def cleanup_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if gc.isenabled():
            gc.collect()

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

def print_memory_stats(prefix=""):
    """Print GPU memory statistics for debugging"""
    if not torch.cuda.is_available():
        return
        
    print(f"{prefix} Memory stats:")
    print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

def setup_cuda_optimizations():
    """Apply optimizations for CUDA environment"""
    if not torch.cuda.is_available():
        return
        
    # Memory management optimizations
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.empty_cache()
    
    # Performance optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    
    # Configure memory pagination
    torch.cuda.set_per_process_memory_fraction(0.95)
    
    # Disable automatic garbage collection
    gc.disable()

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
    
    checkpoint = {
        'model': {k: v.cpu() for k, v in raw_model.state_dict().items()},
        'optimizer': {
            'state': {k: {sk: sv.cpu() if isinstance(sv, torch.Tensor) else sv 
                         for sk, sv in v.items()}
                         for k, v in optimizer.state_dict()['state'].items()},
            'param_groups': optimizer.state_dict()['param_groups']
        },
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    
    checkpoint_name = f'ckpt_iter_{iter_num}'
    if val_loss is not None:
        checkpoint_name += f'_loss_{val_loss:.4f}'
    checkpoint_path = os.path.join(out_dir, f'{checkpoint_name}.pt')
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path

def load_checkpoint(ckpt_path, model, optimizer=None, map_location='cpu'):
    """
    Load checkpoint with memory optimization
    Args:
        ckpt_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        map_location: Device to load tensors onto
    Returns:
        tuple: (model, optimizer, iter_num, best_val_loss)
    """
    checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    
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
        # Clean optimizer states that might be in float32
        for state in optimizer_state['state'].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(dtype=torch.get_default_dtype())
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

def get_context_manager(device_type, dtype):
    """
    Get appropriate context manager for mixed precision training
    Args:
        device_type: 'cuda' or 'cpu'
        dtype: Desired dtype string ('float16', 'bfloat16', or 'float32')
    Returns:
        Context manager for mixed precision
    """
    from contextlib import nullcontext
    
    if device_type == 'cpu':
        return nullcontext()
        
    ptdtype = {
        'float32': torch.float32, 
        'bfloat16': torch.bfloat16, 
        'float16': torch.float16
    }[dtype]
    
    return torch.amp.autocast(enabled=True, device_type=device_type, dtype=ptdtype)

def estimate_loss(model, train_dataset, val_dataset, eval_iters, device, ctx, ddp=False, ddp_world_size=1):
    """
    Estimate loss on training and validation datasets
    Args:
        model: Model to evaluate
        train_dataset: Training dataset
        val_dataset: Validation dataset
        eval_iters: Number of evaluation iterations
        device: Device to run evaluation on
        ctx: Context manager for mixed precision
        ddp: Whether running in DDP mode
        ddp_world_size: World size for DDP
    Returns:
        dict: Dictionary with loss values
    """
    out = {}
    model.eval()
    
    with torch.no_grad():
        for split, dataset in [('train', train_dataset), ('val', val_dataset)]:
            losses = torch.zeros(eval_iters, device=device)
            valid_iters = 0
            for k in range(eval_iters):
                encoder_input, decoder_input, target = next(iter(dataset))
                with ctx:
                    logits, loss = model(encoder_input, decoder_input, target)
                    if loss is not None:
                        losses[valid_iters] = loss.item()
                        valid_iters += 1
            
            # Average over valid iterations
            if valid_iters > 0:
                mean_loss = losses[:valid_iters].mean()
                # Reduce loss across all processes if in DDP mode
                if ddp:
                    mean_loss = reduce_metrics(mean_loss, ddp_world_size)
                out[split] = mean_loss
                # Calculate and store perplexity
                out[f'{split}_ppl'] = calculate_perplexity(mean_loss)
            else:
                out[split] = float('inf')
                out[f'{split}_ppl'] = float('inf')
    
    model.train()
    return out

def generate_text(model, encoder_input, max_new_tokens=50, temperature=0.8, top_k=40, tokenizer=None):
    """
    Generate text using an encoder-decoder model
    Args:
        model: Encoder-decoder model
        encoder_input: Input tensor for the encoder [batch, seq_len]
        max_new_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        top_k: Top-k for sampling
        tokenizer: Tokenizer for decoding (optional)
    Returns:
        str: Generated text
    """
    model.eval()
    device = encoder_input.device
    batch_size = encoder_input.size(0)
    
    # Create initial decoder input (start token)
    pad_token_id = tokenizer.pad_token_id if tokenizer else 0
    bos_token_id = tokenizer.bos_token_id if tokenizer else 1
    
    # Initialize decoder input with start token
    decoder_input = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
    
    with torch.no_grad():
        with torch.amp.autocast(enabled=True, device_type='cuda'):
            # Get encoder outputs (only need to do this once)
            encoder_outputs = model.encoder(encoder_input)
            
            # Generate tokens one by one
            for _ in range(max_new_tokens):
                # Forward pass through decoder
                logits = model.decoder(decoder_input, encoder_outputs=encoder_outputs)
                
                # Get logits for the next token (last token in sequence)
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Sample from the distribution
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to decoder input
                decoder_input = torch.cat((decoder_input, next_token), dim=1)
                
                # Stop if we generate an EOS token
                if tokenizer and (next_token == tokenizer.eos_token_id).any():
                    break
    
    # Convert token IDs to text
    if tokenizer:
        output_text = tokenizer.decode(decoder_input[0].tolist(), skip_special_tokens=True)
    else:
        output_text = decoder_input[0].tolist()
    
    return output_text

def evaluate_model(model, eval_dataset, ctx, num_examples=5, max_tokens=50, temperature=0.8, tokenizer=None):
    """
    Evaluate model by generating samples
    Args:
        model: Model to evaluate
        eval_dataset: Evaluation dataset
        ctx: Context manager for mixed precision
        num_examples: Number of examples to generate
        max_tokens: Maximum number of tokens to generate per example
        temperature: Temperature for sampling
        tokenizer: Tokenizer for encoding/decoding
    Returns:
        list: List of dictionaries with input and generated output
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for _ in range(num_examples):
            encoder_input, _, _ = next(iter(eval_dataset))
            
            # Generate text
            generated = generate_text(
                model, 
                encoder_input, 
                max_new_tokens=max_tokens,
                temperature=temperature,
                tokenizer=tokenizer
            )
            
            # Decode input
            if tokenizer:
                input_text = tokenizer.decode(encoder_input[0].tolist(), skip_special_tokens=True)
            else:
                input_text = encoder_input[0].tolist()
            
            results.append({
                'input': input_text,
                'generated': generated
            })
    
    return results 