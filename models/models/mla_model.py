"""
MLA-Model: Un modèle sparse Mixture-of-Experts utilisant Multi-head Latent Attention.

Ce modèle combine:
- Multi-head Latent Attention (MLA) pour une attention efficace en mémoire
- Mixture-of-Experts (MoE) extrêmement sparse pour le scaling des paramètres
- RoPE (Rotary Positional Encoding) pour le positionnement
- Support pour l'entraînement en FP8 (avec paramètres critiques en FP16)
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Optional, Dict, List, Tuple, Union, Any
from dataclasses import dataclass

# Import components from blocks
from models.blocks.mla import MLA
from models.blocks.mla_block import MLABlock
from models.blocks.normalization import RMSNorm
from models.blocks.positional_encoding import RoPE
from models.blocks.mlp import MLP
from models.blocks.tensor_utils import isolate_tensor, prevent_backward_reuse

# Import utility functions
from train.train_utils import estimate_mfu as utils_estimate_mfu

@dataclass
class MLAModelConfig:
    """Configuration for MLA-Model."""
    # Architecture
    n_layer: int = 24
    n_embd: int = 2048
    n_head: int = 16
    n_inner: Optional[int] = None  # Inner dimension for MLP. If None, will be 4*n_embd
    vocab_size: int = 50304
    block_size: int = 4096
    
    # MLA specific parameters
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    
    # MoE parameters - kept but not used in dense model
    use_moe: bool = False  # Set to False for dense model
    
    # Méthodes d'attention
    attention_backend: Optional[str] = None  # Si None, utilisera automatiquement le meilleur backend
    attn_impl: str = "absorb"  # "absorb" (optimisé) ou "naive" (standard)
    
    # RoPE
    original_max_seq_len: int = 4096
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: float = 10000.0
    
    # Précision
    fp8_params: bool = True
    fp8_mla_params: bool = False  # Gardez MLA en FP16 pour la stabilité numérique
    
    # Dropout et régularisation
    dropout: float = 0.0
    bias: bool = False
    
    # Training
    use_gradient_checkpointing: bool = True
    init_device: str = 'cuda'
    label_smoothing: float = 0.0
    
    def __post_init__(self):
        # Set inner dimension if not provided
        if self.n_inner is None:
            self.n_inner = 4 * self.n_embd

class FP8Module(nn.Module):
    """Module wrapper that manages FP8 precision for specific parameters."""
    def __init__(self, module, exclude_patterns=None):
        super().__init__()
        self.module = module
        self.exclude_patterns = exclude_patterns or []
        
        # Convert eligible parameters to FP8
        self._convert_params_to_fp8()
    
    def _convert_params_to_fp8(self):
        for name, param in self.module.named_parameters():
            should_exclude = any(pattern in name for pattern in self.exclude_patterns)
            
            if not should_exclude and hasattr(torch, 'float8_e4m3fn'):
                # Convert to FP8 precision if supported
                param.data = param.data.to(torch.float8_e4m3fn)
                print(f"Converted {name} to FP8 precision")
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def to(self, device_or_dtype):
        # Special handling for converting to device or dtype
        if isinstance(device_or_dtype, torch.dtype):
            # Don't change dtype of FP8 parameters
            # This is a no-op for FP8 parameters
            pass
        return super().to(device_or_dtype)

# MoE removed - using only dense model

class MLAModelBlock(nn.Module):
    """
    MLA-Model block combining MLA attention and dense MLP feed-forward network.
    """
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        
        # Layer normalization
        self.norm1 = RMSNorm(config.n_embd)
        self.norm2 = RMSNorm(config.n_embd)
        
        # Multi-head Latent Attention
        self.attn = MLA(config)
        
        # Regular MLP (all layers are dense)
        self.ffn = MLP(config)
        
        # Gradient checkpointing
        self.use_checkpoint = config.use_gradient_checkpointing
    
    def _attn_block(self, x, start_pos=0, freqs_cis=None, mask=None):
        return self.attn(self.norm1(x), start_pos, freqs_cis, mask)
    
    def _ffn_block(self, x):
        normalized = self.norm2(x)
        # Always use standard MLP
        return self.ffn(normalized)
    
    def forward(self, x, start_pos=0, freqs_cis=None, mask=None):
        # Apply attention
        if self.use_checkpoint and self.training:
            attn_output = checkpoint.checkpoint(
                self._attn_block, x, start_pos, freqs_cis, mask,
                use_reentrant=False
            )
        else:
            attn_output = self._attn_block(x, start_pos, freqs_cis, mask)
        
        x = x + attn_output
        
        # Apply feed-forward network
        if self.use_checkpoint and self.training:
            ffn_output = checkpoint.checkpoint(
                self._ffn_block, x,
                use_reentrant=False
            )
        else:
            ffn_output = self._ffn_block(x)
        
        x = x + ffn_output
        
        return x

class MLAModel(nn.Module):
    """
    MLA-Model: Un modèle sparse Mixture-of-Experts utilisant Multi-head Latent Attention.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([MLAModelBlock(config, i) for i in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd)
        ))
        
        # LM head (weight tied with embeddings)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        
        # Router loss tracking
        self.last_router_loss = None
        
        # Rotary embeddings for positional encoding
        rope_theta = config.rope_theta
        max_seq_len = config.block_size
        head_dim = config.qk_rope_head_dim
        
        # Compute rotary frequencies
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Apply rope scaling if provided
        if config.rope_scaling is not None:
            scaling_type = config.rope_scaling["type"]
            scaling_factor = config.rope_scaling["factor"]
            
            if scaling_type == "linear":
                self.register_buffer("freqs_cis", precompute_freqs_cis_with_linear_scaling(
                    head_dim, max_seq_len, rope_theta, scaling_factor, 
                    config.original_max_seq_len
                ))
            else:
                raise ValueError(f"Unknown RoPE scaling type: {scaling_type}")
        else:
            self.register_buffer("freqs_cis", precompute_freqs_cis(
                head_dim, max_seq_len, rope_theta
            ))
        
        # Tracker for MFU estimation
        self.timing_stats = None
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Convert to FP8 if requested and if we're in a CUDA context
        # Skip FP8 conversion for CPU-only operation as it's not supported
        if config.fp8_params and torch.cuda.is_available() and torch.cuda.device_count() > 0:
            exclude_patterns = []
            if not config.fp8_mla_params:
                exclude_patterns.extend(["attn", "wkv", "wq"])
            
            try:
                self._apply_fp8_to_params(exclude_patterns)
            except Exception as e:
                print(f"Warning: FP8 conversion failed: {e}")
                print("Continuing with standard precision")
        
        # Parameter count
        self.param_count = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {self.param_count/1e6:.2f}M")
    
    def _apply_fp8_to_params(self, exclude_patterns):
        """Apply FP8 precision to eligible parameters."""
        if not hasattr(torch, 'float8_e4m3fn'):
            print("Warning: FP8 precision requested but not supported by PyTorch. Skipping conversion.")
            return
        
        # Keep track of converted parameters
        fp8_param_count = 0
        total_param_count = 0
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                should_exclude = any(pattern in name for pattern in exclude_patterns)
                
                if not should_exclude:
                    # Convert parameters to FP8
                    if hasattr(module, 'weight') and module.weight is not None:
                        module.weight.data = module.weight.data.to(torch.float8_e4m3fn)
                        fp8_param_count += module.weight.numel()
                    
                total_param_count += sum(p.numel() for p in module.parameters())
        
        print(f"Converted {fp8_param_count / 1e6:.2f}M parameters to FP8 precision "
              f"({fp8_param_count / total_param_count * 100:.2f}% of model parameters)")
    
    def _init_weights(self, module):
        """Initialize weights with scaled initialization."""
        if isinstance(module, nn.Linear):
            # Use scaled initialization for better gradient flow
            scale = 1.0 / math.sqrt(self.config.n_embd)
            nn.init.normal_(module.weight, mean=0.0, std=scale)
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def set_timing_stats(self, timing_stats):
        """Set timing stats object for MFU estimation."""
        self.timing_stats = timing_stats
    
    def estimate_mfu(self, batch_size, dt, seq_length=None):
        """
        Estimate model FLOPS utilization (MFU) in percentage.
        
        Args:
            batch_size (int): Batch size for the calculation
            dt (float): Time taken for the forward+backward pass
            seq_length (int, optional): Sequence length. If None, uses block_size
        """
        # If seq_length is not provided, use the model's block_size
        if seq_length is None:
            seq_length = self.config.block_size
            
        return utils_estimate_mfu(
            model=self,
            batch_size=batch_size,
            seq_length=seq_length,
            dt=dt,
            # Use mixed precision
            dtype=torch.bfloat16 if not self.config.fp8_params else torch.float16
        )
    
    def set_gradient_checkpointing(self, value: bool):
        """Set gradient checkpointing for all transformer blocks."""
        self.config.use_gradient_checkpointing = value
        for block in self.transformer.h:
            block.use_checkpoint = value
    
    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        # Always ensure we're in training mode when doing a forward pass
        # This prevents KV caching which causes memory leaks
        if self.training:
            self._set_inference_mode(False)
        
        # Completely isolate input tensors to ensure no connections to previous computation graphs
        with torch.no_grad():
            idx = idx.detach().clone().requires_grad_(False)
            if targets is not None:
                targets = targets.detach().clone().requires_grad_(False)

        # Use the context manager to prevent backward reuse
        with prevent_backward_reuse():
            device = idx.device
            b, t = idx.size()
            assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            
            # Generate position indices - no grad needed
            with torch.no_grad():
                pos = torch.arange(0, t, dtype=torch.long, device=device)
            
            # Forward the MLA model
            tok_emb = self.transformer.wte(idx)
            x = self.transformer.drop(tok_emb)
            
            # Prepare attention mask if needed (for causal attention) - no grad needed
            with torch.no_grad():
                mask = None
                if t > 1:
                    mask = torch.full((t, t), float("-inf"), device=device).triu_(1)
                
                # Extract rotary embeddings for this sequence length
                freqs_cis = self.freqs_cis[:t].detach().clone()
            
            # Process through layers while maintaining gradient flow
            for i, block in enumerate(self.transformer.h):
                # Process through the block, maintaining gradient flow
                x = block(x, 0, freqs_cis, mask)
                
                # Periodically clear CUDA cache to help with memory usage
                # But don't do it too frequently as it can slow down training
                if i % 6 == 5 and i > 0:
                    torch.cuda.empty_cache()
            
            # Keep gradient flow for final normalization  
            x = self.transformer.ln_f(x)
            
            # Compute logits and loss if needed
            if targets is not None:
                # Compute logits directly with gradient flow
                logits = self.lm_head(x)
                
                if self.config.label_smoothing > 0.0:
                    # Apply label smoothing
                    num_classes = logits.size(-1)
                    smoothing = self.config.label_smoothing
                    
                    # Create smoothed labels
                    confidence = 1.0 - smoothing
                    smoothing_value = smoothing / (num_classes - 1)
                    
                    # One-hot with label smoothing
                    with torch.no_grad():
                        true_dist = torch.zeros_like(logits)
                        true_dist.fill_(smoothing_value)
                        true_dist.scatter_(
                            -1, 
                            targets.unsqueeze(-1), 
                            confidence
                        )
                    
                    # Calculate KL divergence
                    log_probs = F.log_softmax(logits.view(-1, logits.size(-1)), dim=-1)
                    loss = -(true_dist.view(-1, true_dist.size(-1)) * log_probs).sum(-1)
                    
                    # Mask padding positions
                    with torch.no_grad():
                        mask = (targets != -1).float()
                    loss = (loss * mask.view(-1)).sum() / mask.sum()
                else:
                    # Standard loss without label smoothing
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
                
            else:
                # Inference-time optimization: only compute logits for last position
                logits = self.lm_head(x[:, [-1], :])
                loss = None
        
        # Create fresh tensor outputs with no connections to the previous graph
        # Create copies that properly preserve gradient flow
        if loss is not None:
            final_loss = loss.clone()  # Keep the gradient connection
        else:
            final_loss = None
            
        final_logits = logits.clone()  # Keep the gradient connection
        
        # Clean up intermediate tensors to prevent memory leaks
        del x, logits
        if loss is not None:
            del loss
        
        # Force cleanup to prevent memory growth
        # This is crucial for the training loop memory stability
        torch.cuda.empty_cache()
        
        return final_logits, final_loss
    
    def _set_inference_mode(self, use_inference=True):
        """Set the MLA blocks to inference mode for KV caching"""
        try:
            # First explicitly set training mode correctly
            if use_inference:
                self.eval()  # Set to evaluation mode
            else:
                self.train()  # Set to training mode
                
            # Then walk through all MLA modules and set their inference mode
            for name, module in self.named_modules():
                if hasattr(module, 'set_inference_mode') and module is not self:
                    try:
                        module.set_inference_mode(use_inference)
                    except Exception as internal_e:
                        print(f"Error setting inference mode for module {name}: {internal_e}")
        except Exception as e:
            print(f"Error setting inference mode: {e}")
            # Continue without failing if there's an issue
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=None, temperature=1.0, top_k=None, prompt=None, gen_length=None):
        """
        Generate text autoregressively.
        
        Args:
            idx (torch.LongTensor, optional): Context tokens [batch_size, seq_len]
            max_new_tokens (int, optional): Number of new tokens to generate
            temperature (float): Sampling temperature
            top_k (int, optional): If set, use top-k sampling
            prompt (torch.LongTensor, optional): Alternative way to specify context tokens
            gen_length (int, optional): Alternative way to specify number of tokens to generate
            
        Returns:
            tuple: (torch.LongTensor: Generated sequence with original context, None)
        """
        # Handle compatibility with train_utils.generate_text
        if prompt is not None:
            idx = prompt
        
        # Determine how many tokens to generate
        tokens_to_generate = max_new_tokens
        if tokens_to_generate is None and gen_length is not None:
            tokens_to_generate = gen_length
        if tokens_to_generate is None:
            tokens_to_generate = 20  # Default if no length specified
        
        # Set model to evaluation mode
        self.eval()
        
        for _ in range(tokens_to_generate):
            # Ensure sequence doesn't exceed maximum length
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass to get logits
            logits, _ = self(idx_cond)
            
            # Extract logits for the last position only
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k sampling if needed
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        # Return a tuple to match the expected return signature from other models
        result = (idx, None)
        
        # Reset to training mode and clear caches after generation
        self._set_inference_mode(False)
        
        # Manually trigger garbage collection to free memory
        torch.cuda.empty_cache()
        
        return result
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, optimizer_type=None):
        """
        Configure optimizer with specialized parameter groups.
        """
        # Always use AdamW for dense MLA models (more stable)
        optimizer_type = "adamw"
            
        # For dense models, use a simpler optimizer configuration 
        # Create parameter groups with weight decay only for non-biases
        params_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Create standard parameter groups (no MoE special handling needed)
        decay_params = []
        no_decay_params = []
        
        # Categorize parameters
        for param_name, param in params_dict.items():
            # Skip if not requires grad
            if not param.requires_grad:
                continue
                
            # Standard decay/no_decay split
            is_no_decay = 'bias' in param_name or 'ln' in param_name or 'norm' in param_name
            
            if is_no_decay:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        # Create optimizer param groups
        optimizer_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        # Create optimizer directly (no special configuration needed)
        if device_type == 'cuda':
            optimizer = torch.optim.AdamW(
                optimizer_groups,
                lr=learning_rate,
                betas=betas,
                fused=True  # Use fused implementation for better performance
            )
        else:
            optimizer = torch.optim.AdamW(
                optimizer_groups,
                lr=learning_rate,
                betas=betas
            )
        
        print(f"Using AdamW optimizer for Dense MLA model")
        return optimizer

def precompute_freqs_cis(dim, max_seq_len, theta):
    """Precompute the frequency tensor for complex exponentials (RoPE)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    
    # Complex exponentials
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def precompute_freqs_cis_with_linear_scaling(dim, max_seq_len, theta, scaling_factor, original_max_seq_len):
    """Precompute rotary embeddings with linear scaling for extended context."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    
    # Scale frequencies for extended context
    if max_seq_len > original_max_seq_len:
        # Linear scaling: reduce frequencies by the scaling factor
        scale_factor = 1.0 / scaling_factor
        freqs = freqs * scale_factor
    
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    
    # Complex exponentials
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def create_mla_model(
    size: str = 'small',
    n_layer: Optional[int] = None,
    n_embd: Optional[int] = None,
    n_head: Optional[int] = None,
    vocab_size: int = 50304,
    block_size: int = 4096,
    dropout: float = 0.0,
    fp8_params: bool = True,
    **kwargs
):
    """
    Create an MLA-Model with predefined sizes.
    
    Args:
        size: Model size ('small', 'medium', 'large', 'xl')
        n_layer: Override number of layers
        n_embd: Override embedding dimension
        n_head: Override number of heads
        vocab_size: Vocabulary size
        block_size: Maximum sequence length
        dropout: Dropout probability
        fp8_params: Whether to use FP8 precision for eligible parameters
        **kwargs: Additional configuration arguments
    
    Returns:
        MLAModel: Configured model
    """
    # Define model sizes
    sizes = {
        'small': {
            'n_layer': 12,
            'n_embd': 768,
            'n_head': 12,
        },
        'medium': {
            'n_layer': 24,
            'n_embd': 1024,
            'n_head': 16,
        },
        'large': {
            'n_layer': 32,
            'n_embd': 2048,
            'n_head': 16,
        },
        'xl': {
            'n_layer': 40,
            'n_embd': 2560,
            'n_head': 20,
        }
    }
    
    # Start with the selected size
    if size not in sizes:
        raise ValueError(f"Unknown model size: {size}, valid sizes are {list(sizes.keys())}")
    
    config_dict = sizes[size].copy()
    
    # Override with explicit parameters
    if n_layer is not None:
        config_dict['n_layer'] = n_layer
    if n_embd is not None:
        config_dict['n_embd'] = n_embd
    if n_head is not None:
        config_dict['n_head'] = n_head
    
    # Add standard parameters
    config_dict.update({
        'vocab_size': vocab_size,
        'block_size': block_size,
        'dropout': dropout,
        'fp8_params': fp8_params,
        'use_moe': False,  # Explicitly set to False for dense model
    })
    
    # Add any additional parameters
    config_dict.update(kwargs)
    
    # Create config and model
    config = MLAModelConfig(**config_dict)
    model = MLAModel(config)
    
    return model