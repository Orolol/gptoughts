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
from models.blocks.mla_fp8 import MLA_FP8
from models.blocks.mla_block import MLABlock
from models.blocks.normalization import RMSNorm, DynamicTanh
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
    use_fp8: bool = False  # Master switch for FP8 usage
    fp8_tile_size: int = 128  # Tile size for FP8 quantization
    
    # Dropout et régularisation
    dropout: float = 0.0
    bias: bool = False
    
    # Training
    use_gradient_checkpointing: bool = True
    init_device: str = 'cuda'
    label_smoothing: float = 0.0
    
    # Dynamic Tanh (DyT) options
    use_dyt: bool = False  # Whether to use Dynamic Tanh (DyT) instead of RMSNorm
    dyt_alpha_init: float = 0.5  # Initial value for DyT alpha parameter
    
    def __post_init__(self):
        # Set inner dimension if not provided
        if self.n_inner is None:
            self.n_inner = 4 * self.n_embd

# FP8Module removed - FP8 will only be used during computation, not for parameter storage
# This ensures compatibility with standard optimizers like Adam/AdamW

class MLAModelBlock(nn.Module):
    """
    MLA-Model block combining MLA attention and dense MLP feed-forward network.
    """
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        
        # Layer normalization (use DynamicTanh if configured)
        if config.use_dyt:
            self.norm1 = DynamicTanh(config.n_embd, alpha_init=config.dyt_alpha_init)
            self.norm2 = DynamicTanh(config.n_embd, alpha_init=config.dyt_alpha_init)
        else:
            self.norm1 = RMSNorm(config.n_embd)
            self.norm2 = RMSNorm(config.n_embd)
        
        # Multi-head Latent Attention
        # Use FP8 MLA if use_fp8 is enabled (regardless of fp8_mla_params which controls the linear layer type)
        if getattr(config, 'use_fp8', False):
            print("Using FP8 MLA")
            self.attn = MLA_FP8(config)
        else:
            print("Using standard MLA")
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
        
        # Handle FP8 to BFloat16/Float16 conversion for residual connection
        if attn_output.dtype in [torch.float8_e4m3fn, torch.float8_e5m2] and x.dtype != attn_output.dtype:
            attn_output = attn_output.to(x.dtype)
        
        x = x + attn_output
        
        # Apply feed-forward network
        if self.use_checkpoint and self.training:
            ffn_output = checkpoint.checkpoint(
                self._ffn_block, x,
                use_reentrant=False
            )
        else:
            ffn_output = self._ffn_block(x)
        
        # Handle FP8 to BFloat16/Float16 conversion for residual connection
        if ffn_output.dtype in [torch.float8_e4m3fn, torch.float8_e5m2] and x.dtype != ffn_output.dtype:
            ffn_output = ffn_output.to(x.dtype)
        
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
            ln_f = DynamicTanh(config.n_embd, alpha_init=config.dyt_alpha_init) if config.use_dyt else RMSNorm(config.n_embd)
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
        
        # FP8 configuration handling
        if config.fp8_params:
            print("Note: FP8 parameter storage requested but is not compatible with most optimizers.")
            print("FP8 will be used only during forward pass computations if transformer_engine is available.")
            print("Model parameters will remain in BFloat16/Float32 for optimizer compatibility.")
            # Set flag to use FP8 in forward pass but not for parameter storage
            self.use_fp8_compute = True
            # Disable fp8_params to prevent parameter conversion
            config.fp8_params = False
        else:
            self.use_fp8_compute = False
        
        # Parameter count
        self.param_count = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {self.param_count/1e6:.2f}M")
    
    
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
        # Always ensure we're in training mode when doing a forward pass during training
        # This prevents KV caching which causes memory leaks
        if self.training:
            self._set_inference_mode(False)
        
        # Don't isolate inputs - we need gradient flow!
        # The previous implementation was breaking gradient computation
        
        # Use the context manager to prevent backward reuse
        with prevent_backward_reuse():
            device = idx.device
            b, t = idx.size()
            assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            
            # Generate position indices
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            
            # Forward the MLA model
            tok_emb = self.transformer.wte(idx)
            x = self.transformer.drop(tok_emb)
            
            # Prepare attention mask if needed (for causal attention)
            mask = None
            if t > 1:
                # Create mask without requiring gradients
                mask = torch.full((t, t), float("-inf"), device=device, requires_grad=False).triu_(1)
            
            # Extract rotary embeddings for this sequence length (no gradient needed)
            freqs_cis = self.freqs_cis[:t].detach()
            
            # Process through layers while maintaining gradient flow
            for i, block in enumerate(self.transformer.h):
                # Process through the block, maintaining gradient flow
                x = block(x, 0, freqs_cis, mask)
                
                # REMOVED: torch.cuda.empty_cache() calls during forward pass
                # These were causing VRAM fluctuations and performance issues
            
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
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
                
            else:
                # Inference-time optimization: only compute logits for last position
                logits = self.lm_head(x[:, [-1], :])
                loss = None
        
        # Return outputs directly - no need to clone as we're not breaking gradient flow
        return logits, loss
    
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
    
    def clear_cache(self):
        """Clear all KV caches in MLA attention layers to free memory"""
        for name, module in self.named_modules():
            if isinstance(module, MLA):
                # Force clear any existing caches
                module.set_inference_mode(False)
                # Ensure cache attributes exist but are None
                module.k_cache = None
                module.v_cache = None
                module.kv_cache = None
                module.pe_cache = None
    
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
        
        # Reset to training mode after generation
        self._set_inference_mode(False)
        
        # Let PyTorch's memory allocator handle memory management
        # Removed manual torch.cuda.empty_cache() call
        
        return result
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, optimizer_type=None, **kwargs):
        """
        Configure optimizer with specialized parameter groups.
        
        Args:
            weight_decay: Weight decay coefficient
            learning_rate: Learning rate
            betas: Adam beta parameters
            device_type: Device type ('cuda' or 'cpu')
            optimizer_type: Type of optimizer to use
            **kwargs: Additional optimizer-specific configuration
        """
        # Import optimizer configuration utilities
        from models.optimizers import configure_optimizer_for_gpt
        
        # Default to AdamW if no optimizer type specified
        if optimizer_type is None:
            optimizer_type = "adamw"
            print("No optimizer type specified, defaulting to AdamW")
        
        # Extract GaLore configuration from kwargs if present
        galore_config = None
        galore_quantize_proj = None
        if optimizer_type in ["galore", "galore-8bit", "galore2"]:
            galore_config = {
                "rank": kwargs.get("galore_rank", 128),
                "update_proj_gap": kwargs.get("galore_update_proj_gap", 200),
                "scale": kwargs.get("galore_scale", 0.25),
                "proj_type": kwargs.get("galore_proj_type", "std")
            }
            if optimizer_type == "galore2":
                galore_quantize_proj = kwargs.get("galore_quantize_proj", None)
        
        # Use the GPT optimizer configuration which supports multiple optimizers
        # MLA models work well with the same optimizer configurations as GPT models
        optimizer = configure_optimizer_for_gpt(
            model=self,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
            device_type=device_type,
            optimizer_type=optimizer_type,
            galore_config=galore_config,
            galore_quantize_proj=galore_quantize_proj
        )
        
        print(f"Configured {optimizer_type} optimizer for MLA model")
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