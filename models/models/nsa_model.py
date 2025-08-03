"""
NSA-Model: A transformer model using Native Sparse Attention.

This model implements the Native Sparse Attention mechanism which combines:
- Compressed attention for global context
- Blockwise selection for important fine-grained tokens
- Sliding window attention for local context
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Optional, Dict, List, Tuple, Union, Any
from dataclasses import dataclass, field

# Import components from blocks
from models.blocks.nsa_optimized import NSAConfig
from models.blocks.nsa_block import NSABlock
from models.blocks.normalization import RMSNorm, DynamicTanh
from models.blocks.tensor_utils import prevent_backward_reuse

# Import utility functions
from train.train_utils import estimate_mfu as utils_estimate_mfu


@dataclass
class NSAModelConfig:
    """Configuration for NSA-Model."""
    # Architecture
    n_layer: int = 24
    n_embd: int = 2048
    n_head: int = 16
    n_inner: Optional[int] = None  # Inner dimension for MLP. If None, will be 4*n_embd
    vocab_size: int = 50304
    block_size: int = 4096
    
    # NSA specific parameters (will be passed to NSAConfig)
    n_groups: int = 4  # For group query attention
    head_dim: int = 128
    value_dim: int = 128
    compress_block_size: int = 32
    compress_stride: int = 16
    selection_block_size: int = 64
    num_selected_blocks: int = 16
    sliding_window_size: int = 512
    
    # Precision
    use_fp8: bool = False
    use_fp4_inference: bool = False
    fp8_tile_size: int = 128
    enable_microscaling: bool = False
    
    # Dropout and regularization
    dropout: float = 0.0
    bias: bool = False
    
    # Training
    use_gradient_checkpointing: bool = True
    init_device: str = 'cuda'
    label_smoothing: float = 0.0
    
    # Dynamic Tanh (DyT) options
    use_dyt: bool = False
    dyt_alpha_init: float = 0.5
    
    def __post_init__(self):
        # Set inner dimension if not provided
        if self.n_inner is None:
            self.n_inner = 4 * self.n_embd
        
        # Compute head_dim if not set
        if self.head_dim is None:
            self.head_dim = self.n_embd // self.n_head
    
    def to_nsa_config(self) -> NSAConfig:
        """Convert to NSAConfig for attention blocks."""
        return NSAConfig(
            hidden_dim=self.n_embd,
            n_heads=self.n_head,
            n_groups=self.n_groups,
            head_dim=self.head_dim,
            value_dim=self.value_dim,
            compress_block_size=self.compress_block_size,
            compress_stride=self.compress_stride,
            selection_block_size=self.selection_block_size,
            num_selected_blocks=self.num_selected_blocks,
            sliding_window_size=self.sliding_window_size,
            use_fp8=self.use_fp8,
            use_fp4_inference=self.use_fp4_inference,
            tensor_core_block_size=self.fp8_tile_size,
            enable_microscaling=self.enable_microscaling,
            dropout=self.dropout,
            bias=self.bias
        )


class NSAModel(nn.Module):
    """
    NSA-Model: A transformer model using Native Sparse Attention.
    """
    
    def __init__(self, config: NSAModelConfig):
        super().__init__()
        self.config = config
        
        # Convert to NSA config for blocks
        self.nsa_config = config.to_nsa_config()
        
        # Add additional fields to NSA config for compatibility
        self.nsa_config.n_inner = config.n_inner
        self.nsa_config.use_dyt = config.use_dyt
        self.nsa_config.dyt_alpha_init = config.dyt_alpha_init
        self.nsa_config.use_gradient_checkpointing = config.use_gradient_checkpointing
        
        # Embeddings
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([NSABlock(self.nsa_config) for _ in range(config.n_layer)]),
            ln_f=DynamicTanh(config.n_embd, alpha_init=config.dyt_alpha_init) if config.use_dyt else RMSNorm(config.n_embd)
        ))
        
        # LM head (weight tied with embeddings)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        
        # Timing stats for MFU estimation
        self.timing_stats = None
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Parameter count
        self.param_count = sum(p.numel() for p in self.parameters())
        print(f"NSA Model - Number of parameters: {self.param_count/1e6:.2f}M")
        
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
        if seq_length is None:
            seq_length = self.config.block_size
            
        return utils_estimate_mfu(
            model=self,
            batch_size=batch_size,
            seq_length=seq_length,
            dt=dt,
            dtype=torch.bfloat16 if not self.config.use_fp8 else torch.float16
        )
    
    def set_gradient_checkpointing(self, value: bool):
        """Set gradient checkpointing for all transformer blocks."""
        self.config.use_gradient_checkpointing = value
        self.nsa_config.use_gradient_checkpointing = value
        for block in self.transformer.h:
            block.use_checkpoint = value
    
    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params
    
    def forward(self, idx, targets=None):
        """
        Forward pass of NSA model.
        
        Args:
            idx: Input token indices [batch_size, seq_len]
            targets: Target token indices for loss computation [batch_size, seq_len]
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            loss: Language modeling loss (if targets provided)
        """
        # Use the context manager to prevent backward reuse
        with prevent_backward_reuse():
            device = idx.device
            b, t = idx.size()
            assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            
            # Token embeddings
            tok_emb = self.transformer.wte(idx)
            x = self.transformer.drop(tok_emb)
            
            # Create causal mask
            mask = None
            if t > 1:
                # Create causal mask for NSA
                mask = torch.ones(b, t, dtype=torch.bool, device=device)
                # NSA handles causality internally, so we just pass a simple mask
                # indicating which positions are valid
            
            # Process through transformer layers
            for i, block in enumerate(self.transformer.h):
                x = block(x, mask=mask)
            
            # Final layer norm
            x = self.transformer.ln_f(x)
            
            # Compute logits and loss if needed
            if targets is not None:
                # Compute logits for all positions
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
                    # Standard cross-entropy loss
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        ignore_index=-1
                    )
            else:
                # Inference mode: only compute logits for last position
                logits = self.lm_head(x[:, [-1], :])
                loss = None
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=None,
        temperature=1.0,
        top_k=None,
        prompt=None,
        gen_length=None
    ):
        """
        Generate text autoregressively.
        
        Args:
            idx: Context tokens [batch_size, seq_len]
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature
            top_k: If set, use top-k sampling
            prompt: Alternative way to specify context tokens
            gen_length: Alternative way to specify number of tokens to generate
            
        Returns:
            tuple: (Generated sequence with original context, None)
        """
        # Handle compatibility
        if prompt is not None:
            idx = prompt
        
        # Determine how many tokens to generate
        tokens_to_generate = max_new_tokens
        if tokens_to_generate is None and gen_length is not None:
            tokens_to_generate = gen_length
        if tokens_to_generate is None:
            tokens_to_generate = 20  # Default
        
        # Set model to evaluation mode
        self.eval()
        
        for _ in range(tokens_to_generate):
            # Ensure sequence doesn't exceed maximum length
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass to get logits
            logits, _ = self(idx_cond)
            
            # Extract logits for the last position
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
        
        # Return tuple for compatibility
        return (idx, None)
    
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
        
        # Default to AdamW
        if optimizer_type is None:
            optimizer_type = "adamw"
            print("No optimizer type specified, defaulting to AdamW")
        
        # Use the GPT optimizer configuration
        optimizer = configure_optimizer_for_gpt(
            model=self,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
            device_type=device_type,
            optimizer_type=optimizer_type,
            **kwargs
        )
        
        print(f"Configured {optimizer_type} optimizer for NSA model")
        return optimizer


def create_nsa_model(
    size: str = 'small',
    n_layer: Optional[int] = None,
    n_embd: Optional[int] = None,
    n_head: Optional[int] = None,
    vocab_size: int = 50304,
    block_size: int = 4096,
    dropout: float = 0.0,
    use_fp8: bool = False,
    **kwargs
):
    """
    Create an NSA-Model with predefined sizes.
    
    Args:
        size: Model size ('small', 'medium', 'large', 'xl')
        n_layer: Override number of layers
        n_embd: Override embedding dimension
        n_head: Override number of heads
        vocab_size: Vocabulary size
        block_size: Maximum sequence length
        dropout: Dropout probability
        use_fp8: Whether to use FP8 precision
        **kwargs: Additional configuration arguments
    
    Returns:
        NSAModel: Configured model
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
        'use_fp8': use_fp8,
    })
    
    # NSA-specific defaults based on size
    nsa_defaults = {
        'small': {
            'compress_block_size': 16,
            'compress_stride': 8,
            'selection_block_size': 32,
            'num_selected_blocks': 8,
            'sliding_window_size': 256,
        },
        'medium': {
            'compress_block_size': 32,
            'compress_stride': 16,
            'selection_block_size': 64,
            'num_selected_blocks': 12,
            'sliding_window_size': 384,
        },
        'large': {
            'compress_block_size': 32,
            'compress_stride': 16,
            'selection_block_size': 64,
            'num_selected_blocks': 16,
            'sliding_window_size': 512,
        },
        'xl': {
            'compress_block_size': 64,
            'compress_stride': 32,
            'selection_block_size': 128,
            'num_selected_blocks': 20,
            'sliding_window_size': 640,
        }
    }
    
    # Add NSA-specific defaults
    config_dict.update(nsa_defaults.get(size, nsa_defaults['medium']))
    
    # Add any additional parameters
    config_dict.update(kwargs)
    
    # Create config and model
    config = NSAModelConfig(**config_dict)
    model = NSAModel(config)
    
    return model
