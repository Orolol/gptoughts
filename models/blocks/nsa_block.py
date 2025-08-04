"""NSA Block combining Native Sparse Attention with feed-forward network."""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from typing import Optional

from .nsa_optimized import NSAAttention, NSAConfig
from .mlp import MLP
from .normalization import RMSNorm, DynamicTanh


class NSABlock(nn.Module):
    """
    NSA Block combining Native Sparse Attention and feed-forward network.
    
    This block follows the standard transformer architecture:
    1. Layer norm + attention + residual
    2. Layer norm + feed-forward + residual
    """
    
    def __init__(self, config: NSAConfig):
        super().__init__()
        self.config = config
        
        # Layer normalization
        norm_cls = DynamicTanh if getattr(config, 'use_dyt', False) else RMSNorm
        norm_kwargs = {'alpha_init': getattr(config, 'dyt_alpha_init', 0.5)} if norm_cls == DynamicTanh else {}
        
        self.norm1 = norm_cls(config.hidden_dim, **norm_kwargs)
        self.norm2 = norm_cls(config.hidden_dim, **norm_kwargs)
        
        # Native Sparse Attention from our optimized file
        self.attn = NSAAttention(config)
        
        # Feed-forward network
        # Create a compatible config for MLP if needed
        mlp_config = type('MLPConfig', (), {
            'n_embd': config.hidden_dim,
            'n_inner': getattr(config, 'n_inner', 4 * config.hidden_dim),
            'dropout': config.dropout,
            'bias': config.bias,
            'use_fp8': config.use_fp8,
            'fp8_tile_size': getattr(config, 'fp8_tile_size', 128)
        })()
        
        self.ffn = MLP(mlp_config)
        
        # Gradient checkpointing
        self.use_checkpoint = getattr(config, 'use_gradient_checkpointing', False)
        
    def _attn_block(
        self, 
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Attention sub-block with normalization."""
        return self.attn(self.norm1(x), freqs_cis, mask)
    
    def _ffn_block(self, x: torch.Tensor) -> torch.Tensor:
        """Feed-forward sub-block with normalization."""
        return self.ffn(self.norm2(x))
    
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs  # For compatibility with other blocks that might pass additional args
    ) -> torch.Tensor:
        """
        Forward pass of NSA block.
        
        Args:
            x: Input tensor [B, T, D]
            freqs_cis: Precomputed RoPE frequencies
            mask: Optional attention mask
            **kwargs: Additional arguments for compatibility
            
        Returns:
            Output tensor [B, T, D]
        """
        # Attention with residual
        if self.use_checkpoint and self.training:
            attn_output = checkpoint.checkpoint(
                self._attn_block, x, freqs_cis, mask,
                use_reentrant=False
            )
        else:
            attn_output = self._attn_block(x, freqs_cis, mask)
        
        # Handle potential dtype conversion for residual
        if attn_output.dtype != x.dtype:
            attn_output = attn_output.to(x.dtype)
        
        x = x + attn_output
        
        # Feed-forward with residual
        if self.use_checkpoint and self.training:
            ffn_output = checkpoint.checkpoint(
                self._ffn_block, x,
                use_reentrant=False
            )
        else:
            ffn_output = self._ffn_block(x)
    
        # Handle potential dtype conversion for residual
        if ffn_output.dtype != x.dtype:
            ffn_output = ffn_output.to(x.dtype)
        
        x = x + ffn_output
        
        return x
