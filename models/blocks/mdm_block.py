"""MDM Transformer Block with Pre-LayerNorm architecture."""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from ..llada.attention import LLaDAAttention
from .mlp import MLP
from .normalization import RMSNorm


class MDMBlock(nn.Module):
    """
    Transformer block for Masked Diffusion Model with pre-LayerNorm.
    
    Uses RMSNorm before attention and MLP layers (pre-LayerNorm architecture).
    Supports bidirectional (non-causal) attention for encoder-only architecture.
    """
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = LLaDAAttention(config)  # Already non-causal
        self.ln_2 = RMSNorm(config.n_embd) 
        self.mlp = MLP(config)
        
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through MDM block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            attention_mask: Optional attention mask
            position_ids: Optional position IDs for RoPE
            
        Returns:
            Tuple of (output tensor, attention weights if returned)
        """
        # Pre-LayerNorm attention
        residual = x
        x = self.ln_1(x)
        attn_out, _ = self.attn(
            x, 
            attn_mask=attention_mask,  # LLaDAAttention uses attn_mask
            past_key_value=None,
            use_kv_cache=False
        )
        x = residual + attn_out
        
        # Pre-LayerNorm MLP
        residual = x
        x = self.ln_2(x)
        x = residual + self.mlp(x)
        
        return x