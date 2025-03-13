"""Block module for the LLaDA model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks.normalization import RMSNorm
from .attention import LLaDAAttention
from .router import LLaDARouter
from .expert import LLaDAExpertGroup

class LLaDABlock(nn.Module):
    """
    Transformer block for LLaDA that combines:
    1. Bidirectional attention (non-causal)
    2. MoE FFN with specialized routing
    
    Main difference from standard blocks: No causal masking in attention
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = LLaDAAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        
        # MoE components
        self.router = LLaDARouter(config.n_embd, config.num_experts, config.k)
        self.expert_group = LLaDAExpertGroup(config)
        
        # Router loss coefficient
        self.router_z_loss = 0.001
        
        # Gradient checkpointing flag
        self.use_checkpoint = False

    def forward(self, x, use_kv_cache=False):
        """
        Forward pass with optimized memory usage
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            use_kv_cache: Whether to use KV caching during generation
            
        Returns:
            output: Processed tensor [batch_size, seq_len, hidden_dim]
            router_loss: Auxiliary loss from router
        """
        # Make sure input is contiguous for better performance
        if not x.is_contiguous():
            x = x.contiguous()
        
        # 1. Attention block
        residual_attn = x
        x_ln1 = self.ln_1(x)
        attn_out = self.attn(x_ln1, use_kv_cache=use_kv_cache)
        x = residual_attn + attn_out
        
        # 2. MoE block
        residual_moe = x
        x_ln2 = self.ln_2(x)
        
        # Run router
        routing_weights, dispatch_mask, router_loss = self.router(x_ln2)
        
        # Run expert group
        expert_output = self.expert_group(x_ln2, dispatch_mask)
        
        # Add residual
        x = residual_moe + expert_output
        
        return x, router_loss 