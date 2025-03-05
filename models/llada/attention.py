"""Attention mechanism for the LLaDA model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks.positional_encoding import RoPE

class LLaDAAttention(nn.Module):
    """
    Self-attention module for LLaDA with bidirectional attention (no causal masking).
    This differs from CausalSelfAttention by removing the causal mask.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Ensure embedding dimension is divisible by number of heads
        assert config.n_embd % config.n_head == 0
        
        # Key, query, value projections - same as standard attention
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Attention params
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Head size
        self.head_size = config.n_embd // config.n_head
        
        # Add RoPE positional embeddings - reuse from blocks
        self.rope = RoPE(self.head_size, config.block_size)
        
        # For KV caching during generation
        self.kv_cache_enabled = False
        self._cached_k = None
        self._cached_v = None

    def forward(self, x, use_kv_cache=False):
        batch_size, seq_len, n_embd = x.shape
        
        # Ensure inputs are contiguous for better performance
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Use a single matrix multiply for qkv projection to maximize GPU utilization
        with torch.amp.autocast(enabled=True, device_type='cuda'):
            # Calculate query, key, values with a single operation
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)
            
            # Reshape to multiple heads
            k = k.view(batch_size, seq_len, self.n_head, self.head_size).transpose(1, 2).contiguous()
            q = q.view(batch_size, seq_len, self.n_head, self.head_size).transpose(1, 2).contiguous()
            v = v.view(batch_size, seq_len, self.n_head, self.head_size).transpose(1, 2).contiguous()
            
            # Apply RoPE to queries and keys
            q = self.rope(q, seq_len)
            k = self.rope(k, seq_len)
            
            # Free memory
            del qkv
            
            # Handle KV caching for faster generation
            if use_kv_cache and self.kv_cache_enabled:
                if self._cached_k is not None and self._cached_v is not None:
                    # Concatenate current k,v with cached k,v
                    k = torch.cat([self._cached_k, k], dim=2)
                    v = torch.cat([self._cached_v, v], dim=2)
                
                # Update the cache
                self._cached_k = k
                self._cached_v = v
            
            # Use PyTorch's SDPA with no causal mask (KEY DIFFERENCE FROM GPT)
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,  # No causal mask here (bidirectional attention)
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False  # Not causal - this is LLaDA's main difference
            )
            
            # Reshape back to original
            y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
            
            # Output projection
            y = self.c_proj(y)
            
            # Apply dropout last
            if self.dropout > 0 and self.training:
                y = self.resid_dropout(y)
        
        return y 