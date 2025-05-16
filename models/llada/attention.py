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
        # Ensure max_seq_len accommodates BD3 training (2 * block_size)
        self.rope = RoPE(self.head_size, max_seq_len=2 * config.block_size)
        
        # For KV caching during generation
        self.kv_cache_enabled = False
        self._cached_k = None
        self._cached_v = None

    def forward(self, x, attn_mask=None, past_key_value=None, use_kv_cache=False): # Added past_key_value
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
            # Handle KV caching for faster generation
            # Logic priority: 1. explicit past_key_value, 2. internal cache, 3. no cache
            if past_key_value is not None:
                # Use explicit past K/V passed for this block (e.g., during BD3 generation)
                past_k, past_v = past_key_value
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            elif use_kv_cache and self.kv_cache_enabled and self._cached_k is not None:
                # Use internal cache (for original LLaDA generation)
                k = torch.cat([self._cached_k, k], dim=2)
                v = torch.cat([self._cached_v, v], dim=2)
                 
            # Update internal cache if enabled (for original LLaDA generation)
            if use_kv_cache and self.kv_cache_enabled:
                 self._cached_k = k
                 self._cached_v = v
            
            # Use PyTorch's SDPA with no causal mask (KEY DIFFERENCE FROM GPT)
            # Modified to accept an optional attention mask for BD3-LM
            y = F.scaled_dot_product_attention(
                 q, k, v,
                 attn_mask=attn_mask, # Pass the provided mask
                 dropout_p=self.dropout if self.training else 0.0,
                 is_causal=False # Still explicitly not causal by default, mask handles structure
             )
            
            # Reshape back to original
            y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
            
            # Output projection
            y = self.c_proj(y)
            
            # Apply dropout last
            if self.dropout > 0 and self.training:
                y = self.resid_dropout(y)
        
        # Return output and the K/V state for this input (needed for BD3 cache update)
        # Note: k, v here might include past_key_value if it was provided.
        present_key_value = (k, v)
        return y, present_key_value