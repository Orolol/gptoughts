"""Positional encoding methods for transformer models."""

import math
from typing import Optional

import torch
import torch.nn as nn

class RoPE(nn.Module):
    """
    Rotary Position Embeddings implementation.
    Based on the paper: https://arxiv.org/abs/2104.09864
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Cache cos and sin values
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Reshape for broadcasting: [1, 1, seq_len, dim]
        self.register_buffer('cos_cached', emb.cos().view(1, 1, max_seq_len, dim), persistent=False)
        self.register_buffer('sin_cached', emb.sin().view(1, 1, max_seq_len, dim), persistent=False)

    def _rotate_half(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        B, H, T, D = x.shape
        
        # Reshape avec des dimensions explicites
        x_reshaped = x.view(B, H, T, D // 2, 2)
        x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]
        
        # Get the cos and sin values for the current sequence length
        cos = self.cos_cached[:, :, :seq_len, :(D//2)]
        sin = self.sin_cached[:, :, :seq_len, :(D//2)]
        
        # Ensure broadcasting works correctly
        cos = cos.expand(B, H, -1, -1)
        sin = sin.expand(B, H, -1, -1)
        
        # Apply rotation
        rotated = torch.stack([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin,
        ], dim=-1)
        
        return rotated.view(B, H, T, D)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        if seq_len is None:
            seq_len = x.shape[-2]
        return self._rotate_half(x, seq_len)

class AlibiPositionalBias(nn.Module):
    """
    ALiBi (Attention with Linear Biases) implementation.
    Paper: https://arxiv.org/abs/2108.12409
    """
    def __init__(self, num_heads: int, max_seq_len: int):
        super().__init__()
        
        # Calculate ALiBi slopes
        def get_slopes(n_heads: int) -> torch.Tensor:
            def get_slopes_power_of_2(n_heads: int) -> list:
                start = 2 ** (-(2 ** -(math.log2(n_heads) - 3)))
                ratio = start
                return [start * (ratio ** i) for i in range(n_heads)]

            if math.log2(n_heads).is_integer():
                return torch.tensor(get_slopes_power_of_2(n_heads))
            
            closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
            base_slopes = torch.tensor(get_slopes_power_of_2(closest_power_of_2))
            
            if n_heads <= 2 * closest_power_of_2:
                slopes = torch.concat([base_slopes, base_slopes[0::2]])[:n_heads]
            else:
                slopes = torch.concat([base_slopes, base_slopes[0::2], base_slopes[1::2]])[:n_heads]
            
            return slopes

        slopes = get_slopes(num_heads)
        # Create position bias matrix
        pos = torch.arange(max_seq_len)
        rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)  # [seq_len, seq_len]
        
        # [num_heads, seq_len, seq_len]
        alibi = slopes.unsqueeze(1).unsqueeze(1) * rel_pos.unsqueeze(0)
        
        # Register as buffer since it's position-dependent but not trainable
        self.register_buffer('alibi', alibi)
        
    def get_bias(self, T: int, device: torch.device) -> torch.Tensor:
        return self.alibi.to(device)[:, :T, :T] 