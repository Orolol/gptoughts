"""
SEDD (Score Entropy Discrete Diffusion) specific building blocks.

This module contains the core components for SEDD models including:
- DDiT blocks with adaptive layer normalization
- Timestep embedders
- Final output layers with modulation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from .normalization import RMSNorm
from .positional_encoding import apply_rope, precompute_freqs_cis


class LayerNorm(nn.Module):
    """Simple LayerNorm implementation for SEDD."""
    
    def __init__(self, dim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias, self.eps)


class SimpleAttention(nn.Module):
    """Simple multi-head attention for SEDD."""
    
    def __init__(
        self,
        dim: int,
        n_heads: int,
        bias: bool = False,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        assert dim % n_heads == 0
        
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # QKV projection
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)  # Always use Dropout, even if p=0
        
    def forward(
        self, 
        x: torch.Tensor, 
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C = x.shape
        
        # Compute QKV
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        
        # Apply rotary embeddings if provided (simplified for now)
        # if freqs_cis is not None:
        #     q = apply_rope(q, freqs_cis)
        #     k = apply_rope(k, freqs_cis)
        
        # Attention computation using PyTorch's SDPA for efficiency
        if hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's optimized attention
            y = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False  # SEDD typically doesn't use causal masking
            )
        else:
            # Fallback to manual attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if mask is not None:
                att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v
        
        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        
        return y


class SimpleMLP(nn.Module):
    """Simple MLP implementation for SEDD."""
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
        **kwargs
    ):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)  # Always use Dropout, even if p=0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")  # Use tanh approximation for speed
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer normalization modulation."""
    # shift and scale already have the right shape from [:, None] in the calling code
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations using sinusoidal embeddings.
    """
    
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            t: a 1-D Tensor of N indices, one per batch element
            dim: the dimension of the output
            max_period: controls the minimum frequency of the embeddings
            
        Returns:
            an (N, D) Tensor of positional embeddings
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of timestep embedder.
        
        Args:
            t: Timestep tensor of shape (batch_size,)
            
        Returns:
            Embedded timesteps of shape (batch_size, hidden_size)
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DDiTBlock(nn.Module):
    """
    Diffusion Transformer Block with Adaptive Layer Normalization.
    
    This is the core building block of the SEDD model, featuring:
    - Adaptive layer normalization conditioned on timestep embeddings
    - Multi-head self-attention
    - MLP with residual connections
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int,
        cond_dim: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        bias: bool = False,
        use_dyt: bool = False,
        dyt_alpha_init: float = 0.5,
        attention_backend: Optional[str] = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        
        # Normalization layers
        if use_dyt:
            from .normalization import DynamicTanhNorm
            self.norm1 = DynamicTanhNorm(dim, alpha_init=dyt_alpha_init)
            self.norm2 = DynamicTanhNorm(dim, alpha_init=dyt_alpha_init)
        else:
            self.norm1 = LayerNorm(dim, bias=bias)
            self.norm2 = LayerNorm(dim, bias=bias)
        
        # Attention
        self.attn = SimpleAttention(
            dim=dim,
            n_heads=n_heads,
            bias=bias,
            dropout=dropout
        )
        
        # MLP
        self.mlp = SimpleMLP(
            dim=dim,
            hidden_dim=mlp_ratio * dim,
            dropout=dropout,
            bias=bias
        )
        
        # Adaptive Layer Norm modulation
        # 6 parameters: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)
        
        self.dropout_layer = nn.Dropout(dropout)  # Always use Dropout, even if p=0

    def forward(
        self, 
        x: torch.Tensor, 
        c: torch.Tensor,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of DDiT block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            c: Conditioning tensor of shape (batch_size, cond_dim)
            freqs_cis: Optional rotary position embeddings
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        
        # Multi-head self-attention with adaptive layer norm
        x_skip = x
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        
        # Apply attention
        attn_out = self.attn(x_norm, freqs_cis=freqs_cis, mask=mask)
        
        # Apply dropout and gating, then residual
        x = x_skip + gate_msa * self.dropout_layer(attn_out)
        
        # MLP with adaptive layer norm
        x_skip = x
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        
        # Apply MLP
        mlp_out = self.mlp(x_norm)
        
        # Apply dropout and gating, then residual
        x = x_skip + gate_mlp * self.dropout_layer(mlp_out)
        
        return x


class DDiTFinalLayer(nn.Module):
    """
    Final output layer for SEDD model with adaptive normalization.
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        out_channels: int, 
        cond_dim: int,
        bias: bool = False,
        use_dyt: bool = False,
        dyt_alpha_init: float = 0.5
    ):
        super().__init__()
        
        # Final normalization
        if use_dyt:
            from .normalization import DynamicTanhNorm
            self.norm_final = DynamicTanhNorm(hidden_size, alpha_init=dyt_alpha_init)
        else:
            self.norm_final = LayerNorm(hidden_size, bias=bias)
        
        # Output projection
        self.linear = nn.Linear(hidden_size, out_channels, bias=bias)
        nn.init.zeros_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)
        
        # Adaptive modulation for final layer
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of final layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            c: Conditioning tensor of shape (batch_size, cond_dim)
            
        Returns:
            Output logits of shape (batch_size, seq_len, out_channels)
        """
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class EmbeddingLayer(nn.Module):
    """Token embedding layer for SEDD model."""
    
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        # Use Kaiming uniform initialization like in the original
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of embedding layer.
        
        Args:
            x: Token indices of shape (batch_size, seq_len)
            
        Returns:
            Embeddings of shape (batch_size, seq_len, dim)
        """
        return self.embedding(x)