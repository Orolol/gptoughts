"""
Adaptive MoE Transformer Block
Combines adaptive MoE attention with standard FFN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from .adaptive_attention_moe import AdaptiveAttentionMoE
from .normalization import RMSNorm


class SimpleMLP(nn.Module):
    """Simple MLP with configurable activation for Adaptive MoE"""

    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 activation: str = "gelu",
                 dropout: float = 0.1,
                 bias: bool = False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout)

        # Activation function
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class AdaptiveMoEBlock(nn.Module):
    """Bloc transformer complet avec attention adaptative MoE pour LLM"""

    def __init__(self,
                 dim: int = 768,
                 heads: int = 12,
                 ff_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 router_temperature: float = 1.0,
                 k_peripheral: int = 32,
                 k_focal: int = 64,
                 k_reflective: int = 128,
                 use_bias: bool = False,
                 norm_type: str = "rmsnorm",
                 activation: str = "gelu"):
        super().__init__()

        self.dim = dim

        # Default ff_dim to 4x dim if not specified
        if ff_dim is None:
            ff_dim = dim * 4

        # Normalization layers
        if norm_type == "rmsnorm":
            self.norm1 = RMSNorm(dim)
            self.norm2 = RMSNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

        # Attention adaptative MoE
        self.attention = AdaptiveAttentionMoE(
            dim=dim,
            heads=heads,
            dropout=dropout,
            temperature=router_temperature,
            k_peripheral=k_peripheral,
            k_focal=k_focal,
            k_reflective=k_reflective,
            use_bias=use_bias
        )

        # FFN standard (pas de MoE ici - différence clé avec MoE traditionnel)
        self.ffn = SimpleMLP(
            in_features=dim,
            hidden_features=ff_dim,
            out_features=dim,
            activation=activation,
            dropout=dropout,
            bias=use_bias
        )

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x: [batch, seq_len, dim]
        attention_mask: [batch, seq_len]
        Returns: (output, routing_info)
        """
        # Attention avec connexion résiduelle
        x_norm = self.norm1(x)
        attn_out, routing_info = self.attention(x_norm, attention_mask)
        x = x + self.dropout(attn_out)

        # FFN avec connexion résiduelle
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)

        return x, routing_info