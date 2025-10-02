"""HSEBlock: Orchestrator attention (standard) + Expert MoE.

This block composes the standard CausalSelfAttention (from attention.py)
with a MoE feed-forward stage controlled by a top-k router (k=2 by default),
matching the HSE design (Experts with top-2 gating + load-balance loss),
without NSA-specific logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .attention import CausalSelfAttention
from .normalization import RMSNorm, DynamicTanh
from .moe import MoELayer
from .moe_switch import SwitchMoELayer


@dataclass
class HSEBlockConfig:
    # Attention parameters (see CausalSelfAttention)
    n_embd: int
    n_head: int
    block_size: int
    dropout: float = 0.0
    bias: bool = False
    ratio_kv: int = 8
    attention_backend: Optional[str] = None
    # MoE parameters
    num_experts: int = 8
    experts_per_token: int = 2
    moe_type: str = "standard"  # "standard" | "switch"
    moe_capacity_factor: float = 1.25
    moe_drop_tokens: bool = True
    moe_router_z_loss: float = 1e-2
    moe_load_balance_loss: float = 1e-2
    # Aux
    use_gradient_checkpointing: bool = False
    use_dyt: bool = False
    dyt_alpha_init: float = 0.5


class HSEBlock(nn.Module):
    def __init__(self, config: HSEBlockConfig):
        super().__init__()
        self.config = config

        norm_cls = DynamicTanh if getattr(config, 'use_dyt', False) else RMSNorm
        norm_kwargs = {'alpha_init': getattr(config, 'dyt_alpha_init', 0.5)} if norm_cls == DynamicTanh else {}

        self.norm1 = norm_cls(config.n_embd, **norm_kwargs)
        self.norm2 = norm_cls(config.n_embd, **norm_kwargs)

        attn_cfg = type('AttnConfig', (), {
            'n_embd': config.n_embd,
            'n_head': config.n_head,
            'ratio_kv': config.ratio_kv,
            'bias': config.bias,
            'dropout': config.dropout,
            'block_size': config.block_size,
            'attention_backend': config.attention_backend,
        })()
        self.attn = CausalSelfAttention(attn_cfg)

        moe_cfg = type(
            'MoEConfig',
            (),
            {
                'n_embd': config.n_embd,
                'bias': config.bias,
                'dropout': config.dropout,
            },
        )()
        if getattr(config, "moe_type", "standard") == "switch":
            self.moe = SwitchMoELayer(
                moe_cfg,
                num_experts=config.num_experts,
                capacity_factor=config.moe_capacity_factor,
                drop_tokens=config.moe_drop_tokens,
                router_z_loss_coef=config.moe_router_z_loss,
                load_balance_coef=config.moe_load_balance_loss,
            )
            self._moe_requires_prenorm = False
        else:
            self.moe = MoELayer(
                moe_cfg,
                num_experts=config.num_experts,
                k=config.experts_per_token,
            )
            self._moe_requires_prenorm = True

        self.use_checkpoint = getattr(config, 'use_gradient_checkpointing', False)

    def _attn_block(self, x: torch.Tensor, rope: nn.Module, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # rope/mask are unused in standard CausalSelfAttention; kept for API compatibility
        return self.attn(self.norm1(x))

    def _moe_block(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "_moe_requires_prenorm", True):
            return self.moe(self.norm2(x))
        return self.moe(x)

    def forward(self, x: torch.Tensor, rope: nn.Module, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Attention + residual (disable checkpointing under torch.compile capture)
        use_cp = self.use_checkpoint and self.training
        try:
            import torch._dynamo as _dynamo  # type: ignore
            if getattr(_dynamo, 'is_compiling', None) and _dynamo.is_compiling():
                use_cp = False
        except Exception:
            pass
        if use_cp:
            attn_out = checkpoint.checkpoint(self._attn_block, x, rope, mask, use_reentrant=False)
        else:
            attn_out = self._attn_block(x, rope, mask)

        if attn_out.dtype != x.dtype:
            attn_out = attn_out.to(x.dtype)
        x = x + attn_out

        # MoE + residual and router loss (same compile-aware handling)
        if use_cp:
            moe_out, router_loss = checkpoint.checkpoint(self._moe_block, x, use_reentrant=False)
        else:
            moe_out, router_loss = self._moe_block(x)

        if moe_out.dtype != x.dtype:
            moe_out = moe_out.to(x.dtype)
        x = x + moe_out

        return x, router_loss
