"""Switch-style Mixture-of-Experts layer (top-1 gating with capacity).

This module mirrors the API of the existing MoE layer but follows the Switch
Transformer recipe: each token is routed to a single expert, a per-expert
capacity enforces sparsity, and tokens beyond capacity are dropped (residual
only). The design keeps activations close to those of a dense MLP layer while
retaining sparse parameters.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP


class SwitchRouter(nn.Module):
    """Top-1 router with auxiliary losses for load balancing."""

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        router_z_loss_coef: float = 1e-2,
        load_balance_coef: float = 1e-2,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_experts, bias=False)
        self.router_z_loss_coef = router_z_loss_coef
        self.load_balance_coef = load_balance_coef

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.linear(x)
        probs = F.softmax(logits, dim=-1)
        top1_prob, top1_idx = torch.max(probs, dim=-1)

        # Auxiliary losses mirror Switch Transformer
        density = probs.mean(dim=0)
        density_proxy = F.one_hot(top1_idx, num_classes=probs.size(-1)).float().mean(dim=0)
        load_balance_loss = (density * density_proxy).sum() * probs.size(-1)
        router_z_loss = torch.mean(logits.pow(2))
        aux_loss = (
            self.load_balance_coef * load_balance_loss
            + self.router_z_loss_coef * router_z_loss
        )
        return top1_idx, top1_prob, aux_loss


class SwitchMoELayer(nn.Module):
    """Top-1 capacity-limited MoE layer compatible with existing blocks."""

    def __init__(
        self,
        config,
        num_experts: int = 8,
        capacity_factor: float = 1.25,
        drop_tokens: bool = True,
        router_z_loss_coef: float = 1e-2,
        load_balance_coef: float = 1e-2,
        use_dropout: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens

        self.norm = nn.LayerNorm(config.n_embd)
        self.router = SwitchRouter(
            input_dim=config.n_embd,
            num_experts=num_experts,
            router_z_loss_coef=router_z_loss_coef,
            load_balance_coef=load_balance_coef,
        )
        self.experts = nn.ModuleList([MLP(config) for _ in range(num_experts)])
        dropout_p = config.dropout if use_dropout is None else use_dropout
        self.dropout = nn.Dropout(dropout_p)
        self.residual_scale = nn.Parameter(torch.ones(1))

    def _compute_capacity(self, tokens: int) -> int:
        base = tokens / max(1, self.num_experts)
        cap = int(math.ceil(self.capacity_factor * base))
        return max(1, cap)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        B, S, D = x.shape
        normed = self.norm(x)
        flat_inputs = normed.reshape(-1, D)

        expert_indices, expert_gates, router_loss = self.router(flat_inputs)

        # Positions inside each expert buffer (zero-based)
        assign_mask = F.one_hot(expert_indices, num_classes=self.num_experts)
        positions = torch.cumsum(assign_mask, dim=0) - 1
        positions = (positions * assign_mask).sum(dim=-1)
        capacity = self._compute_capacity(flat_inputs.size(0))

        mask = positions < capacity
        if not self.drop_tokens:
            positions = torch.clamp(positions, min=0, max=capacity - 1)
            mask = torch.ones_like(mask, dtype=torch.bool)

        kept_idx = mask.nonzero(as_tuple=False).squeeze(-1)
        expert_output = torch.zeros_like(flat_inputs)

        kept_experts = torch.index_select(expert_indices, 0, kept_idx)
        kept_gates = torch.index_select(expert_gates, 0, kept_idx)
        inputs_kept = torch.index_select(flat_inputs, 0, kept_idx)

        for expert_id, expert in enumerate(self.experts):
            token_mask = kept_experts == expert_id
            token_positions = torch.nonzero(token_mask, as_tuple=False).squeeze(-1)
            expert_in = torch.index_select(inputs_kept, 0, token_positions)
            expert_out = expert(expert_in)
            gates = torch.index_select(kept_gates, 0, token_positions).unsqueeze(-1)
            expert_out = expert_out * gates
            dest_indices = torch.index_select(kept_idx, 0, token_positions)
            expert_output.index_copy_(0, dest_indices, expert_out)

        expert_output = expert_output.view(B, S, D)
        expert_output = self.dropout(expert_output)
        out = residual + self.residual_scale.to(residual.dtype) * expert_output.to(residual.dtype)
        return out, router_loss
