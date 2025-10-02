"""
HRM-Model: Hierarchical Reasoning Model implemented to match the project model API.

This module mirrors the structure and training interface of other models like
`MLAModel` and `NSAModel` so it can be trained via `train/lightning_module.py`.
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse existing project utilities
from models.blocks.normalization import RMSNorm
from models.blocks.mla import MLA
from types import SimpleNamespace
from models.blocks.tensor_utils import prevent_backward_reuse


@dataclass
class HRMModelConfig:
    """Configuration for HRM-Model.

    Notes:
    - Uses learned token and positional embeddings
    - Two-level hierarchy (High- and Low-level modules)
    - Adaptive Computation Time over segments
    """

    # Tokenizer/sequence
    vocab_size: int = 50304
    block_size: int = 4096

    # Hidden dimensions
    d_model: int = 1024
    n_heads: int = 16
    d_ff: Optional[int] = None

    # Temporal hierarchy
    max_segments: int = 6
    cycles_per_segment: int = 2  # N
    steps_per_cycle: int = 3     # T

    # Regularization
    dropout: float = 0.0
    bias: bool = False

    # Training/runtime
    use_gradient_checkpointing: bool = True
    label_smoothing: float = 0.0
    ponder_loss_weight: float = 0.01
    halt_bias_init: float = -2.0
    
    # DEQ-style 1-step gradient within a segment
    deq_one_step: bool = False
    
    # Deep supervision across segments
    use_deep_supervision: bool = False
    n_supervision_segments: int = 1

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_model, d_ff, bias=bias)
        self.w3 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        value = self.w2(x)
        hidden = F.silu(gate) * value
        out = self.w3(hidden)
        return self.dropout(out)


class HRMBlock(nn.Module):
    """Transformer-like block with RMSNorm + MLA + SwiGLU (post-norm style)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, bias: bool, max_seq_len: int = 4096):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        # Minimal config shim for MLA
        mla_cfg = SimpleNamespace(
            n_embd=d_model,
            n_head=n_heads,
            dropout=dropout,
            bias=bias,
            max_seq_len=max_seq_len,
            attn_impl="absorb",
        )
        self.attn = MLA(mla_cfg)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLUFFN(d_model, d_ff, dropout=dropout, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention sublayer (use MLA)
        attn_in = self.norm1(x)
        attn_out = self.attn(attn_in, start_pos=0, freqs_cis=None, mask=attn_mask)
        x = x + self.dropout(attn_out)

        # FFN sublayer
        ffn_in = self.norm2(x)
        ffn_out = self.mlp(ffn_in)
        x = x + self.dropout(ffn_out)

        return x


class LowLevelModule(nn.Module):
    def __init__(self, config: HRMModelConfig):
        super().__init__()
        self.block = HRMBlock(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            bias=config.bias,
            max_seq_len=config.block_size,
        )

    def forward(
        self,
        z_L: torch.Tensor,
        z_H: torch.Tensor,
        x_tilde: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z_input = z_L + z_H + x_tilde
        return self.block(z_input, attn_mask=attn_mask, key_padding_mask=key_padding_mask)


class HighLevelModule(nn.Module):
    def __init__(self, config: HRMModelConfig):
        super().__init__()
        self.block = HRMBlock(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            bias=config.bias,
            max_seq_len=config.block_size,
        )

    def forward(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z_input = z_H + z_L
        return self.block(z_input, attn_mask=attn_mask, key_padding_mask=key_padding_mask)


class HRMInner(nn.Module):
    def __init__(self, config: HRMModelConfig):
        super().__init__()
        self.config = config
        self.L_module = LowLevelModule(config)
        self.H_module = HighLevelModule(config)

    def forward(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        x_tilde: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        one_step_last_only: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, list]]:
        # Run one full segment consisting of N cycles and T low-level steps per cycle
        seq_len = x_tilde.size(1)
        convergence_metrics = {"l_residuals": [], "h_residuals": []}

        for cycle in range(self.config.cycles_per_segment):
            for step in range(self.config.steps_per_cycle):
                is_last_step = (cycle == self.config.cyles_per_segment - 1) if hasattr(self.config, 'cyles_per_segment') else False
                is_last_step = (cycle == self.config.cycles_per_segment - 1 and step == self.config.steps_per_cycle - 1)

                # Low-level update
                if one_step_last_only and not is_last_step:
                    with torch.no_grad():
                        z_L_next = self.L_module(z_L, z_H, x_tilde, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
                else:
                    z_L_next = self.L_module(z_L, z_H, x_tilde, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
                l_residual = (z_L_next - z_L).norm(dim=-1).mean()
                convergence_metrics["l_residuals"].append(l_residual)
                z_L = z_L_next

                # High-level update at end of cycle
                if step == self.config.steps_per_cycle - 1:
                    if one_step_last_only and not is_last_step:
                        with torch.no_grad():
                            z_H_next = self.H_module(z_H, z_L, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
                    else:
                        z_H_next = self.H_module(z_H, z_L, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
                    h_residual = (z_H_next - z_H).norm(dim=-1).mean()
                    convergence_metrics["h_residuals"].append(h_residual)
                    z_H = z_H_next

        return z_H, z_L, convergence_metrics


class HRMModel(nn.Module):
    """Hierarchical Reasoning Model compatible with project training API."""

    def __init__(self, config: HRMModelConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.block_size, config.d_model)
        self.register_buffer("pos_ids", torch.arange(config.block_size).unsqueeze(0), persistent=False)
        self.drop = nn.Dropout(config.dropout)

        # Hierarchical inner model
        self.inner = HRMInner(config)

        # Output heads
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        # Halting head for ACT
        self.halt_head = nn.Sequential(nn.Linear(config.d_model, 1), nn.Sigmoid())
        with torch.no_grad():
            self.halt_head[0].bias.fill_(config.halt_bias_init)

        # Initialize weights
        self.apply(self._init_weights)

        # Parameter count logging
        param_count = sum(p.numel() for p in self.parameters())
        print(f"HRM Model - Number of parameters: {param_count/1e6:.2f}M")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            # LeCun truncated normal initialization
            fan_in = module.weight.shape[1]
            std = 1.0 / math.sqrt(fan_in)
            nn.init.trunc_normal_(module.weight, std=std, a=-2 * std, b=2 * std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)

    def _compute_embeddings(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.size()
        tok = self.wte(idx)
        pos = self.wpe(self.pos_ids[:, :t])
        return self.drop(tok + pos)

    def _compute_halting_probability(self, z_H: torch.Tensor, step_idx: int) -> torch.Tensor:
        p_halt = self.halt_head(z_H).squeeze(-1)
        eps = 1e-6
        p_halt = p_halt.clamp(eps, 1 - eps)
        if step_idx >= self.config.max_segments - 1:
            p_halt = torch.ones_like(p_halt)
        return p_halt

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        with prevent_backward_reuse():
            device = idx.device
            b, t = idx.size()
            assert t <= self.config.block_size, (
                f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            )

            x_emb = self._compute_embeddings(idx)

            # Initialize hidden states
            z_L = torch.zeros_like(x_emb)
            z_H = torch.zeros_like(x_emb)

            # Attention masks
            key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
            causal_mask = None
            if t > 1:
                causal_mask = torch.triu(
                    torch.ones(t, t, device=device, dtype=torch.bool), diagonal=1
                )

            # Deep Supervision Branch
            if getattr(self.config, 'use_deep_supervision', False):
                total_loss = None
                last_logits = None
                n_seg = max(1, int(getattr(self.config, 'n_supervision_segments', 1)))
                for seg_idx in range(n_seg):
                    # Run one segment; use 1-step gradient if enabled
                    z_H, z_L, _ = self.inner(
                        z_H, z_L, x_emb,
                        attn_mask=causal_mask,
                        key_padding_mask=key_padding_mask,
                        one_step_last_only=getattr(self.config, 'deq_one_step', False),
                    )
                    # Per-segment logits via normalized z_H
                    seg_logits = self.lm_head(self.ln_f(z_H))
                    last_logits = seg_logits
                    if targets is not None:
                        shift_logits = seg_logits[:, :-1, :].contiguous()
                        shift_labels = targets[:, 1:].contiguous()
                        if self.config.label_smoothing > 0.0:
                            num_classes = shift_logits.size(-1)
                            smoothing = self.config.label_smoothing
                            confidence = 1.0 - smoothing
                            smoothing_value = smoothing / (num_classes - 1)
                            with torch.no_grad():
                                true_dist = torch.zeros_like(shift_logits)
                                true_dist.fill_(smoothing_value)
                                true_dist.scatter_(-1, shift_labels.unsqueeze(-1), confidence)
                            log_probs = F.log_softmax(shift_logits.view(-1, num_classes), dim=-1)
                            seg_loss = -(true_dist.view(-1, num_classes) * log_probs).sum(-1)
                            with torch.no_grad():
                                mask = (shift_labels != -100).float()
                            seg_loss = (seg_loss * mask.view(-1)).sum() / mask.sum()
                        else:
                            seg_loss = F.cross_entropy(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1),
                                ignore_index=-100,
                            )
                        total_loss = seg_loss if total_loss is None else (total_loss + seg_loss)
                    # Detach states between segments
                    z_H = z_H.detach()
                    z_L = z_L.detach()

                if targets is not None:
                    # Average across segments
                    total_loss = total_loss / n_seg
                return last_logits, total_loss

            # Default ACT Branch (weighted aggregation)
            remainders = torch.ones(b, t, device=device)
            accumulated_z_H = torch.zeros_like(z_H)
            n_updates = torch.zeros(b, t, device=device)

            for seg_idx in range(self.config.max_segments):
                z_H, z_L, _ = self.inner(
                    z_H, z_L, x_emb,
                    attn_mask=causal_mask,
                    key_padding_mask=key_padding_mask,
                    one_step_last_only=getattr(self.config, 'deq_one_step', False),
                )

                p_halt = self._compute_halting_probability(z_H, seg_idx)
                is_last = seg_idx == self.config.max_segments - 1
                contrib = remainders if is_last else (remainders * p_halt)
                accumulated_z_H = accumulated_z_H + contrib.unsqueeze(-1) * z_H

                if not is_last:
                    remainders = remainders * (1.0 - p_halt)
                    n_updates = n_updates + remainders

                if torch.all(remainders < 1e-6):
                    break

            # Output head
            x = self.ln_f(accumulated_z_H)
            logits = self.lm_head(x)

            loss = None
            if targets is not None:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = targets[:, 1:].contiguous()

                if self.config.label_smoothing > 0.0:
                    num_classes = shift_logits.size(-1)
                    smoothing = self.config.label_smoothing
                    confidence = 1.0 - smoothing
                    smoothing_value = smoothing / (num_classes - 1)
                    with torch.no_grad():
                        true_dist = torch.zeros_like(shift_logits)
                        true_dist.fill_(smoothing_value)
                        true_dist.scatter_(-1, shift_labels.unsqueeze(-1), confidence)
                    log_probs = F.log_softmax(shift_logits.view(-1, num_classes), dim=-1)
                    loss = -(true_dist.view(-1, num_classes) * log_probs).sum(-1)
                    with torch.no_grad():
                        mask = (shift_labels != -100).float()
                    loss = (loss * mask.view(-1)).sum() / mask.sum()
                else:
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )

                ponder_loss = torch.mean(n_updates) * self.config.ponder_loss_weight
                loss = loss + ponder_loss

            return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        prompt: Optional[torch.Tensor] = None,
        gen_length: Optional[int] = None,
    ):
        # Support both signatures: idx=... or prompt=...
        if idx is None and prompt is not None:
            idx = prompt
        if idx is None:
            raise ValueError("HRMModel.generate requires either idx or prompt to be provided")

        tokens_to_generate = max_new_tokens if max_new_tokens is not None else gen_length
        if tokens_to_generate is None:
            tokens_to_generate = 20

        self.eval()
        for _ in range(tokens_to_generate):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return (idx, None)

    def configure_optimizers(
        self,
        weight_decay,
        learning_rate,
        betas,
        device_type,
        optimizer_type: Optional[str] = None,
        **kwargs,
    ):
        from models.optimizers import configure_optimizer_for_gpt

        if optimizer_type is None:
            optimizer_type = "adamw"
            print("No optimizer type specified, defaulting to AdamW")

        optimizer = configure_optimizer_for_gpt(
            model=self,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
            device_type=device_type,
            optimizer_type=optimizer_type,
            **kwargs,
        )
        print(f"Configured {optimizer_type} optimizer for HRM model")
        return optimizer

