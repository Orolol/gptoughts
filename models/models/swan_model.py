"""Implementation of SWAN-GPT architecture (arXiv:2504.08719)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from models.blocks.normalization import RMSNorm
from models.blocks.attention import CausalSelfAttention
from models.blocks.mlp import MLP
from models.optimizers import configure_optimizer_for_gpt


@dataclass
class SWANLayerConfig:
    """Per-layer configuration forwarded to attention/MLP blocks."""

    n_embd: int
    n_head: int
    block_size: int
    dropout: float
    bias: bool
    ratio_kv: int
    attention_backend: Optional[str]
    use_rope: bool
    attention_window: Optional[int]
    rope_theta: float
    logit_scale_base: Optional[float]
    logit_scale_window: int
    logit_scale_offset: int
    logit_scale_min: float
    logit_scale_max: Optional[float]
    logit_scale_during_training: bool
    use_gradient_checkpointing: bool


class SWANBlock(nn.Module):
    """Transformer block tailored for SWAN layering."""

    def __init__(self, config: SWANLayerConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)
        self.use_checkpoint = config.use_gradient_checkpointing

    def _attn_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(self.ln_1(x))

    def _mlp_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.ln_2(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            attn_out = checkpoint.checkpoint(self._attn_block, x, use_reentrant=False)
            x = x + attn_out
            mlp_out = checkpoint.checkpoint(self._mlp_block, x, use_reentrant=False)
            x = x + mlp_out
            return x

        x = x + self._attn_block(x)
        x = x + self._mlp_block(x)
        return x


@dataclass
class SWANConfig:
    """Top-level configuration for SWAN-GPT models."""

    vocab_size: int = 50304
    block_size: int = 4096
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1536
    dropout: float = 0.0
    bias: bool = False
    ratio_kv: int = 1
    attention_backend: Optional[str] = None
    use_gradient_checkpointing: bool = True
    global_layers_per_cycle: int = 1
    local_layers_per_cycle: int = 3
    swa_window: int = 512
    rope_theta: float = 10000.0
    logit_scale_base: float = 128.0
    logit_scale_window: int = 128
    logit_scale_offset: int = 0
    logit_scale_min: float = 1.0
    logit_scale_max: Optional[float] = None
    apply_logit_scale_during_training: bool = False
    label_smoothing: float = 0.0

    def __post_init__(self) -> None:
        if self.global_layers_per_cycle < 1 or self.local_layers_per_cycle < 1:
            raise ValueError("SWAN requires at least one global and one local layer per cycle.")


class SWANModel(nn.Module):
    """Implementation of SWAN-GPT hybrid attention architecture."""

    def __init__(self, config: SWANConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(),
                ln_f=RMSNorm(config.n_embd),
            )
        )

        global_cfg = SWANLayerConfig(
            n_embd=config.n_embd,
            n_head=config.n_head,
            block_size=config.block_size,
            dropout=config.dropout,
            bias=config.bias,
            ratio_kv=config.ratio_kv,
            attention_backend=config.attention_backend,
            use_rope=False,
            attention_window=None,
            rope_theta=config.rope_theta,
            logit_scale_base=config.logit_scale_base,
            logit_scale_window=config.logit_scale_window,
            logit_scale_offset=config.logit_scale_offset,
            logit_scale_min=config.logit_scale_min,
            logit_scale_max=config.logit_scale_max,
            logit_scale_during_training=config.apply_logit_scale_during_training,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
        )

        local_cfg = SWANLayerConfig(
            n_embd=config.n_embd,
            n_head=config.n_head,
            block_size=config.block_size,
            dropout=config.dropout,
            bias=config.bias,
            ratio_kv=config.ratio_kv,
            attention_backend=config.attention_backend,
            use_rope=True,
            attention_window=config.swa_window,
            rope_theta=config.rope_theta,
            logit_scale_base=None,
            logit_scale_window=config.logit_scale_window,
            logit_scale_offset=config.logit_scale_offset,
            logit_scale_min=config.logit_scale_min,
            logit_scale_max=config.logit_scale_max,
            logit_scale_during_training=False,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
        )

        cycle_len = config.global_layers_per_cycle + config.local_layers_per_cycle
        global_per_cycle = config.global_layers_per_cycle

        for layer_idx in range(config.n_layer):
            cycle_pos = layer_idx % cycle_len
            layer_cfg = global_cfg if cycle_pos < global_per_cycle else local_cfg
            # Clone configuration to avoid shared state where we mutate inside attention
            cfg_dict: Dict[str, object] = layer_cfg.__dict__.copy()  # type: ignore[arg-type]
            block = SWANBlock(SWANLayerConfig(**cfg_dict))
            block.use_checkpoint = config.use_gradient_checkpointing
            self.transformer.h.append(block)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)
        self.param_count = sum(p.numel() for p in self.parameters())
        print(f"SWAN Model - Number of parameters: {self.param_count / 1e6:.2f}M")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        device = idx.device
        b, t = idx.size()
        if t > self.config.block_size:
            raise ValueError(f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}")

        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            if self.config.label_smoothing > 0.0:
                num_classes = logits.size(-1)
                smoothing = self.config.label_smoothing
                confidence = 1.0 - smoothing
                smoothing_value = smoothing / (num_classes - 1)
                with torch.no_grad():
                    true_dist = torch.zeros_like(logits)
                    true_dist.fill_(smoothing_value)
                    true_dist.scatter_(-1, targets.unsqueeze(-1), confidence)
                log_probs = F.log_softmax(logits.view(-1, num_classes), dim=-1)
                loss = -(true_dist.view(-1, num_classes) * log_probs).sum(-1)
                with torch.no_grad():
                    mask = (targets != -100).float()
                loss = (loss * mask.view(-1)).sum() / (mask.sum() + 1e-6)
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        prompt=None,
        gen_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if idx is None and prompt is not None:
            idx = prompt
        if idx is None:
            raise TypeError("SWANModel.generate requires either 'idx' or 'prompt'.")

        tokens_to_generate = max_new_tokens if max_new_tokens is not None else (gen_length if gen_length is not None else 20)
        self.eval()
        for _ in range(tokens_to_generate):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx, None

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas,
        device_type: str,
        optimizer_type: Optional[str] = None,
        **kwargs,
    ):
        optimizer = configure_optimizer_for_gpt(
            model=self,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
            device_type=device_type,
            optimizer_type=optimizer_type or "adamw",
            **kwargs,
        )
        print("Configured optimizer for SWAN model")
        return optimizer


def create_swan_model(
    size: str = 'base',
    vocab_size: int = 50304,
    block_size: int = 4096,
    dropout: float = 0.0,
    **kwargs,
) -> SWANModel:
    size = size.lower()
    presets = {
        'small': dict(n_layer=16, n_embd=1024, n_head=16),
        'base': dict(n_layer=24, n_embd=1536, n_head=16),
        '1b': dict(n_layer=24, n_embd=1536, n_head=16),
        'large': dict(n_layer=28, n_embd=2048, n_head=24),
        'xl': dict(n_layer=32, n_embd=4096, n_head=32),
        '8b': dict(n_layer=32, n_embd=4096, n_head=32),
    }
    if size not in presets:
        raise ValueError(f"Unknown SWAN model size: {size}")

    cfg_kwargs = presets[size].copy()
    cfg_kwargs.update(
        dict(
            vocab_size=vocab_size,
            block_size=block_size,
            dropout=dropout,
        )
    )
    cfg_kwargs.update(kwargs)
    config = SWANConfig(**cfg_kwargs)
    return SWANModel(config)
