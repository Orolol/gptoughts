"""HSE Model: Hierarchical Sparse Experts (PoC implementation)

This model follows the framework used in the repo (cf. NSA model), integrates:
- Scribe local encoder producing summaries/evidence (not used in loss yet)
- Orchestrator attention using optimized NSA
- Expert MoE with top-2 routing and load-balance loss
- QAP controller stub with budgeting (no real async IO)

The forward API is consistent with other models: returns (logits, loss, router_loss).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from models.blocks.positional_encoding import RoPE
from models.blocks.normalization import RMSNorm, DynamicTanh
from models.blocks.hse_block import HSEBlock, HSEBlockConfig
from models.blocks.hse_scribe import HSEScribe, HSEScribeConfig
from models.blocks.hse_qap import QAPController
from models.blocks.hse_types import QAPBudget
from models.blocks.hse_cache import HierarchicalCache
from models.blocks.tensor_utils import prevent_backward_reuse
from models.optimizers import configure_optimizer_for_gpt


@dataclass
class HSEConfig:
    # Architecture
    n_layer: int = 24
    n_embd: int = 1024
    n_head: int = 16
    vocab_size: int = 50304
    block_size: int = 4096
    n_inner: Optional[int] = None

    # NSA params (orchestrator)
    n_groups: int = 4
    head_dim: Optional[int] = None
    value_dim: int = 128
    qk_rope_head_dim: int = 64
    compress_block_size: int = 32
    compress_stride: int = 16
    selection_block_size: int = 64
    num_selected_blocks: int = 16
    sliding_window_size: int = 512
    rope_theta: float = 10000.0
    original_max_seq_len: int = 4096
    # Standard attention extras
    ratio_kv: int = 8
    attention_backend: Optional[str] = None

    # Experts (MoE)
    num_experts: int = 8
    experts_per_token: int = 2

    # Scribe
    scribe_chunk_size: int = 2048
    scribe_summary_len: int = 128

    # QAP budgets
    qap_per_step: int = 12
    qap_per_expert: int = 6
    qap_max_queries: int = 20

    # Training / precision
    dropout: float = 0.0
    bias: bool = False
    use_gradient_checkpointing: bool = True
    use_fp8: bool = False
    fp8_tile_size: int = 128
    use_dyt: bool = False
    dyt_alpha_init: float = 0.5
    label_smoothing: float = 0.0

    def __post_init__(self):
        if self.n_inner is None:
            self.n_inner = 4 * self.n_embd
        if self.head_dim is None:
            self.head_dim = self.n_embd // self.n_head

    # NSA config conversion removed; standard attention is used in HSEBlock


class HSEModel(nn.Module):
    def __init__(self, config: HSEConfig):
        super().__init__()
        self.config = config

        # Embedding + blocks
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([
                    HSEBlock(HSEBlockConfig(
                        n_embd=config.n_embd,
                        n_head=config.n_head,
                        block_size=config.block_size,
                        dropout=config.dropout,
                        bias=config.bias,
                        ratio_kv=config.ratio_kv,
                        attention_backend=config.attention_backend,
                        num_experts=config.num_experts,
                        experts_per_token=config.experts_per_token,
                        use_gradient_checkpointing=config.use_gradient_checkpointing,
                        use_dyt=config.use_dyt,
                        dyt_alpha_init=config.dyt_alpha_init,
                    ))
                    for _ in range(config.n_layer)
                ]),
                ln_f=DynamicTanh(config.n_embd, alpha_init=config.dyt_alpha_init)
                if config.use_dyt else RMSNorm(config.n_embd),
            )
        )

        # LM head tied
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

        # RoPE unused by standard attention in HSEBlock, kept for compatibility
        self.rope = RoPE(dim=config.qk_rope_head_dim, max_seq_len=config.block_size, base=int(config.rope_theta))

        # Scribe + cache + QAP
        self.scribe = HSEScribe(HSEScribeConfig(
            hidden_dim=config.n_embd,
            chunk_size=config.scribe_chunk_size,
            summary_len=config.scribe_summary_len,
        ))
        self.cache_manager = HierarchicalCache()
        self.qap = QAPController(budget=QAPBudget(
            per_step=config.qap_per_step,
            per_expert=config.qap_per_expert,
            max_queries=config.qap_max_queries,
        ))

        # Init weights
        self.apply(self._init_weights)
        self.param_count = sum(p.numel() for p in self.parameters())
        print(f"HSE Model - Number of parameters: {self.param_count/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            scale = 1.0 / math.sqrt(self.config.n_embd)
            nn.init.normal_(module.weight, mean=0.0, std=scale)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _run_scribes(self, tok_emb: torch.Tensor):
        """Run scribe on non-overlapping chunks for metadata and caching.

        Args:
            tok_emb: [B, T, D]
        Returns:
            list of per-batch lists of ScribeOutput
        """
        B, T, D = tok_emb.shape
        w = max(1, self.config.scribe_chunk_size)
        outputs: List[List[Any]] = []
        for b in range(B):
            outs: List[Any] = []
            cur = tok_emb[b]
            chunk_id = 0
            for s in range(0, T, w):
                e = min(T, s + w)
                if s >= e:
                    break
                out = self.scribe(cur[s:e].detach(), chunk_id=chunk_id)
                outs.append(out)
                # Store into L3 cache (raw tokens metadata)
                self.cache_manager.L3.put(out.index_ref, {
                    'semantic_embedding': out.semantic_embedding,
                    'routing_indices': out.routing_indices,
                    'importance': out.importance_score,
                })
                chunk_id += 1
            outputs.append(outs)
        return outputs

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Token embeddings
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)

        # Scribes run per chunk (metadata only). For torch.compile stability,
        # skip during Dynamo capture to avoid graph breaks.
        try:
            is_compiling = False
            try:
                import torch._dynamo as _dynamo  # type: ignore
                is_compiling = bool(getattr(_dynamo, 'is_compiling', lambda: False)())
            except Exception:
                is_compiling = False
            if not is_compiling:
                _ = self._run_scribes(tok_emb.detach())
        except Exception:
            pass

        # Simple valid mask
        mask = torch.ones(b, t, dtype=torch.bool, device=device) if t > 1 else None

        router_loss_total = None
        for block in self.transformer.h:
            x, rloss = block(x, rope=self.rope, mask=mask)
            if rloss is not None:
                router_loss_total = (router_loss_total + rloss) if router_loss_total is not None else rloss

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = None
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
                    m = (targets != -100).float()
                loss = (loss * m.view(-1)).sum() / (m.sum() + 1e-6)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, router_loss_total

    @torch.no_grad()
    def generate(self, idx: Optional[torch.Tensor] = None, max_new_tokens: Optional[int] = None, temperature: float = 1.0, top_k: Optional[int] = None, prompt=None, gen_length=None):
        # Support both idx=... and prompt=... signatures
        if idx is None and prompt is not None:
            idx = prompt
        if idx is None:
            raise TypeError("HSEModel.generate requires either 'idx' or 'prompt'.")
        tokens_to_generate = max_new_tokens if max_new_tokens is not None else (gen_length if gen_length is not None else 20)
        self.eval()
        for _ in range(tokens_to_generate):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _loss, _rl = self(idx_cond)
            logits = logits[:, -1, :] / max(1e-6, temperature)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return (idx, None)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, optimizer_type=None, **kwargs):
        if optimizer_type is None:
            optimizer_type = "adamw"
        optimizer = configure_optimizer_for_gpt(
            model=self,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
            device_type=device_type,
            optimizer_type=optimizer_type,
            **kwargs,
        )
        print(f"Configured {optimizer_type} optimizer for HSE model")
        return optimizer


def create_hse_model(
    size: str = 'medium',
    vocab_size: int = 50304,
    block_size: int = 4096,
    dropout: float = 0.0,
    use_fp8: bool = False,
    **kwargs,
):
    sizes = {
        'small': dict(n_layer=12, n_embd=768, n_head=12, compress_block_size=16, compress_stride=8, selection_block_size=32, num_selected_blocks=8, sliding_window_size=256),
        'medium': dict(n_layer=24, n_embd=1024, n_head=16, compress_block_size=32, compress_stride=16, selection_block_size=64, num_selected_blocks=12, sliding_window_size=384),
        'large': dict(n_layer=32, n_embd=2048, n_head=16, compress_block_size=32, compress_stride=16, selection_block_size=64, num_selected_blocks=16, sliding_window_size=512),
        'xl': dict(n_layer=40, n_embd=2560, n_head=20, compress_block_size=64, compress_stride=32, selection_block_size=128, num_selected_blocks=20, sliding_window_size=640),
    }
    if size not in sizes:
        raise ValueError(f"Unknown model size: {size}")
    cfg = sizes[size].copy()
    cfg.update(dict(vocab_size=vocab_size, block_size=block_size, dropout=dropout, use_fp8=use_fp8))
    cfg.update(kwargs)
    model = HSEModel(HSEConfig(**cfg))
    return model
