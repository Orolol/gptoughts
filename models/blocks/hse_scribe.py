"""Scribe block for HSE.

This module implements a small encoder that operates on local windows (chunks)
to produce ScribeOutput entries: embeddings, routing hints, summaries, and
evidence spans. It is intentionally simple and fast for PoC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hse_types import EvidenceSpan, ScribeOutput


@dataclass
class HSEScribeConfig:
    hidden_dim: int
    chunk_size: int = 2048
    summary_len: int = 128
    num_domains: int = 4  # e.g. [code, math, dialogue, analytical]


class HSEScribe(nn.Module):
    def __init__(self, config: HSEScribeConfig):
        super().__init__()
        self.config = config

        # Lightweight conv for local pattern mixing
        self.local_conv = nn.Conv1d(
            in_channels=config.hidden_dim,
            out_channels=config.hidden_dim,
            kernel_size=5,
            padding=2,
            groups=1,
            bias=False,
        )
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.proj_summary = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        # Routing head (domain logits)
        self.routing_head = nn.Linear(config.hidden_dim, config.num_domains, bias=False)
        # Importance score
        self.importance_head = nn.Linear(config.hidden_dim, 1, bias=False)

        # Token importance conv to pick spans (strided pooling-like scoring)
        self.imp_conv = nn.Conv1d(
            in_channels=config.hidden_dim,
            out_channels=1,
            kernel_size=17,
            padding=8,
            bias=False,
        )

        # Initialize gently
        nn.init.normal_(self.local_conv.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.proj_summary.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.routing_head.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.importance_head.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.imp_conv.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor, chunk_id: int = 0) -> ScribeOutput:
        """
        Args:
            x: [T, D] token embeddings for a local chunk (no batch to keep simple)
        Returns:
            ScribeOutput (dataclass of basic tensors and metadata)
        """
        T, D = x.shape
        h = self.local_conv(x.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)  # [T, D]
        h = self.norm(h)

        # Summary via average pooling into summary_len slots
        summary_len = min(self.config.summary_len, max(1, T))
        # linspace bins
        bins = torch.linspace(0, T, steps=summary_len + 1, device=x.device).long()
        summaries = []
        for i in range(summary_len):
            s, e = bins[i].item(), bins[i + 1].item()
            if s >= e:
                e = min(T, s + 1)
            summaries.append(h[s:e].mean(dim=0, keepdim=True))
        summary_tokens = torch.cat(summaries, dim=0)  # [summary_len, D]
        summary_tokens = self.proj_summary(summary_tokens)

        # Semantic embedding as mean-pooled vector
        semantic_embedding = h.mean(dim=0)  # [D]

        # Routing domain logits -> softmax into distribution
        # Use chunk-level aggregated representation (mean over time)
        route_logits = self.routing_head(h.mean(dim=0))  # [num_domains]
        routing_probs = F.softmax(route_logits, dim=-1)
        routing_indices = {
            "code": float(routing_probs[0].item()) if routing_probs.numel() > 0 else 0.0,
            "math": float(routing_probs[1].item()) if routing_probs.numel() > 1 else 0.0,
            "dialogue": float(routing_probs[2].item()) if routing_probs.numel() > 2 else 0.0,
            "analysis": float(routing_probs[3].item()) if routing_probs.numel() > 3 else 0.0,
        }

        # Importance score (normalized sigmoid of head on mean)
        importance_score = torch.sigmoid(self.importance_head(h.mean(dim=0))).item()

        # Temporal markers: pick a few peaks from token-level scores
        token_scores = self.imp_conv(h.transpose(0, 1).unsqueeze(0)).squeeze(0).squeeze(0)  # [T]
        k = min(8, T)
        _, top_idx = torch.topk(token_scores, k)
        temporal_markers = sorted([int(i.item()) for i in top_idx])

        # Evidence spans: build small spans of width w around top markers
        w = max(4, self.config.chunk_size // 64)
        spans: List[EvidenceSpan] = []
        for i in temporal_markers[: min(4, len(temporal_markers))]:
            s = max(0, i - w // 2)
            e = min(T, s + w)
            spans.append(EvidenceSpan(start=s, end=e, score=float(token_scores[i].item())))

        out: ScribeOutput = ScribeOutput(
            semantic_embedding=semantic_embedding,
            routing_indices=routing_indices,
            importance_score=float(importance_score),
            temporal_markers=temporal_markers,
            summary_tokens=summary_tokens,
            evidence_spans=spans,
            index_ref=f"chunk:{chunk_id}:0-{T}",
        )
        return out

