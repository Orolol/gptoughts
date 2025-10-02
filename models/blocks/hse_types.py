"""Types and lightweight structures for HSE components.

This file defines helper dataclasses/typed dicts used by the HSE implementation.
They are intentionally lightweight to avoid introducing runtime dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal, Any, Optional


@dataclass
class EvidenceSpan:
    start: int
    end: int
    score: float


@dataclass
class ScribeOutput:
    # Dense summary/embedding of the chunk
    semantic_embedding: Any  # Tensor[768] (kept generic for torch.Tensor)
    routing_indices: Dict[str, float]
    importance_score: float
    temporal_markers: List[int]
    summary_tokens: Any      # Tensor[summary_len, hidden_dim]
    evidence_spans: List[EvidenceSpan]
    index_ref: str


# QAP protocol skeleton (minimal runtime-only variants)
@dataclass
class QAPQuery:
    query_id: str
    source_level: Literal[3, 2, 1]
    target_level: Literal[2, 1]
    query_type: Literal["DETAIL", "CONTEXT", "VERIFICATION", "EXPANSION"]
    resource: Literal["SPANS", "TOKENS", "STATE"]
    specificity: Dict[str, Any]
    priority: float
    deadline_ms: int
    budget_cost: float


@dataclass
class QAPBudget:
    per_step: int = 12
    per_expert: int = 6
    max_queries: int = 20


@dataclass
class QAPMetrics:
    efficiency_per_1k: float = 0.0
    hit_rate: float = 0.0
    evidence_recall_k: float = 0.0
    latency_ms: Dict[str, float] | None = None

