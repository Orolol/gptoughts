"""QAP controller (asynchronous-like) for HSE PoC.

Implements a minimal budgeting and request API compatible with the HSE design.
No real async IO is performed; the controller tracks budgets/metrics and stores
requested keys for potential downstream usage.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .hse_types import QAPQuery, QAPBudget, QAPMetrics


@dataclass
class QAPController:
    budget: QAPBudget
    metrics: QAPMetrics = field(default_factory=QAPMetrics)

    def __post_init__(self):
        self._step_budget_remaining = self.budget.per_step
        self._issued_queries: List[QAPQuery] = []

    @contextmanager
    def session(self, budget_step: Optional[int] = None):
        prev = self._step_budget_remaining
        if budget_step is not None:
            self._step_budget_remaining = budget_step
        start = time.time()
        try:
            yield self
        finally:
            dt = time.time() - start
            # Very rough placeholder metrics update
            self.metrics.latency_ms = self.metrics.latency_ms or {}
            self.metrics.latency_ms["qap"] = 1000.0 * dt
            # Reset to previous if any
            if budget_step is not None:
                self._step_budget_remaining = prev

    def can_issue(self, cost: float) -> bool:
        return self._step_budget_remaining >= cost

    def issue(self, query: QAPQuery) -> Dict[str, Any]:
        # Budget check
        if not self.can_issue(query.budget_cost):
            return {"status": "budget_exceeded"}
        self._step_budget_remaining -= query.budget_cost
        self._issued_queries.append(query)
        # Stubbed response
        return {"status": "ok", "resource": query.resource, "specificity": query.specificity}

    def reset_step_budget(self):
        self._step_budget_remaining = self.budget.per_step
