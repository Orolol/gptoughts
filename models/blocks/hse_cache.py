"""Hierarchical cache skeleton for HSE.

This is a lightweight runtime cache used to store:
- L1 (Orchestrator): recent expert results + quantized KV (not implemented, stub)
- L2 (Experts): scribe summaries + compressed states
- L3 (Scribes): raw tokens + index_ref + meta

Eviction policies are simplified (LRU-like using insertion order); the API is
minimal and tailored to PoC-level usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CacheLevel:
    capacity: int
    store: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str) -> Optional[Any]:
        return self.store.get(key)

    def put(self, key: str, value: Any):
        if len(self.store) >= self.capacity:
            # Evict first inserted key (FIFO as simple LRU-ish)
            try:
                oldest_key = next(iter(self.store.keys()))
                self.store.pop(oldest_key, None)
            except StopIteration:
                pass
        self.store[key] = value

    def clear(self):
        self.store.clear()


class HierarchicalCache:
    def __init__(self, l1_capacity: int = 1024, l2_capacity: int = 4096, l3_capacity: int = 16384):
        self.L1 = CacheLevel(l1_capacity)
        self.L2 = CacheLevel(l2_capacity)
        self.L3 = CacheLevel(l3_capacity)

    def clear(self):
        self.L1.clear()
        self.L2.clear()
        self.L3.clear()

