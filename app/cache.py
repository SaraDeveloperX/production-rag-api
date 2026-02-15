"""Lightweight in-memory TTL cache for RAG responses.

Disabled by default (CACHE_TTL_SECONDS=0). Key = SHA-256(question + config).
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any

from app.settings import settings


def _make_key(question: str) -> str:
    """Deterministic cache key from question + model config."""
    raw = f"{question}|{settings.NAMESPACE}|{settings.TOP_K}|{settings.OPENAI_MODEL}"
    return hashlib.sha256(raw.encode()).hexdigest()


class _TTLCache:
    """Thread-safe LRU cache with per-entry TTL expiry."""

    def __init__(self, max_items: int, ttl_seconds: int) -> None:
        self._max = max_items
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._ttl > 0

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        with self._lock:
            if key not in self._store:
                return None
            ts, value = self._store[key]
            if time.time() - ts > self._ttl:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return value

    def put(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._store[key] = (time.time(), value)
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# Module-level singleton
response_cache = _TTLCache(
    max_items=settings.CACHE_MAX_ITEMS,
    ttl_seconds=settings.CACHE_TTL_SECONDS,
)
