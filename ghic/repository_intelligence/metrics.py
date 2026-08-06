"""Metrics and correlation context for the Repository Intelligence Engine.

In-process counters and timers, exposed through `/stats` alongside the
webhook latency percentiles the service already reports. Deliberately not
a Prometheus client: adding a metrics dependency and a scrape endpoint to
a service that currently reports through one JSON endpoint would be a
bigger operational change than the metrics are worth, and `snapshot()` is
shaped so a Prometheus exporter can be written over it later without
touching a single call site.

The honest caveat, stated here because it decides how these numbers should
be read: counters are **per process**. On a horizontally-scaled or
serverless deploy each container reports its own, and they reset on
recycle. Durable per-repository facts (chunk counts, index times, retrieval
hit rate) live in the state store instead, which is why
`RepositoryRecord` carries them.

Correlation IDs come from the webhook's own `X-GitHub-Delivery` header
where one exists -- inventing a second identifier when GitHub already
supplies a unique, user-visible one just makes two things to correlate.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from .. import utils

logger = utils.get_logger(__name__)

# Set per request/job so every log line inside one unit of work can be tied
# back to the delivery that caused it. A ContextVar (not a thread-local) so
# it survives the sync/async boundary FastAPI puts in the middle.
_correlation_id: ContextVar[str] = ContextVar("ghic_correlation_id", default="")
_repo_context: ContextVar[str] = ContextVar("ghic_repo", default="")


def set_correlation_id(value: str) -> str:
    _correlation_id.set(value or uuid.uuid4().hex[:16])
    return _correlation_id.get()


def get_correlation_id() -> str:
    return _correlation_id.get()


def set_repo_context(repo: str) -> None:
    _repo_context.set(repo or "")


def log_context() -> dict[str, str]:
    """Correlation fields for a structured log record."""
    context = {}
    if _correlation_id.get():
        context["correlation_id"] = _correlation_id.get()
    if _repo_context.get():
        context["repo"] = _repo_context.get()
    return context


@dataclass
class _Timings:
    """Rolling window of durations. Bounded so a long-lived process can't
    grow this without limit; percentiles over the last N are what an
    operator actually wants anyway."""
    samples: deque = field(default_factory=lambda: deque(maxlen=1000))

    def add(self, seconds: float) -> None:
        self.samples.append(seconds)

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {}
        ordered = sorted(self.samples)

        def pct(p: float) -> float:
            return round(ordered[min(len(ordered) - 1, int(len(ordered) * p))] * 1000, 1)

        return {
            "n": len(ordered),
            "p50_ms": pct(0.50),
            "p95_ms": pct(0.95),
            "p99_ms": pct(0.99),
            "mean_ms": round(sum(ordered) / len(ordered) * 1000, 1),
        }


class RepositoryIntelligenceMetrics:
    """Thread-safe counters and timers for one process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._timers: dict[str, _Timings] = {}
        self._gauges: dict[str, float] = {}

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def observe(self, name: str, seconds: float) -> None:
        with self._lock:
            self._timers.setdefault(name, _Timings()).add(seconds)

    @contextmanager
    def timed(self, name: str):
        """Time a block, recording it even when the block raises -- a slow
        failure is exactly the timing an operator needs to see."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, time.perf_counter() - started)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            timers = {name: t.summary() for name, t in self._timers.items()}

        retrievals = counters.get("retrievals_total", 0)
        hits = counters.get("retrievals_with_results", 0)
        cache_lookups = counters.get("index_cache_lookups", 0)
        cache_hits = counters.get("index_cache_hits", 0)
        return {
            "scope": "process",   # see the module docstring: not cluster-wide
            "counters": counters,
            "gauges": gauges,
            "timings": timers,
            "derived": {
                "retrieval_hit_rate": round(hits / retrievals, 4) if retrievals else 0.0,
                "index_cache_hit_rate": (
                    round(cache_hits / cache_lookups, 4) if cache_lookups else 0.0
                ),
            },
        }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._timers.clear()
            self._gauges.clear()


# One shared instance; injectable everywhere it's used, so tests never
# depend on global state (see RepositoryIntelligenceService's constructor).
METRICS = RepositoryIntelligenceMetrics()
