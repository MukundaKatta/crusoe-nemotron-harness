"""Trace recorder for Nemotron agent runs.

Each call inside a run becomes a CallEvent. The Trace groups events under a
single run, and roll-up helpers compute p50, p95, total tokens, and total
latency. This is the same shape the author's agenttrace crate exposes for
Rust.

The Trace is intentionally append-only inside a run. Callers use
`trace.record_llm(...)` and `trace.record_tool(...)` so the harness can be
loud at the seam where data enters, instead of a generic `add(event)` that
hides what kind of event went in.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CallEvent:
    """A single LLM or tool call inside a Nemotron agent run."""

    step: int
    kind: str  # "llm" or "tool"
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    latency_ms: float = 0.0
    ok: bool = True
    error_code: str | None = None
    retried: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceSummary:
    """Rolled-up numbers for one run, computed by Trace.summary()."""

    total_calls: int
    llm_calls: int
    tool_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_cached_input_tokens: int
    total_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    tool_failures: int


@dataclass
class Trace:
    """Trace for a single Nemotron run. Append-only inside the run."""

    events: list[CallEvent] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def record_llm(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cached_input_tokens: int = 0,
        ok: bool = True,
        error_code: str | None = None,
        retried: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> CallEvent:
        event = CallEvent(
            step=len(self.events) + 1,
            kind="llm",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            latency_ms=latency_ms,
            ok=ok,
            error_code=error_code,
            retried=retried,
            metadata=metadata or {},
        )
        self.events.append(event)
        return event

    def record_tool(
        self,
        *,
        tool: str,
        latency_ms: float,
        ok: bool = True,
        error_code: str | None = None,
        retried: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> CallEvent:
        meta = dict(metadata or {})
        meta.setdefault("tool", tool)
        event = CallEvent(
            step=len(self.events) + 1,
            kind="tool",
            model=tool,
            latency_ms=latency_ms,
            ok=ok,
            error_code=error_code,
            retried=retried,
            metadata=meta,
        )
        self.events.append(event)
        return event

    def finish(self) -> None:
        self.finished_at = time.time()

    def summary(self) -> TraceSummary:
        latencies = [e.latency_ms for e in self.events if e.latency_ms > 0]
        return TraceSummary(
            total_calls=len(self.events),
            llm_calls=sum(1 for e in self.events if e.kind == "llm"),
            tool_calls=sum(1 for e in self.events if e.kind == "tool"),
            total_input_tokens=sum(e.input_tokens for e in self.events),
            total_output_tokens=sum(e.output_tokens for e in self.events),
            total_cached_input_tokens=sum(e.cached_input_tokens for e in self.events),
            total_latency_ms=sum(latencies),
            p50_latency_ms=percentile(latencies, 50),
            p95_latency_ms=percentile(latencies, 95),
            tool_failures=sum(1 for e in self.events if e.kind == "tool" and not e.ok),
        )


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile across a list of floats.

    `pct` is 0..100. Empty input returns 0.0.
    """

    if not values:
        return 0.0
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    ordered = sorted(values)
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac
