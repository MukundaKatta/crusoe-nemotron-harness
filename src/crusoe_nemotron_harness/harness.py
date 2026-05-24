"""NemotronHarness: the facade.

Wraps any NemotronProvider and weaves the six concerns (cost, egress, vet,
snap, trace, budget) into one cohesive run.

Typical use:

```python
from crusoe_nemotron_harness import NemotronHarness, FakeNemotronProvider

harness = NemotronHarness(
    provider=FakeNemotronProvider(seed=3),
    max_tokens_per_run=5_000,
    max_usd_per_run=0.50,
    allowed_hosts=["api.example.com"],
)
with harness.run() as run:
    result = run.complete("Summarize Nemotron in one sentence.")
    run.fetch_url("https://api.example.com/v1/data")  # passes
    print(run.report())
```

The harness never silently swallows failures: budget overruns raise
BudgetExceeded, disallowed hosts raise EgressDenied, bad tool args raise
ToolArgError. Snapshot mismatches surface on `run.snapshot(path)`.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from .budget import Budget, BudgetStatus
from .cost import DEFAULT_PRICES, ModelPrice, cost_usd
from .egress import EgressPolicy
from .snap import RunSnapshot, SnapshotMismatch
from .trace import Trace, TraceSummary
from .vet import ToolSpec, vet_or_raise
from .providers import CompletionResult, NemotronProvider


@dataclass
class RunReport:
    """Everything the harness collected about a run, in one object."""

    trace: TraceSummary
    budget: BudgetStatus
    total_cost_usd: float
    wall_time_ms: float
    snapshot_events: int
    aborted: bool
    abort_reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_calls": self.trace.total_calls,
            "llm_calls": self.trace.llm_calls,
            "tool_calls": self.trace.tool_calls,
            "total_input_tokens": self.trace.total_input_tokens,
            "total_output_tokens": self.trace.total_output_tokens,
            "p50_latency_ms": self.trace.p50_latency_ms,
            "p95_latency_ms": self.trace.p95_latency_ms,
            "tool_failures": self.trace.tool_failures,
            "tokens_used": self.budget.tokens_used,
            "tokens_cap": self.budget.tokens_cap,
            "usd_used": self.budget.usd_used,
            "usd_cap": self.budget.usd_cap,
            "total_cost_usd": self.total_cost_usd,
            "wall_time_ms": self.wall_time_ms,
            "snapshot_events": self.snapshot_events,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
        }


@dataclass
class _RunContext:
    """Per-run state. Callers get this from `with harness.run() as ctx:`."""

    provider: NemotronProvider
    trace: Trace
    budget: Budget
    snapshot: RunSnapshot
    egress: EgressPolicy
    tools: dict[str, ToolSpec]
    prices: dict[str, ModelPrice]
    started_at: float

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> CompletionResult:
        """Run a single Nemotron completion, charging budget and tracing.

        Order of operations matters:
            1. Call the provider.
            2. Compute cost.
            3. Charge the budget. If charge raises, the trace still has the
               event recorded but the run aborts before any subsequent
               complete().
            4. Record the trace event.
            5. Record the snapshot event.
        """

        self.snapshot.record("prompt", {"text": prompt, "model": model or getattr(self.provider, "model", "unknown")})
        result = self.provider.complete(
            prompt, model=model, max_tokens=max_tokens, temperature=temperature
        )
        call_cost = cost_usd(
            result.model,
            result.input_tokens,
            result.output_tokens,
            result.cached_input_tokens,
            self.prices,
        )
        tokens_charge = result.input_tokens + result.output_tokens
        # Charge first. If it raises, we never record success.
        self.budget.charge(tokens=tokens_charge, usd=call_cost)
        self.trace.record_llm(
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cached_input_tokens=result.cached_input_tokens,
            latency_ms=result.latency_ms,
            metadata={"usd": call_cost},
        )
        self.snapshot.record(
            "response",
            {"model": result.model, "text": result.text, "output_tokens": result.output_tokens},
        )
        return result

    def call_tool(
        self,
        tool: str,
        args: dict[str, Any],
        *,
        latency_ms: float = 0.0,
        url: str | None = None,
    ) -> None:
        """Vet a tool call, check egress if a URL is given, record it.

        This is a recording helper; the actual tool body runs in the caller's
        code. The harness only enforces the contracts.
        """

        if self.tools:
            vet_or_raise(tool, args, self.tools)
        if url is not None:
            self.egress.check_url(url)
        self.snapshot.record("tool_call", {"tool": tool, "args": args})
        self.trace.record_tool(tool=tool, latency_ms=latency_ms, ok=True)

    def fetch_url(self, url: str) -> str:
        """Egress-check a URL and return the host. No actual fetch.

        Egress is enforced at the policy boundary, not at the wire. Real
        callers wrap their own HTTP client and gate it on `harness.run.fetch_url`.
        """

        return self.egress.check_url(url)

    def snapshot_compare(self, fixture_path: str, update: bool | None = None) -> SnapshotMismatch:
        """Compare the in-flight snapshot against a saved fixture."""

        return self.snapshot.compare(fixture_path, update=update)

    def report(self) -> RunReport:
        self.trace.finish()
        summary = self.trace.summary()
        status = self.budget.status()
        total_cost = status.usd_used
        wall_time_ms = (time.time() - self.started_at) * 1000.0
        return RunReport(
            trace=summary,
            budget=status,
            total_cost_usd=total_cost,
            wall_time_ms=wall_time_ms,
            snapshot_events=len(self.snapshot.events),
            aborted=self.budget.aborted,
            abort_reason=status.abort_reason,
        )


@dataclass
class NemotronHarness:
    """Facade around a NemotronProvider.

    Args:
        provider: any object implementing the NemotronProvider protocol.
        max_tokens_per_run: 0 means no token cap.
        max_usd_per_run: 0 means no USD cap.
        allowed_hosts: list of hosts tools may fetch. Empty = deny all.
        tools: optional tool spec registry for vetting.
        prices: per-model price table. Defaults to DEFAULT_PRICES.
    """

    provider: NemotronProvider
    max_tokens_per_run: int = 0
    max_usd_per_run: float = 0.0
    allowed_hosts: list[str] = field(default_factory=list)
    tools: dict[str, ToolSpec] = field(default_factory=dict)
    prices: dict[str, ModelPrice] = field(default_factory=lambda: dict(DEFAULT_PRICES))

    @contextmanager
    def run(self) -> Iterator[_RunContext]:
        """Context-managed run. Always produces a report on exit, even on error.

        Use:
            with harness.run() as ctx:
                ctx.complete(...)
            # `ctx.report()` is callable inside the block; after exit the
            # report is still computable because the trace and budget objects
            # live on the context.
        """

        ctx = _RunContext(
            provider=self.provider,
            trace=Trace(),
            budget=Budget(max_tokens=self.max_tokens_per_run, max_usd=self.max_usd_per_run),
            snapshot=RunSnapshot(),
            egress=EgressPolicy.from_iterable(self.allowed_hosts),
            tools=dict(self.tools),
            prices=dict(self.prices),
            started_at=time.time(),
        )
        try:
            yield ctx
        finally:
            if ctx.trace.finished_at is None:
                ctx.trace.finish()


def format_report(report: RunReport) -> str:
    """Render a RunReport as a plain-text leaderboard-style table for demos."""

    rows = [
        ("total_calls", str(report.trace.total_calls)),
        ("llm_calls", str(report.trace.llm_calls)),
        ("tool_calls", str(report.trace.tool_calls)),
        ("total_input_tokens", str(report.trace.total_input_tokens)),
        ("total_output_tokens", str(report.trace.total_output_tokens)),
        ("p50_latency_ms", f"{report.trace.p50_latency_ms:.0f} ms"),
        ("p95_latency_ms", f"{report.trace.p95_latency_ms:.0f} ms"),
        ("tool_failures", str(report.trace.tool_failures)),
        ("tokens_used", f"{report.budget.tokens_used} / {report.budget.tokens_cap or 'no cap'}"),
        ("usd_used", f"${report.budget.usd_used:.6f} / ${report.budget.usd_cap:.4f}" if report.budget.usd_cap else f"${report.budget.usd_used:.6f}"),
        ("total_cost_usd", f"${report.total_cost_usd:.6f}"),
        ("wall_time_ms", f"{report.wall_time_ms:.0f} ms"),
        ("snapshot_events", str(report.snapshot_events)),
        ("aborted", "yes" if report.aborted else "no"),
        ("abort_reason", report.abort_reason or "-"),
    ]
    width = max(len(name) for name, _ in rows)
    lines = ["RunReport", "-" * (width + 22)]
    for name, value in rows:
        lines.append(f"{name.ljust(width)}  {value}")
    return "\n".join(lines)
