"""End-to-end NemotronHarness tests."""

import pytest

from crusoe_nemotron_harness import (
    BudgetExceeded,
    EgressDenied,
    FakeNemotronProvider,
    NemotronHarness,
    ToolArgError,
    ToolSpec,
    format_report,
)


def _harness(**kwargs) -> NemotronHarness:
    defaults = dict(
        provider=FakeNemotronProvider(seed=3),
        max_tokens_per_run=10_000,
        max_usd_per_run=1.0,
        allowed_hosts=["api.example.com"],
    )
    defaults.update(kwargs)
    return NemotronHarness(**defaults)


def test_basic_complete_records_trace_and_charges_budget() -> None:
    h = _harness()
    with h.run() as ctx:
        ctx.complete("hello world")
        report = ctx.report()
    assert report.trace.llm_calls == 1
    assert report.budget.tokens_used > 0
    assert report.total_cost_usd > 0
    assert not report.aborted


def test_run_is_deterministic_for_same_seed() -> None:
    a = _harness()
    b = _harness()
    with a.run() as ctx_a:
        for i in range(3):
            ctx_a.complete(f"prompt {i}")
        report_a = ctx_a.report()
    with b.run() as ctx_b:
        for i in range(3):
            ctx_b.complete(f"prompt {i}")
        report_b = ctx_b.report()
    assert report_a.budget.tokens_used == report_b.budget.tokens_used
    assert report_a.total_cost_usd == report_b.total_cost_usd


def test_budget_overrun_aborts_run() -> None:
    h = _harness(max_tokens_per_run=20)
    with h.run() as ctx:
        with pytest.raises(BudgetExceeded):
            for i in range(20):
                ctx.complete(f"prompt {i} with some payload to make tokens add up")
        report = ctx.report()
    assert report.aborted
    assert report.abort_reason == "tokens"


def test_egress_blocks_disallowed_host() -> None:
    h = _harness(allowed_hosts=["api.example.com"])
    with h.run() as ctx:
        ctx.fetch_url("https://api.example.com/v1/x")  # allowed
        with pytest.raises(EgressDenied):
            ctx.fetch_url("https://evil.example.com/exfil")


def test_call_tool_vets_args_when_specs_registered() -> None:
    specs = {"search": ToolSpec("search", required=("query",), types={"query": "str"})}
    h = _harness(tools=specs)
    with h.run() as ctx:
        ctx.call_tool("search", {"query": "nemotron"}, latency_ms=10)
        with pytest.raises(ToolArgError):
            ctx.call_tool("search", {}, latency_ms=10)


def test_call_tool_url_egress_enforced() -> None:
    h = _harness(allowed_hosts=["api.example.com"])
    with h.run() as ctx:
        ctx.call_tool("http_get", {}, latency_ms=5, url="https://api.example.com/v1/y")
        with pytest.raises(EgressDenied):
            ctx.call_tool("http_get", {}, latency_ms=5, url="https://evil.example.com")


def test_snapshot_compare_works_inside_run(tmp_path) -> None:
    h = _harness()
    fixture = tmp_path / "run.json"
    with h.run() as ctx:
        ctx.complete("snapshot me")
        first = ctx.snapshot_compare(str(fixture))
    assert first.ok
    assert fixture.exists()

    with h.run() as ctx2:
        ctx2.complete("snapshot me")
        # Second time around with same seed must match the saved fixture.
        h2_provider_state_matters = True
        assert h2_provider_state_matters  # sanity
        second = ctx2.snapshot_compare(str(fixture))
    assert second.ok


def test_format_report_renders_all_rows() -> None:
    h = _harness()
    with h.run() as ctx:
        ctx.complete("hi")
        report = ctx.report()
    rendered = format_report(report)
    for label in (
        "total_calls",
        "llm_calls",
        "tool_calls",
        "p50_latency_ms",
        "p95_latency_ms",
        "tokens_used",
        "total_cost_usd",
        "wall_time_ms",
        "snapshot_events",
        "aborted",
    ):
        assert label in rendered


def test_report_as_dict_round_trips_keys() -> None:
    h = _harness()
    with h.run() as ctx:
        ctx.complete("hi")
        d = ctx.report().as_dict()
    for key in (
        "total_calls",
        "llm_calls",
        "tool_calls",
        "p50_latency_ms",
        "p95_latency_ms",
        "tokens_used",
        "tokens_cap",
        "usd_used",
        "total_cost_usd",
        "wall_time_ms",
        "snapshot_events",
        "aborted",
    ):
        assert key in d
