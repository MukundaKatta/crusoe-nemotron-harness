"""Trace tests."""

import pytest

from crusoe_nemotron_harness.trace import Trace, percentile


def test_empty_trace_summary() -> None:
    summary = Trace().summary()
    assert summary.total_calls == 0
    assert summary.p50_latency_ms == 0.0
    assert summary.p95_latency_ms == 0.0


def test_record_llm_increments_step() -> None:
    trace = Trace()
    trace.record_llm(model="m", input_tokens=10, output_tokens=20, latency_ms=100)
    trace.record_llm(model="m", input_tokens=5, output_tokens=5, latency_ms=50)
    assert [e.step for e in trace.events] == [1, 2]


def test_summary_rolls_up_tokens_and_latency() -> None:
    trace = Trace()
    trace.record_llm(model="m", input_tokens=10, output_tokens=20, latency_ms=100)
    trace.record_llm(model="m", input_tokens=30, output_tokens=40, latency_ms=200)
    summary = trace.summary()
    assert summary.total_input_tokens == 40
    assert summary.total_output_tokens == 60
    assert summary.total_latency_ms == 300


def test_tool_failures_are_counted() -> None:
    trace = Trace()
    trace.record_tool(tool="x", latency_ms=10, ok=True)
    trace.record_tool(tool="x", latency_ms=10, ok=False, error_code="bad_arg")
    trace.record_tool(tool="y", latency_ms=10, ok=False, error_code="timeout")
    assert trace.summary().tool_failures == 2


def test_percentile_basic() -> None:
    vals = [10, 20, 30, 40, 50]
    assert percentile(vals, 50) == pytest.approx(30)
    assert percentile(vals, 95) == pytest.approx(48)
    assert percentile([], 50) == 0.0


def test_percentile_edge_cases() -> None:
    assert percentile([42], 50) == 42
    assert percentile([1, 2], 0) == 1
    assert percentile([1, 2], 100) == 2


def test_p95_at_least_p50() -> None:
    trace = Trace()
    for ms in (50, 100, 150, 200, 1000):
        trace.record_llm(model="m", input_tokens=1, output_tokens=1, latency_ms=ms)
    summary = trace.summary()
    assert summary.p95_latency_ms >= summary.p50_latency_ms
