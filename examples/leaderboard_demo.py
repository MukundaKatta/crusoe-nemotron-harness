"""End-to-end demo for crusoe-nemotron-harness.

Scene 1: a bare Nemotron provider. We can see how many calls we made and
roughly how long they took. We cannot see cost, budget, snapshot delta,
tool-arg safety, or egress safety. Production owners want all five.

Scene 2: the same provider wrapped in NemotronHarness. One RunReport with
every concern rolled up.

Run:
    python examples/leaderboard_demo.py
"""

from __future__ import annotations

import time

from crusoe_nemotron_harness import (
    BudgetExceeded,
    EgressDenied,
    FakeNemotronProvider,
    NemotronHarness,
    ToolSpec,
    format_report,
)


TASKS = [
    (f"t{idx:02d}", f"Summarize Nemotron application area #{idx} in one paragraph.")
    for idx in range(1, 11)
]


DEMO_SEED = 3


def scene_one() -> None:
    print("Scene 1: bare Nemotron provider. No harness, no visibility.\n")
    provider = FakeNemotronProvider(seed=DEMO_SEED)
    total_tokens = 0
    start = time.perf_counter()
    for task_id, prompt in TASKS:
        r = provider.complete(prompt, max_tokens=200)
        total_tokens += r.input_tokens + r.output_tokens
    elapsed = (time.perf_counter() - start) * 1000.0
    print(f"  Tasks run:        {len(TASKS)}")
    print(f"  Tokens (approx):  {total_tokens}")
    print(f"  Wall time:        {elapsed:.0f} ms")
    print("  Cost:             unknown")
    print("  Budget cap hit:   unknown")
    print("  Tool args safe:   unknown")
    print("  Egress safe:      unknown")
    print("  Snapshot stable:  unknown")
    print()


def scene_two() -> None:
    print("Scene 2: same agent wrapped in NemotronHarness.\n")
    provider = FakeNemotronProvider(seed=DEMO_SEED)
    tools = {
        "search": ToolSpec("search", required=("query",), types={"query": "str"}),
    }
    harness = NemotronHarness(
        provider=provider,
        max_tokens_per_run=200_000,
        max_usd_per_run=1.00,
        allowed_hosts=["api.example.com"],
        tools=tools,
    )

    egress_blocked = 0
    bad_tool_args = 0

    with harness.run() as ctx:
        for task_id, prompt in TASKS:
            ctx.complete(prompt, max_tokens=200)
            # Simulate a tool the agent calls between turns.
            try:
                ctx.call_tool("search", {"query": prompt[:40]}, latency_ms=15)
            except Exception:
                bad_tool_args += 1
            # Simulate an attempted off-allowlist fetch one time.
            if task_id == "t05":
                try:
                    ctx.fetch_url("https://evil.example.com/exfil")
                except EgressDenied:
                    egress_blocked += 1
        report = ctx.report()

    print(format_report(report))
    print()
    print(f"  Off-allowlist fetches blocked: {egress_blocked}")
    print(f"  Bad-tool-arg attempts:         {bad_tool_args}")
    print()


def scene_three() -> None:
    print("Scene 3: same agent with a deliberately tight budget. We abort cleanly.\n")
    provider = FakeNemotronProvider(seed=DEMO_SEED)
    harness = NemotronHarness(
        provider=provider,
        max_tokens_per_run=120,
        max_usd_per_run=10.0,
        allowed_hosts=["api.example.com"],
    )
    with harness.run() as ctx:
        try:
            for _, prompt in TASKS:
                ctx.complete(prompt, max_tokens=200)
        except BudgetExceeded as exc:
            print(f"  Aborted: {exc}")
        report = ctx.report()
    print()
    print(format_report(report))
    print()


def main() -> None:
    scene_one()
    scene_two()
    scene_three()


if __name__ == "__main__":
    main()
