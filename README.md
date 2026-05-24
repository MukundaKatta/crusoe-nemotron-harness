# crusoe-nemotron-harness

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![tests: 60 passing](https://img.shields.io/badge/tests-60%20passing-brightgreen.svg)](#tests)
[![runtime deps: 0](https://img.shields.io/badge/runtime%20deps-0-blue.svg)](pyproject.toml)

Production harness for Nemotron agents on Crusoe Cloud Managed Inference.

Wrap any Nemotron provider in `NemotronHarness` and every run produces one auditable `RunReport` that covers the six concerns every production agent needs: cost, egress allowlist, tool-arg vetting, run snapshots, traces, and budget caps. No code changes inside the agent.

Built for the DevNetwork [AI+ML] Hackathon 2026, Crusoe sponsor track ("Your Harness, Our Inference: Build a Nemotron Agent").

## Quickstart

```bash
git clone https://github.com/MukundaKatta/crusoe-nemotron-harness.git
cd crusoe-nemotron-harness
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
.venv/bin/python examples/leaderboard_demo.py
```

## What you get

A `FakeNemotronProvider` lets you see the whole flow without any API keys. The real `CrusoeNemotronProvider` is a one-line swap (see `DEPLOY.md`).

```python
from crusoe_nemotron_harness import (
    FakeNemotronProvider,
    NemotronHarness,
    ToolSpec,
)

harness = NemotronHarness(
    provider=FakeNemotronProvider(seed=3),
    max_tokens_per_run=200_000,
    max_usd_per_run=1.00,
    allowed_hosts=["api.example.com"],
    tools={"search": ToolSpec("search", required=("query",), types={"query": "str"})},
)

with harness.run() as ctx:
    result = ctx.complete("Summarize Nemotron in one paragraph.", max_tokens=200)
    ctx.call_tool("search", {"query": "nemotron benchmarks"}, latency_ms=15)
    ctx.fetch_url("https://api.example.com/v1/data")
    report = ctx.report()

print(report.as_dict())
```

## Sample run

`examples/leaderboard_demo.py` runs three scenes against the same seeded provider.

```
Scene 1: bare Nemotron provider. No harness, no visibility.

  Tasks run:        10
  Tokens (approx):  344
  Cost:             unknown
  Budget cap hit:   unknown
  Tool args safe:   unknown
  Egress safe:      unknown
  Snapshot stable:  unknown

Scene 2: same agent wrapped in NemotronHarness.

RunReport
-----------------------------------------
total_calls          20
llm_calls            10
tool_calls           10
total_input_tokens   140
total_output_tokens  204
p50_latency_ms       124 ms
p95_latency_ms       1003 ms
tool_failures        0
tokens_used          344 / 200000
usd_used             $0.000654 / $1.0000
total_cost_usd       $0.000654
wall_time_ms         0 ms
snapshot_events      30
aborted              no
abort_reason         -

  Off-allowlist fetches blocked: 1
  Bad-tool-arg attempts:         0

Scene 3: same agent with a deliberately tight budget. We abort cleanly.

  Aborted: Budget exceeded (tokens): current=110 + attempted=42 > cap=120.

RunReport
-----------------------------------------
total_calls          3
llm_calls            3
tokens_used          110 / 120
total_cost_usd       $0.000221
aborted              yes
abort_reason         tokens
```

## How each row gets computed

Every metric maps to a small, single-purpose library the author already shipped. The harness is the seam that pulls them together for Crusoe + Nemotron.

| Concern              | Module       | Sibling library shipped earlier         |
| -------------------- | ------------ | --------------------------------------- |
| `total_cost_usd`     | `cost.py`    | claude-cost, bedrock-cost, bedrock-kit  |
| `allowed_hosts`      | `egress.py`  | agentguard, agentguard-rs, birddog      |
| `tool_failures`      | `vet.py`     | agentvet, agentvet-rs                   |
| `snapshot_events`    | `snap.py`    | agentsnap, agentsnap-rs                 |
| `p50/p95_latency_ms` | `trace.py`   | agenttrace, agenttrace-rs               |
| `tokens_used`        | `budget.py`  | token-budget-pool, token-budget-py, llm-budget-window |

## Swap the fake for real Crusoe Managed Inference

`CrusoeNemotronProvider` ships with the request shape Crusoe Managed Inference expects (OpenAI-compatible chat completions). Inject a transport callable for the actual HTTP call:

```python
import os
import requests

from crusoe_nemotron_harness import CrusoeNemotronProvider, NemotronHarness


def requests_transport(url: str, headers: dict[str, str], body: bytes) -> dict:
    response = requests.post(url, headers=headers, data=body, timeout=60)
    response.raise_for_status()
    return response.json()


provider = CrusoeNemotronProvider(
    model="nemotron-70b-instruct",
    api_key=os.environ["CRUSOE_API_KEY"],
    url=os.environ["CRUSOE_INFERENCE_URL"],
    transport=requests_transport,
)
harness = NemotronHarness(
    provider=provider,
    max_tokens_per_run=200_000,
    max_usd_per_run=10.00,
    allowed_hosts=["api.crusoe.example.com"],
)
```

See `DEPLOY.md` for the full path (env vars, model id, retry hints, real pricing).

## Why this fits the Crusoe + Nemotron track

The track asks for harnesses that make Nemotron agents production-ready on Crusoe Managed Inference. This package answers six of the questions every production owner asks on day one:

- How much did this run cost?
- Did it try to call hosts I never approved?
- Did it call tools with bad args?
- Did the output drift from a saved baseline?
- What was the p95 latency?
- Did it stay under the cap I funded?

One context-managed `with harness.run() as ctx:` block, one `RunReport`, no extra moving parts.

## Tests

```bash
.venv/bin/pytest -q
```

60 tests cover cost math, egress allowlist, tool-arg vetting, snapshot save / compare / update, trace percentiles, budget aborts, provider determinism, the Crusoe HTTP shape, and the end-to-end facade.

## License

MIT. See `LICENSE`.
