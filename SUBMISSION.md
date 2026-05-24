# crusoe-nemotron-harness: DevNetwork [AI+ML] Hackathon 2026 submission

Track: Crusoe sponsor track. "Your Harness, Our Inference: Build a Nemotron Agent."
Author: Mukunda Katta
Repo: https://github.com/MukundaKatta/crusoe-nemotron-harness

## The problem

You can stand up a Nemotron agent on Crusoe Managed Inference in an afternoon. What you cannot do in an afternoon is answer the questions a real owner has after the first ten production runs.

- How much did each run actually cost?
- Did any tool inside the agent reach a host I never approved?
- How often did the model hand a tool the wrong args?
- Did the agent's outputs drift from a saved baseline?
- What were p50 and p95 latency on this build?
- Did the run stay under the per-customer cap I funded?

Today these answers come from grepping logs, copying numbers into a spreadsheet, and squinting. They live in five different tools. None of them know what a "run" is for your agent.

crusoe-nemotron-harness is a single harness that wraps any Nemotron provider on Crusoe and produces one `RunReport` with all of those numbers.

## The approach

A judge can drop one object around any Nemotron provider:

```python
with NemotronHarness(provider=provider, max_usd_per_run=1.00,
                     allowed_hosts=["api.example.com"]).run() as ctx:
    ctx.complete("Summarize Nemotron in one paragraph.")
    ctx.call_tool("search", {"query": "..."}, latency_ms=15)
    ctx.fetch_url("https://api.example.com/v1/data")
    print(ctx.report().as_dict())
```

Inside the facade, six small modules each own one concern:

| Concern              | This repo module | Sibling library shipped earlier         |
| -------------------- | ---------------- | --------------------------------------- |
| `total_cost_usd`     | `cost.py`        | claude-cost, bedrock-cost, bedrock-kit  |
| `allowed_hosts`      | `egress.py`      | agentguard, agentguard-rs, birddog      |
| `tool_failures`      | `vet.py`         | agentvet, agentvet-rs                   |
| `snapshot_events`    | `snap.py`        | agentsnap, agentsnap-rs                 |
| `p50/p95_latency_ms` | `trace.py`       | agenttrace, agenttrace-rs               |
| `tokens_used`        | `budget.py`      | token-budget-pool, token-budget-py, llm-budget-window |

That part matters: the harness is not magic. It is a thin Nemotron-shaped glue layer over building blocks the author has already shipped to PyPI and crates.io under MukundaKatta. The hackathon contribution is the integration that gets all six right at once on Crusoe.

## Demo

The repo ships `examples/leaderboard_demo.py`. Same seeded provider, three scenes.

Scene 1, bare provider, no harness:

```
Tasks run:        10
Tokens (approx):  344
Cost:             unknown
Budget cap hit:   unknown
Tool args safe:   unknown
Egress safe:      unknown
Snapshot stable:  unknown
```

Scene 2, the same provider wrapped in NemotronHarness:

```
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
snapshot_events      30
aborted              no
abort_reason         -

  Off-allowlist fetches blocked: 1
  Bad-tool-arg attempts:         0
```

Scene 3, deliberately tight budget, the harness aborts on the fourth call before the run can burn through more tokens:

```
Aborted: Budget exceeded (tokens): current=110 + attempted=42 > cap=120.

RunReport
tokens_used     110 / 120
total_cost_usd  $0.000221
aborted         yes
abort_reason    tokens
```

The fake provider is seed-deterministic, so every judge that runs `examples/leaderboard_demo.py` sees the same numbers. The 60-test suite uses the same seam.

## Why this matters for Crusoe specifically

Crusoe Managed Inference is the path of least resistance to a real Nemotron deployment. What it does not give you out of the box is a sharp, judge-ready answer to "is this run any good, did it stay safe, and is it worth the spend." This harness answers that in one screen.

The `CrusoeNemotronProvider` shape is OpenAI-compatible chat completions, which is what Crusoe Managed Inference serves Nemotron through today. The provider already constructs the wire body the right way; injecting a transport callable is the final step:

```python
provider = CrusoeNemotronProvider(
    model="nemotron-70b-instruct",
    api_key=os.environ["CRUSOE_API_KEY"],
    url=os.environ["CRUSOE_INFERENCE_URL"],
    transport=requests_transport,
)
```

Full transport code is 10 lines and lives in `DEPLOY.md`.

## Deploy story

`DEPLOY.md` walks the full path from a fresh shell to a working Crusoe run:

1. Provision a Nemotron model in Crusoe Managed Inference. Note the endpoint URL.
2. Export `CRUSOE_API_KEY` and `CRUSOE_INFERENCE_URL`.
3. Drop the 10-line `requests_transport` shim into your code.
4. Pass `CrusoeNemotronProvider(transport=requests_transport)` into `NemotronHarness`.
5. Run the same harness API the demo uses. The `RunReport` shape is identical.

## What is in the repo

- `src/crusoe_nemotron_harness/`: 8 small modules (cost, egress, vet, snap, trace, budget, harness, providers).
- `tests/`: 60 passing tests across all six concerns plus the facade.
- `examples/leaderboard_demo.py`: three-scene end-to-end demo.
- `README.md`: quickstart, concern table, and an ASCII RunReport.
- `DEPLOY.md`: the swap from FakeNemotronProvider to live Crusoe Managed Inference.
- `DEMO_SCRIPT.md`: 90-second video script in three shots.

## What I am asking the judges to look at

1. Run `pytest -q`. Sixty tests, under a tenth of a second, no API keys.
2. Run `examples/leaderboard_demo.py`. Three scenes, all numbers reproducible from seed 3.
3. Read `DEPLOY.md`. Confirm the swap from fake to live Crusoe is one provider line plus 10 lines of requests glue.

If those land, the harness is doing exactly what the pitch promised: it is a thin, opinionated, six-in-one production layer around Nemotron agents on Crusoe Managed Inference that a team can keep extending without breaking the public seam.
