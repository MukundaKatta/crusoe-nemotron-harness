# Deploying against real Crusoe Cloud Managed Inference

The harness ships with `FakeNemotronProvider` so the demo and tests run with no API keys. Below is how you point it at a real Nemotron model on Crusoe Managed Inference.

## Prerequisites

- A Crusoe Cloud account with Managed Inference enabled.
- A Nemotron model provisioned in your project (for example `nemotron-70b-instruct` or `nemotron-mini-4b-instruct`).
- The Managed Inference URL for that model (chat completions endpoint).
- An API key with permission to call inference.

```bash
export CRUSOE_API_KEY="..."
export CRUSOE_INFERENCE_URL="https://<your-crusoe-region>.crusoe.example.com/v1/chat/completions"
```

The exact host and path will match whatever Crusoe Managed Inference shows in the model's deployment view. The URL above is illustrative.

## Wire up a transport

`CrusoeNemotronProvider` does not import `requests` or `httpx` so the package keeps zero runtime dependencies. You supply a tiny transport callable, the provider does the rest.

```python
# crusoe_runtime.py
import requests


def requests_transport(url: str, headers: dict[str, str], body: bytes) -> dict:
    response = requests.post(url, headers=headers, data=body, timeout=60)
    response.raise_for_status()
    return response.json()
```

Any HTTP client works. The function only needs to take `(url, headers, body_bytes)` and return the parsed JSON body.

## Build the provider and harness

```python
import os

from crusoe_nemotron_harness import (
    CrusoeNemotronProvider,
    NemotronHarness,
    ToolSpec,
)
from crusoe_runtime import requests_transport


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
    tools={
        "search": ToolSpec("search", required=("query",), types={"query": "str"}),
    },
)

with harness.run() as ctx:
    out = ctx.complete("Summarize Nemotron in one paragraph.", max_tokens=256)
    print(out.text)
    print(ctx.report().as_dict())
```

That is the whole swap. Everything else (budget, egress, snapshot, vet, trace, cost) is wired through the harness the same way as in the demo.

## Pricing

`cost.py` ships placeholder Nemotron prices pinned in code. Replace `DEFAULT_PRICES` with your real Crusoe Managed Inference contract rates before reporting cost to anyone who pays.

```python
from crusoe_nemotron_harness.cost import ModelPrice

prices = {
    "nemotron-70b-instruct": ModelPrice(
        input_per_million=0.90,
        output_per_million=2.70,
        cached_input_per_million=0.10,
    ),
}
harness = NemotronHarness(provider=provider, prices=prices)
```

## Snapshots in CI

Save the first known-good snapshot into the repo so future runs detect drift:

```bash
CRUSOE_HARNESS_UPDATE=1 .venv/bin/python my_script.py
```

After that, any change to the recorded (prompt, tool_call, response) sequence shows up as a mismatch in CI without `CRUSOE_HARNESS_UPDATE=1`.

## Multiple models

Pass the model explicitly per call. The provider will use it instead of its default:

```python
ctx.complete("classify this", model="nemotron-mini-4b-instruct", max_tokens=32)
ctx.complete("write the brief", model="nemotron-70b-instruct", max_tokens=512)
```

Both calls roll up into the same `RunReport`. Per-model cost math uses each call's own model id.

## What you should not do

- Do not check `CRUSOE_API_KEY` into the repo. The provider already reads it from the environment.
- Do not call real Crusoe inference from the test suite. Tests use `FakeNemotronProvider` on purpose so they stay seed-deterministic and zero-key.
- Do not skip `allowed_hosts`. An empty allowlist denies all egress. If you want to be permissive, list each hostname explicitly so the audit log shows intent.
