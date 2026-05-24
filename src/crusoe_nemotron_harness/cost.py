"""Cost calculator for Nemotron models on Crusoe Cloud Managed Inference.

Mirrors the shape of the author's claude-cost and bedrock-cost crates: a small
per-model price table plus a function that turns token counts into dollars.

Default prices are placeholders pinned in code so callers notice missing rates
instead of silently zeroing cost. Replace `DEFAULT_PRICES` with your real
Crusoe contract rates in production.

Nemotron model ids covered out of the box:
    - nemotron-mini-4b-instruct
    - nemotron-70b-instruct
    - nemotron-340b-reward

If you point the harness at a model id we do not know, register a price
explicitly or pass a custom `prices` dict.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    """Per-million-token USD price for a single Nemotron model."""

    input_per_million: float
    output_per_million: float
    cached_input_per_million: float = 0.0


# Defaults are placeholders pinned in code. Swap in real Crusoe Managed
# Inference contract rates before reporting cost numbers to anyone who pays.
DEFAULT_PRICES: dict[str, ModelPrice] = {
    # Small Nemotron, fast path. Placeholder, single-digit cents per million.
    "nemotron-mini-4b-instruct": ModelPrice(0.10, 0.30, 0.01),
    # Mid Nemotron, the workhorse most agents will hit on Crusoe.
    "nemotron-70b-instruct": ModelPrice(0.90, 2.70, 0.10),
    # Reward / judge model. Output dominated. Kept here so judging cost rolls
    # into the same RunReport as generation cost.
    "nemotron-340b-reward": ModelPrice(3.00, 9.00, 0.30),
}


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    prices: dict[str, ModelPrice] | None = None,
) -> float:
    """Compute the USD cost for a single Nemotron call.

    Cached input tokens are billed at the cached rate and excluded from the
    standard input bucket. Unknown models raise KeyError on purpose so callers
    notice missing price entries instead of silently zeroing cost.

    Negative token counts are clamped to zero; we never bill someone for
    negative usage, even if a provider returns garbage.
    """

    table = prices if prices is not None else DEFAULT_PRICES
    price = table[model]
    input_tokens = max(input_tokens, 0)
    output_tokens = max(output_tokens, 0)
    cached_input_tokens = max(cached_input_tokens, 0)
    billable_input = max(input_tokens - cached_input_tokens, 0)
    return (
        billable_input * price.input_per_million / 1_000_000.0
        + cached_input_tokens * price.cached_input_per_million / 1_000_000.0
        + output_tokens * price.output_per_million / 1_000_000.0
    )


def total_cost(records: list[dict], prices: dict[str, ModelPrice] | None = None) -> float:
    """Sum cost across a list of call records.

    Each record needs `model`, `input_tokens`, `output_tokens`, and optionally
    `cached_input_tokens`. Records missing a `model` key are skipped, which is
    handy when a tool-call record gets sent through accidentally.
    """

    total = 0.0
    for record in records:
        model = record.get("model")
        if not model:
            continue
        total += cost_usd(
            model,
            record.get("input_tokens", 0),
            record.get("output_tokens", 0),
            record.get("cached_input_tokens", 0),
            prices,
        )
    return total
