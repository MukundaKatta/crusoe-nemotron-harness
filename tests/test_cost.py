"""Cost calculator tests."""

import pytest

from crusoe_nemotron_harness.cost import DEFAULT_PRICES, ModelPrice, cost_usd, total_cost


def test_cost_usd_basic_arithmetic() -> None:
    price = ModelPrice(input_per_million=0.50, output_per_million=2.00)
    cost = cost_usd("x", 1_000_000, 1_000_000, prices={"x": price})
    assert cost == pytest.approx(2.50)


def test_cost_usd_with_cache_hits_uses_cached_rate() -> None:
    price = ModelPrice(
        input_per_million=1.0, output_per_million=0.0, cached_input_per_million=0.10
    )
    # 1M input of which 500k are cached. Bill 500k at $1/M + 500k at $0.10/M.
    cost = cost_usd("x", 1_000_000, 0, cached_input_tokens=500_000, prices={"x": price})
    assert cost == pytest.approx(0.55)


def test_cost_usd_unknown_model_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        cost_usd("does-not-exist", 100, 100)


def test_cost_usd_negative_tokens_clamped_to_zero() -> None:
    cost = cost_usd("nemotron-mini-4b-instruct", -50, -50, cached_input_tokens=-10)
    assert cost == 0.0


def test_default_prices_cover_nemotron_models() -> None:
    for model in ("nemotron-mini-4b-instruct", "nemotron-70b-instruct", "nemotron-340b-reward"):
        assert model in DEFAULT_PRICES
        assert DEFAULT_PRICES[model].input_per_million > 0


def test_total_cost_sums_records_and_skips_missing_model() -> None:
    records = [
        {"model": "nemotron-mini-4b-instruct", "input_tokens": 1000, "output_tokens": 0},
        {"model": "nemotron-mini-4b-instruct", "input_tokens": 0, "output_tokens": 1000},
        {"input_tokens": 999, "output_tokens": 999},  # missing model, skipped
    ]
    expected = (
        1000 * DEFAULT_PRICES["nemotron-mini-4b-instruct"].input_per_million / 1_000_000
        + 1000 * DEFAULT_PRICES["nemotron-mini-4b-instruct"].output_per_million / 1_000_000
    )
    assert total_cost(records) == pytest.approx(expected)
