"""Budget tests."""

import pytest

from crusoe_nemotron_harness.budget import Budget, BudgetExceeded


def test_charge_within_caps_passes() -> None:
    b = Budget(max_tokens=1000, max_usd=1.0)
    status = b.charge(tokens=100, usd=0.01)
    assert status.tokens_used == 100
    assert status.usd_used == pytest.approx(0.01)
    assert not status.aborted


def test_token_overrun_raises() -> None:
    b = Budget(max_tokens=100)
    with pytest.raises(BudgetExceeded) as exc:
        b.charge(tokens=101)
    assert exc.value.kind == "tokens"
    assert b.aborted
    assert b.status().abort_reason == "tokens"


def test_usd_overrun_raises() -> None:
    b = Budget(max_usd=0.10)
    with pytest.raises(BudgetExceeded) as exc:
        b.charge(usd=0.11)
    assert exc.value.kind == "usd"


def test_zero_caps_mean_unlimited() -> None:
    b = Budget()
    status = b.charge(tokens=10_000_000, usd=999.99)
    assert status.tokens_used == 10_000_000
    assert status.tokens_remaining == -1  # sentinel for unlimited
    assert status.usd_remaining == -1.0


def test_negative_charge_rejected() -> None:
    b = Budget()
    with pytest.raises(ValueError):
        b.charge(tokens=-1)
    with pytest.raises(ValueError):
        b.charge(usd=-0.01)


def test_negative_caps_rejected() -> None:
    with pytest.raises(ValueError):
        Budget(max_tokens=-1)
    with pytest.raises(ValueError):
        Budget(max_usd=-0.01)


def test_partial_overrun_does_not_commit() -> None:
    b = Budget(max_tokens=100)
    b.charge(tokens=50)
    with pytest.raises(BudgetExceeded):
        b.charge(tokens=60)
    # The failed charge must not have moved the counter.
    assert b.status().tokens_used == 50
