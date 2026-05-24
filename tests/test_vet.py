"""Tool-arg vet tests."""

import pytest

from crusoe_nemotron_harness.vet import (
    ToolArgError,
    ToolSpec,
    vet_args,
    vet_or_raise,
)


SPECS = {
    "search": ToolSpec(
        name="search",
        required=("query",),
        types={"query": "str", "limit": "int"},
    ),
    "set_price": ToolSpec(
        name="set_price",
        required=("amount",),
        types={"amount": "float"},
    ),
}


def test_valid_args_pass() -> None:
    result = vet_args("search", {"query": "nemotron", "limit": 10}, SPECS)
    assert result.ok
    assert result.hint is None


def test_missing_required_arg_returns_hint() -> None:
    result = vet_args("search", {"limit": 5}, SPECS)
    assert not result.ok
    assert result.error_code == "missing_arg"
    assert "query" in (result.hint or "")


def test_unknown_tool_lists_available_tools() -> None:
    result = vet_args("unknown_tool", {}, SPECS)
    assert not result.ok
    assert result.error_code == "unknown_tool"
    assert "search" in (result.hint or "")


def test_bad_type_int_vs_str_rejected() -> None:
    result = vet_args("search", {"query": 123}, SPECS)
    assert not result.ok
    assert result.error_code == "bad_type"


def test_bool_is_not_int_even_though_isinstance_says_so() -> None:
    result = vet_args("search", {"query": "q", "limit": True}, SPECS)
    assert not result.ok
    assert result.error_code == "bad_type"


def test_int_is_accepted_where_float_is_required() -> None:
    result = vet_args("set_price", {"amount": 5}, SPECS)
    assert result.ok


def test_bool_is_not_float() -> None:
    result = vet_args("set_price", {"amount": True}, SPECS)
    assert not result.ok
    assert result.error_code == "bad_type"


def test_extra_optional_args_tolerated() -> None:
    # An undeclared arg should not flunk the call.
    result = vet_args("search", {"query": "q", "trace_id": "abc"}, SPECS)
    assert result.ok


def test_vet_or_raise_throws_on_failure() -> None:
    with pytest.raises(ToolArgError) as exc:
        vet_or_raise("search", {}, SPECS)
    assert exc.value.error_code == "missing_arg"
    assert "search" in str(exc.value)
