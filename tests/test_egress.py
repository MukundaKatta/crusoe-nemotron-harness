"""Egress allowlist tests."""

import pytest

from crusoe_nemotron_harness.egress import EgressDenied, EgressPolicy


def test_exact_host_match_passes() -> None:
    policy = EgressPolicy.from_iterable(["api.example.com"])
    assert policy.is_allowed("api.example.com")
    assert policy.check_url("https://api.example.com/v1/data") == "api.example.com"


def test_unlisted_host_raises_egress_denied() -> None:
    policy = EgressPolicy.from_iterable(["api.example.com"])
    with pytest.raises(EgressDenied) as exc:
        policy.check_url("https://evil.example.com/exfil")
    assert exc.value.host == "evil.example.com"


def test_wildcard_suffix_matches_subdomains_but_not_bare() -> None:
    policy = EgressPolicy.from_iterable(["*.example.com"])
    assert policy.is_allowed("api.example.com")
    assert policy.is_allowed("a.b.example.com")
    # The bare host requires its own entry.
    assert not policy.is_allowed("example.com")


def test_bare_host_input_works_without_scheme() -> None:
    policy = EgressPolicy.from_iterable(["api.example.com"])
    assert policy.check_url("api.example.com") == "api.example.com"


def test_empty_allowlist_denies_everything() -> None:
    policy = EgressPolicy.from_iterable([])
    with pytest.raises(EgressDenied):
        policy.check_url("https://anything.example.com")


def test_case_insensitive_and_whitespace_tolerant() -> None:
    policy = EgressPolicy.from_iterable(["  API.Example.com  "])
    assert policy.is_allowed("api.example.com")
    assert policy.is_allowed("API.EXAMPLE.COM")


def test_policy_allowlist_is_sorted_and_deduped() -> None:
    policy = EgressPolicy.from_iterable(["b.com", "a.com", "b.com"])
    assert policy.allowed == ("a.com", "b.com")
