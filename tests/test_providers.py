"""Provider tests for FakeNemotronProvider and CrusoeNemotronProvider."""

import json

import pytest

from crusoe_nemotron_harness.providers import (
    CrusoeNemotronProvider,
    FakeNemotronProvider,
    _parse_chat_response,
)


def test_fake_provider_is_deterministic_for_same_seed() -> None:
    a = FakeNemotronProvider(seed=7)
    b = FakeNemotronProvider(seed=7)
    ra = a.complete("hello", max_tokens=64)
    rb = b.complete("hello", max_tokens=64)
    assert ra.text == rb.text
    assert ra.input_tokens == rb.input_tokens
    assert ra.output_tokens == rb.output_tokens


def test_fake_provider_changes_with_seed() -> None:
    ra = FakeNemotronProvider(seed=1).complete("hello")
    rb = FakeNemotronProvider(seed=2).complete("hello")
    assert ra.text != rb.text


def test_fake_provider_emits_cache_hits_after_warmup() -> None:
    p = FakeNemotronProvider(seed=0, cache_after=2)
    # First 2 calls: no cache. Then cache kicks in.
    for _ in range(2):
        r = p.complete("q")
        assert r.cached_input_tokens == 0
    r3 = p.complete("q")
    assert r3.cached_input_tokens > 0


def test_crusoe_provider_build_request_shape() -> None:
    p = CrusoeNemotronProvider(
        api_key="test-key",
        url="https://inference.crusoe.example.com/v1/chat/completions",
    )
    url, headers, body = p.build_request("hi there", max_tokens=128, temperature=0.2)
    assert url == "https://inference.crusoe.example.com/v1/chat/completions"
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Content-Type"] == "application/json"
    parsed = json.loads(body)
    assert parsed["model"] == "nemotron-70b-instruct"
    assert parsed["messages"] == [{"role": "user", "content": "hi there"}]
    assert parsed["max_tokens"] == 128
    assert parsed["temperature"] == 0.2


def test_crusoe_provider_missing_creds_raises() -> None:
    p = CrusoeNemotronProvider()
    # No api_key, no env var.
    with pytest.raises(RuntimeError, match="API key"):
        p.build_request("hi")


def test_crusoe_provider_missing_url_raises() -> None:
    p = CrusoeNemotronProvider(api_key="abc")
    with pytest.raises(RuntimeError, match="URL"):
        p.build_request("hi")


def test_crusoe_provider_uses_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("CRUSOE_API_KEY", "env-key")
    monkeypatch.setenv("CRUSOE_INFERENCE_URL", "https://env-url.example.com/v1/chat")
    url, headers, _ = CrusoeNemotronProvider().build_request("x")
    assert url == "https://env-url.example.com/v1/chat"
    assert headers["Authorization"] == "Bearer env-key"


def test_crusoe_provider_complete_without_transport_raises() -> None:
    p = CrusoeNemotronProvider(api_key="k", url="https://u.example.com")
    with pytest.raises(NotImplementedError):
        p.complete("hi")


def test_crusoe_provider_complete_with_injected_transport() -> None:
    captured: dict[str, object] = {}

    def transport(url: str, headers: dict[str, str], body: bytes) -> dict[str, object]:
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = body
        return {
            "choices": [{"message": {"content": "nemotron says hi"}}],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "cached_prompt_tokens": 4,
            },
        }

    p = CrusoeNemotronProvider(
        api_key="k", url="https://u.example.com/v1/chat", transport=transport
    )
    result = p.complete("hi")
    assert result.text == "nemotron says hi"
    assert result.input_tokens == 12
    assert result.output_tokens == 8
    assert result.cached_input_tokens == 4
    assert captured["url"] == "https://u.example.com/v1/chat"


def test_parse_chat_response_tolerates_missing_usage() -> None:
    out = _parse_chat_response(
        {"choices": [{"message": {"content": "hi"}}]}, "nemotron-70b-instruct", 42.0
    )
    assert out.text == "hi"
    assert out.input_tokens == 0
    assert out.output_tokens == 0
    assert out.latency_ms == 42.0
