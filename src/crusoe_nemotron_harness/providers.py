"""Nemotron provider implementations.

Two concrete providers ship in the box:

- FakeNemotronProvider: deterministic, seeded, no API key. Used by the demo
  and the test suite so the harness story works offline.
- CrusoeNemotronProvider: documents the real Crusoe Cloud Managed Inference
  HTTP shape. It reads CRUSOE_API_KEY and CRUSOE_INFERENCE_URL from the
  environment and constructs the request body the way Crusoe's Managed
  Inference endpoint expects (OpenAI-compatible chat completions). It does
  not require a live key at import time. Set `transport=` in production to
  inject a real HTTP client.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol


@dataclass
class CompletionResult:
    """What every NemotronProvider returns from `complete`."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


class NemotronProvider(Protocol):
    """Minimal contract any Nemotron provider implements."""

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> CompletionResult: ...


@dataclass
class FakeNemotronProvider:
    """Deterministic stand-in for a real Nemotron call.

    The text it returns is derived from a hash of (seed, model, prompt) so
    the demo and tests get the same output every run. Token counts are
    derived from prompt length so cost math is stable too.
    """

    seed: int = 0
    model: str = "nemotron-70b-instruct"
    base_latency_ms: float = 250.0
    slow_every: int = 5
    cache_after: int = 3
    _calls: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> CompletionResult:
        use_model = model or self.model
        # Per-prompt RNG so reordering tasks does not perturb earlier ones.
        per_call_seed = int(
            hashlib.sha256(f"{self.seed}:{use_model}:{prompt}".encode()).hexdigest()[:8],
            16,
        )
        rng = random.Random(per_call_seed)

        # Token math derived from prompt length so cost numbers stay stable.
        input_tokens = max(10, len(prompt) // 4)
        output_tokens = min(max_tokens, max(10, input_tokens // 2 + rng.randint(0, 30)))

        cache_warm = self._calls >= self.cache_after
        cached_input_tokens = input_tokens // 3 if cache_warm else 0

        is_slow = (self._calls + 1) % self.slow_every == 0
        latency = self.base_latency_ms * (4.0 if is_slow else 1.0) + rng.uniform(-30.0, 60.0)
        latency = max(latency, 1.0)

        # The "text" is a deterministic synthesized response. We avoid putting
        # the raw prompt in so snapshots stay short and clean.
        digest = hashlib.sha256(f"{per_call_seed}:{prompt}".encode()).hexdigest()[:12]
        text = f"nemotron[{use_model}]::{digest}"

        self._calls += 1
        return CompletionResult(
            text=text,
            model=use_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            latency_ms=latency,
            raw={"seed": self.seed, "fake": True},
        )


Transport = Callable[[str, dict[str, str], bytes], dict[str, Any]]
"""Transport callable: (url, headers, body) -> parsed-JSON response.

Production deployments pass a real HTTP client wrapped in this shape. The
provider does not import requests/httpx so the package keeps zero deps.
"""


@dataclass
class CrusoeNemotronProvider:
    """Adapter for Crusoe Cloud Managed Inference.

    Reads CRUSOE_API_KEY and CRUSOE_INFERENCE_URL from the environment unless
    they are passed in explicitly. The request body uses the OpenAI-compatible
    chat-completions shape, which Crusoe Managed Inference supports for
    Nemotron deployments.

    A `transport` callable is injected to do the actual HTTP call. The
    default `transport` raises NotImplementedError so the class is usable
    out of the box for shape inspection without pulling in a network stack.
    """

    model: str = "nemotron-70b-instruct"
    api_key_env: str = "CRUSOE_API_KEY"
    url_env: str = "CRUSOE_INFERENCE_URL"
    api_key: str | None = None
    url: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    transport: Transport | None = None

    def _resolve_auth(self) -> tuple[str, str]:
        key = self.api_key or os.environ.get(self.api_key_env)
        url = self.url or os.environ.get(self.url_env)
        if not key:
            raise RuntimeError(
                f"Crusoe API key not set. Provide api_key= or set ${self.api_key_env}."
            )
        if not url:
            raise RuntimeError(
                f"Crusoe inference URL not set. Provide url= or set ${self.url_env}."
            )
        return key, url

    def build_request(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> tuple[str, dict[str, str], bytes]:
        """Return (url, headers, body) for the Crusoe Managed Inference call.

        Exposed publicly so callers can inspect the wire shape without
        sending a real request, and so unit tests can assert on the body.
        """

        key, url = self._resolve_auth()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        headers.update(self.extra_headers)
        body = json.dumps(
            {
                "model": model or self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        ).encode("utf-8")
        return url, headers, body

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> CompletionResult:
        if self.transport is None:
            raise NotImplementedError(
                "CrusoeNemotronProvider needs a `transport` callable for live calls. "
                "See DEPLOY.md for a 10-line requests-based wiring."
            )
        url, headers, body = self.build_request(
            prompt, model=model, max_tokens=max_tokens, temperature=temperature
        )
        started = time.perf_counter()
        response = self.transport(url, headers, body)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return _parse_chat_response(response, model or self.model, latency_ms)


def _parse_chat_response(
    response: dict[str, Any], model: str, latency_ms: float
) -> CompletionResult:
    """Pull text + usage out of an OpenAI-compatible chat response.

    Tolerates missing usage fields so a partial response from Crusoe still
    produces a CompletionResult instead of a KeyError that kills the run.
    """

    choices = response.get("choices") or []
    text = ""
    if choices:
        first = choices[0]
        msg = first.get("message") or {}
        text = msg.get("content") or first.get("text") or ""

    usage = response.get("usage") or {}
    return CompletionResult(
        text=text,
        model=model,
        input_tokens=int(usage.get("prompt_tokens", 0)),
        output_tokens=int(usage.get("completion_tokens", 0)),
        cached_input_tokens=int(usage.get("cached_prompt_tokens", 0)),
        latency_ms=latency_ms,
        raw=response,
    )


def iter_models(provider: NemotronProvider) -> Iterable[str]:
    """Best-effort model listing.

    Real Crusoe inference URLs publish a `/models` endpoint, but exposing
    that requires the transport, so we just return the provider's default
    model when nothing better is available. Tests rely on this fallback.
    """

    model = getattr(provider, "model", None)
    if model:
        yield model
