"""Egress allowlist for tools the Nemotron agent can fetch.

Mirrors the shape of the author's agentguard library. Declare a list of
allowed domains up front. Any tool that tries to fetch a URL outside that
allowlist raises EgressDenied with a clear message and the offending host.

This module is deliberately host-only: we do not validate scheme, path, or
port. The contract is "you can hit this host or you cannot", which is the
trust boundary most production agents actually want enforced.

Wildcard suffix matching is supported via a leading "*.": "*.example.com"
matches "api.example.com" and "a.b.example.com" but not "example.com" itself.
Add the bare host as a separate entry if you want both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse


class EgressDenied(Exception):
    """Raised when a tool tries to call a host outside the allowlist."""

    def __init__(self, host: str, allowed: tuple[str, ...]):
        self.host = host
        self.allowed = allowed
        super().__init__(
            f"Egress denied: host '{host}' is not in the allowlist. "
            f"Allowed hosts: {list(allowed)}."
        )


@dataclass(frozen=True)
class EgressPolicy:
    """Frozen allowlist of hosts the agent's tools may fetch.

    Stored as a sorted tuple so the policy hashes cleanly and shows up in
    snapshots and logs in the same order every run.
    """

    allowed: tuple[str, ...]

    @classmethod
    def from_iterable(cls, hosts: Iterable[str]) -> "EgressPolicy":
        cleaned = tuple(sorted({h.strip().lower() for h in hosts if h and h.strip()}))
        return cls(allowed=cleaned)

    def is_allowed(self, host: str) -> bool:
        host_norm = host.strip().lower()
        if not host_norm:
            return False
        for entry in self.allowed:
            if entry.startswith("*."):
                suffix = entry[1:]  # ".example.com"
                if host_norm.endswith(suffix) and host_norm != suffix.lstrip("."):
                    return True
            elif entry == host_norm:
                return True
        return False

    def check_url(self, url: str) -> str:
        """Return the host if allowed, raise EgressDenied otherwise.

        Returns the parsed host so callers can use it for logging without
        re-parsing the URL.
        """

        host = _extract_host(url)
        if not self.is_allowed(host):
            raise EgressDenied(host, self.allowed)
        return host


def _extract_host(url: str) -> str:
    """Pull the lowercased hostname out of a URL.

    We accept either a full URL (with scheme) or a bare host. Bare hosts pass
    through unchanged so callers do not have to wrap them in fake schemes.
    """

    if "://" not in url:
        # Bare host. Strip any port the caller stuck on the end.
        candidate = url.split("/", 1)[0]
        return candidate.split(":", 1)[0].strip().lower()
    parsed = urlparse(url)
    host = parsed.hostname or ""
    return host.lower()
