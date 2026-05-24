"""Token + USD budget cap for a Nemotron run.

Mirrors the shape of the author's token-budget-pool crate. Declare a token
cap and a USD cap up front. Every recorded call charges against both
counters. The first call that would push either counter past its cap raises
BudgetExceeded and aborts the run cleanly.

The harness wires Budget into the facade so callers do not have to remember
to charge it manually for normal `complete(...)` calls. Manual `charge(...)`
is exposed for tools and custom inference paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExceeded(Exception):
    """Raised when a charge would push a counter past its cap."""

    def __init__(self, kind: str, current: float, attempted: float, cap: float):
        self.kind = kind
        self.current = current
        self.attempted = attempted
        self.cap = cap
        super().__init__(
            f"Budget exceeded ({kind}): current={current:.6f} + attempted={attempted:.6f} "
            f"> cap={cap:.6f}."
        )


@dataclass
class BudgetStatus:
    """Snapshot of how much of each cap has been spent."""

    tokens_used: int
    tokens_cap: int
    tokens_remaining: int
    usd_used: float
    usd_cap: float
    usd_remaining: float
    aborted: bool = False
    abort_reason: str | None = None


@dataclass
class Budget:
    """Token + USD cap.

    `max_tokens` and `max_usd` of 0 mean "no cap on this axis". Negative
    values are rejected at construction time.
    """

    max_tokens: int = 0
    max_usd: float = 0.0
    _tokens_used: int = field(default=0, init=False, repr=False)
    _usd_used: float = field(default=0.0, init=False, repr=False)
    _aborted: bool = field(default=False, init=False, repr=False)
    _abort_reason: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")
        if self.max_usd < 0.0:
            raise ValueError("max_usd must be >= 0")

    def charge(self, tokens: int = 0, usd: float = 0.0) -> BudgetStatus:
        """Add `tokens` and `usd` to the running totals.

        Negative arguments are rejected so callers cannot "refund" a budget
        to sneak past a cap.
        """

        if tokens < 0:
            raise ValueError("tokens must be >= 0")
        if usd < 0.0:
            raise ValueError("usd must be >= 0")

        if self.max_tokens > 0 and self._tokens_used + tokens > self.max_tokens:
            self._aborted = True
            self._abort_reason = "tokens"
            raise BudgetExceeded(
                kind="tokens",
                current=float(self._tokens_used),
                attempted=float(tokens),
                cap=float(self.max_tokens),
            )
        if self.max_usd > 0.0 and self._usd_used + usd > self.max_usd:
            self._aborted = True
            self._abort_reason = "usd"
            raise BudgetExceeded(
                kind="usd",
                current=self._usd_used,
                attempted=usd,
                cap=self.max_usd,
            )

        self._tokens_used += tokens
        self._usd_used += usd
        return self.status()

    def status(self) -> BudgetStatus:
        token_remaining = (
            max(self.max_tokens - self._tokens_used, 0) if self.max_tokens > 0 else -1
        )
        usd_remaining = (
            max(self.max_usd - self._usd_used, 0.0) if self.max_usd > 0.0 else -1.0
        )
        return BudgetStatus(
            tokens_used=self._tokens_used,
            tokens_cap=self.max_tokens,
            tokens_remaining=token_remaining,
            usd_used=self._usd_used,
            usd_cap=self.max_usd,
            usd_remaining=usd_remaining,
            aborted=self._aborted,
            abort_reason=self._abort_reason,
        )

    @property
    def aborted(self) -> bool:
        return self._aborted
