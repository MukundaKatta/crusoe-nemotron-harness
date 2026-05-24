"""Jest-style run snapshots for Nemotron agents.

Mirrors the shape of the author's agentsnap library. The harness records the
ordered sequence of (kind, payload) events for a run and diffs the result
against a saved fixture on disk.

Pass `CRUSOE_HARNESS_UPDATE=1` (env var) or `update=True` (kwarg) to
overwrite the fixture with the current run. Default is read-only: if the
fixture is missing the first call writes it, subsequent calls assert.

Snapshots are JSON. Big numbers (latency, tokens) are intentionally NOT
included so the snapshot stays stable across machines. The trace module
handles those numbers separately.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_UPDATE_ENV = "CRUSOE_HARNESS_UPDATE"


@dataclass
class SnapshotEvent:
    """One event in a snapshot. Stable fields only.

    `kind` is "prompt", "tool_call", or "response".
    `payload` is a small dict the snap module trusts the caller to keep stable
    (tool name, args, response text, etc.). Numbers that vary across machines
    should NOT go here; put them in the trace.
    """

    kind: str
    payload: dict[str, Any]


@dataclass
class SnapshotMismatch:
    """What `compare` returns when a snapshot diverges from the fixture."""

    ok: bool
    diff: str
    expected: list[dict[str, Any]] | None
    actual: list[dict[str, Any]]


@dataclass
class RunSnapshot:
    """Ordered event log for a single run."""

    events: list[SnapshotEvent] = field(default_factory=list)

    def record(self, kind: str, payload: dict[str, Any]) -> None:
        self.events.append(SnapshotEvent(kind=kind, payload=payload))

    def to_serializable(self) -> list[dict[str, Any]]:
        return [asdict(event) for event in self.events]

    def compare(self, fixture_path: str | os.PathLike, update: bool | None = None) -> SnapshotMismatch:
        """Compare this snapshot to a fixture on disk.

        Returns SnapshotMismatch(ok=True, diff="") on a clean match. On
        mismatch returns the human-readable diff plus both serialized event
        lists for the caller to log.

        Update mode (write current snapshot to disk):
            - update=True
            - update is None and env var CRUSOE_HARNESS_UPDATE=1
            - fixture file does not exist yet (always write on first run)
        """

        path = Path(fixture_path)
        env_update = os.environ.get(_UPDATE_ENV) == "1"
        should_update = update is True or (update is None and env_update)
        actual = self.to_serializable()

        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(actual, indent=2, sort_keys=True))
            return SnapshotMismatch(ok=True, diff="", expected=None, actual=actual)

        if should_update:
            path.write_text(json.dumps(actual, indent=2, sort_keys=True))
            return SnapshotMismatch(ok=True, diff="", expected=actual, actual=actual)

        expected = json.loads(path.read_text())
        if expected == actual:
            return SnapshotMismatch(ok=True, diff="", expected=expected, actual=actual)

        return SnapshotMismatch(
            ok=False,
            diff=_diff_lines(expected, actual),
            expected=expected,
            actual=actual,
        )


def _diff_lines(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> str:
    """Tiny unified-style diff suitable for surfacing in test output.

    We do not pull in `difflib` because zero runtime deps is the contract.
    `difflib` is stdlib so this restriction is style, not necessity, but the
    output stays small enough that a hand-rolled diff is clearer.
    """

    exp_lines = json.dumps(expected, indent=2, sort_keys=True).splitlines()
    act_lines = json.dumps(actual, indent=2, sort_keys=True).splitlines()
    out: list[str] = []
    max_len = max(len(exp_lines), len(act_lines))
    for i in range(max_len):
        e = exp_lines[i] if i < len(exp_lines) else None
        a = act_lines[i] if i < len(act_lines) else None
        if e == a:
            continue
        if e is not None:
            out.append(f"- {e}")
        if a is not None:
            out.append(f"+ {a}")
    return "\n".join(out)
