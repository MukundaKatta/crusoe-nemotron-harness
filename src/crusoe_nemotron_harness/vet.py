"""Tool-arg validator for Nemotron tool calls.

Mirrors the shape of the author's agentvet library. Each tool registers a
small JSON-shape spec (required keys plus a type-tag per key). `vet_args`
returns a ToolArgResult with either ok=True or an LLM-friendly hint that the
agent can feed back into the next turn.

The point: when Nemotron calls a tool with wrong args, surface that as a
recoverable error and a retry hint instead of a hard crash.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ToolArgError(Exception):
    """Raised when a tool call's args fail validation in strict mode.

    Carries the same hint string a ToolArgResult would carry so the caller can
    pipe it straight back into the model on retry.
    """

    def __init__(self, tool: str, hint: str, error_code: str):
        self.tool = tool
        self.hint = hint
        self.error_code = error_code
        super().__init__(f"[{error_code}] {hint}")


@dataclass(frozen=True)
class ToolSpec:
    """Tool argument spec.

    `required` lists the keys that must be present. `types` maps key -> simple
    type tag ("str", "int", "float", "bool", "list", "dict"). Keys missing
    from `types` skip type-checking. Keys in args that are not declared in
    `types` and not in `required` are tolerated; tools commonly accept extra
    optional args.
    """

    name: str
    required: tuple[str, ...]
    types: dict[str, str]


@dataclass(frozen=True)
class ToolArgResult:
    """Outcome of a single arg-vet call."""

    ok: bool
    hint: str | None = None
    # Closed union: "missing_arg" | "bad_type" | "unknown_tool"
    error_code: str | None = None


_TYPE_CHECKS: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
}


def vet_args(tool: str, args: dict[str, Any], specs: dict[str, ToolSpec]) -> ToolArgResult:
    """Validate a single tool call's args against a registered spec.

    Returns ToolArgResult(ok=True) on success, or a result with an
    LLM-friendly hint string on failure. The hint is plain English so the
    model can read it and retry without help.
    """

    spec = specs.get(tool)
    if spec is None:
        return ToolArgResult(
            ok=False,
            hint=f"Tool '{tool}' is not registered. Available tools: {sorted(specs)}.",
            error_code="unknown_tool",
        )

    missing = [key for key in spec.required if key not in args]
    if missing:
        return ToolArgResult(
            ok=False,
            hint=f"Tool '{tool}' is missing required args: {missing}.",
            error_code="missing_arg",
        )

    for key, type_tag in spec.types.items():
        if key not in args:
            continue
        expected = _TYPE_CHECKS.get(type_tag)
        if expected is None:
            continue
        value = args[key]
        # bool is a subclass of int in Python; treat them as separate types
        # because tool authors almost always mean it that way.
        if type_tag == "int" and isinstance(value, bool):
            return ToolArgResult(
                ok=False,
                hint=f"Tool '{tool}' arg '{key}' should be int, got bool.",
                error_code="bad_type",
            )
        if type_tag == "float" and isinstance(value, bool):
            return ToolArgResult(
                ok=False,
                hint=f"Tool '{tool}' arg '{key}' should be float, got bool.",
                error_code="bad_type",
            )
        # Allow int where float is requested. Reject in the other direction.
        if type_tag == "float" and isinstance(value, int):
            continue
        if not isinstance(value, expected):
            return ToolArgResult(
                ok=False,
                hint=(
                    f"Tool '{tool}' arg '{key}' should be {type_tag}, "
                    f"got {type(value).__name__}."
                ),
                error_code="bad_type",
            )

    return ToolArgResult(ok=True)


def vet_or_raise(tool: str, args: dict[str, Any], specs: dict[str, ToolSpec]) -> None:
    """Strict-mode variant. Raises ToolArgError on any validation failure."""

    result = vet_args(tool, args, specs)
    if not result.ok:
        raise ToolArgError(tool=tool, hint=result.hint or "", error_code=result.error_code or "")
