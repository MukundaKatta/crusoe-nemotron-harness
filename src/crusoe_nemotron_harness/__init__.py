"""crusoe-nemotron-harness: production harness for Nemotron agents on Crusoe.

Wraps any Nemotron provider with cost, egress, tool-arg vet, snapshot, trace,
and budget cap so an agent run produces a single auditable RunReport.

Public surface:
    - NemotronHarness: the facade
    - RunReport, format_report: result type and renderer
    - FakeNemotronProvider, CrusoeNemotronProvider, NemotronProvider, CompletionResult
    - Cost: ModelPrice, cost_usd, total_cost, DEFAULT_PRICES
    - Egress: EgressPolicy, EgressDenied
    - Vet: ToolSpec, ToolArgResult, ToolArgError, vet_args, vet_or_raise
    - Snap: RunSnapshot, SnapshotEvent, SnapshotMismatch
    - Trace: Trace, CallEvent, TraceSummary, percentile
    - Budget: Budget, BudgetStatus, BudgetExceeded
"""

from .budget import Budget, BudgetExceeded, BudgetStatus
from .cost import DEFAULT_PRICES, ModelPrice, cost_usd, total_cost
from .egress import EgressDenied, EgressPolicy
from .harness import NemotronHarness, RunReport, format_report
from .providers import (
    CompletionResult,
    CrusoeNemotronProvider,
    FakeNemotronProvider,
    NemotronProvider,
)
from .snap import RunSnapshot, SnapshotEvent, SnapshotMismatch
from .trace import CallEvent, Trace, TraceSummary, percentile
from .vet import ToolArgError, ToolArgResult, ToolSpec, vet_args, vet_or_raise

__all__ = [
    "Budget",
    "BudgetExceeded",
    "BudgetStatus",
    "CallEvent",
    "CompletionResult",
    "CrusoeNemotronProvider",
    "DEFAULT_PRICES",
    "EgressDenied",
    "EgressPolicy",
    "FakeNemotronProvider",
    "ModelPrice",
    "NemotronHarness",
    "NemotronProvider",
    "RunReport",
    "RunSnapshot",
    "SnapshotEvent",
    "SnapshotMismatch",
    "Trace",
    "TraceSummary",
    "ToolArgError",
    "ToolArgResult",
    "ToolSpec",
    "cost_usd",
    "format_report",
    "percentile",
    "total_cost",
    "vet_args",
    "vet_or_raise",
]

__version__ = "0.1.0"
