"""Recursive Language Models for efficient long-context processing."""

from typing import TYPE_CHECKING, Any

from .budget import RunBudget
from .errors import (
    BudgetExceededError,
    MaxDepthError,
    MaxIterationsError,
    ProviderResponseError,
    RLMError,
)
from .repl import REPLError, REPLTimeoutError, WorkerResourceLimits
from .results import CompletionResult, FailedCompletionResult, RunResult, TrajectoryEvent

if TYPE_CHECKING:
    from .core import RLM as RLM


def __getattr__(name: str) -> Any:
    """Load the model engine only when requested, keeping REPL workers light."""
    if name == "RLM":
        from .core import RLM

        globals()[name] = RLM
        return RLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__version__ = "0.5.0"

__all__ = [
    "RLM",
    "RLMError",
    "MaxIterationsError",
    "MaxDepthError",
    "BudgetExceededError",
    "ProviderResponseError",
    "RunBudget",
    "CompletionResult",
    "FailedCompletionResult",
    "RunResult",
    "TrajectoryEvent",
    "REPLError",
    "REPLTimeoutError",
    "WorkerResourceLimits",
]
