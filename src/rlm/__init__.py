"""Recursive Language Models for efficient long-context processing."""

from .budget import RunBudget
from .core import RLM
from .errors import (
    BudgetExceededError,
    MaxDepthError,
    MaxIterationsError,
    ProviderResponseError,
    RLMError,
)
from .repl import REPLError, REPLTimeoutError, WorkerResourceLimits
from .results import CompletionResult, TrajectoryEvent

__version__ = "0.2.0"

__all__ = [
    "RLM",
    "RLMError",
    "MaxIterationsError",
    "MaxDepthError",
    "BudgetExceededError",
    "ProviderResponseError",
    "RunBudget",
    "CompletionResult",
    "TrajectoryEvent",
    "REPLError",
    "REPLTimeoutError",
    "WorkerResourceLimits",
]
