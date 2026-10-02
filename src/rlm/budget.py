"""Thread-safe budgets shared by an entire RLM completion tree."""

from __future__ import annotations

import time
from threading import Lock
from typing import Any, Dict, Optional

from .errors import BudgetExceededError


class RunBudget:
    """Track hard call limits and best-effort usage, cost, and time limits.

    Call limits are checked before a provider request. Token and cost limits rely
    on provider response metadata, so the response that crosses a limit is still
    counted before the recursion tree is stopped.
    """

    def __init__(
        self,
        *,
        max_calls: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_cost_usd: Optional[float] = None,
        max_elapsed_seconds: Optional[float] = None,
    ) -> None:
        self._validate_positive("max_calls", max_calls)
        self._validate_positive("max_tokens", max_tokens)
        self._validate_positive("max_cost_usd", max_cost_usd)
        self._validate_positive("max_elapsed_seconds", max_elapsed_seconds)
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.max_cost_usd = max_cost_usd
        self.max_elapsed_seconds = max_elapsed_seconds
        self._started_at = time.monotonic()
        self._calls = 0
        self._observed_tokens = 0
        self._observed_cost_usd: Optional[float] = None
        self._exhausted: Optional[tuple[str, float, float]] = None
        self._lock = Lock()

    @staticmethod
    def _validate_positive(name: str, value: Optional[float]) -> None:
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be greater than zero when provided")

    def reserve_call(self) -> None:
        """Reserve one provider call atomically before it starts."""
        with self._lock:
            self._check_exhausted_locked()
            self._check_deadline_locked()
            next_calls = self._calls + 1
            if self.max_calls is not None and next_calls > self.max_calls:
                self._exceed_locked("llm_calls", self.max_calls, next_calls)
            self._calls = next_calls

    def record_usage(self, total_tokens: int, estimated_cost_usd: Optional[float]) -> None:
        """Record aggregate observed usage and stop when a limit was crossed."""
        with self._lock:
            self._observed_tokens = total_tokens
            self._observed_cost_usd = estimated_cost_usd
            self._check_exhausted_locked()
            if self.max_tokens is not None and total_tokens > self.max_tokens:
                self._exceed_locked("total_tokens", self.max_tokens, total_tokens)
            if (
                self.max_cost_usd is not None
                and estimated_cost_usd is not None
                and estimated_cost_usd > self.max_cost_usd
            ):
                self._exceed_locked("estimated_cost_usd", self.max_cost_usd, estimated_cost_usd)
            self._check_deadline_locked()

    def check_deadline(self) -> None:
        """Raise when the elapsed-time limit has been exhausted."""
        with self._lock:
            self._check_exhausted_locked()
            self._check_deadline_locked()

    def remaining_seconds(self) -> Optional[float]:
        """Return time remaining, or ``None`` when no deadline is configured."""
        with self._lock:
            if self.max_elapsed_seconds is None:
                return None
            return max(0.0, self.max_elapsed_seconds - (time.monotonic() - self._started_at))

    def snapshot(self) -> Dict[str, Any]:
        """Return current limits and observed consumption."""
        with self._lock:
            return {
                "max_calls": self.max_calls,
                "max_tokens": self.max_tokens,
                "max_cost_usd": self.max_cost_usd,
                "max_elapsed_seconds": self.max_elapsed_seconds,
                "reserved_calls": self._calls,
                "observed_tokens": self._observed_tokens,
                "observed_cost_usd": self._observed_cost_usd,
                "elapsed_seconds": round(time.monotonic() - self._started_at, 6),
            }

    def _check_deadline_locked(self) -> None:
        if self.max_elapsed_seconds is None:
            return
        elapsed = time.monotonic() - self._started_at
        if elapsed >= self.max_elapsed_seconds:
            self._exceed_locked("elapsed_seconds", self.max_elapsed_seconds, elapsed)

    def _check_exhausted_locked(self) -> None:
        if self._exhausted is not None:
            raise BudgetExceededError(*self._exhausted)

    def _exceed_locked(self, metric: str, limit: float, observed: float) -> None:
        """Latch exhaustion so later requests cannot spend a failed run's budget."""
        self._exhausted = (metric, limit, observed)
        raise BudgetExceededError(metric, limit, observed)
