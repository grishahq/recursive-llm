"""Usage statistics shared across an RLM recursion tree."""

from __future__ import annotations

from copy import deepcopy
from threading import Lock
from typing import Any, Dict, Optional


def _get_value(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a mapping or an object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_int(value: Any) -> int:
    """Convert an API usage value to an integer without raising."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


class UsageTracker:
    """Thread-safe aggregate statistics for one root RLM invocation tree."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._llm_calls = 0
        self._root_calls = 0
        self._recursive_calls = 0
        self._total_iterations = 0
        self._max_depth_reached = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._cached_tokens = 0
        self._usage_calls = 0
        self._priced_calls = 0
        self._estimated_cost_usd = 0.0
        self._by_model: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _empty_model_stats() -> Dict[str, Any]:
        return {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "usage_calls": 0,
            "priced_calls": 0,
            "estimated_cost_usd": None,
        }

    def record_iteration(self) -> None:
        """Record one loop iteration at any recursion depth."""
        with self._lock:
            self._total_iterations += 1

    def record_call(self, model: str, depth: int) -> None:
        """Record an attempted model call."""
        with self._lock:
            self._llm_calls += 1
            if depth == 0:
                self._root_calls += 1
            else:
                self._recursive_calls += 1
            self._max_depth_reached = max(self._max_depth_reached, depth)

            model_stats = self._by_model.setdefault(model, self._empty_model_stats())
            model_stats["calls"] += 1

    def record_response(
        self,
        model: str,
        response: Any,
        estimated_cost_usd: Optional[float],
    ) -> None:
        """Aggregate usage and optional cost from a successful response."""
        usage = _get_value(response, "usage")
        prompt_tokens = _as_int(_get_value(usage, "prompt_tokens"))
        completion_tokens = _as_int(_get_value(usage, "completion_tokens"))
        total_tokens = _as_int(_get_value(usage, "total_tokens"))

        prompt_details = _get_value(usage, "prompt_tokens_details")
        cached_tokens = _as_int(_get_value(prompt_details, "cached_tokens"))
        if not cached_tokens:
            cached_tokens = _as_int(_get_value(usage, "cache_read_input_tokens"))

        has_usage = usage is not None

        with self._lock:
            model_stats = self._by_model.setdefault(model, self._empty_model_stats())

            if has_usage:
                self._usage_calls += 1
                self._prompt_tokens += prompt_tokens
                self._completion_tokens += completion_tokens
                self._total_tokens += total_tokens
                self._cached_tokens += cached_tokens

                model_stats["usage_calls"] += 1
                model_stats["prompt_tokens"] += prompt_tokens
                model_stats["completion_tokens"] += completion_tokens
                model_stats["total_tokens"] += total_tokens
                model_stats["cached_tokens"] += cached_tokens

            if estimated_cost_usd is not None:
                self._priced_calls += 1
                self._estimated_cost_usd += estimated_cost_usd
                model_stats["priced_calls"] += 1
                previous_cost = model_stats["estimated_cost_usd"] or 0.0
                model_stats["estimated_cost_usd"] = previous_cost + estimated_cost_usd

    def snapshot(self) -> Dict[str, Any]:
        """Return a detached snapshot safe for callers to modify."""
        with self._lock:
            by_model = deepcopy(self._by_model)
            for model_stats in by_model.values():
                cost = model_stats["estimated_cost_usd"]
                if cost is not None:
                    model_stats["estimated_cost_usd"] = round(cost, 10)

            return {
                "llm_calls": self._llm_calls,
                "root_calls": self._root_calls,
                "recursive_calls": self._recursive_calls,
                "total_iterations": self._total_iterations,
                "max_depth_reached": self._max_depth_reached,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "total_tokens": self._total_tokens,
                "cached_tokens": self._cached_tokens,
                "usage_calls": self._usage_calls,
                "priced_calls": self._priced_calls,
                "estimated_cost_usd": (
                    round(self._estimated_cost_usd, 10) if self._priced_calls else None
                ),
                "by_model": by_model,
            }
