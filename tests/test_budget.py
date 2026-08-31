"""Tests for completion-tree budgets."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from rlm import BudgetExceededError, RLM, RunBudget


class MockResponse:
    """Minimal LiteLLM-compatible response."""

    def __init__(self, content, usage=None, response_cost=None):
        self.choices = [MagicMock(message=MagicMock(content=content))]
        self.usage = usage
        self._hidden_params = {}
        if response_cost is not None:
            self._hidden_params["response_cost"] = response_cost


@pytest.mark.asyncio
async def test_call_budget_stops_before_an_extra_provider_request() -> None:
    """A call cap must be exact, not a best-effort post-response limit."""
    with patch("rlm.core.litellm.acompletion") as completion:
        completion.side_effect = [MockResponse("x = 1"), MockResponse("x = 2")]
        rlm = RLM(model="test-model", max_iterations=5, max_total_calls=2)

        with pytest.raises(BudgetExceededError) as raised:
            await rlm.acomplete("Test", "Context")

    assert completion.call_count == 2
    assert raised.value.metric == "llm_calls"
    assert raised.value.stats is not None
    assert raised.value.stats["llm_calls"] == 2
    assert rlm.stats["budget"]["reserved_calls"] == 2


@pytest.mark.asyncio
async def test_call_budget_is_shared_with_repl_subcalls(capfd) -> None:
    """A child call cannot escape the root completion-tree budget."""
    with patch("rlm.core.litellm.acompletion") as completion:
        completion.return_value = MockResponse("result = llm_query('Sub-task', context)")
        rlm = RLM(model="root", recursive_model="leaf", max_depth=1, max_total_calls=1)

        with pytest.raises(BudgetExceededError) as raised:
            await rlm.acomplete("Test", "Context")

    assert completion.call_count == 1
    assert raised.value.metric == "llm_calls"
    assert raised.value.stats is not None
    assert raised.value.stats["root_calls"] == 1
    assert raised.value.stats["leaf_calls"] == 0
    await asyncio.sleep(0.1)
    assert "BrokenPipeError" not in capfd.readouterr().err


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "metric"),
    [
        ({"max_total_tokens": 10}, "total_tokens"),
        ({"max_total_cost_usd": 0.001}, "estimated_cost_usd"),
    ],
)
async def test_usage_budget_stops_after_crossing_response_is_counted(kwargs, metric) -> None:
    """Usage limits retain the response metadata that proves the overrun."""
    response = MockResponse(
        'FINAL("too expensive")',
        usage={"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
        response_cost=0.002,
    )
    with patch("rlm.core.litellm.acompletion", return_value=response):
        rlm = RLM(model="test-model", **kwargs)
        with pytest.raises(BudgetExceededError) as raised:
            await rlm.acomplete("Test", "Context")

    assert raised.value.metric == metric
    assert raised.value.stats is not None
    assert raised.value.stats["total_tokens"] == 11
    assert raised.value.stats["estimated_cost_usd"] == pytest.approx(0.002)


@pytest.mark.asyncio
async def test_call_reservations_are_atomic_under_concurrency() -> None:
    """Concurrent subcalls cannot race past the shared hard call cap."""
    with patch(
        "rlm.core.litellm.acompletion", return_value=MockResponse("leaf answer")
    ) as completion:
        rlm = RLM(model="root", recursive_model="leaf", max_total_calls=2)
        results = await asyncio.gather(
            *(rlm._call_leaf(f"Task {index}") for index in range(6)),
            return_exceptions=True,
        )

    assert completion.call_count == 2
    assert sum(result == "leaf answer" for result in results) == 2
    errors = [result for result in results if isinstance(result, BudgetExceededError)]
    assert len(errors) == 4
    assert all(error.metric == "llm_calls" for error in errors)


@pytest.mark.asyncio
async def test_elapsed_budget_cancels_an_in_flight_provider_request() -> None:
    """The shared deadline must bound a provider coroutine that does not return."""

    async def slow_completion(**_kwargs):
        await asyncio.sleep(5)
        return MockResponse('FINAL("late")')

    with patch("rlm.core.litellm.acompletion", side_effect=slow_completion):
        rlm = RLM(model="test-model", max_elapsed_seconds=0.05)
        started = time.monotonic()
        with pytest.raises(BudgetExceededError) as raised:
            await rlm.acomplete("Test", "Context")

    assert time.monotonic() - started < 0.5
    assert raised.value.metric == "elapsed_seconds"
    assert raised.value.stats is not None
    assert raised.value.stats["llm_calls"] == 1


@pytest.mark.asyncio
async def test_elapsed_budget_rejects_answer_finished_by_late_validator() -> None:
    """A validator result completed after the tree deadline must not be accepted."""

    def slow_validator(_answer: str):
        time.sleep(0.08)
        return None

    with patch("rlm.core.litellm.acompletion", return_value=MockResponse('FINAL("late")')):
        rlm = RLM(
            model="test-model",
            max_elapsed_seconds=0.03,
            final_answer_validator=slow_validator,
        )
        with pytest.raises(BudgetExceededError) as raised:
            await rlm.acomplete("Test", "Context")

    assert raised.value.metric == "elapsed_seconds"
    assert raised.value.stats is not None


@pytest.mark.asyncio
async def test_elapsed_budget_caps_repl_execution_by_remaining_tree_time() -> None:
    """A long local step must use the remaining tree deadline, not only repl_timeout."""
    with patch(
        "rlm.core.litellm.acompletion",
        return_value=MockResponse("while True: pass"),
    ):
        rlm = RLM(
            model="test-model",
            max_iterations=2,
            repl_timeout=2,
            max_elapsed_seconds=0.15,
        )
        started = time.monotonic()
        with pytest.raises(BudgetExceededError) as raised:
            await rlm.acomplete("Test", "Context")

    assert time.monotonic() - started < 1
    assert raised.value.metric == "elapsed_seconds"
    assert raised.value.stats is not None


def test_elapsed_budget_uses_a_monotonic_deadline() -> None:
    """Elapsed time is checked against a monotonic clock."""
    with patch("rlm.budget.time.monotonic", side_effect=[10.0, 11.01]):
        budget = RunBudget(max_elapsed_seconds=1.0)
        with pytest.raises(BudgetExceededError, match="elapsed_seconds"):
            budget.check_deadline()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_total_calls": 0},
        {"max_total_tokens": 0},
        {"max_total_cost_usd": 0},
        {"max_elapsed_seconds": 0},
    ],
)
def test_non_positive_budget_limits_are_rejected(kwargs) -> None:
    """Every configured budget limit must be positive."""
    with pytest.raises(ValueError):
        RLM(model="test-model", **kwargs)
