"""Tests for provider response hardening and bounded retries."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rlm import BudgetExceededError, ProviderResponseError, RLM


class MockResponse:
    """Minimal LiteLLM-compatible response."""

    def __init__(self, content, usage=None):
        self.choices = [MagicMock(message=MagicMock(content=content))]
        self.usage = usage
        self._hidden_params = {}


class HTTPFailure(RuntimeError):
    """Provider-like error exposing status and response headers."""

    def __init__(self, status_code, *, retry_after=None):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        headers = {} if retry_after is None else {"Retry-After": str(retry_after)}
        self.response = MagicMock(status_code=status_code, headers=headers)


@pytest.mark.asyncio
async def test_retryable_failure_is_retried_and_counted() -> None:
    """Each provider retry must be visible in usage, budgets, and trajectories."""
    with patch(
        "rlm.core.litellm.acompletion",
        side_effect=[HTTPFailure(429), MockResponse('FINAL("done")')],
    ) as completion:
        result = await RLM(
            model="test-model",
            max_retries=1,
            retry_backoff_seconds=0,
        ).acomplete_result("Test", "Context")

    assert result.answer == "done"
    assert completion.call_count == 2
    assert all(call.kwargs["num_retries"] == 0 for call in completion.call_args_list)
    assert result.stats["llm_calls"] == 2
    assert result.stats["retry_calls"] == 1
    assert result.stats["by_model"]["test-model"]["retry_calls"] == 1
    starts = [event for event in result.trajectory if event.kind == "model_call_start"]
    assert [event.data["attempt"] for event in starts] == [1, 2]
    retry = next(event for event in result.trajectory if event.kind == "model_retry")
    assert retry.data["failed_attempt"] == 1
    assert retry.data["next_attempt"] == 2


@pytest.mark.asyncio
async def test_non_retryable_failure_is_not_retried() -> None:
    """Application and authentication-style errors should fail immediately."""
    with patch("rlm.core.litellm.acompletion", side_effect=ValueError("bad request")) as completion:
        with pytest.raises(ValueError, match="bad request"):
            await RLM(model="test-model", max_retries=3).acomplete("Test", "Context")

    assert completion.call_count == 1


@pytest.mark.asyncio
async def test_provider_timeout_is_retryable_within_a_longer_run_deadline() -> None:
    """A provider timeout must not be confused with the tree-wide deadline."""
    with patch(
        "rlm.core.litellm.acompletion",
        side_effect=[asyncio.TimeoutError(), MockResponse('FINAL("done")')],
    ) as completion:
        result = await RLM(
            model="test-model",
            max_retries=1,
            retry_backoff_seconds=0,
            max_elapsed_seconds=5,
        ).acomplete_result("Test", "Context")

    assert result.answer == "done"
    assert completion.call_count == 2
    assert result.stats["retry_calls"] == 1


@pytest.mark.asyncio
async def test_malformed_response_can_use_the_bounded_retry_path() -> None:
    """A transient malformed payload should be repairable when retries are enabled."""
    with patch(
        "rlm.core.litellm.acompletion",
        side_effect=[{"choices": []}, MockResponse('FINAL("done")')],
    ):
        result = await RLM(
            model="test-model",
            max_retries=1,
            retry_backoff_seconds=0,
        ).acomplete_result("Test", "Context")

    assert result.answer == "done"
    assert result.stats["retry_calls"] == 1
    error = next(event for event in result.trajectory if event.kind == "model_call_error")
    assert error.data["error_type"] == ProviderResponseError.__name__
    assert error.data["retrying"] is True


@pytest.mark.asyncio
async def test_null_content_is_normalized_without_a_hidden_retry() -> None:
    """Null content should enter the existing empty-response repair iteration."""
    with patch(
        "rlm.core.litellm.acompletion",
        side_effect=[MockResponse(None), MockResponse('FINAL("done")')],
    ) as completion:
        result = await RLM(model="test-model").acomplete_result("Test", "Context")

    assert result.answer == "done"
    assert completion.call_count == 2
    assert result.stats["retry_calls"] == 0
    normalized = [
        event for event in result.trajectory if event.kind == "provider_response_normalized"
    ]
    assert len(normalized) == 1
    second_messages = completion.call_args_list[1].kwargs["messages"]
    assert "response was empty" in second_messages[-1]["content"]


@pytest.mark.asyncio
async def test_retry_cannot_escape_the_shared_call_budget() -> None:
    """A retry must reserve capacity before making another provider request."""
    with patch(
        "rlm.core.litellm.acompletion",
        side_effect=[HTTPFailure(503), MockResponse('FINAL("unexpected")')],
    ) as completion:
        rlm = RLM(
            model="test-model",
            max_retries=2,
            retry_backoff_seconds=0,
            max_total_calls=1,
        )
        with pytest.raises(BudgetExceededError) as raised:
            await rlm.acomplete("Test", "Context")

    assert completion.call_count == 1
    assert raised.value.metric == "llm_calls"
    assert raised.value.stats is not None
    assert raised.value.stats["llm_calls"] == 1
    assert raised.value.stats["retry_calls"] == 0


@pytest.mark.asyncio
async def test_retry_after_is_respected() -> None:
    """A numeric Retry-After header should override a shorter exponential delay."""
    sleep = AsyncMock()
    with (
        patch("rlm.core.asyncio.sleep", sleep),
        patch(
            "rlm.core.litellm.acompletion",
            side_effect=[HTTPFailure(429, retry_after=2.5), MockResponse('FINAL("done")')],
        ),
    ):
        result = await RLM(
            model="test-model",
            max_retries=1,
            retry_backoff_seconds=0.1,
        ).acomplete_result("Test", "Context")

    assert result.answer == "done"
    sleep.assert_awaited_once_with(2.5)


@pytest.mark.asyncio
async def test_retry_backoff_cannot_cross_the_run_deadline() -> None:
    """A retry that cannot start before the deadline should fail without sleeping."""
    with (
        patch("rlm.core.asyncio.sleep", AsyncMock()) as sleep,
        patch(
            "rlm.core.litellm.acompletion",
            side_effect=HTTPFailure(503, retry_after=10),
        ) as completion,
    ):
        with pytest.raises(BudgetExceededError) as raised:
            await RLM(
                model="test-model",
                max_retries=1,
                max_elapsed_seconds=0.1,
            ).acomplete("Test", "Context")

    assert raised.value.metric == "elapsed_seconds"
    assert completion.call_count == 1
    sleep.assert_not_awaited()


def test_litellm_retry_options_are_rejected() -> None:
    """Hidden LiteLLM retries would invalidate exact provider-call accounting."""
    with pytest.raises(ValueError, match="RLM max_retries"):
        RLM(model="test-model", num_retries=2)


@pytest.mark.asyncio
async def test_per_completion_retry_options_are_rejected() -> None:
    """Per-completion LiteLLM retry overrides must not bypass RLM accounting."""
    with patch("rlm.core.litellm.acompletion") as completion:
        with pytest.raises(ValueError, match="RLM max_retries"):
            await RLM(model="test-model").acomplete("Test", "Context", max_retries=2)

    completion.assert_not_called()


def test_async_cancellation_is_not_retryable() -> None:
    """Caller cancellation should always propagate immediately."""
    assert RLM._is_retryable_error(asyncio.CancelledError()) is False
