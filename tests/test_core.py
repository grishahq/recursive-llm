"""Tests for core RLM."""

import pytest
from unittest.mock import MagicMock, patch
from rlm import RLM, MaxIterationsError, MaxDepthError


class MockResponse:
    """Mock LLM response."""

    def __init__(self, content, usage=None, response_cost=None):
        self.choices = [MagicMock(message=MagicMock(content=content))]
        self.usage = usage
        self._hidden_params = {}
        if response_cost is not None:
            self._hidden_params["response_cost"] = response_cost


@pytest.fixture
def mock_litellm():
    """Mock litellm.acompletion."""
    with patch("rlm.core.litellm.acompletion") as mock:
        yield mock


@pytest.mark.asyncio
async def test_simple_complete(mock_litellm):
    """Test simple complete with FINAL."""
    mock_litellm.return_value = MockResponse('FINAL("The answer")')

    rlm = RLM(model="test-model")
    result = await rlm.acomplete("What is the answer?", "Some context")

    assert result == "The answer"
    assert mock_litellm.called


@pytest.mark.asyncio
async def test_multi_step_complete(mock_litellm):
    """Test multi-step complete."""
    responses = [
        MockResponse("x = context[:10]\nprint(x)"),
        MockResponse('FINAL("Done")'),
    ]
    mock_litellm.side_effect = responses

    rlm = RLM(model="test-model")
    result = await rlm.acomplete("Test", "Hello World Test")

    assert result == "Done"
    assert mock_litellm.call_count == 2


@pytest.mark.asyncio
async def test_max_iterations_error(mock_litellm):
    """Test max iterations exceeded."""
    mock_litellm.return_value = MockResponse("x = 1")  # Never returns FINAL

    rlm = RLM(model="test-model", max_iterations=3)

    with pytest.raises(MaxIterationsError):
        await rlm.acomplete("Test", "Context")


@pytest.mark.asyncio
async def test_max_depth_error(mock_litellm):
    """Test max depth exceeded."""
    rlm = RLM(model="test-model", max_depth=2, _current_depth=2)

    with pytest.raises(MaxDepthError):
        await rlm.acomplete("Test", "Context")


@pytest.mark.asyncio
async def test_final_var(mock_litellm):
    """Test FINAL_VAR extraction."""
    responses = [
        MockResponse('result = "Test Answer"\nprint(result)'),
        MockResponse("FINAL_VAR(result)"),
    ]
    mock_litellm.side_effect = responses

    rlm = RLM(model="test-model")
    result = await rlm.acomplete("Test", "Context")

    assert result == "Test Answer"


@pytest.mark.asyncio
async def test_final_var_does_not_use_deleted_parent_snapshot(mock_litellm):
    """Test that FINAL_VAR reads current worker state instead of a stale snapshot."""
    mock_litellm.side_effect = [
        MockResponse("result = 'old'"),
        MockResponse("del result"),
        MockResponse("FINAL_VAR(result)"),
        MockResponse('FINAL("not stale")'),
    ]
    rlm = RLM(model="test-model")

    assert await rlm.acomplete("Test", "Context") == "not stale"


@pytest.mark.asyncio
async def test_repl_error_handling(mock_litellm):
    """Test REPL error handling."""
    responses = [
        MockResponse("x = 1 / 0"),  # This will cause error
        MockResponse('FINAL("Recovered")'),
    ]
    mock_litellm.side_effect = responses

    rlm = RLM(model="test-model")
    result = await rlm.acomplete("Test", "Context")

    assert result == "Recovered"


@pytest.mark.asyncio
async def test_context_operations(mock_litellm):
    """Test context operations in REPL."""
    responses = [
        MockResponse("first_10 = context[:10]"),
        MockResponse("FINAL_VAR(first_10)"),
    ]
    mock_litellm.side_effect = responses

    rlm = RLM(model="test-model")
    result = await rlm.acomplete("Get first 10 chars", "Hello World Example")

    assert result == "Hello Worl"


def test_sync_complete():
    """Test sync complete wrapper."""
    with patch("rlm.core.litellm.acompletion") as mock:
        mock.return_value = MockResponse('FINAL("Sync result")')

        rlm = RLM(model="test-model")
        result = rlm.complete("Test", "Context")

        assert result == "Sync result"


@pytest.mark.asyncio
async def test_two_models(mock_litellm):
    """Test using different models for root and recursive."""
    mock_litellm.return_value = MockResponse('FINAL("Answer")')

    rlm = RLM(model="expensive-model", recursive_model="cheap-model", _current_depth=0)

    await rlm.acomplete("Test", "Context")

    # First call should use expensive model
    call_args = mock_litellm.call_args_list[0]
    assert call_args[1]["model"] == "expensive-model"


@pytest.mark.asyncio
async def test_stats(mock_litellm):
    """Test statistics tracking."""
    responses = [
        MockResponse("x = 1"),
        MockResponse("y = 2"),
        MockResponse('FINAL("Done")'),
    ]
    mock_litellm.side_effect = responses

    rlm = RLM(model="test-model")
    await rlm.acomplete("Test", "Context")

    stats = rlm.stats
    assert stats["llm_calls"] == 3
    assert stats["root_calls"] == 3
    assert stats["recursive_calls"] == 0
    assert stats["total_iterations"] == 3
    assert stats["max_depth_reached"] == 0
    assert stats["iterations"] == 3
    assert stats["depth"] == 0
    assert stats["by_model"]["test-model"]["calls"] == 3


@pytest.mark.asyncio
async def test_stats_aggregate_recursive_calls(mock_litellm):
    """Test that child RLM calls contribute to root statistics."""
    mock_litellm.side_effect = [
        MockResponse('result = recursive_llm("Sub-task", context)'),
        MockResponse('FINAL("Child answer")'),
        MockResponse("FINAL_VAR(result)"),
    ]

    rlm = RLM(model="root-model", recursive_model="child-model", max_depth=2)
    result = await rlm.acomplete("Test", "Context")

    assert result == "Child answer"
    stats = rlm.stats
    assert stats["llm_calls"] == 3
    assert stats["root_calls"] == 2
    assert stats["recursive_calls"] == 1
    assert stats["leaf_calls"] == 0
    assert stats["total_iterations"] == 3
    assert stats["max_depth_reached"] == 1
    assert stats["by_model"]["root-model"]["calls"] == 2
    assert stats["by_model"]["child-model"]["calls"] == 1


@pytest.mark.asyncio
async def test_depth_zero_has_repl_without_subcalls(mock_litellm):
    """Test the max_depth=0 contract."""
    mock_litellm.side_effect = [
        MockResponse("llm_query('not available')"),
        MockResponse('FINAL("root only")'),
    ]
    rlm = RLM(model="root-model", recursive_model="leaf-model", max_depth=0)

    assert await rlm.acomplete("Test", "Context") == "root only"
    assert [call.kwargs["model"] for call in mock_litellm.call_args_list] == [
        "root-model",
        "root-model",
    ]
    assert rlm.stats["leaf_calls"] == 0
    assert rlm.stats["max_depth_reached"] == 0


@pytest.mark.asyncio
async def test_depth_one_uses_plain_lm_subcall(mock_litellm):
    """Test that max_depth=1 permits a plain LM but no child RLM."""
    mock_litellm.side_effect = [
        MockResponse("result = rlm_query('Sub-task', context)"),
        MockResponse("leaf answer"),
        MockResponse("FINAL_VAR(result)"),
    ]
    rlm = RLM(model="root-model", recursive_model="leaf-model", max_depth=1)

    assert await rlm.acomplete("Test", "Context") == "leaf answer"
    assert [call.kwargs["model"] for call in mock_litellm.call_args_list] == [
        "root-model",
        "leaf-model",
        "root-model",
    ]
    assert rlm.stats["leaf_calls"] == 1
    assert rlm.stats["max_depth_reached"] == 1


@pytest.mark.asyncio
async def test_depth_two_creates_child_then_falls_back_to_leaf(mock_litellm):
    """Test the complete max_depth=2 recursion boundary."""
    mock_litellm.side_effect = [
        MockResponse("outer = rlm_query('Child task', context)"),
        MockResponse("inner = rlm_query('Leaf task', context)"),
        MockResponse("leaf answer"),
        MockResponse("FINAL_VAR(inner)"),
        MockResponse("FINAL_VAR(outer)"),
    ]
    rlm = RLM(model="root-model", recursive_model="recursive-model", max_depth=2)

    assert await rlm.acomplete("Test", "Context") == "leaf answer"
    assert [call.kwargs["model"] for call in mock_litellm.call_args_list] == [
        "root-model",
        "recursive-model",
        "recursive-model",
        "recursive-model",
        "root-model",
    ]
    stats = rlm.stats
    assert stats["leaf_calls"] == 1
    assert stats["max_depth_reached"] == 2


@pytest.mark.asyncio
async def test_answer_object_finishes_without_another_model_call(mock_litellm):
    """Test final-answer publication directly from REPL code."""
    mock_litellm.return_value = MockResponse(
        "answer['content'] = 'published'; answer['ready'] = True"
    )
    rlm = RLM(model="test-model")

    assert await rlm.acomplete("Test", "Context") == "published"
    assert mock_litellm.call_count == 1


@pytest.mark.asyncio
async def test_empty_model_response_gets_explicit_protocol_feedback(mock_litellm):
    """Test recovery from reasoning-only or otherwise empty provider output."""
    mock_litellm.side_effect = [MockResponse(""), MockResponse('FINAL("recovered")')]
    rlm = RLM(model="test-model")

    assert await rlm.acomplete("Test", "Context") == "recovered"
    second_messages = mock_litellm.call_args_list[1].kwargs["messages"]
    assert "response was empty" in second_messages[-1]["content"]


@pytest.mark.asyncio
async def test_root_stats_reset_for_each_completion(mock_litellm):
    """Test that stats describe the latest root completion, not lifetime usage."""
    mock_litellm.side_effect = [
        MockResponse("x = 1"),
        MockResponse('FINAL("first")'),
        MockResponse('FINAL("second")'),
    ]
    rlm = RLM(model="test-model")

    assert await rlm.acomplete("First", "Context") == "first"
    assert rlm.stats["llm_calls"] == 2
    assert await rlm.acomplete("Second", "Context") == "second"
    assert rlm.stats["llm_calls"] == 1


def test_batched_queries_preserve_order_bound_concurrency_and_capture_errors():
    """Test the semantic guarantees of batched subcalls."""
    import threading
    import time

    active = 0
    peak = 0
    lock = threading.Lock()

    def query_fn(query, _context):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            if query == "bad":
                raise RuntimeError("failed")
            time.sleep(0.08 if query == "slow" else 0.01)
            return query.upper()
        finally:
            with lock:
                active -= 1

    rlm = RLM(model="test-model", max_concurrent_subcalls=2)
    batched = rlm._make_batched_query(query_fn)

    assert batched(["slow", "fast", "bad"]) == ["SLOW", "FAST", "Error: failed"]
    assert peak == 2


@pytest.mark.parametrize("max_depth", [-1, -2])
def test_negative_max_depth_is_rejected(max_depth):
    """Test public depth validation."""
    with pytest.raises(ValueError, match="max_depth"):
        RLM(model="test-model", max_depth=max_depth)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iterations": 0},
        {"repl_timeout": 0},
        {"max_output_chars": 0},
        {"max_concurrent_subcalls": 0},
        {"_current_depth": -1},
    ],
)
def test_invalid_runtime_configuration_is_rejected(kwargs):
    """Test validation for execution and concurrency limits."""
    with pytest.raises(ValueError):
        RLM(model="test-model", **kwargs)


@pytest.mark.asyncio
async def test_single_argument_is_treated_as_context(mock_litellm):
    """Test the documented one-argument convenience form."""
    mock_litellm.return_value = MockResponse('FINAL("done")')
    rlm = RLM(model="test-model")

    assert await rlm.acomplete("Embedded task and context") == "done"
    messages = mock_litellm.call_args.kwargs["messages"]
    assert messages[1]["content"] == ""
    assert "25 characters" in messages[0]["content"]


@pytest.mark.asyncio
async def test_complete_works_inside_running_event_loop(mock_litellm):
    """Test the synchronous wrapper's running-loop compatibility path."""
    mock_litellm.return_value = MockResponse('FINAL("threaded")')
    rlm = RLM(model="test-model")

    assert rlm.complete("Test", "Context") == "threaded"


@pytest.mark.asyncio
async def test_leaf_query_without_context_and_with_model_override(mock_litellm):
    """Test direct plain-LM prompt and per-call model selection."""
    mock_litellm.return_value = MockResponse("leaf")
    rlm = RLM(model="root", recursive_model="default-leaf")

    result = await rlm._call_leaf("Question", model="override-leaf")

    assert result == "leaf"
    assert mock_litellm.call_args.kwargs["model"] == "override-leaf"
    assert mock_litellm.call_args.kwargs["messages"][1]["content"] == "Question"


def test_batched_queries_reject_mismatched_contexts():
    """Test batch input shape validation."""
    rlm = RLM(model="test-model")
    batched = rlm._make_batched_query(lambda query, context: query + context)

    with pytest.raises(ValueError, match="same length"):
        batched(["one", "two"], ["context"])


def test_response_without_text_is_rejected():
    """Test provider responses that contain no textual choice content."""
    response = MockResponse(None)

    with pytest.raises(Exception, match="did not contain text"):
        RLM._response_text(response)


@pytest.mark.asyncio
async def test_stats_aggregate_usage_and_cost(mock_litellm):
    """Test token, cache, and cost aggregation from provider responses."""
    usage_one = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "prompt_tokens_details": {"cached_tokens": 40},
    }
    usage_two = {
        "prompt_tokens": 150,
        "completion_tokens": 30,
        "total_tokens": 180,
        "cache_read_input_tokens": 50,
    }
    mock_litellm.side_effect = [
        MockResponse("x = 1", usage=usage_one, response_cost=0.001),
        MockResponse('FINAL("Done")', usage=usage_two, response_cost=0.002),
    ]

    rlm = RLM(model="priced-model")
    await rlm.acomplete("Test", "Context")

    stats = rlm.stats
    assert stats["prompt_tokens"] == 250
    assert stats["completion_tokens"] == 50
    assert stats["total_tokens"] == 300
    assert stats["cached_tokens"] == 90
    assert stats["usage_calls"] == 2
    assert stats["priced_calls"] == 2
    assert stats["estimated_cost_usd"] == pytest.approx(0.003)

    model_stats = stats["by_model"]["priced-model"]
    assert model_stats["prompt_tokens"] == 250
    assert model_stats["completion_tokens"] == 50
    assert model_stats["estimated_cost_usd"] == pytest.approx(0.003)


@pytest.mark.asyncio
async def test_stats_handle_missing_usage_and_price(mock_litellm):
    """Test that providers without usage or pricing metadata remain supported."""
    mock_litellm.return_value = MockResponse('FINAL("Done")')

    rlm = RLM(model="unknown-model")
    await rlm.acomplete("Test", "Context")

    stats = rlm.stats
    assert stats["llm_calls"] == 1
    assert stats["usage_calls"] == 0
    assert stats["priced_calls"] == 0
    assert stats["estimated_cost_usd"] is None


@pytest.mark.asyncio
async def test_stats_use_litellm_cost_fallback(mock_litellm):
    """Test cost calculation when the response has no embedded cost."""
    mock_litellm.return_value = MockResponse('FINAL("Done")')

    with patch("rlm.core.litellm.completion_cost", return_value=0.004):
        rlm = RLM(model="known-model")
        await rlm.acomplete("Test", "Context")

    assert rlm.stats["priced_calls"] == 1
    assert rlm.stats["estimated_cost_usd"] == pytest.approx(0.004)


@pytest.mark.asyncio
async def test_api_base_and_key(mock_litellm):
    """Test API base and key passing."""
    mock_litellm.return_value = MockResponse('FINAL("Answer")')

    rlm = RLM(model="test-model", api_base="http://localhost:8000", api_key="test-key")

    await rlm.acomplete("Test", "Context")

    call_kwargs = mock_litellm.call_args[1]
    assert call_kwargs["api_base"] == "http://localhost:8000"
    assert call_kwargs["api_key"] == "test-key"
