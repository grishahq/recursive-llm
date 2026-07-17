"""Tests for structured completion results and trajectories."""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from rlm import CompletionResult, MaxIterationsError, RLM, TrajectoryEvent


class MockResponse:
    """Minimal LiteLLM-compatible response."""

    def __init__(self, content):
        self.choices = [MagicMock(message=MagicMock(content=content))]
        self.usage = None
        self._hidden_params = {}


@pytest.mark.asyncio
async def test_structured_result_contains_the_full_recursive_tree() -> None:
    """Root, child RLM, and boundary leaf calls must share one trajectory."""
    responses = [
        MockResponse("outer = rlm_query('Child task', context)"),
        MockResponse("inner = rlm_query('Leaf task', context)"),
        MockResponse("leaf answer"),
        MockResponse("FINAL_VAR(inner)"),
        MockResponse("FINAL_VAR(outer)"),
    ]
    with patch("rlm.core.litellm.acompletion", side_effect=responses):
        result = await RLM(model="root", recursive_model="recursive", max_depth=2).acomplete_result(
            "Test", "Sensitive context"
        )

    assert isinstance(result, CompletionResult)
    assert result.answer == "leaf answer"
    assert result.stats["llm_calls"] == 5
    assert result.stats["max_depth_reached"] == 2

    rlm_starts = [event for event in result.trajectory if event.kind == "rlm_start"]
    assert [event.depth for event in rlm_starts] == [0, 1]
    assert rlm_starts[1].parent_id == rlm_starts[0].node_id

    model_starts = [event for event in result.trajectory if event.kind == "model_call_start"]
    assert [event.depth for event in model_starts] == [0, 1, 2, 1, 0]
    leaf = next(event for event in model_starts if event.data["is_leaf"])
    assert leaf.parent_id == rlm_starts[1].node_id
    assert [event.sequence for event in result.trajectory] == list(
        range(1, len(result.trajectory) + 1)
    )
    assert result.trajectory[0].kind == "run_start"
    assert result.trajectory[-1].kind == "run_end"
    json.dumps(result.to_dict())


@pytest.mark.asyncio
async def test_trajectory_redacts_content_by_default() -> None:
    """Diagnostics should expose lengths, not user or model content, by default."""
    with patch(
        "rlm.core.litellm.acompletion",
        side_effect=[MockResponse("x = context[:3]"), MockResponse("FINAL_VAR(x)")],
    ):
        result = await RLM(model="test-model").acomplete_result(
            "Sensitive query", "Sensitive context"
        )

    data_keys = {key for event in result.trajectory for key in event.data}
    assert "query" not in data_keys
    assert "context" not in data_keys
    assert "messages" not in data_keys
    assert "response" not in data_keys
    assert "code" not in data_keys
    assert "output" not in data_keys
    assert {"query_chars", "context_chars", "response_chars", "code_chars"} <= data_keys


@pytest.mark.asyncio
async def test_jsonl_export_is_versioned_redacted_and_appendable(tmp_path) -> None:
    """Each exported line should be a complete, safe, versioned run record."""
    with patch(
        "rlm.core.litellm.acompletion",
        side_effect=[MockResponse("x = context[:3]"), MockResponse('FINAL("answer")')],
    ):
        result = await RLM(model="test-model").acomplete_result(
            "Sensitive query", "Sensitive context"
        )

    output = tmp_path / "runs.jsonl"
    result.write_jsonl(output)
    result.write_jsonl(output)

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    record = records[0]
    assert record["schema_version"] == 1
    assert record["termination_reason"] == "completed"
    assert record["answer"] == "answer"
    assert record["config"]["model"] == "test-model"
    assert record["config"]["capture_trajectory_content"] is False
    assert record["config"]["final_answer_validator"] is False
    serialized = json.dumps(record)
    assert "Sensitive query" not in serialized
    assert "Sensitive context" not in serialized


@pytest.mark.asyncio
async def test_jsonl_export_can_replace_an_existing_file(tmp_path) -> None:
    """Explicit replacement should produce exactly one JSONL record."""
    with patch("rlm.core.litellm.acompletion", return_value=MockResponse('FINAL("answer")')):
        result = await RLM(model="test-model").acomplete_result("Test", "Context")

    output = tmp_path / "runs.jsonl"
    output.write_text("old record\n", encoding="utf-8")
    result.write_jsonl(output, append=False)

    records = output.read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
    assert json.loads(records[0])["answer"] == "answer"


@pytest.mark.asyncio
async def test_content_capture_and_event_handler_are_explicit_opt_ins() -> None:
    """Opt-in diagnostics should include payloads and stream every event."""
    streamed = []

    def handler(event: TrajectoryEvent) -> None:
        streamed.append(event)

    with patch(
        "rlm.core.litellm.acompletion",
        side_effect=[MockResponse("x = context[:3]"), MockResponse("FINAL_VAR(x)")],
    ):
        result = await RLM(
            model="test-model",
            capture_trajectory_content=True,
            event_handler=handler,
        ).acomplete_result("Query", "Context")

    assert [event.sequence for event in streamed] == [event.sequence for event in result.trajectory]
    run_start = result.trajectory[0]
    assert run_start.data["query"] == "Query"
    assert run_start.data["context"] == "Context"
    model_end = next(event for event in result.trajectory if event.kind == "model_call_end")
    assert model_end.data["response"] == "x = context[:3]"
    repl_step = next(event for event in result.trajectory if event.kind == "repl_step")
    assert repl_step.data["code"] == "x = context[:3]"


@pytest.mark.asyncio
async def test_handler_failure_does_not_change_model_completion() -> None:
    """Observability callbacks are best-effort and cannot fail a completion."""

    def failing_handler(_event: TrajectoryEvent) -> None:
        raise RuntimeError("logging backend unavailable")

    with patch("rlm.core.litellm.acompletion", return_value=MockResponse('FINAL("answer")')):
        result = await RLM(model="test-model", event_handler=failing_handler).acomplete_result(
            "Test", "Context"
        )

    assert result.answer == "answer"
    assert result.trajectory[-1].kind == "run_end"


@pytest.mark.asyncio
async def test_latest_trajectory_includes_failed_run_events() -> None:
    """A failed run remains inspectable after its exception is raised."""
    rlm = RLM(model="test-model", max_iterations=1, capture_trajectory_content=True)

    with patch("rlm.core.litellm.acompletion", return_value=MockResponse("print(context[:3])")):
        with pytest.raises(MaxIterationsError):
            await rlm.acomplete_result("Test", "Sensitive context")

    assert rlm.trajectory[-1].kind == "run_error"
    repl_step = next(event for event in rlm.trajectory if event.kind == "repl_step")
    assert repl_step.data["code"] == "print(context[:3])"


@pytest.mark.asyncio
async def test_concurrent_structured_results_have_exact_per_run_stats() -> None:
    """Structured results remove ambiguity from the latest-run stats property."""

    async def completion(*, messages, **_kwargs):
        await asyncio.sleep(0.01)
        if len(messages) == 2:
            return MockResponse("value = query")
        return MockResponse("FINAL_VAR(value)")

    with patch("rlm.core.litellm.acompletion", side_effect=completion):
        rlm = RLM(model="test-model")
        results = await asyncio.gather(
            rlm.acomplete_result("first", "Context"),
            rlm.acomplete_result("second", "Context"),
        )

    assert [result.answer for result in results] == ["first", "second"]
    assert [result.stats["llm_calls"] for result in results] == [2, 2]
    assert all(result.trajectory[0].kind == "run_start" for result in results)


def test_sync_structured_result_api_preserves_string_api() -> None:
    """The structured API is additive and the existing API still returns a string."""
    with patch("rlm.core.litellm.acompletion", return_value=MockResponse('FINAL("answer")')):
        rlm = RLM(model="test-model")
        result = rlm.complete_result("Test", "Context")
        answer = rlm.complete("Test", "Context")

    assert result.answer == "answer"
    assert answer == "answer"
