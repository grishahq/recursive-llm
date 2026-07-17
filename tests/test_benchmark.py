"""Tests for deterministic benchmark grading and aggregation."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.compare_same_model import (
    FULL_TASKS,
    SMOKE_TASKS,
    ValidationResult,
    _nearest_rank,
    _write_jsonl,
    aggregate_results,
    default_run_limits,
    run_task,
    validate_division_rollup,
    validate_incident_filter,
    validate_largest_value,
)
from rlm import CompletionResult, FailedCompletionResult


@pytest.mark.parametrize(
    ("answer", "passed"),
    [
        ("ITEM-B has value 19", True),
        ("ITEM-B has value 19.", True),
        ("ITEM-B has value 19.0.", True),
        ("ITEM-B has value 19.5", False),
        ("ITEM-B has value 190", False),
        ("ITEM-A has value 19", False),
    ],
)
def test_largest_value_validator_uses_numeric_boundaries(answer, passed):
    """Identifiers and numbers must not pass through accidental substrings."""
    assert validate_largest_value(answer).passed is passed


def test_division_validator_accepts_decimal_formatting() -> None:
    """Equivalent trailing-zero formatting should remain valid."""
    assert validate_division_rollup("Profit 514.0; DIV-J has 4.80% defects").passed


def test_division_task_explicitly_requests_every_graded_field() -> None:
    """A grader must not require information omitted from the task contract."""
    division = next(task for task in FULL_TASKS if task.name == "division_rollup")
    assert "and that exact percentage" in division.query


def test_incident_validator_requires_an_explicit_count() -> None:
    """The digit in INC-003 must not masquerade as the requested count."""
    ids_only = "INC-003, INC-007, INC-009"
    validation = validate_incident_filter(ids_only)

    assert not validation.passed
    assert "missing explicit incident count 3" in validation.failures
    assert validate_incident_filter(f"{ids_only}; count: 3").passed


def test_incident_task_explicitly_requests_the_graded_count_label() -> None:
    """The count-label grader must match both RLM and direct task contracts."""
    incident = next(task for task in FULL_TASKS if task.name == "incident_filter")
    assert "Count: <integer>" in incident.query
    assert incident.direct_query is not None
    assert "Count: <integer>" in incident.direct_query


def test_incident_validator_rejects_extra_ids_even_with_correct_count() -> None:
    """Citation precision is part of the task contract."""
    answer = "INC-003, INC-007, INC-009, INC-010; count: 3"
    assert not validate_incident_filter(answer).passed


def test_nearest_rank_handles_small_repeated_samples() -> None:
    """p95 must remain deterministic for the small run counts used locally."""
    assert _nearest_rank([], 0.95) is None
    assert _nearest_rank([2.0], 0.95) == 2.0
    assert _nearest_rank([1.0, 3.0, 2.0], 0.95) == 3.0


def _result(task: str, passed: bool, elapsed: float, calls: int, tokens: int, cost: float):
    return {
        "task": task,
        "passed": passed,
        "elapsed_seconds": elapsed,
        "stats": {
            "llm_calls": calls,
            "total_tokens": tokens,
            "estimated_cost_usd": cost,
        },
    }


def test_aggregate_results_reports_quality_latency_usage_and_cost() -> None:
    """Repeated runs should produce comparable aggregate metrics."""
    results = [
        _result("one", True, 1.0, 2, 100, 0.01),
        _result("one", False, 3.0, 4, 200, 0.03),
        _result("two", True, 2.0, 3, 150, 0.02),
    ]

    summary = aggregate_results("model", results, max_depth=2)

    assert summary["pass_rate"] == pytest.approx(2 / 3, abs=0.0001)
    assert summary["latency_p50_seconds"] == 2.0
    assert summary["latency_p95_seconds"] == 3.0
    assert summary["llm_calls"] == 9
    assert summary["total_tokens"] == 450
    assert summary["estimated_cost_usd"] == pytest.approx(0.06)
    assert summary["per_task"]["one"]["pass_rate"] == 0.5


def test_jsonl_writer_emits_one_parseable_record_per_line(tmp_path: Path) -> None:
    """Raw artifacts should be easy to diff and process incrementally."""
    path = tmp_path / "nested" / "results.jsonl"
    records = [{"record_type": "result", "passed": True}, {"record_type": "summary"}]

    _write_jsonl(path, records)

    assert [json.loads(line) for line in path.read_text().splitlines()] == records


def test_run_task_uses_structured_result_and_task_specific_grader() -> None:
    """The live runner must not fall back to substring grading."""

    class FakeRLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.stats = {"llm_calls": 1, "estimated_cost_usd": 0.0}

        def try_complete_result(self, **_kwargs):
            return CompletionResult(
                answer="INC-003, INC-007, INC-009",
                stats=self.stats,
                trajectory=(),
            )

    incident = next(task for task in FULL_TASKS if task.name == "incident_filter")
    with patch("benchmarks.compare_same_model.RLM", FakeRLM):
        result = run_task(
            "model",
            incident,
            run_index=1,
            max_depth=2,
            max_iterations=4,
            max_tokens=100,
            max_total_calls=8,
            max_elapsed_seconds=30,
        )

    assert not result["passed"]
    assert result["validation_failures"] == ["missing explicit incident count 3"]


def test_run_task_serializes_a_structured_failure() -> None:
    """Expected RLM failures should become benchmark records without an exception fallback."""

    class FakeRLM:
        def __init__(self, **_kwargs):
            self.stats = {"llm_calls": 99, "estimated_cost_usd": 99.0}

        def try_complete_result(self, **_kwargs):
            return FailedCompletionResult(
                error_type="MaxIterationsError",
                error="limit reached",
                stats={"llm_calls": 3, "estimated_cost_usd": 0.1},
                trajectory=(),
            )

    with patch("benchmarks.compare_same_model.RLM", FakeRLM):
        result = run_task(
            "model",
            SMOKE_TASKS[0],
            run_index=1,
            max_depth=2,
            max_iterations=4,
            max_tokens=100,
            max_total_calls=8,
            max_elapsed_seconds=30,
        )

    assert result["error"] == "MaxIterationsError: limit reached"
    assert result["stats"] == {"llm_calls": 3, "estimated_cost_usd": 0.1}


def test_direct_mode_is_a_single_long_context_call() -> None:
    """The direct baseline must bypass the REPL and expose the full context once."""
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="ITEM-B has value 19"))]
    response.usage = {"prompt_tokens": 50, "completion_tokens": 8, "total_tokens": 58}
    response._hidden_params = {"response_cost": 0.001}

    with patch("benchmarks.compare_same_model.litellm.completion", return_value=response) as call:
        result = run_task(
            "model",
            SMOKE_TASKS[0],
            run_index=1,
            max_depth=2,
            max_iterations=4,
            max_tokens=100,
            max_total_calls=8,
            max_elapsed_seconds=30,
            mode="direct",
        )

    assert result["passed"]
    assert result["mode"] == "direct"
    assert result["stats"]["llm_calls"] == 1
    direct_prompt = call.call_args.kwargs["messages"][1]["content"]
    assert "ITEM-C value=11" in direct_prompt
    assert "recursive_llm" not in direct_prompt


def test_run_task_rejects_overrides_of_controlled_options() -> None:
    """Benchmark variants must not silently change common run limits."""
    with pytest.raises(ValueError, match="benchmark-controlled options: max_depth"):
        run_task(
            "model",
            SMOKE_TASKS[0],
            run_index=1,
            max_depth=2,
            max_iterations=4,
            max_tokens=1_000,
            max_total_calls=24,
            max_elapsed_seconds=300,
            rlm_options={"max_depth": 3},
        )


def test_validation_result_is_immutable_and_tuple_backed() -> None:
    """Grader output should have a stable serialization-friendly shape."""
    result = ValidationResult(False, ("failure",))
    assert result.failures == ("failure",)


def test_long_context_suite_gets_a_larger_iteration_budget() -> None:
    """Long deterministic aggregation should not inherit the small smoke cutoff."""
    assert default_run_limits(generated=False, full=False) == (4, 1_000)
    assert default_run_limits(generated=False, full=True) == (6, 4_000)
    assert default_run_limits(generated=True, full=False) == (10, 4_000)
