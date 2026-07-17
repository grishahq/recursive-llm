"""Repeated live comparison of same-model RLM configurations.

Run from the repository root after configuring provider keys in ``.env``:

    python benchmarks/compare_same_model.py gpt-5-mini --runs 3
    python benchmarks/compare_same_model.py deepseek/deepseek-v4-flash --full --runs 3

The script uses task-specific graders, writes optional JSONL artifacts, and
reports pass rate plus p50/p95 latency. Live runs make paid provider calls.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import litellm
from dotenv import load_dotenv

from rlm import RLM
from rlm.stats import UsageTracker

try:
    from .generated_long_context import generate_long_context
except ImportError:  # Support direct execution from the repository root.
    from generated_long_context import generate_long_context


@dataclass(frozen=True)
class ValidationResult:
    """Deterministic grading result for one model answer."""

    passed: bool
    failures: Tuple[str, ...]


Validator = Callable[[str], ValidationResult]


@dataclass(frozen=True)
class Task:
    """One deterministic RLM evaluation task."""

    name: str
    query: str
    context: str
    validator: Validator
    metadata: Optional[Dict[str, Any]] = None
    direct_query: Optional[str] = None


def _contains_number(answer: str, expected: str) -> bool:
    """Return whether a decimal appears as a standalone numeric value."""
    if "." in expected:
        pattern = rf"(?<![\d.]){re.escape(expected)}(?:0+)?(?!\d)"
    else:
        pattern = rf"(?<![\d.]){re.escape(expected)}(?:\.0+)?(?!\d|\.\d)"
    return re.search(pattern, answer) is not None


def _validate_required_values(
    answer: str, *, identifiers: Sequence[str], numbers: Sequence[str]
) -> ValidationResult:
    failures = [
        f"missing identifier {identifier}"
        for identifier in identifiers
        if re.search(rf"\b{re.escape(identifier)}\b", answer) is None
    ]
    failures.extend(
        f"missing numeric value {number}"
        for number in numbers
        if not _contains_number(answer, number)
    )
    return ValidationResult(not failures, tuple(failures))


def validate_largest_value(answer: str) -> ValidationResult:
    """Grade the exact winner and value for the smoke task."""
    return _validate_required_values(answer, identifiers=("ITEM-B",), numbers=("19",))


def validate_division_rollup(answer: str) -> ValidationResult:
    """Grade the aggregate profit, highest-defect division, and percentage."""
    return _validate_required_values(answer, identifiers=("DIV-J",), numbers=("514", "4.8"))


def validate_incident_filter(answer: str) -> ValidationResult:
    """Grade the exact incident set and an explicitly stated count."""
    expected_ids = {"INC-003", "INC-007", "INC-009"}
    observed_ids = set(re.findall(r"\bINC-\d{3}\b", answer.upper()))
    failures: List[str] = []
    if observed_ids != expected_ids:
        failures.append(
            f"incident IDs differ: expected={sorted(expected_ids)}, "
            f"observed={sorted(observed_ids)}"
        )

    count_patterns = (
        r"\b(?:count|total|number(?:\s+of\s+incidents)?)\s*(?:is|=|:)?\s*(\d+)\b",
        r"\b(\d+)\s+(?:matching\s+)?incidents?\b",
    )
    counts = [
        int(match.group(1))
        for pattern in count_patterns
        for match in re.finditer(pattern, answer, flags=re.IGNORECASE)
    ]
    if 3 not in counts:
        failures.append("missing explicit incident count 3")
    return ValidationResult(not failures, tuple(failures))


SMOKE_TASKS = (
    Task(
        name="largest_value",
        query=(
            "Use recursive_llm exactly once on the full context. Return the ID whose value is "
            "largest and its value."
        ),
        context="""ITEM-A value=7
ITEM-B value=19
ITEM-C value=11""",
        validator=validate_largest_value,
        direct_query="Return the ID whose value is largest and its value.",
    ),
)


FULL_TASKS = (
    Task(
        name="division_rollup",
        query=(
            "Use recursive_llm exactly once to analyze all division records. Verify the child's "
            "evidence and return the total profit across all divisions plus the division with "
            "the highest defect percentage and that exact percentage. Profit is revenue minus "
            "cost."
        ),
        context="""DIV-A revenue=120 cost=80 defect_pct=1.8
DIV-B revenue=95 cost=60 defect_pct=2.1
DIV-C revenue=140 cost=92 defect_pct=1.4
DIV-D revenue=88 cost=51 defect_pct=3.7
DIV-E revenue=155 cost=101 defect_pct=1.2
DIV-F revenue=110 cost=74 defect_pct=2.5
DIV-G revenue=130 cost=89 defect_pct=1.6
DIV-H revenue=102 cost=65 defect_pct=2.9
DIV-I revenue=170 cost=108 defect_pct=1.1
DIV-J revenue=76 cost=45 defect_pct=4.8
DIV-K revenue=145 cost=93 defect_pct=1.5
DIV-L revenue=118 cost=77 defect_pct=2.2""",
        validator=validate_division_rollup,
        direct_query=(
            "Analyze all division records. Return the total profit across all divisions plus "
            "the division with the highest defect percentage and that exact percentage. Profit "
            "is revenue minus cost."
        ),
    ),
    Task(
        name="incident_filter",
        query=(
            "Use recursive_llm exactly once to inspect all incidents. Return the incident IDs "
            "where deploy_api=v3 and cache_mode=legacy, together with the count. Cite only IDs "
            "supported by the context. End with `Count: <integer>`."
        ),
        context="""INC-001 deploy_api=v2 cache_mode=modern latency_ms=180
INC-002 deploy_api=v3 cache_mode=modern latency_ms=220
INC-003 deploy_api=v3 cache_mode=legacy latency_ms=910
INC-004 deploy_api=v2 cache_mode=legacy latency_ms=340
INC-005 deploy_api=v3 cache_mode=modern latency_ms=205
INC-006 deploy_api=v4 cache_mode=legacy latency_ms=290
INC-007 deploy_api=v3 cache_mode=legacy latency_ms=870
INC-008 deploy_api=v2 cache_mode=modern latency_ms=160
INC-009 deploy_api=v3 cache_mode=legacy latency_ms=940
INC-010 deploy_api=v4 cache_mode=modern latency_ms=175""",
        validator=validate_incident_filter,
        direct_query=(
            "Inspect all incidents. Return the incident IDs where deploy_api=v3 and "
            "cache_mode=legacy, together with the count. Cite only matching IDs. End with "
            "`Count: <integer>`."
        ),
    ),
)


def build_generated_task(target_chars: int, seed: int) -> Task:
    """Adapt a reproducible generated corpus to the common benchmark task API."""
    generated = generate_long_context(target_chars=target_chars, seed=seed)

    def validate(answer: str) -> ValidationResult:
        failures = generated.validate(answer)
        return ValidationResult(not failures, failures)

    return Task(
        name="generated_transaction_rollup",
        query=generated.query,
        context=generated.context,
        validator=validate,
        metadata={
            "seed": generated.seed,
            "target_chars": generated.target_chars,
            "actual_chars": generated.actual_chars,
            "sha256": generated.sha256,
            "truth": {
                "count": generated.truth.count,
                "total_amount_cents": generated.truth.total_amount_cents,
                "max_transaction_id": generated.truth.max_transaction_id,
                "max_amount_cents": generated.truth.max_amount_cents,
            },
        },
    )


def run_task(
    model: str,
    task: Task,
    *,
    run_index: int,
    max_depth: int,
    max_iterations: int,
    max_tokens: int,
    max_total_calls: int,
    max_elapsed_seconds: float,
    mode: str = "rlm",
    trace: bool = False,
) -> Dict[str, Any]:
    """Run and deterministically grade one same-model RLM task."""
    if mode not in {"rlm", "direct"}:
        raise ValueError("mode must be 'rlm' or 'direct'")
    rlm: Optional[RLM] = None
    if mode == "rlm":
        rlm = RLM(
            model=model,
            max_depth=max_depth,
            max_iterations=max_iterations,
            max_total_calls=max_total_calls,
            max_elapsed_seconds=max_elapsed_seconds,
            capture_trajectory_content=trace,
            max_retries=0,
            timeout=60,
            max_tokens=max_tokens,
        )

    started = time.perf_counter()
    trajectory: List[Dict[str, Any]] = []
    try:
        if rlm is not None:
            completion = rlm.complete_result(query=task.query, context=task.context)
            answer = completion.answer
            stats = completion.stats
            if trace:
                trajectory = [event.to_dict() for event in completion.trajectory]
        else:
            answer, stats = _run_direct(
                model,
                task,
                max_tokens=max_tokens,
                timeout=min(60.0, max_elapsed_seconds),
            )
        error = None
    except Exception as exc:  # Keep later repetitions running after one failure.
        answer = ""
        stats = rlm.stats if rlm is not None else _empty_direct_stats()
        if trace and rlm is not None:
            trajectory = [event.to_dict() for event in rlm.trajectory]
        error = f"{type(exc).__name__}: {exc}"
    elapsed_seconds = time.perf_counter() - started

    validation = task.validator(answer) if error is None else ValidationResult(False, ())
    result = {
        "record_type": "result",
        "model": model,
        "mode": mode,
        "task": task.name,
        "task_metadata": task.metadata or {},
        "run_index": run_index,
        "passed": error is None and validation.passed,
        "validation_failures": list(validation.failures),
        "answer": answer,
        "error": error,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "stats": stats,
    }
    if trace:
        result["trajectory"] = trajectory
    return result


def _empty_direct_stats() -> Dict[str, Any]:
    """Return a UsageTracker-compatible empty direct-call snapshot."""
    stats = UsageTracker().snapshot()
    stats.update({"iterations": 0, "depth": 0})
    return stats


def _run_direct(
    model: str,
    task: Task,
    *,
    max_tokens: int,
    timeout: float,
) -> Tuple[str, Dict[str, Any]]:
    """Run a one-call long-context baseline through LiteLLM."""
    tracker = UsageTracker()
    tracker.record_call(model, 0)
    response = litellm.completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Answer the task using only the supplied context.",
            },
            {
                "role": "user",
                "content": (
                    f"Task:\n{task.direct_query or task.query}\n\nContext:\n{task.context}"
                ),
            },
        ],
        max_tokens=max_tokens,
        timeout=timeout,
        num_retries=0,
    )
    hidden = getattr(response, "_hidden_params", None)
    cost = hidden.get("response_cost") if isinstance(hidden, dict) else None
    if not isinstance(cost, (int, float)):
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = None
    tracker.record_response(model, response, float(cost) if cost is not None else None)
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("Direct model response did not contain text content")
    stats = tracker.snapshot()
    stats.update({"iterations": 0, "depth": 0})
    return cast(str, content), stats


def _nearest_rank(values: Sequence[float], percentile: float) -> Optional[float]:
    """Return a deterministic nearest-rank percentile for non-empty values."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 3)


def _sum_stat(results: Iterable[Dict[str, Any]], name: str) -> float:
    return sum(float(result["stats"].get(name) or 0) for result in results)


def aggregate_results(
    model: str,
    results: Sequence[Dict[str, Any]],
    *,
    max_depth: int,
    mode: str = "rlm",
) -> Dict[str, Any]:
    """Aggregate repeated runs into quality, latency, usage, and cost metrics."""
    elapsed = [float(result["elapsed_seconds"]) for result in results]
    passed = sum(bool(result["passed"]) for result in results)
    per_task: Dict[str, Dict[str, Any]] = {}
    for task_name in sorted({str(result["task"]) for result in results}):
        task_results = [result for result in results if result["task"] == task_name]
        task_passed = sum(bool(result["passed"]) for result in task_results)
        task_elapsed = [float(result["elapsed_seconds"]) for result in task_results]
        per_task[task_name] = {
            "passed": task_passed,
            "runs": len(task_results),
            "pass_rate": round(task_passed / len(task_results), 4),
            "latency_p50_seconds": round(statistics.median(task_elapsed), 3),
            "latency_p95_seconds": _nearest_rank(task_elapsed, 0.95),
        }

    total_cost = _sum_stat(results, "estimated_cost_usd")
    total_calls = int(_sum_stat(results, "llm_calls"))
    total_tokens = int(_sum_stat(results, "total_tokens"))
    return {
        "record_type": "summary",
        "model": model,
        "mode": mode,
        "max_depth": max_depth,
        "passed": passed,
        "runs": len(results),
        "pass_rate": round(passed / len(results), 4) if results else 0.0,
        "latency_p50_seconds": round(statistics.median(elapsed), 3) if elapsed else None,
        "latency_p95_seconds": _nearest_rank(elapsed, 0.95),
        "llm_calls": total_calls,
        "mean_llm_calls": round(total_calls / len(results), 3) if results else 0.0,
        "total_tokens": total_tokens,
        "mean_total_tokens": round(total_tokens / len(results), 3) if results else 0.0,
        "estimated_cost_usd": round(total_cost, 10),
        "mean_estimated_cost_usd": (round(total_cost / len(results), 10) if results else 0.0),
        "per_task": per_task,
    }


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    """Write benchmark records as newline-delimited JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")


def default_run_limits(*, generated: bool, full: bool) -> Tuple[int, int]:
    """Return benchmark-specific iteration and output-token defaults."""
    if generated:
        return 10, 4_000
    if full:
        return 6, 4_000
    return 4, 1_000


def main() -> None:
    """Run repeated tasks for one model and print machine-readable results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="LiteLLM model identifier")
    parser.add_argument(
        "--full",
        action="store_true",
        help="run the slower two-task suite instead of the default smoke test",
    )
    all_tasks = SMOKE_TASKS + FULL_TASKS
    parser.add_argument(
        "--task",
        choices=[task.name for task in all_tasks],
        help="run one named task",
    )
    parser.add_argument("--runs", type=int, default=1, help="repetitions per task")
    parser.add_argument(
        "--mode",
        choices=("rlm", "direct"),
        default="rlm",
        help="RLM execution or a one-call long-context baseline",
    )
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument(
        "--generated-chars",
        type=int,
        help="run one deterministic generated context of at least this many characters",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-total-calls", type=int, default=24)
    parser.add_argument("--max-elapsed-seconds", type=float, default=300)
    parser.add_argument("--jsonl", type=Path, help="write result and summary records")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="include full content-bearing trajectories in output",
    )
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be greater than zero")
    if args.max_depth < 0:
        parser.error("--max-depth must be zero or greater")
    if args.generated_chars is not None and args.generated_chars <= 0:
        parser.error("--generated-chars must be greater than zero")
    for name, value in (
        ("--max-iterations", args.max_iterations),
        ("--max-tokens", args.max_tokens),
        ("--max-total-calls", args.max_total_calls),
        ("--max-elapsed-seconds", args.max_elapsed_seconds),
    ):
        if value is not None and value <= 0:
            parser.error(f"{name} must be greater than zero")

    load_dotenv()
    if args.generated_chars is not None:
        tasks = (build_generated_task(args.generated_chars, args.seed),)
    elif args.task:
        tasks = tuple(task for task in all_tasks if task.name == args.task)
    else:
        tasks = FULL_TASKS if args.full else SMOKE_TASKS
    use_full_limits = args.full or (args.task and args.task != "largest_value")
    default_iterations, default_tokens = default_run_limits(
        generated=args.generated_chars is not None,
        full=bool(use_full_limits),
    )
    max_iterations = args.max_iterations or default_iterations
    max_tokens = args.max_tokens or default_tokens

    results: List[Dict[str, Any]] = []
    for task in tasks:
        for run_index in range(1, args.runs + 1):
            print(
                f"Running {args.model}: {task.name} ({run_index}/{args.runs})",
                flush=True,
            )
            result = run_task(
                args.model,
                task,
                run_index=run_index,
                max_depth=args.max_depth,
                max_iterations=max_iterations,
                max_tokens=max_tokens,
                max_total_calls=args.max_total_calls,
                max_elapsed_seconds=args.max_elapsed_seconds,
                mode=args.mode,
                trace=args.trace,
            )
            results.append(result)
            print(
                f"Finished {task.name}: passed={result['passed']} "
                f"calls={result['stats']['llm_calls']} "
                f"cost=${result['stats']['estimated_cost_usd']}",
                flush=True,
            )

    summary = aggregate_results(args.model, results, max_depth=args.max_depth, mode=args.mode)
    if args.jsonl:
        _write_jsonl(args.jsonl, [*results, summary])
    print(json.dumps({"summary": summary, "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
