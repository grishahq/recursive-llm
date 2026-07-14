"""Live comparison of same-model recursive RLM configurations.

Run from the repository root after configuring provider keys in ``.env``:

    python benchmarks/compare_same_model.py gpt-5-mini
    python benchmarks/compare_same_model.py deepseek/deepseek-v4-flash

Pass ``--full`` to run the slower two-task suite.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from dotenv import load_dotenv

from rlm import RLM


@dataclass(frozen=True)
class Task:
    """One deterministic RLM evaluation task."""

    name: str
    query: str
    context: str
    expected_fragments: Sequence[str]


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
        expected_fragments=("ITEM-B", "19"),
    ),
)


FULL_TASKS = (
    Task(
        name="division_rollup",
        query=(
            "Use recursive_llm exactly once to analyze all division records. Verify the child's "
            "evidence and return the total profit across all divisions plus the division with "
            "the highest defect percentage. Profit is revenue minus cost."
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
        expected_fragments=("514", "DIV-J", "4.8"),
    ),
    Task(
        name="incident_filter",
        query=(
            "Use recursive_llm exactly once to inspect all incidents. Return the incident IDs "
            "where deploy_api=v3 and cache_mode=legacy, together with the count. Cite only IDs "
            "supported by the context."
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
        expected_fragments=("INC-003", "INC-007", "INC-009", "3"),
    ),
)


def run_task(
    model: str,
    task: Task,
    *,
    max_iterations: int,
    max_tokens: int,
    trace: bool = False,
) -> Dict[str, Any]:
    """Run a task using the same model at every recursion depth."""
    rlm = RLM(
        model=model,
        max_depth=3,
        max_iterations=max_iterations,
        timeout=60,
        max_tokens=max_tokens,
        num_retries=0,
    )
    if trace:
        original_call = rlm._call_llm

        async def traced_call(messages: List[Dict[str, str]], **kwargs: Any) -> str:
            response = await original_call(messages, **kwargs)
            print(f"\n[depth={rlm._current_depth} response]\n{response}\n", flush=True)
            return response

        rlm._call_llm = traced_call  # type: ignore[method-assign]

    started = time.perf_counter()
    try:
        answer = rlm.complete(query=task.query, context=task.context)
        error = None
    except Exception as exc:  # Keep the second task running after a provider/protocol failure.
        answer = ""
        error = f"{type(exc).__name__}: {exc}"
    elapsed_seconds = time.perf_counter() - started

    missing = [fragment for fragment in task.expected_fragments if fragment not in answer]
    return {
        "task": task.name,
        "passed": error is None and not missing,
        "missing_expected_fragments": missing,
        "answer": answer,
        "error": error,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "stats": rlm.stats,
    }


def main() -> None:
    """Run all tasks for one model and print machine-readable results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="LiteLLM model identifier")
    parser.add_argument(
        "--full",
        action="store_true",
        help="run the slower two-task suite instead of the default smoke test",
    )
    parser.add_argument(
        "--task",
        choices=[task.name for task in FULL_TASKS],
        help="run one named full-suite task",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="print root-model responses for protocol debugging",
    )
    args = parser.parse_args()

    load_dotenv()
    if args.task:
        tasks = tuple(task for task in FULL_TASKS if task.name == args.task)
    else:
        tasks = FULL_TASKS if args.full else SMOKE_TASKS
    use_full_limits = args.full or args.task is not None
    max_iterations = 6 if use_full_limits else 4
    max_tokens = 4_000 if use_full_limits else 1_000
    results: List[Dict[str, Any]] = []
    for task in tasks:
        print(f"Running {args.model}: {task.name}", flush=True)
        result = run_task(
            args.model,
            task,
            max_iterations=max_iterations,
            max_tokens=max_tokens,
            trace=args.trace,
        )
        results.append(result)
        print(
            f"Finished {task.name}: passed={result['passed']} "
            f"calls={result['stats']['llm_calls']} "
            f"cost=${result['stats']['estimated_cost_usd']}",
            flush=True,
        )

    summary = {
        "model": args.model,
        "passed": sum(result["passed"] for result in results),
        "total": len(results),
        "elapsed_seconds": round(sum(result["elapsed_seconds"] for result in results), 3),
        "llm_calls": sum(result["stats"]["llm_calls"] for result in results),
        "recursive_calls": sum(result["stats"]["recursive_calls"] for result in results),
        "leaf_calls": sum(result["stats"].get("leaf_calls", 0) for result in results),
        "prompt_tokens": sum(result["stats"]["prompt_tokens"] for result in results),
        "completion_tokens": sum(result["stats"]["completion_tokens"] for result in results),
        "estimated_cost_usd": round(
            sum(result["stats"]["estimated_cost_usd"] or 0.0 for result in results), 10
        ),
    }
    print(json.dumps({"summary": summary, "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
