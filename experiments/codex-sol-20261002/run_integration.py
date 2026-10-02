"""Two bounded integration regressions; not an architecture benchmark.

Run from the repository root with ``python experiments/codex-sol-20261002/run_integration.py``.
Each invocation creates a fresh attempt directory and retains failed runs.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from rlm import RLM
from rlm.codex import DISABLED_FEATURES, TEXT_INSTRUCTIONS, CodexCompletion

MODEL = "gpt-6.1-sol"
SCENARIOS = (
    {
        "name": "root_tuple_assignment",
        "max_depth": 0,
        "context": "TX-A amount=7\nTX-B amount=29\nTX-C amount=13\nTX-D amount=23\n",
        "query": (
            "Inspect all transaction records in context. Use Python to parse every record into "
            "(amount, ident) tuples in a variable named records, and execute the tuple assignment "
            "amount, ident = max(records) in the REPL. Return only id=<ID> amount=<integer> for "
            "the record with the largest amount. This checks ordinary tuple assignment support."
        ),
        "expected": "id=TX-B amount=29",
    },
    {
        "name": "child_rlm_delegation",
        "max_depth": 2,
        "context": (
            "Internal synthetic release register.\n"
            "Depot MICA-17: release label INDIGO-206.\n"
            "Depot KESTREL-83: release label ORCHID-4289.\n"
            "Depot CEDAR-51: release label COBALT-731.\n"
        ),
        "query": (
            "Use recursive_llm exactly once on the full context to determine the release label "
            "for depot KESTREL-83. Instruct the child to inspect the supplied text in its own "
            "REPL, use no further subcalls, and return only release_label=<value>. After "
            "receiving its evidence, return that exact labeled answer. This is an integration "
            "check of child RLM execution, so the child must actually run a REPL inspection."
        ),
        "expected": "release_label=ORCHID-4289",
    },
)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write(path: Path, record: dict) -> None:
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def contains_max_unpack(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Tuple) for target in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "max"
        for node in ast.walk(tree)
    )


async def main() -> None:
    project = Path(__file__).resolve().parents[2]
    captured_at = datetime.now(timezone.utc)
    output = Path(__file__).parent / captured_at.strftime("attempt-%H%M%S-%f")
    output.mkdir()
    manifest = {
        "kind": "integration_regression_not_architecture_benchmark",
        "captured_at": captured_at.isoformat(),
        "model": MODEL,
        "reasoning_effort": "low",
        "cli_version": subprocess.check_output(["codex", "--version"], text=True).strip(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": {
            name: sha256((project / name).read_bytes())
            for name in (
                "src/rlm/core.py",
                "src/rlm/repl.py",
                "src/rlm/codex.py",
                "src/rlm/prompts.py",
                "experiments/codex-sol-20261002/run_integration.py",
            )
        },
        "billing_mode": "ChatGPT subscription",
        "api_fallback": False,
        "answer_key_feedback": False,
        "max_total_calls": 12,
        "max_elapsed_seconds": 180,
        "per_call_timeout_seconds": 75,
        "disabled_cli_features": list(DISABLED_FEATURES),
        "wrapper_instructions": TEXT_INSTRUCTIONS,
    }
    transport = CodexCompletion(reasoning_effort="low", timeout=75)
    manifest["login_method"] = await transport.check_login()
    write(output / "manifest.json", manifest)
    print(f"Saving all attempts under {output}", flush=True)
    summaries = []
    for scenario in SCENARIOS:
        transport = CodexCompletion(reasoning_effort="low", timeout=75)
        path = output / f"{scenario['name']}.json"
        record = {
            **scenario,
            "model": MODEL,
            "reasoning_effort": "low",
            "context_sha256": sha256(scenario["context"].encode()),
            "calls": transport.calls,
            "trajectory_checkpoints": [],
            "status": "started",
        }

        def checkpoint(event) -> None:
            record["trajectory_checkpoints"].append(event.to_dict())
            write(path, record)

        write(path, record)
        rlm = RLM(
            model=MODEL,
            completion_handler=transport,
            max_depth=scenario["max_depth"],
            max_iterations=8,
            max_total_calls=12,
            max_elapsed_seconds=180,
            max_retries=0,
            repl_timeout=100,
            capture_trajectory_content=True,
            event_handler=checkpoint,
        )
        result = await rlm.atry_complete_result(scenario["query"], scenario["context"])
        record["result"] = result.to_dict()
        record["status"] = "completed" if result.succeeded else "failed"
        steps = [event for event in result.trajectory if event.kind == "repl_step"]
        item_types = {
            event.get("item", {}).get("type")
            for call in transport.calls
            for event in call["events"]
            if event.get("type", "").startswith("item.")
        }
        checks = {
            "exact_answer": result.succeeded and result.answer.strip() == scenario["expected"],
            "no_repl_errors": bool(steps) and all(event.data["status"] == "ok" for event in steps),
            "text_only_cli": item_types <= {"agent_message", "reasoning"},
            "call_limit": len(transport.calls) <= 12,
        }
        if scenario["max_depth"] == 0:
            checks["tuple_assignment_executed"] = any(
                contains_max_unpack(event.data.get("code", "")) and event.data["status"] == "ok"
                for event in steps
            )
            checks["root_only"] = result.stats["max_depth_reached"] == 0
        else:
            checks["child_repl_executed"] = any(event.depth == 1 for event in steps)
            checks["child_model_called"] = any(
                event.kind == "model_call_start" and event.depth == 1 for event in result.trajectory
            )
        record["checks"] = checks
        record["passed"] = all(checks.values())
        write(path, record)
        summary = {
            "scenario": scenario["name"],
            "passed": record["passed"],
            "checks": checks,
            "answer": result.answer,
            "stats": result.stats,
            "artifact": path.name,
        }
        summaries.append(summary)
        write(output / "summary.json", {"manifest": "manifest.json", "scenarios": summaries})
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
