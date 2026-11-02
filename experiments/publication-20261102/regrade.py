"""Regrade saved answers without model calls or modification of original captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.grounded_documents import (  # noqa: E402
    DEFAULT_DIRECTORY,
    GRADER_VERSION,
    _transport_usage,
    grade_answer,
    load_public_documents,
    public_tasks,
    summarize,
    synthetic_tasks,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = {
        task.name: task
        for task in synthetic_tasks() + public_tasks(load_public_documents(DEFAULT_DIRECTORY))
    }
    reviewed = []
    summary_records = []
    for directory in args.directories:
        for path in sorted(directory.glob("*-*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or "raw_answer" not in record:
                continue
            task_name = record["task"].removesuffix("-forced-child")
            task = tasks[task_name]
            raw = record["raw_answer"]
            if raw is None:
                grade = {
                    "passed": False,
                    "factual_passed": False,
                    "provenance_passed": False,
                    "failures": ["no completed answer"],
                    "grader_version": GRADER_VERSION,
                }
            else:
                grade = grade_answer(task, raw)
            entry = {
                "artifact": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                "model": record["model"],
                "task": record["task"],
                "mode": record["mode"],
                "status": record["status"],
                "raw_answer_sha256": (
                    hashlib.sha256(raw.encode("utf-8")).hexdigest() if raw is not None else None
                ),
                "original_grade": record["grade"],
                "regraded": grade,
                "recursive_calls": (record.get("runtime_stats") or {}).get("recursive_calls", 0),
                "original_transport_usage": record["transport_usage"],
                "audited_transport_usage": _transport_usage(record["transport_calls"]),
            }
            if record.get("integration_checks", {}).get("forced_child"):
                entry["integration_checks"] = record["integration_checks"]
                if not entry["recursive_calls"]:
                    grade["passed"] = False
                    grade["failures"].append("forced child call was not observed")
            reviewed.append(entry)
            adapted = copy.deepcopy(record)
            adapted["grade"] = grade
            adapted["transport_usage"] = entry["audited_transport_usage"]
            summary_records.append(adapted)
    output = {
        "grader_version": GRADER_VERSION,
        "grader_sha256": hashlib.sha256(
            (ROOT / "benchmarks/grounded_documents.py").read_bytes()
        ).hexdigest(),
        "method": "All saved answers regraded; original grades and captures preserved; no new inference.",
        "records": reviewed,
        "summary": summarize(summary_records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
