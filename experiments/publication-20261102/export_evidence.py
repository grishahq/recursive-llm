"""Export compact, explicitly whitelisted evidence without changing raw captures.

Usage: python export_evidence.py INPUT_ROOT OUTPUT_ROOT

Retained answers, selected citations, queries, and error messages are public
content. This is a schema projection, not a general-purpose secret scrubber.
Full prompts, source contexts, CLI output, and trajectory payloads are omitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, cast

_TOP = (
    "dataset_version",
    "grader_version",
    "task",
    "mode",
    "model",
    "reasoning_effort",
    "query",
    "status",
    "elapsed_seconds",
    "raw_answer",
    "started_at",
    "repeat",
    "direct_input_format",
)
_SOURCE = ("document_id", "title", "source_sha256", "page_count", "character_count")
_CONFIG = (
    "model",
    "recursive_model",
    "max_depth",
    "max_iterations",
    "repl_timeout",
    "max_output_chars",
    "repl_max_snapshot_bytes",
    "max_concurrent_subcalls",
    "max_total_calls",
    "max_total_tokens",
    "max_total_cost_usd",
    "max_elapsed_seconds",
    "max_retries",
    "retry_backoff_seconds",
    "capture_trajectory_content",
    "final_answer_validator",
    "call_timeout",
    "token_budget_enforcement",
    "seed",
)
_STATS = (
    "llm_calls",
    "root_calls",
    "recursive_calls",
    "leaf_calls",
    "retry_calls",
    "total_iterations",
    "max_depth_reached",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_tokens",
    "usage_calls",
    "priced_calls",
    "estimated_cost_usd",
    "iterations",
    "depth",
    "calls",
    "billing_mode",
    "unknown_usage_calls",
    "failed_calls_with_known_usage",
)
_BUDGET = (
    "max_calls",
    "max_tokens",
    "max_cost_usd",
    "max_elapsed_seconds",
    "reserved_calls",
    "observed_tokens",
    "observed_cost_usd",
    "elapsed_seconds",
)
_GRADE = (
    "passed",
    "factual_passed",
    "provenance_passed",
    "supported_gold_fields",
    "required_gold_fields",
    "grader_version",
    "tested_inputs",
    "field_contract_passed",
    "citation_structure_passed",
    "evidence_coverage_passed",
    "supported_fields",
    "required_fields",
)
_TRACE_KINDS = {
    "run_start",
    "run_end",
    "run_error",
    "rlm_start",
    "rlm_end",
    "rlm_error",
    "model_call_start",
    "model_call_end",
    "model_call_error",
    "model_retry",
    "provider_response_normalized",
    "repl_step",
    "final_answer",
    "final_answer_validated",
    "final_answer_rejected",
    "final_answer_validation_error",
    "sources_available",
    "source_read",
    "source_search",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _scalars(record: Mapping[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    result = {}
    for key in keys:
        if key not in record:
            continue
        value = record[key]
        if value is not None and not isinstance(value, (str, bool, int, float)):
            raise ValueError(f"Expected scalar public field: {key}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Nonfinite public field: {key}")
        result[key] = value
    return result


def _stats(record: Any) -> Any:
    if record is None:
        return None
    if not isinstance(record, dict):
        raise ValueError("Usage and runtime statistics must be objects or null")
    result = _scalars(record, _STATS)
    if "budget" in record:
        result["budget"] = _scalars(record["budget"], _BUDGET)
    if "by_model" in record:
        result["by_model"] = {
            model: _scalars(values, _STATS) for model, values in record["by_model"].items()
        }
    return result


def _grade(record: Any) -> Any:
    if record is None:
        return None
    if not isinstance(record, dict):
        raise ValueError("grade must be an object or null")
    result = _scalars(record, _GRADE)
    if "failures" in record:
        failures = record["failures"]
        if not isinstance(failures, list) or any(not isinstance(item, str) for item in failures):
            raise ValueError("grade.failures must be a list of strings")
        result["failures"] = list(failures)
    if "answer_grade" in record:
        result["answer_grade"] = _grade(record["answer_grade"])
    if "observed_outputs" in record:

        def public_array(value: Any) -> Any:
            if isinstance(value, list):
                return [public_array(item) for item in value]
            return _scalars({"observed_output": value}, ("observed_output",))["observed_output"]

        if not isinstance(record["observed_outputs"], list):
            raise ValueError("grade.observed_outputs must be an array")
        result["observed_outputs"] = public_array(record["observed_outputs"])
    return result


def _valid_usage(events: Sequence[Any]) -> Optional[Dict[str, int]]:
    completed = [
        event
        for event in events
        if isinstance(event, dict) and event.get("type") == "turn.completed"
    ]
    if len(completed) != 1 or not isinstance(completed[0].get("usage"), dict):
        return None
    usage = completed[0]["usage"]
    keys = ("input_tokens", "cached_input_tokens", "output_tokens")
    if any(type(usage.get(key)) is not int or usage[key] < 0 for key in keys):
        return None
    if usage["cached_input_tokens"] > usage["input_tokens"]:
        return None
    return {key: usage[key] for key in keys}


def _response_text(call: Mapping[str, Any], events: Sequence[Any]) -> Optional[str]:
    response = call.get("response")
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return cast(str, message["content"])
    texts = []
    for event in events:
        if not isinstance(event, dict) or event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == "agent_message":
            texts.append(item.get("text"))
    return texts[0] if len(texts) == 1 and isinstance(texts[0], str) else None


def _call(call: Mapping[str, Any]) -> Dict[str, Any]:
    result = _scalars(
        call, ("model", "reasoning_effort", "status", "elapsed_seconds", "billing_mode")
    )
    events = call.get("events", [])
    if not isinstance(events, list):
        raise ValueError("transport call events must be a list")
    usage = _valid_usage(events)
    text = _response_text(call, events)
    result.update(
        usage=usage,
        usage_known=usage is not None,
        response_chars=len(text) if text is not None else None,
        response_sha256=_sha(text.encode("utf-8")) if text is not None else None,
    )
    return result


def _trace(events: Any) -> Dict[str, Any]:
    counts: Counter[str] = Counter()
    by_depth: Dict[str, Counter[str]] = {}
    reads = []
    for event in events or []:
        if not isinstance(event, dict):
            continue
        raw_kind = event.get("kind")
        kind = raw_kind if isinstance(raw_kind, str) and raw_kind in _TRACE_KINDS else "other"
        counts[kind] += 1
        depth = event.get("depth")
        label = str(depth) if type(depth) is int and depth >= 0 else "unknown"
        by_depth.setdefault(label, Counter())[kind] += 1
        if kind == "source_read" and isinstance(event.get("data"), dict):
            data = event["data"]
            entry = _scalars(
                event, ("sequence", "depth", "node_id", "parent_id", "elapsed_seconds")
            )
            entry.update(
                _scalars(
                    data, ("document_id", "source_sha256", "page", "start", "end", "quote_chars")
                )
            )
            if "quote_chars" not in entry and isinstance(data.get("quote"), str):
                entry["quote_chars"] = len(data["quote"])
            reads.append(entry)
    return {
        "event_count": sum(counts.values()),
        "event_counts": dict(sorted(counts.items())),
        "event_counts_by_depth": {
            key: dict(sorted(value.items())) for key, value in sorted(by_depth.items())
        },
        "source_reads": reads,
    }


def export_record(record: Mapping[str, Any], raw_capture_sha256: str) -> Dict[str, Any]:
    """Project one capture into public schema 1 without modifying the input."""
    result = _scalars(record, _TOP)
    result.update(
        public_schema_version=1,
        raw_capture_sha256=raw_capture_sha256,
        redaction={
            "method": "explicit field whitelists",
            "omitted": [
                "full messages",
                "full source contexts",
                "CLI stdout and stderr",
                "commands",
                "raw trajectory payloads",
                "unlisted fields",
            ],
            "retained_content": [
                "query",
                "raw_answer including selected citations",
                "grade failure messages",
                "error type and message",
                "source metadata",
            ],
            "usage": "Per-call usage requires exactly one valid turn.completed event; unknown usage remains null. Aggregate recorded statistics are retained without invented zero values.",
        },
    )
    if "sources" in record:
        result["sources"] = [_scalars(source, _SOURCE) for source in record["sources"]]
    if "config" in record:
        result["config"] = _scalars(record["config"], _CONFIG)
    for key in ("runtime_stats", "transport_usage"):
        if key in record:
            result[key] = _stats(record[key])
    if "grade" in record:
        result["grade"] = _grade(record["grade"])
    if "error" in record:
        result["error"] = (
            None if record["error"] is None else _scalars(record["error"], ("type", "message"))
        )
    if "integration_checks" in record:
        value = record["integration_checks"]
        result["integration_checks"] = (
            None
            if value is None
            else _scalars(value, ("forced_child", "child_calls", "child_used", "note"))
        )
    calls = [_call(call) for call in record.get("transport_calls", [])]
    result["transport_calls"] = calls
    known = [call["usage"] for call in calls if call["usage_known"]]
    result["call_summary"] = {
        "observed_calls": len(calls),
        "known_usage_calls": len(known),
        "unknown_usage_calls": len(calls) - len(known),
        "observed_total_tokens": (
            sum(usage["input_tokens"] + usage["output_tokens"] for usage in known)
            if known
            else None
        ),
        "usage_complete": bool(calls) and len(known) == len(calls),
    }
    events = record.get("trajectory")
    if events is None:
        document_result = record.get("result", {})
        run = document_result.get("run", {}) if isinstance(document_result, dict) else {}
        events = run.get("trajectory", []) if isinstance(run, dict) else []
    result["trace_summary"] = _trace(events)
    return result


def export_tree(input_root: Path, output_root: Path) -> Dict[str, Any]:
    """Export attempt JSON files into a new, disjoint output directory."""
    input_root, output_root = input_root.resolve(), output_root.resolve()
    if not input_root.is_dir():
        raise ValueError("input root must be an existing directory")
    if (
        input_root == output_root
        or input_root in output_root.parents
        or output_root in input_root.parents
    ):
        raise ValueError("input and output directory trees must be disjoint")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("output directory must be empty")
    outputs: List[Dict[str, Any]] = []
    skipped = 0
    for path in sorted(input_root.rglob("*.json")):
        if path.is_symlink():
            raise ValueError("input JSON symlinks are not supported")
        raw = path.read_bytes()
        record = json.loads(raw)
        if not isinstance(record, dict) or not {"task", "raw_answer"} <= record.keys():
            skipped += 1
            continue
        if "public_schema_version" in record:
            raise ValueError("public exports cannot be used as raw captures")
        projected = export_record(record, _sha(raw))
        public = (
            json.dumps(projected, ensure_ascii=True, indent=2, allow_nan=False) + "\n"
        ).encode()
        relative = path.relative_to(input_root)
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(public)
        outputs.append(
            {
                "path": relative.as_posix(),
                "raw_capture_sha256": _sha(raw),
                "public_sha256": _sha(public),
                "raw_bytes": len(raw),
                "public_bytes": len(public),
            }
        )
    manifest = {
        "public_schema_version": 1,
        "record_count": len(outputs),
        "skipped_json_files": skipped,
        "raw_bytes": sum(item["raw_bytes"] for item in outputs),
        "public_bytes": sum(item["public_bytes"] for item in outputs),
        "records": outputs,
        "note": "Original captures remain unchanged locally. Public records retain answers, scores, identities and usage, not complete prompts or execution replay.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "public-manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    manifest = export_tree(args.input_root, args.output_root)
    print(json.dumps({key: manifest[key] for key in ("record_count", "raw_bytes", "public_bytes")}))


if __name__ == "__main__":
    main()
