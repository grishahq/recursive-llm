"""Record real RLM versus direct runs through an existing ChatGPT subscription.

    python examples/capture_codex_comparison.py --chars 100000 --seed 2026 \
        --output /tmp/rlm-codex-experiment/comparison-100000.json

Requires ``codex login`` using ChatGPT and a recent Codex CLI (tested: 0.149.0).
Produces browser-importable comparison JSON and a sibling ``.raw.json`` containing
the exact generated corpus, ground truth, prompts, CLI events, and RLM trajectory.
No answer-key feedback is given to either model, and failed attempts are retained.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Allow running this example from an editable checkout without installing benchmarks.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.generated_long_context import (
    GeneratedLongContext,
    generate_long_context,
)  # noqa: E402
from rlm import RLM  # noqa: E402
from rlm.codex import DISABLED_FEATURES, MODEL, TEXT_INSTRUCTIONS, CodexCompletion  # noqa: E402


def _write(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _grade(corpus: Optional[GeneratedLongContext], answer: str) -> Optional[str]:
    return f"{4 - len(corpus.validate(answer))} / 4 fields" if corpus else None


def validate_expected_fields(value: Any) -> Dict[str, Any]:
    """Validate a bounded reference map, which must never enter a model prompt."""
    if value is None:
        return {}
    if not isinstance(value, dict) or len(value) > 20:
        raise ValueError("expected_fields must be an object with at most 20 fields")
    for label, expected in value.items():
        if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", label):
            raise ValueError("Reference field labels must be letters, digits, and underscores")
        if isinstance(expected, str):
            if len(expected) > 1000:
                raise ValueError("Reference strings must have at most 1000 characters")
        elif (
            not isinstance(expected, (int, float, bool))
            or (isinstance(expected, float) and not math.isfinite(expected))
            or (isinstance(expected, int) and expected.bit_length() > 4096)
        ):
            raise ValueError("Reference values must be JSON strings, finite numbers, or booleans")
    return dict(value)


def validate_expected_aliases(value: Any, expected: Dict[str, Any]) -> Dict[str, List[str]]:
    """Validate explicit alternate strings for already declared reference fields."""
    if value is None:
        return {}
    if not isinstance(value, dict) or len(value) > 20:
        raise ValueError("expected_aliases must be an object with at most 20 fields")
    aliases = {}
    for label, alternatives in value.items():
        if not isinstance(expected.get(label), str):
            raise ValueError("Aliases require an existing string reference field")
        if (
            not isinstance(alternatives, list)
            or not 1 <= len(alternatives) <= 10
            or any(
                not isinstance(alias, str) or not alias.strip() or len(alias) > 1000
                for alias in alternatives
            )
        ):
            raise ValueError(
                "Each alias field requires 1–10 nonempty strings of at most 1000 characters"
            )
        aliases[label] = list(alternatives)
    return aliases


def grade_fields(
    answer: str,
    expected: Dict[str, Any],
    *,
    failed: bool = False,
    expected_aliases: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Grade JSON or labeled answers without asking a model to judge itself."""
    aliases = validate_expected_aliases(expected_aliases, expected)
    if failed or not expected:
        return {
            "status": "unavailable" if failed else "manual_review",
            "matched": None,
            "total": len(expected) or None,
            "fields": [],
        }
    parsed = None
    duplicate = False

    def pairs(items: Any) -> Any:
        nonlocal duplicate
        result = {}
        for key, value in items:
            if key in result:
                duplicate = True
            result[key] = value
        return result

    candidate = answer.strip()
    if candidate.startswith("```json") and candidate.endswith("```"):
        candidate = candidate[7:-3].strip()
    if candidate.startswith("{"):
        try:
            parsed = json.loads(candidate, object_pairs_hook=pairs)
        except (ValueError, TypeError):
            parsed = {}
        if not isinstance(parsed, dict) or duplicate:
            parsed = {}
    fields = []
    label_boundary = r"(?=\s+[A-Za-z][A-Za-z0-9_]{0,63}\s*(?:=|:)|[\r\n]|$)"
    for label, value in expected.items():
        matches = re.findall(
            rf"(?:^|\s){re.escape(label)}\s*(?:=|:)\s*([^\r\n]*?){label_boundary}",
            answer,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        observed = (
            parsed.get(label)
            if parsed is not None
            else (matches[0].strip() if len(matches) == 1 else None)
        )
        passed = False
        if isinstance(value, bool):
            passed = type(observed) is bool and observed == value
        elif (
            isinstance(value, (int, float))
            and not isinstance(observed, bool)
            and (parsed is None or isinstance(observed, (int, float)))
        ):
            try:
                passed = Decimal(str(observed)) == Decimal(str(value))
            except InvalidOperation:
                pass
        elif isinstance(value, str) and isinstance(observed, str):

            def normalize(text: str) -> str:
                translations = str.maketrans("‘’‐‑–—", "''----")
                return " ".join(text.translate(translations).casefold().split())

            passed = any(
                (
                    observed == allowed
                    if not allowed.strip()
                    else normalize(observed) == normalize(allowed)
                )
                for allowed in [value, *aliases.get(label, [])]
            )
        if isinstance(observed, (dict, list)):
            observed = json.dumps(observed, ensure_ascii=False)
        elif isinstance(observed, float) and not math.isfinite(observed):
            observed = str(observed)
        field = {"label": label, "expected": value, "observed": observed, "passed": passed}
        if label in aliases:
            field["accepted_aliases"] = aliases[label]
        fields.append(field)
    return {
        "status": "graded",
        "matched": sum(field["passed"] for field in fields),
        "total": len(fields),
        "fields": fields,
    }


def _apply_quality(
    lane: Dict[str, Any], expected: Dict[str, Any], aliases: Dict[str, List[str]]
) -> None:
    quality = grade_fields(
        lane["result"], expected, failed=bool(lane.get("error")), expected_aliases=aliases
    )
    lane["quality"] = quality
    lane["passed"] = (
        f"{quality['matched']} / {quality['total']} fields"
        if quality["status"] == "graded"
        else None
    )


def _empty_lane() -> Dict[str, Any]:
    return {
        "tokens": None,
        "usage_complete": False,
        "cached_tokens": 0,
        "cost": None,
        "seconds": 0,
        "calls": 0,
        "result": "",
        "passed": None,
        "steps": [],
        "quality": {"status": "unavailable", "matched": None, "total": None, "fields": []},
    }


def _observed_usage(calls: List[Dict[str, Any]], *, terminal: bool) -> Dict[str, Any]:
    usages = [call["response"]["usage"] for call in calls if call.get("response")]
    complete = terminal and bool(calls) and len(usages) == len(calls)
    observed = sum(usage["total_tokens"] for usage in usages) if usages else None
    return {
        "tokens": observed if complete else None,
        "observed_tokens": observed,
        "usage_complete": complete,
        "calls": len(calls),
        "cached_tokens": sum(
            usage.get("prompt_tokens_details", {}).get("cached_tokens", 0) for usage in usages
        ),
        "prompt_tokens": sum(usage["prompt_tokens"] for usage in usages),
        "completion_tokens": sum(usage["completion_tokens"] for usage in usages),
    }


def _rlm_steps(trajectory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    steps = []
    call_number = 0
    for event in trajectory:
        kind, data = event["kind"], event["data"]
        step = {
            "at": event["elapsed_seconds"],
            "node_id": event["node_id"],
            "parent_id": event["parent_id"],
            "kind": kind,
        }
        if kind == "run_start":
            step.update(
                title="Context stays in Python",
                detail="The model receives the question and context size.",
            )
        elif kind == "model_call_start":
            call_number += 1
            step.update(
                title=f"Model call {call_number}",
                detail=(
                    f"{data['model']} · depth {event['depth']} · "
                    f"{'plain subcall' if data['is_leaf'] else 'RLM step'}"
                ),
            )
        elif kind == "repl_step":
            step.update(
                title="Execute Python" if data["status"] == "ok" else "Python feedback",
                detail="Executed by the library's restricted REPL.",
                code=data.get("code", ""),
                output=data.get("output", ""),
            )
        elif kind == "final_answer":
            step.update(
                title="Final answer",
                detail="RLM returned its answer.",
                output=data.get("answer", ""),
            )
        elif kind in {"model_call_error", "run_error"}:
            step.update(title="Run error", detail=data.get("error", "Unknown error"))
        else:
            continue
        steps.append(step)
    return steps


async def capture_comparison(
    *,
    chars: int = 100_000,
    seed: int = 2026,
    output: Optional[Path] = None,
    model: str = MODEL,
    reasoning_effort: str = "medium",
    max_depth: int = 1,
    context: Optional[str] = None,
    query: Optional[str] = None,
    expected_fields: Optional[Dict[str, Any]] = None,
    expected_aliases: Optional[Dict[str, List[str]]] = None,
    progress_handler: Optional[Callable[[Dict[str, Any]], None]] = None,
    document_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one paired experiment and return the normalized visualizer record.

    Direct uses one fresh text-only CLI turn with the full corpus. RLM uses the
    same model, reasoning effort, question, and corpus through the real RLM loop.
    Each call has a 90-second deadline; the RLM tree has 12 calls / 240 seconds.
    """
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty Codex model identifier")
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 0:
        raise ValueError("max_depth must be a nonnegative integer")
    direct_transport = CodexCompletion(reasoning_effort=reasoning_effort)
    rlm_transport = CodexCompletion(reasoning_effort=reasoning_effort)
    if (context is None) != (query is None):
        raise ValueError("Custom context and query must be supplied together")
    if context is not None and (not context.strip() or not query or not query.strip()):
        raise ValueError("Custom context and query must be nonempty")
    expected = validate_expected_fields(expected_fields)
    output = Path(output) if output else Path(f"/tmp/rlm-codex-experiment/comparison-{chars}.json")
    raw_path = output.with_suffix(".raw.json")
    if output.exists() or raw_path.exists():
        raise FileExistsError("Choose a new output filename to preserve all previous attempts")
    corpus = generate_long_context(target_chars=chars, seed=seed) if context is None else None
    if corpus:
        query, context = corpus.query, corpus.context
        expected = asdict(corpus.truth)
    aliases = validate_expected_aliases(expected_aliases, expected)
    assert context is not None and query is not None
    original_query = query
    if expected and not corpus:
        query += (
            "\n\nReturn one JSON object with exactly these field names: "
            + ", ".join(expected)
            + ". Use JSON strings, numbers, or booleans as appropriate."
        )
    context_chars = len(context)
    context_sha256 = hashlib.sha256(context.encode("utf-8")).hexdigest()
    evaluation_label = (
        "Reference fields: numbers compared exactly; text ignores case, typography, and whitespace. "
        "Whitespace-only strings and booleans are compared exactly. References are withheld from both models."
        if expected
        else "Manual review: no reference answer was supplied, so quality is not scored."
    )
    evaluation_metadata: Dict[str, Any] = {}
    if aliases:
        evaluation_metadata = {
            "evaluation_version": "reference-fields-aliases-v1",
            "reference_aliases": aliases,
        }
        evaluation_label += (
            " Explicit accepted aliases use the same exact text normalization; "
            "see each field's accepted_aliases. Aliases are withheld from both models. "
            "Grader: reference-fields-aliases-v1."
        )
    live: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "recorded",
        "title": "Document comparison in progress",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "max_depth": max_depth,
        "query": query,
        "original_query": original_query,
        "context_chars": context_chars,
        "context_preview": context[:2000],
        "context_sha256": context_sha256,
        "sample_note": "A real local Codex run is in progress.",
        "evaluation_label": evaluation_label,
        "reference_fields": expected,
        **evaluation_metadata,
        "direct": _empty_lane(),
        "rlm": _empty_lane(),
    }
    display_metadata = {
        key: value
        for key, value in (document_metadata or {}).items()
        if key
        in {
            "title",
            "filename",
            "format",
            "bytes",
            "sha256",
            "context_sha256",
            "context_chars",
            "source_url",
        }
    }
    if display_metadata:
        live["document"] = display_metadata
        live["title"] = display_metadata.get(
            "title", display_metadata.get("filename", live["title"])
        )
        if display_metadata.get("source_url"):
            live["source_url"] = display_metadata["source_url"]

    def emit(phase: str, comparison: Optional[Dict[str, Any]] = None) -> None:
        if progress_handler:
            progress_handler({"phase": phase, "comparison": deepcopy(comparison or live)})

    emit("auth")
    await direct_transport.check_login()
    captured_at = datetime.now(timezone.utc).isoformat()
    raw: Dict[str, Any] = {
        "schema_version": 1,
        "captured_at": captured_at,
        "model": model,
        "transport": "codex exec",
        "billing_mode": "ChatGPT subscription",
        "reasoning_effort": reasoning_effort,
        "max_depth": max_depth,
        "wrapper_instructions": TEXT_INSTRUCTIONS,
        "order": ["direct", "rlm"],
        "query": query,
        "context": context,
        "context_sha256": context_sha256,
        "context_chars": context_chars,
        "seed": seed if corpus else None,
        "truth": asdict(corpus.truth) if corpus else None,
        "reference_fields": expected,
        **evaluation_metadata,
        "document": display_metadata,
        "methodology": {
            "same_question_and_corpus": True,
            "grader_feedback_to_models": False,
            "disabled_cli_features": list(DISABLED_FEATURES),
            "required_code_mode_host_initialized": True,
            "non_text_events_rejected": True,
            "direct_calls": 1,
            "rlm_max_calls": 12,
            "rlm_max_depth": max_depth,
            "rlm_max_seconds": 240,
            "per_call_timeout_seconds": 90,
            "usage_note": "CLI-reported input + output, including cached input and CLI prompt overhead. "
            "Tokens are not a measurement of subscription quota remaining. Dollar cost is unknown.",
        },
        "direct": {"calls": direct_transport.calls},
        "rlm": {"calls": rlm_transport.calls},
    }
    _write(raw_path, raw)

    def checkpoint(event: Any) -> None:
        raw["rlm"].setdefault("trajectory_checkpoints", []).append(event.to_dict())
        _write(raw_path, raw)
        lane = live["rlm"]
        lane.update(_observed_usage(rlm_transport.calls, terminal=False))
        lane["seconds"] = event.elapsed_seconds
        lane["steps"] = _rlm_steps(raw["rlm"]["trajectory_checkpoints"])
        if event.kind == "final_answer":
            lane["result"] = event.data.get("answer", "")
        if event.kind == "run_error":
            lane["error"] = event.data.get("error", "Run failed")
        emit("rlm")

    direct_answer = ""
    direct_response = None
    direct_error = None
    started = time.monotonic()
    live["direct"]["calls"] = 1
    live["direct"]["steps"] = [
        {
            "at": 0,
            "title": "Send full context",
            "detail": f"{context_chars:,} characters in one model request.",
            "kind": "model_call_start",
        }
    ]
    emit("direct")
    try:
        direct_response = await direct_transport(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer the task using only the supplied context. "
                    "Return the answer directly. Do not emit code or FINAL directives.",
                },
                {"role": "user", "content": f"Task:\n{query}\n\nContext:\n{context}"},
            ],
        )
        direct_answer = direct_response["choices"][0]["message"]["content"]
    except Exception as exc:
        direct_error = f"{type(exc).__name__}: {exc}"
    except asyncio.CancelledError:
        direct_error = "Cancelled by user"
        raise
    finally:
        direct_seconds = round(time.monotonic() - started, 6)
        raw["direct"].update(
            answer=direct_answer,
            error=direct_error,
            elapsed_seconds=direct_seconds,
            grading_failures=list(corpus.validate(direct_answer)) if corpus else None,
        )
        _write(raw_path, raw)
        lane = live["direct"]
        lane.update(_observed_usage(direct_transport.calls, terminal=direct_response is not None))
        lane.update(seconds=direct_seconds, result=direct_answer)
        if direct_error:
            lane["error"] = direct_error
        lane["steps"].append(
            {
                "at": direct_seconds,
                "title": "Run error" if direct_error else "Final answer",
                "detail": direct_error or "Direct answer received.",
                "output": direct_answer,
                "kind": "run_error" if direct_error else "final_answer",
            }
        )
        _apply_quality(lane, expected, aliases)
        emit("cancelled" if direct_error == "Cancelled by user" else "direct_complete")

    rlm = RLM(
        model=model,
        completion_handler=rlm_transport,
        max_depth=max_depth,
        max_iterations=12,
        max_total_calls=12,
        max_elapsed_seconds=240,
        max_retries=0,
        repl_timeout=95,
        max_output_chars=3000,
        capture_trajectory_content=True,
        event_handler=checkpoint,
    )
    started = time.monotonic()
    emit("rlm")
    try:
        rlm_result = await rlm.atry_complete_result(query, context)
    finally:
        rlm_seconds = round(time.monotonic() - started, 6)
        raw["rlm"].update(elapsed_seconds=rlm_seconds, partial_stats=rlm.stats)
        _write(raw_path, raw)
        if sys.exc_info()[0] is asyncio.CancelledError:
            live["rlm"].update(error="Cancelled by user", seconds=rlm_seconds)
            _apply_quality(live["rlm"], expected, aliases)
            emit("cancelled")
    serialized = rlm_result.to_dict()
    rlm_answer = rlm_result.answer or ""
    raw["rlm"].update(
        result=serialized,
        elapsed_seconds=rlm_seconds,
        grading_failures=list(corpus.validate(rlm_answer)) if corpus else None,
    )
    _write(raw_path, raw)

    direct_usage = direct_response["usage"] if direct_response else {}
    direct_lane: Dict[str, Any] = {
        "tokens": direct_usage.get("total_tokens"),
        "usage_complete": direct_response is not None,
        "cached_tokens": direct_usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
        "prompt_tokens": direct_usage.get("prompt_tokens"),
        "completion_tokens": direct_usage.get("completion_tokens"),
        "cost": None,
        "seconds": direct_seconds,
        "calls": len(direct_transport.calls),
        "result": direct_answer,
        "passed": _grade(corpus, direct_answer) if not direct_error else None,
        "steps": [
            {
                "at": 0,
                "title": "Send full context",
                "detail": f"{context_chars:,} characters in one model request.",
                "kind": "model_call_start",
            },
            {
                "at": direct_seconds,
                "title": "Final answer" if not direct_error else "Run error",
                "detail": "One text-only Codex turn; no external computation.",
                "output": direct_answer or direct_error,
                "kind": "final_answer" if not direct_error else "run_error",
            },
        ],
    }
    if direct_error:
        direct_lane["error"] = direct_error
    stats = rlm_result.stats
    usage_complete = stats["usage_calls"] == stats["llm_calls"] and stats["llm_calls"] > 0
    rlm_lane: Dict[str, Any] = {
        "tokens": stats["total_tokens"] if usage_complete else None,
        "usage_complete": usage_complete,
        "observed_tokens": stats["total_tokens"] if stats["usage_calls"] else None,
        "cached_tokens": stats["cached_tokens"],
        "prompt_tokens": stats["prompt_tokens"],
        "completion_tokens": stats["completion_tokens"],
        "cost": None,
        "seconds": rlm_seconds,
        "calls": stats["llm_calls"],
        "result": rlm_answer,
        "passed": _grade(corpus, rlm_answer) if rlm_result.succeeded else None,
        "steps": _rlm_steps(serialized["trajectory"]),
    }
    if not rlm_result.succeeded:
        rlm_lane["error"] = serialized["error"]["message"]
    _apply_quality(direct_lane, expected, aliases)
    _apply_quality(rlm_lane, expected, aliases)
    normalized = {
        "schema_version": 1,
        "kind": "recorded",
        "title": "A real Codex subscription experiment",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "max_depth": max_depth,
        "query": query,
        "original_query": original_query,
        "reference_fields": expected,
        "evaluation_label": evaluation_label,
        **evaluation_metadata,
        "context_chars": context_chars,
        "context_preview": context[:1400],
        "context_sha256": context_sha256,
        "sample_note": (
            f"{context_chars:,} {'generated' if corpus else 'custom'} characters · "
            f"{'seed ' + str(seed) + ' · ' if corpus else ''}{captured_at[:10]} UTC · "
            f"{reasoning_effort} reasoning · RLM max depth {max_depth} · "
            "one run per mode (direct first). "
            f"{'Reference answer fields' if expected else 'Ungraded custom input'}; no grading "
            "feedback during either run. Codex CLI tokens include cached input and prompt overhead; "
            "subscription dollar cost is unavailable. This single task does not establish a general advantage."
        ),
        "rlm": rlm_lane,
        "direct": direct_lane,
    }
    for key in ("document", "source_url"):
        if key in live:
            normalized[key] = live[key]
    if display_metadata:
        normalized["title"] = live["title"]
    _write(output, normalized)
    emit("completed", normalized)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chars", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default=MODEL, help="Codex model used by both modes")
    parser.add_argument(
        "--reasoning-effort", choices=["low", "medium", "high", "xhigh", "max"], default="medium"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=1,
        help="RLM depth: 0=root REPL only, 1=plain LM leaves, 2=child RLMs",
    )
    parser.add_argument("--context-file", type=Path, help="UTF-8 source text (requires --query)")
    parser.add_argument("--query", help="Question for custom source text (requires --context-file)")
    parser.add_argument(
        "--expected-fields", type=Path, help="JSON reference map, withheld from the model"
    )
    args = parser.parse_args()
    if (args.context_file is None) != (args.query is None):
        parser.error("--context-file and --query must be provided together")
    record = asyncio.run(
        capture_comparison(
            chars=args.chars,
            seed=args.seed,
            output=args.output,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_depth=args.max_depth,
            query=args.query,
            context=args.context_file.read_text(encoding="utf-8") if args.context_file else None,
            expected_fields=(
                json.loads(args.expected_fields.read_text(encoding="utf-8"))
                if args.expected_fields
                else None
            ),
        )
    )
    print(
        json.dumps(
            {
                mode: {key: record[mode][key] for key in ("tokens", "seconds", "calls", "passed")}
                for mode in ("rlm", "direct")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
