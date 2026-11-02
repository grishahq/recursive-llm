"""Public evidence retains answers and accounting while removing full contexts."""

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

EXPORTER_PATH = (
    Path(__file__).resolve().parents[1] / "experiments/publication-20261102/export_evidence.py"
)
SPEC = importlib.util.spec_from_file_location("public_evidence_export", EXPORTER_PATH)
assert SPEC and SPEC.loader
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


def capture():
    secret = "FULL_SOURCE_MUST_NOT_BE_PUBLISHED_" * 2_000
    answer = json.dumps(
        {
            "answer": "fee: 7",
            "claims": [{"claim": "fee: 7", "citations": [{"quote": "The fee is 7."}]}],
            "unknowns": [],
        }
    )
    usage = {"input_tokens": 123, "cached_input_tokens": 23, "output_tokens": 9}
    return {
        "task": "fee",
        "mode": "direct",
        "model": "test-model",
        "reasoning_effort": "low",
        "query": "What is the fee?",
        "status": "completed",
        "raw_answer": answer,
        "elapsed_seconds": 1.2,
        "sources": [{"document_id": "policy", "page_count": 2, "context": secret}],
        "config": {"max_depth": 0, "messages": [secret]},
        "transport_usage": {"llm_calls": 1, "total_tokens": 132, "estimated_cost_usd": None},
        "runtime_stats": None,
        "grade": {
            "passed": True,
            "factual_passed": True,
            "provenance_passed": True,
            "supported_gold_fields": 1,
            "required_gold_fields": 1,
            "failures": [],
            "answer_grade": {"passed": True, "failures": []},
        },
        "transport_calls": [
            {
                "model": "test-model",
                "status": "completed",
                "elapsed_seconds": 1.1,
                "messages": [{"content": secret}],
                "stdout": secret,
                "stderr": secret,
                "command": [secret],
                "events": [
                    {"type": "turn.completed", "usage": usage},
                    {"type": "item.updated", "item": {"text": secret}},
                ],
                "response": {"choices": [{"message": {"content": "short response"}}]},
            }
        ],
        "trajectory": [
            {"kind": "run_start", "depth": 0, "data": {"context": secret}},
            {
                "kind": "source_read",
                "sequence": 2,
                "depth": 1,
                "data": {
                    "document_id": "policy",
                    "page": 2,
                    "start": 0,
                    "end": 12,
                    "quote": "The fee is 7.",
                    "unlisted": secret,
                },
            },
        ],
        "result": {"run": {"context": secret}},
        "unlisted": secret,
    }


def test_public_projection_omits_full_source_and_preserves_answers_grades_and_usage():
    original = capture()
    before = deepcopy(original)
    result = EXPORTER.export_record(original, "a" * 64)
    serialized = json.dumps(result)
    assert "FULL_SOURCE_MUST_NOT_BE_PUBLISHED_" not in serialized
    assert result["raw_answer"] == original["raw_answer"]
    assert "The fee is 7." in result["raw_answer"]
    assert result["grade"] == original["grade"]
    assert result["transport_usage"] == original["transport_usage"]
    assert result["runtime_stats"] is None
    assert result["transport_calls"][0]["usage"] == {
        "input_tokens": 123,
        "cached_input_tokens": 23,
        "output_tokens": 9,
    }
    assert result["transport_calls"][0]["response_chars"] == len("short response")
    assert (
        result["transport_calls"][0]["response_sha256"]
        == hashlib.sha256(b"short response").hexdigest()
    )
    assert result["call_summary"] == {
        "observed_calls": 1,
        "known_usage_calls": 1,
        "unknown_usage_calls": 0,
        "observed_total_tokens": 132,
        "usage_complete": True,
    }
    assert result["trace_summary"]["event_counts_by_depth"] == {
        "0": {"run_start": 1},
        "1": {"source_read": 1},
    }
    assert result["trace_summary"]["source_reads"][0]["quote_chars"] == 13
    assert original == before


@pytest.mark.parametrize(
    "events",
    [
        [],
        [{"type": "turn.completed", "usage": {}}],
        [
            {
                "type": "turn.completed",
                "usage": {"input_tokens": True, "cached_input_tokens": 0, "output_tokens": 1},
            }
        ],
        [
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 1, "cached_input_tokens": 2, "output_tokens": 1},
            }
        ],
        [
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1},
            }
        ]
        * 2,
    ],
)
def test_unknown_or_ambiguous_usage_remains_unknown_instead_of_zero(events):
    raw = capture()
    raw["transport_calls"][0]["events"] = events
    raw["transport_calls"][0].pop("response")
    result = EXPORTER.export_record(raw, "raw-hash")
    assert result["transport_calls"][0]["usage"] is None
    assert result["transport_calls"][0]["usage_known"] is False
    assert result["transport_calls"][0]["response_chars"] is None
    assert result["call_summary"]["observed_total_tokens"] is None
    assert result["call_summary"]["unknown_usage_calls"] == 1
    assert result["call_summary"]["usage_complete"] is False
    assert result["transport_usage"] == raw["transport_usage"]


def test_failed_call_usage_empty_response_and_recorded_unknown_counts_are_preserved():
    raw = capture()
    call = raw["transport_calls"][0]
    call["status"] = "failed"
    call["response"]["choices"][0]["message"]["content"] = ""
    raw["transport_calls"].append({"model": "test-model", "status": "failed", "events": []})
    raw["transport_usage"].update(unknown_usage_calls=1, failed_calls_with_known_usage=1)
    raw["runtime_stats"] = {
        "llm_calls": 2,
        "total_tokens": 132,
        "budget": {"max_calls": 2, "reserved_calls": 2},
        "by_model": {"test-model": {"calls": 2, "total_tokens": 132}},
    }
    result = EXPORTER.export_record(raw, "raw-hash")
    assert result["transport_usage"] == raw["transport_usage"]
    assert result["runtime_stats"] == raw["runtime_stats"]
    assert result["transport_calls"][0]["response_chars"] == 0
    assert result["transport_calls"][0]["response_sha256"] == hashlib.sha256(b"").hexdigest()
    assert result["call_summary"]["observed_total_tokens"] == 132
    assert result["call_summary"]["unknown_usage_calls"] == 1
    assert result["call_summary"]["usage_complete"] is False


def test_code_grading_retains_test_counts_and_observed_numeric_outputs():
    raw = capture()
    raw["grade"]["answer_grade"].update(
        tested_inputs=4, observed_outputs=[[120, 20], [360, 5], [210, 14], [390, 14]]
    )
    assert EXPORTER.export_record(raw, "raw-hash")["grade"] == raw["grade"]


def test_deep_grading_retains_contract_citations_and_evidence_dimensions():
    raw = capture()
    raw["grade"] = {
        "passed": False,
        "field_contract_passed": True,
        "citation_structure_passed": True,
        "evidence_coverage_passed": False,
        "supported_fields": 2,
        "required_fields": 3,
        "failures": ["Required exception evidence missing"],
        "grader_version": "deep-document-fields-v1",
    }
    assert EXPORTER.export_record(raw, "raw-hash")["grade"] == raw["grade"]


def test_event_response_fallback_and_result_trajectory_do_not_copy_payloads():
    raw = capture()
    raw["transport_calls"][0].pop("response")
    raw["transport_calls"][0]["events"].append(
        {"type": "item.completed", "item": {"type": "agent_message", "text": "fallback"}}
    )
    raw["result"] = {"run": {"trajectory": raw.pop("trajectory")}}
    raw["error"] = {"type": "TimeoutError", "message": "Call timed out", "debug": "omitted"}
    raw["integration_checks"] = {
        "forced_child": True,
        "child_calls": 3,
        "child_used": True,
        "note": "Integration only",
        "debug": "omitted",
    }
    result = EXPORTER.export_record(raw, "raw-hash")
    assert result["transport_calls"][0]["response_chars"] == 8
    assert result["trace_summary"]["event_count"] == 2
    assert result["error"] == {"type": "TimeoutError", "message": "Call timed out"}
    assert "debug" not in result["integration_checks"]


def test_export_tree_hashes_original_bytes_mirrors_paths_and_never_rewrites_inputs(tmp_path):
    input_root = tmp_path / "raw"
    output_root = tmp_path / "public"
    (input_root / "runs").mkdir(parents=True)
    raw = json.dumps(capture(), ensure_ascii=False).encode()
    source = input_root / "runs" / "attempt.json"
    source.write_bytes(raw)
    (input_root / "summary.json").write_text('{"summary":true}')
    manifest = EXPORTER.export_tree(input_root, output_root)
    exported_bytes = (output_root / "runs" / "attempt.json").read_bytes()
    exported = json.loads(exported_bytes)
    assert source.read_bytes() == raw
    assert manifest["record_count"] == 1 and manifest["skipped_json_files"] == 1
    assert exported["raw_capture_sha256"] == hashlib.sha256(raw).hexdigest()
    assert manifest["records"][0]["public_sha256"] == hashlib.sha256(exported_bytes).hexdigest()
    assert manifest["public_bytes"] < manifest["raw_bytes"] / 10
    assert json.loads((output_root / "public-manifest.json").read_text()) == manifest
    with pytest.raises(ValueError, match="empty"):
        EXPORTER.export_tree(input_root, output_root)
    assert source.read_bytes() == raw


@pytest.mark.parametrize("relationship", ["same", "child", "parent"])
def test_export_tree_rejects_overlapping_paths_before_writing(tmp_path, relationship):
    input_root = tmp_path / "raw"
    input_root.mkdir()
    output = {"same": input_root, "child": input_root / "public", "parent": tmp_path}[relationship]
    with pytest.raises(ValueError, match="disjoint"):
        EXPORTER.export_tree(input_root, output)
    assert list(input_root.iterdir()) == []
