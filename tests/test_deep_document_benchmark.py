"""Adversarial controls for the deeper document evaluation contract."""

import asyncio
import copy
import json
from argparse import Namespace

import pytest

from benchmarks.deep_document_validation import (
    DeepTask,
    _empty_cli_control,
    _fixed_attempt,
    adversarial_fixtures,
    fixture_task,
    grade,
    live_suite,
    oracle,
    search_controls,
)
from benchmarks.grounded_documents import GoldEvidence, GroundedTask
from rlm.sources import SourceCorpus, SourceDocument


def test_valid_control_and_legitimate_alternative_source_both_pass():
    document = SourceDocument("r", ("revenue: 7", "reported total: 7"))
    alternatives = {
        "value": (
            (GoldEvidence("r", 1, "revenue: 7"),),
            (GoldEvidence("r", 2, "reported total: 7"),),
        )
    }
    task = DeepTask(
        GroundedTask(
            "alternatives",
            "value?",
            (document,),
            {"value": ("7",)},
            {"value": alternatives["value"][0]},
        ),
        alternatives,
    )
    assert grade(task, oracle(task, 0))["passed"]
    assert grade(task, oracle(task, 1))["passed"]


@pytest.mark.parametrize("replacement", ["-7", "17", "not 7", "7 or 9", "7, but actually 9"])
def test_wrong_or_contradictory_values_cannot_pass_by_containing_the_gold_number(replacement):
    task = fixture_task()
    response = json.loads(oracle(task))
    response["answer"] = f"value: {replacement}"
    assert not grade(task, json.dumps(response))["passed"]


def test_valid_but_unrelated_quote_fails_evidence_coverage():
    task = fixture_task()
    response = json.loads(oracle(task))
    response["claims"][0]["citations"] = [
        SourceCorpus(task.task.documents).read("fixture", 1, 9, 22)
    ]
    result = grade(task, json.dumps(response))
    assert result["citation_structure_passed"]
    assert result["field_contract_passed"]
    assert not result["evidence_coverage_passed"]
    assert not result["passed"]


@pytest.mark.parametrize("mutation", ["quote", "hash", "bool_page", "offset", "id"])
def test_adversarial_reference_cannot_bypass_the_deep_grader(mutation):
    task = fixture_task()
    response = json.loads(oracle(task))
    citation = response["claims"][0]["citations"][0]
    if mutation == "quote":
        citation["quote"] = "value: 9"
    elif mutation == "hash":
        citation["source_sha256"] = "0" * 64
    elif mutation == "bool_page":
        citation["page"] = True
    elif mutation == "offset":
        citation["start"] = 1
    else:
        citation["document_id"] = "unknown"
    result = grade(task, json.dumps(response))
    assert result["field_contract_passed"]
    assert not result["citation_structure_passed"]
    assert not result["passed"]


def test_duplicate_claim_or_field_cannot_increase_the_score():
    task = fixture_task()
    response = json.loads(oracle(task))
    response["claims"].append(copy.deepcopy(response["claims"][0]))
    assert not grade(task, json.dumps(response))["passed"]
    response = json.loads(oracle(task))
    response["answer"] += "\nvalue: 7"
    assert not grade(task, json.dumps(response))["passed"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [item for item in adversarial_fixtures() if item["expected_current"] == "reject_then_repair"],
    ids=lambda case: case["name"],
)
async def test_current_api_rejects_fixed_forgery_before_accepting_repair(case):
    result = await _fixed_attempt("current-document-api", None, case)
    assert result["status"] == "accepted"
    assert result["grade"]["passed"]
    assert result["runtime_stats"]["llm_calls"] == 2
    assert any(event["kind"] == "final_answer_rejected" for event in result["trajectory"])
    assert result["live_model_calls"] == 0


@pytest.mark.asyncio
async def test_real_quote_can_still_be_used_for_a_wrong_claim_and_must_remain_a_negative_control():
    case = next(case for case in adversarial_fixtures() if case.get("negative_control"))
    result = await _fixed_attempt("current-document-api", None, case)
    assert result["status"] == "accepted"
    assert result["grade"]["citation_structure_passed"]
    assert not result["grade"]["field_contract_passed"]
    assert not result["grade"]["passed"]


@pytest.mark.asyncio
async def test_empty_cli_fixture_recovers_with_all_known_usage_preserved():
    result = await _empty_cli_control(None, True)
    assert result["status"] == "accepted"
    assert result["answer"] == "recovered"
    assert result["runtime_stats"]["total_tokens"] == 224
    assert result["live_model_calls"] == 0


def test_normalized_search_returns_original_spelling_and_correct_combining_spans():
    records = search_controls()
    assert len(records) == 4
    assert all(item["head_literal_find"] == -1 for item in records)
    assert all(item["correct_original_span"] for item in records)


@pytest.mark.asyncio
async def test_persistent_429_stops_new_attempts_with_at_most_two_in_flight(tmp_path, monkeypatch):
    import benchmarks.deep_document_validation as deep

    calls = []

    async def login(_self):
        return "ChatGPT"

    async def fake_run(task, mode, args):
        calls.append((task.name, mode))
        await asyncio.sleep(0.01)
        return {
            "task": task.name,
            "mode": mode,
            "status": "failed",
            "raw_answer": None,
            "grade": {},
            "transport_calls": [],
            "runtime_stats": {},
            "elapsed_seconds": 0.01,
            "error": {"type": "CodexError", "message": "429 Too Many Requests"},
        }

    monkeypatch.setattr(deep, "prepare", lambda _path: {"gold_sha256": "fixed"})
    monkeypatch.setattr(deep, "build_tasks", lambda _documents: (fixture_task(),))
    monkeypatch.setattr(deep, "load_public_documents", lambda _path: {})
    monkeypatch.setattr(deep, "snapshot_runtime", lambda _path: {})
    monkeypatch.setattr(deep.CodexCompletion, "check_login", login)
    monkeypatch.setattr(deep, "run_one", fake_run)
    args = Namespace(
        directory=tmp_path,
        task=None,
        output=tmp_path / "runs",
        model="fake",
        reasoning_effort="low",
        call_timeout=1,
        repeats=4,
        parallel=2,
        max_calls=10,
        max_tokens=150000,
        timeout=240,
    )
    summary = await live_suite(args)
    assert summary["circuit_breaker_triggered"]
    assert len(calls) <= 3
    assert len(summary["skipped"]) >= 5
