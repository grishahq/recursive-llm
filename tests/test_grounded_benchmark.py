"""Evidence-bound task grading and bounded generated-policy execution."""

import copy
import json

import pytest

from benchmarks.grounded_documents import (
    DATASET_VERSION,
    SOURCE_SPECS,
    _transport_usage,
    direct_sources,
    grade_answer,
    grade_python_function,
    synthetic_tasks,
)
from rlm.sources import SourceCorpus

CORRECT_FUNCTION = """def application_quote(application_count, approved_nonprofit, expedited):
    fee = 120 * (0.25 if approved_nonprofit else 1)
    days = 14 if approved_nonprofit else (5 if expedited else 20)
    return (fee * application_count, days)
"""


def golden(task):
    corpus = SourceCorpus(task.documents)
    documents = {document.document_id: document for document in task.documents}
    claims = []
    for label, evidence in task.evidence.items():
        citations = []
        for item in evidence:
            original = documents[item.document_id].pages[item.page - 1]
            first = original.index(item.text)
            citations.append(
                corpus.read(item.document_id, item.page, first, first + len(item.text))
            )
        claims.append({"claim": f"{label}: {task.expected[label][0]}", "citations": citations})
    return {
        "answer": (
            CORRECT_FUNCTION
            if task.answer_kind == "python"
            else "\n".join(f"{label}: {values[0]}" for label, values in task.expected.items())
        ),
        "claims": claims,
        "unknowns": list(task.unknowns),
    }


@pytest.mark.parametrize("task", synthetic_tasks(), ids=lambda task: task.name)
def test_gold_answers_cover_distributed_rules_exception_unicode_and_absence(task):
    grade = grade_answer(task, json.dumps(golden(task)))
    assert grade["passed"], grade
    assert grade["supported_gold_fields"] == len(task.evidence)


@pytest.mark.parametrize("value", ["-30", "130", "30, actually 120", "not 30", "30 or 120"])
def test_correct_number_inside_a_wrong_answer_does_not_pass(value):
    task = synthetic_tasks()[0]
    response = golden(task)
    response["answer"] = f"approved_nonprofit_fee_eur: {value}"
    assert not grade_answer(task, json.dumps(response))["passed"]


def test_correct_answer_does_not_hide_a_contradictory_claim():
    task = synthetic_tasks()[0]
    response = golden(task)
    response["claims"][0]["claim"] = "approved_nonprofit_fee_eur: 120"
    assert not grade_answer(task, json.dumps(response))["passed"]


def test_folded_name_search_does_not_allow_losing_the_original_accent_in_answer():
    task = next(task for task in synthetic_tasks() if task.name == "synthetic-unicode")
    response = golden(task)
    response["answer"] = "review_owner: Petya Sokolova"
    response["claims"][0]["claim"] = response["answer"]
    assert not grade_answer(task, json.dumps(response))["passed"]


def test_valid_unrelated_quotation_is_not_supporting_evidence():
    task = synthetic_tasks()[0]
    response = golden(task)
    response["claims"][0]["citations"] = [
        SourceCorpus(task.documents).read("aurora-policy", 1, 0, 80)
    ]
    grade = grade_answer(task, json.dumps(response))
    assert grade["provenance_passed"]
    assert grade["factual_passed"]
    assert not grade["passed"]
    assert grade["supported_gold_fields"] == 0


def test_computed_fee_requires_both_distributed_rules():
    task = synthetic_tasks()[0]
    response = golden(task)
    response["claims"][0]["citations"].pop()
    assert not grade_answer(task, json.dumps(response))["passed"]


@pytest.mark.parametrize("change", ["hash", "page", "offset", "quote_type"])
def test_invalid_citation_fails_provenance_without_losing_factual_score(change):
    task = synthetic_tasks()[0]
    response = golden(task)
    citation = response["claims"][0]["citations"][0]
    if change == "hash":
        citation["source_sha256"] = "0" * 64
    elif change == "page":
        citation["page"] = 1
    elif change == "offset":
        citation["start"] += 1
    else:
        citation["quote"] = None
    grade = grade_answer(task, json.dumps(response))
    assert grade["factual_passed"]
    assert not grade["provenance_passed"]
    assert not grade["passed"]


def test_duplicate_claims_and_duplicate_fields_are_rejected():
    task = synthetic_tasks()[0]
    response = golden(task)
    response["claims"].append(copy.deepcopy(response["claims"][0]))
    assert not grade_answer(task, json.dumps(response))["passed"]
    response = golden(task)
    response["answer"] += "\napproved_nonprofit_fee_eur: 30"
    assert not grade_answer(task, json.dumps(response))["passed"]


def test_absence_is_explicit_and_cannot_be_converted_to_zero():
    task = next(task for task in synthetic_tasks() if task.name == "synthetic-absence")
    response = golden(task)
    response["answer"] = "renewal_fee_eur: 0"
    assert not grade_answer(task, json.dumps(response))["passed"]
    response = golden(task)
    response["unknowns"] = []
    assert not grade_answer(task, json.dumps(response))["passed"]


def test_generated_policy_function_is_executed_on_hidden_numeric_inputs():
    correct = grade_python_function(CORRECT_FUNCTION)
    assert correct["passed"]
    assert correct["tested_inputs"] == 16
    hardcoded_one_application = CORRECT_FUNCTION.replace("fee * application_count", "fee")
    assert not grade_python_function(hardcoded_one_application)["passed"]
    wrong_exception = CORRECT_FUNCTION.replace(
        "14 if approved_nonprofit else (5 if expedited else 20)",
        "5 if expedited else (14 if approved_nonprofit else 20)",
    )
    assert not grade_python_function(wrong_exception)["passed"]


def test_generated_function_cannot_infer_flags_from_correlated_quantity_examples():
    count_cheat = """def application_quote(application_count, approved_nonprofit, expedited):
    if application_count == 1:
        return (120, 20)
    if application_count == 3:
        return (360, 5)
    if application_count == 7:
        return (210, 14)
    return (390, 14)
"""
    assert not grade_python_function(count_cheat)["passed"]


def test_generated_function_rejects_imports_top_level_effects_and_runaway_code():
    assert not grade_python_function("import os\n" + CORRECT_FUNCTION)["passed"]
    forbidden = "def application_quote(application_count, approved_nonprofit, expedited):\n    import os\n    return (120, 20)"
    assert not grade_python_function(forbidden)["passed"]
    runaway = "def application_quote(application_count, approved_nonprofit, expedited):\n    while True:\n        pass"
    grade = grade_python_function(runaway)
    assert not grade["passed"]
    assert "Timeout" in grade["failures"][0]


def test_direct_segments_preserve_every_source_character_and_citation_offsets():
    task = synthetic_tasks()[0]
    corpus = SourceCorpus(task.documents)
    for document, supplied in zip(task.documents, direct_sources(task.documents, 600)):
        for page, original in enumerate(document.pages, 1):
            segments = [item for item in supplied["segments"] if item["page"] == page]
            assert "".join(segment["quote"] for segment in segments) == original
            for segment in segments:
                citation = {
                    "document_id": supplied["document_id"],
                    "source_sha256": supplied["source_sha256"],
                    **segment,
                }
                assert citation == corpus.read(
                    document.document_id, page, segment["start"], segment["end"]
                )


def test_dataset_pins_raw_and_extracted_public_sources():
    assert DATASET_VERSION == "grounded-documents-v1"
    for spec in SOURCE_SPECS.values():
        assert len(spec["raw_sha256"]) == 64
        assert len(spec["content_sha256"]) == 64


@pytest.mark.parametrize(
    "raw",
    [
        '{"answer":"x","answer":"y","claims":[],"unknowns":["x"]}',
        '{"answer":"x","claims":[],"unknowns":[NaN]}',
    ],
)
def test_malformed_json_cannot_bypass_the_grader(raw):
    assert not grade_answer(synthetic_tasks()[0], raw)["passed"]


def test_failed_transport_contract_still_accounts_for_reported_cli_tokens():
    call = {
        "status": "failed",
        "error": {"message": "empty response"},
        "events": [
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 11330, "cached_input_tokens": 0, "output_tokens": 24},
            }
        ],
    }
    usage = _transport_usage([call])
    assert usage["total_tokens"] == 11354
    assert usage["failed_calls_with_known_usage"] == 1
    assert usage["unknown_usage_calls"] == 0
    assert call["status"] == "failed"


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": True, "cached_input_tokens": 0, "output_tokens": 24},
        {"input_tokens": 1, "cached_input_tokens": 2, "output_tokens": 24},
        {"input_tokens": 11330, "cached_input_tokens": 0, "output_tokens": -1},
    ],
)
def test_failed_call_accounting_rejects_invalid_usage(usage):
    result = _transport_usage(
        [{"status": "failed", "events": [{"type": "turn.completed", "usage": usage}]}]
    )
    assert result["total_tokens"] == 0
    assert result["unknown_usage_calls"] == 1
