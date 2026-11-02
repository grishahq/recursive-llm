"""End-to-end source validation across real REPL workers and recursive nodes."""

import asyncio
import json
from copy import deepcopy

import pytest

from rlm import DocumentResult, MaxIterationsError, RLM, SourceCorpus, SourceDocument


def response(text):
    return {"choices": [{"message": {"content": text}}]}


def supported(citation, answer="value=7"):
    return json.dumps(
        {
            "answer": answer,
            "claims": [{"claim": answer, "citations": [citation]}],
            "unknowns": [],
        }
    )


@pytest.mark.asyncio
async def test_fabricated_citation_is_rejected_and_repaired_with_original_source(tmp_path):
    doc = SourceDocument("report", ("value=7",), "Original report")
    citation = SourceCorpus([doc]).read("report", 1)
    forged = {**citation, "quote": "value=9"}
    replies = iter(
        [
            f"FINAL({supported(forged)!r})",
            "citation = read_source('report', 1)\n"
            "out = json.dumps({'answer': 'value=7', 'claims': "
            "[{'claim': 'value=7', 'citations': [citation]}], 'unknowns': []})",
            "FINAL_VAR(out)",
        ]
    )
    requests = []

    async def completion(**kwargs):
        requests.append(deepcopy(kwargs))
        return response(next(replies))

    result = await RLM(model="mock", completion_handler=completion).acomplete_documents(
        "What is the value?", [doc]
    )
    assert isinstance(result, DocumentResult)
    assert result.answer == "value=7"
    assert result.claims[0]["citations"][0] == citation
    assert result.stats["llm_calls"] == 3
    assert any(e.kind == "final_answer_rejected" for e in result.trajectory)
    assert any(e.kind == "source_read" for e in result.trajectory)
    source_event = next(e for e in result.trajectory if e.kind == "source_read")
    assert "quote" not in source_event.data and source_event.data["quote_chars"] == 7
    assert "Final answer rejected" in requests[1]["messages"][-1]["content"]
    assert "source_manifest()" in requests[0]["messages"][0]["content"]
    output = tmp_path / "documents.jsonl"
    result.write_jsonl(output)
    persisted = json.loads(output.read_text())
    assert persisted["document_schema_version"] == 1
    assert persisted["claims"][0]["citations"][0] == citation
    assert persisted["run"]["answer"] == result.completion.answer
    persisted["sources"][0]["title"] = "modified"
    assert result.sources[0]["title"] == "Original report"
    assert result.config["model"] == "mock" and result.succeeded


@pytest.mark.asyncio
async def test_children_share_source_identity_and_validate_their_own_answers():
    doc = SourceDocument("report", ("value=7",))
    citation = SourceCorpus([doc]).read("report", 1)
    forged = {**citation, "source_sha256": "0" * 64}
    replies = iter(
        [
            "child = rlm_query('Find the value using original source citations')",
            f"FINAL({supported(forged)!r})",
            "c = search_sources('VALUE', limit=1)[0]\n"
            "out = json.dumps({'answer': 'value=7', 'claims': "
            "[{'claim': 'value=7', 'citations': [c]}], 'unknowns': []})",
            "FINAL_VAR(out)",
            "FINAL_VAR(child)",
        ]
    )
    child_queries = []

    async def completion(*, messages, **kwargs):
        child_queries.append(messages[1]["content"])
        return response(next(replies))

    result = await RLM(
        model="mock", max_depth=2, completion_handler=completion
    ).acomplete_documents("Delegate the value lookup", [doc])
    assert result.answer == "value=7" and result.stats["llm_calls"] == 5
    assert "Find the value" in child_queries[1]
    assert any(e.kind == "final_answer_rejected" and e.depth == 1 for e in result.trajectory)
    assert any(e.kind == "source_search" and e.depth == 1 for e in result.trajectory)
    validated = [e.depth for e in result.trajectory if e.kind == "final_answer_validated"]
    assert validated == [1, 0]


@pytest.mark.asyncio
async def test_concurrent_document_runs_and_plain_runs_do_not_share_corpora():
    async def completion(*, messages, **kwargs):
        if "source_manifest()" not in messages[0]["content"]:
            return response('FINAL("plain")')
        if len(messages) == 2:
            return response(
                "c = read_source('same-id', 1)\n"
                "out = json.dumps({'answer': c['quote'], 'claims': "
                "[{'claim': c['quote'], 'citations': [c]}], 'unknowns': []})"
            )
        await asyncio.sleep(0.01)
        return response("FINAL_VAR(out)")

    engine = RLM(model="mock", max_depth=0, completion_handler=completion)
    left, right, plain = await asyncio.gather(
        engine.acomplete_documents("left", [SourceDocument("same-id", ("left=1",))]),
        engine.acomplete_documents("right", [SourceDocument("same-id", ("right=2",))]),
        engine.acomplete("ordinary task", "ordinary context"),
    )
    assert left.answer == "left=1" and right.answer == "right=2" and plain == "plain"
    assert (
        left.claims[0]["citations"][0]["source_sha256"]
        != right.claims[0]["citations"][0]["source_sha256"]
    )


def test_document_sync_api_and_custom_validator_are_both_enforced():
    doc = SourceDocument("x", ("value=7",))
    valid = supported(SourceCorpus([doc]).read("x", 1))
    replies = iter([f"FINAL({valid!r})", f"FINAL({valid!r})"])
    seen = []

    def validator(answer):
        seen.append(answer)
        return "Perform one more review" if len(seen) == 1 else None

    async def completion(**kwargs):
        return response(next(replies))

    result = RLM(
        model="mock", completion_handler=completion, final_answer_validator=validator
    ).complete_documents("value?", [doc])
    assert result.answer == "value=7" and len(seen) == 2


@pytest.mark.asyncio
async def test_invalid_final_exhausts_budget_and_preserves_diagnostics():
    async def completion(**kwargs):
        return response('FINAL("unsupported answer without evidence")')

    engine = RLM(model="mock", max_iterations=1, completion_handler=completion)
    with pytest.raises(MaxIterationsError):
        await engine.acomplete_documents("value?", [SourceDocument("x", ("value=7",))])
    assert any(e.kind == "final_answer_rejected" for e in engine.trajectory)
    assert engine.stats["llm_calls"] == 1


@pytest.mark.asyncio
async def test_empty_document_request_fails_before_any_model_call():
    async def forbidden(**kwargs):
        raise AssertionError("Invalid document requests must not invoke the provider")

    with pytest.raises(ValueError, match="at least one"):
        await RLM(model="mock", completion_handler=forbidden).acomplete_documents("query", [])


@pytest.mark.asyncio
async def test_native_document_json_uses_the_same_citation_and_custom_validation():
    doc = SourceDocument("x", ("value=7",))
    citation = SourceCorpus([doc]).read("x", 1)
    replies = iter([supported({**citation, "quote": "value=9"}), supported(citation)])
    seen = []

    def validator(raw):
        seen.append(raw)
        return None

    async def completion(**kwargs):
        return response(next(replies))

    result = await RLM(
        model="mock", completion_handler=completion, final_answer_validator=validator
    ).acomplete_documents("value?", [doc])
    assert result.answer == "value=7" and len(seen) == 1
    assert any(e.kind == "final_answer_rejected" for e in result.trajectory)
    assert all(e.kind != "repl_step" for e in result.trajectory)
    final = next(e for e in result.trajectory if e.kind == "final_answer")
    assert final.data["method"] == "document_json"


@pytest.mark.asyncio
async def test_native_json_shortcut_does_not_change_ordinary_string_context_runs():
    raw = supported(SourceCorpus([SourceDocument("x", ("value=7",))]).read("x", 1))
    replies = iter([raw, 'FINAL("ordinary completion")'])

    async def completion(**kwargs):
        return response(next(replies))

    result = await RLM(model="mock", completion_handler=completion).acomplete_result(
        "query", "source"
    )
    assert result.answer == "ordinary completion" and result.stats["llm_calls"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_kind", ["duplicate", "nonfinite", "oversized"])
async def test_native_json_cannot_bypass_strict_or_size_validation(invalid_kind):
    doc = SourceDocument("x", ("value=7",))
    valid = supported(SourceCorpus([doc]).read("x", 1))
    if invalid_kind == "duplicate":
        invalid = valid.replace('"answer":', '"answer": "unsupported", "answer":', 1)
    elif invalid_kind == "nonfinite":
        invalid = valid.replace('"unknowns": []', '"unknowns": [NaN]')
    else:
        invalid = valid.replace('"value=7"', json.dumps("x" * 1_000_001), 1)
    replies = iter([invalid, valid])

    async def completion(**kwargs):
        return response(next(replies))

    result = await RLM(model="mock", completion_handler=completion).acomplete_documents(
        "value?", [doc]
    )
    assert result.answer == "value=7" and result.stats["llm_calls"] == 2
    assert sum(e.kind == "final_answer_rejected" for e in result.trajectory) == 1
    assert all(e.kind != "repl_step" for e in result.trajectory)
