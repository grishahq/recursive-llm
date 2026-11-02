"""Independent document checks for aggregate resources, final paths, and Unicode offsets."""

import asyncio
import io
import json
import random
import threading
import time
import unicodedata
from copy import deepcopy

import pytest

from rlm import MaxIterationsError, RLM, SourceCorpus, SourceDocument
from rlm.document_io import (
    DocumentExtractionWarning,
    DocumentLimits,
    DocumentLoadError,
    read_document_bytes,
)
from rlm.repl import REPLExecutor


def form_resources_pdf(
    shared_form=False, resource_names=("/Form0", "/Form1"), nested=False, resource_cycle=False
):
    """Two compressed forms exceed 1,500 bytes together, not individually."""
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import (
        ArrayObject,
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
        NumberObject,
    )

    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    resources = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    if resource_cycle:
        resource_reference = writer._add_object(resources)
        resources[NameObject("/Cycle")] = resource_reference
    xobjects = DictionaryObject()
    previous = None
    for index in range(2):
        if shared_form and previous is not None:
            xobjects[NameObject(resource_names[index])] = previous
            continue
        form = DecodedStreamObject()
        body = b"BT /F1 12 Tf 0 0 Td (Visible) Tj ET\n"
        form_resources = resources
        if nested and index == 1:
            body = b"/Nested Do\n"
            form_resources = DictionaryObject(
                {NameObject("/XObject"): DictionaryObject({NameObject("/Nested"): previous})}
            )
        form.set_data(body + b" " * 900)
        form.update(
            {
                NameObject("/Type"): NameObject("/XObject"),
                NameObject("/Subtype"): NameObject("/Form"),
                NameObject("/BBox"): ArrayObject(
                    [NumberObject(value) for value in (0, 0, 600, 700)]
                ),
                NameObject("/Resources"): form_resources,
            }
        )
        previous = writer._add_object(form.flate_encode())
        xobjects[NameObject(resource_names[index])] = previous
    if nested:
        xobjects = DictionaryObject({NameObject(resource_names[1]): previous})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/XObject"): xobjects})
    contents = DecodedStreamObject()
    used_names = resource_names[1:] if nested else resource_names
    contents.set_data("".join(f"{name} Do\n" for name in used_names).encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(contents.flate_encode())
    stream = io.BytesIO()
    writer.write(stream)
    return stream.getvalue()


def test_pdf_expanded_limit_includes_form_resources_used_for_text():
    data = form_resources_pdf()
    # Two independently compressed form streams are 936 bytes each. The page's
    # own content stream is tiny, so a page-contents-only sum misses this excess.
    with pytest.raises(DocumentLoadError, match="expanded"):
        read_document_bytes("forms.pdf", data, limits=DocumentLimits(max_expanded_bytes=1_500))
    with pytest.warns(DocumentExtractionWarning):
        document = read_document_bytes(
            "forms.pdf", data, limits=DocumentLimits(max_expanded_bytes=2_000)
        )
    assert document.pages == ("Visible\nVisible",)


def test_reusing_one_form_does_not_charge_its_decoded_bytes_twice():
    with pytest.warns(DocumentExtractionWarning):
        document = read_document_bytes(
            "shared-form.pdf",
            form_resources_pdf(shared_form=True),
            limits=DocumentLimits(max_expanded_bytes=1_500),
        )
    assert document.pages == ("Visible\nVisible",)


@pytest.mark.parametrize("names", [("/P", "/Pg"), ("/Parent", "/Annots")])
def test_pdf_resource_names_are_not_mistaken_for_structural_backreferences(names):
    with pytest.raises(DocumentLoadError, match="expanded"):
        read_document_bytes(
            "named-forms.pdf",
            form_resources_pdf(resource_names=names),
            limits=DocumentLimits(max_expanded_bytes=1_500),
        )


def test_nested_form_bytes_are_counted_and_benign_resource_cycles_terminate():
    data = form_resources_pdf(nested=True)
    with pytest.raises(DocumentLoadError, match="expanded"):
        read_document_bytes("nested.pdf", data, limits=DocumentLimits(max_expanded_bytes=1_500))
    with pytest.warns(DocumentExtractionWarning):
        document = read_document_bytes(
            "nested.pdf", data, limits=DocumentLimits(max_expanded_bytes=2_000)
        )
    assert document.pages == ("Visible",)
    with pytest.warns(DocumentExtractionWarning):
        document = read_document_bytes(
            "resource-cycle.pdf",
            form_resources_pdf(resource_cycle=True),
            limits=DocumentLimits(max_expanded_bytes=2_000),
        )
    assert document.pages == ("Visible\nVisible",)


def independent_normalize(text):
    decomposed = unicodedata.normalize("NFD", text.casefold())
    without_marks = "".join(
        character for character in decomposed if unicodedata.category(character)[0] != "M"
    )
    return " ".join(without_marks.split())


def test_unicode_search_offsets_match_original_across_400_generated_variants():
    randomizer = random.Random(20261102)
    glyphs = [
        "Pétya",
        "Pe\u0301tya",
        "ß",
        "ẞ",
        "İ",
        "ﬃ",
        "ᾲ",
        "ǰ",
        "Å",
        "A\u030a",
        "한",
        "Σ",
        "ς",
        "\r\n\t",
        "\u00a0",
        "\u0301",
        "🙂",
    ]
    for index in range(400):
        original = "Q" + "".join(randomizer.choices(glyphs, k=randomizer.randrange(1, 12))) + "Z"
        prefix = f"<<<{index}:"
        document = SourceDocument("unicode", (prefix + original + ">>>",))
        corpus = SourceCorpus([document])
        matches = corpus.search(independent_normalize(original), window=len(original))
        assert len(matches) == 1
        assert matches[0]["start"] == len(prefix)
        assert matches[0]["end"] == len(prefix) + len(original)
        assert matches[0]["quote"] == original


def supported(citation):
    return json.dumps(
        {
            "answer": "value=7",
            "claims": [{"claim": "value=7", "citations": [citation]}],
            "unknowns": [],
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["native", "directive", "variable", "mutable"])
async def test_all_final_publication_paths_reject_changed_source_then_repair(method):
    document = SourceDocument("report", ("value=7",))
    citation = SourceCorpus([document]).read("report", 1)
    invalid = supported({**citation, "source_sha256": "f" * 64})
    valid = supported(citation)
    if method == "native":
        replies = [invalid, valid]
    elif method == "directive":
        replies = [f"FINAL({invalid!r})", valid]
    elif method == "variable":
        replies = [f"out = {invalid!r}", "FINAL_VAR(out)", valid]
    else:
        replies = [
            f"answer['content'] = {invalid!r}\nanswer['ready'] = True",
            "print('checking the source again')",
            valid,
        ]
    responses = iter(replies)
    requests = []

    async def complete(**kwargs):
        requests.append(deepcopy(kwargs))
        return {"choices": [{"message": {"content": next(responses)}}]}

    result = await RLM(model="mock", completion_handler=complete).acomplete_documents(
        "value?", [document]
    )
    assert result.answer == "value=7" and result.claims[0]["citations"][0] == citation
    assert result.stats["llm_calls"] == len(replies)
    assert sum(event.kind == "final_answer_rejected" for event in result.trajectory) == 1
    assert any("source_sha256" in request["messages"][-1]["content"] for request in requests[1:])


@pytest.mark.asyncio
async def test_duplicate_document_ids_are_rejected_before_provider_calls():
    calls = []

    async def forbidden(**kwargs):
        calls.append(kwargs)
        raise AssertionError("A duplicate input must never reach inference")

    engine = RLM(model="mock", completion_handler=forbidden)
    with pytest.raises(ValueError, match="unique"):
        await engine.acomplete_documents(
            "value?", [SourceDocument("same", ("one",)), SourceDocument("same", ("two",))]
        )
    assert calls == []


@pytest.mark.asyncio
async def test_document_cancellation_kills_source_search_and_freezes_trajectory(monkeypatch):
    executing = threading.Event()
    workers = []

    class ObservedREPL(REPLExecutor):
        def _wait_for_message(self, budget):
            workers.append(self._process)
            executing.set()
            return super()._wait_for_message(budget)

    async def complete(**kwargs):
        return {"choices": [{"message": {"content": "search_sources('missing')"}}]}

    engine = RLM(model="mock", completion_handler=complete, repl_timeout=30)
    monkeypatch.setattr("rlm.core.REPLExecutor", ObservedREPL)
    task = asyncio.create_task(
        engine.acomplete_documents("search", [SourceDocument("large", ("x" * 4_000_000,))])
    )
    try:
        assert await asyncio.to_thread(executing.wait, 15)
        started = time.monotonic()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert time.monotonic() - started < 1.5
        assert workers and all(worker is not None and not worker.is_alive() for worker in workers)
        snapshot = engine.trajectory
        assert snapshot[-1].kind == "run_error"
        await asyncio.sleep(0.05)
        assert engine.trajectory == snapshot
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_all_rejected_document_finals_consume_configured_iterations():
    async def complete(**kwargs):
        return {
            "choices": [{"message": {"content": '{"answer":"uncited","claims":[],"unknowns":[]}'}}]
        }

    engine = RLM(model="mock", completion_handler=complete, max_iterations=3)
    with pytest.raises(MaxIterationsError):
        await engine.acomplete_documents("value?", [SourceDocument("report", ("value=7",))])
    assert engine.stats["llm_calls"] == 3
    assert sum(event.kind == "final_answer_rejected" for event in engine.trajectory) == 3
