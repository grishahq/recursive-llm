"""Source identity, Unicode span mapping, and adversarial citation validation."""

import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest

from rlm.sources import (
    MAX_ANSWER_CHARS,
    MAX_READ_CHARS,
    MAX_SEARCH_LIMIT,
    MAX_SEARCH_TERM_CHARS,
    MAX_SEARCH_WINDOW,
    SourceCorpus,
    SourceDocument,
)


def make_answer(citation, claim="The source contains the quoted text"):
    return {
        "answer": "A supported answer.",
        "claims": [{"claim": claim, "citations": [citation]}],
        "unknowns": [],
    }


def test_sources_snapshot_input_and_content_hash_preserves_page_boundaries():
    pages = ["ab", "c"]
    document = SourceDocument("report", pages, title="Report")
    documents = [document]
    corpus = SourceCorpus(documents)
    pages[0] = "changed"
    documents.clear()

    assert corpus.documents == (document,)
    assert document.pages == ("ab", "c")
    assert document.content_sha256 != SourceDocument("report", ("a", "bc")).content_sha256
    assert document.content_sha256 != SourceDocument("report", ("abc",)).content_sha256
    assert document.content_sha256 == SourceDocument("other", ("ab", "c"), "Other").content_sha256
    digest = hashlib.sha256(b"recursive-llm/source-pages/v1\0")
    digest.update((2).to_bytes(8, "big"))
    for page in (b"ab", b"c"):
        digest.update(len(page).to_bytes(8, "big"))
        digest.update(page)
    assert document.content_sha256 == digest.hexdigest()
    with pytest.raises(FrozenInstanceError):
        document.title = "Replacement"
    with pytest.raises(FrozenInstanceError):
        corpus.documents = ()
    with pytest.raises(TypeError):
        corpus._by_id["other"] = document


@pytest.mark.parametrize(
    "document_id,pages,title",
    [
        ("", ["page"], ""),
        (" \n", ["page"], ""),
        (None, ["page"], ""),
        ("id", [], ""),
        ("id", "one page", ""),
        ("id", None, ""),
        ("id", [None], ""),
        ("id", ["\ud800"], ""),
        ("id", ["page"], None),
    ],
)
def test_source_rejects_invalid_identity_and_page_inputs(document_id, pages, title):
    with pytest.raises(ValueError):
        SourceDocument(document_id, pages, title)


def test_corpus_requires_unique_ids_and_source_documents():
    with pytest.raises(ValueError, match="unique"):
        SourceCorpus([SourceDocument("same", ["first"]), SourceDocument("same", ["second"])])
    with pytest.raises(ValueError, match="SourceDocument"):
        SourceCorpus(["text"])
    with pytest.raises(ValueError, match="sequence"):
        SourceCorpus(None)
    empty = SourceCorpus([])
    assert empty.manifest() == []
    assert empty.context == ""
    assert empty.search("missing") == []


def test_manifest_is_detached_and_flat_context_keeps_physical_page_markers():
    corpus = SourceCorpus([SourceDocument("report", ["first\r\nline", "", "last"], "Title")])
    metadata = corpus.manifest()
    assert metadata == [
        {
            "document_id": "report",
            "title": "Title",
            "source_sha256": corpus.documents[0].content_sha256,
            "page_count": 3,
            "character_count": 15,
        }
    ]
    metadata[0]["title"] = "Changed"
    assert corpus.manifest()[0]["title"] == "Title"
    assert '=== DOCUMENT {"document_id": "report", "title": "Title"} ===' in corpus.context
    assert "=== PAGE 1 ===\nfirst\r\nline" in corpus.context
    assert "=== PAGE 2 ===\n" in corpus.context
    assert "=== PAGE 3 ===\nlast" in corpus.context


def test_read_uses_original_unicode_codepoint_offsets_and_half_open_spans():
    document = SourceDocument("unicode", ["A🙂Pétya\r\n", ""])
    corpus = SourceCorpus([document])
    assert corpus.read("unicode", 1, 1, 3) == {
        "document_id": "unicode",
        "source_sha256": document.content_sha256,
        "page": 1,
        "start": 1,
        "end": 3,
        "quote": "🙂P",
    }
    assert corpus.read("unicode", 1, 7)["quote"] == "\r\n"
    assert corpus.read("unicode", 2)["quote"] == ""


@pytest.mark.parametrize(
    "page,start,end",
    [
        (True, 0, 1),
        (0, 0, 1),
        (2, 0, 1),
        (1.0, 0, 1),
        (1, True, 1),
        (1, -1, 1),
        (1, 0.0, 1),
        (1, 0, True),
        (1, 0, 2.0),
        (1, 0, 4),
        (1, 2, 1),
    ],
)
def test_read_rejects_invalid_coordinates_instead_of_python_slice_coercion(page, start, end):
    corpus = SourceCorpus([SourceDocument("id", ["abc"])])
    with pytest.raises(ValueError):
        corpus.read("id", page, start, end)


def test_read_is_bounded_even_when_end_is_omitted():
    corpus = SourceCorpus([SourceDocument("large", ["a" * (MAX_READ_CHARS + 1)])])
    with pytest.raises(ValueError, match="at most"):
        corpus.read("large", 1)
    with pytest.raises(ValueError, match="at most"):
        corpus.read("large", 1, 0, MAX_READ_CHARS + 1)
    assert len(corpus.read("large", 1, 0, MAX_READ_CHARS)["quote"]) == MAX_READ_CHARS
    with pytest.raises(ValueError, match="not present"):
        corpus.read("missing", 1)


@pytest.mark.parametrize(
    "source,term,expected",
    [
        ("Pétya arrives", "PETYA", "Pétya"),
        ("Pe\u0301tya arrives", "PÉTYA", "Pe\u0301tya"),
        ("Straße", "STRASSE", "Straße"),
        ("first\r\n\t second", "FIRST second", "first\r\n\t second"),
        ("Cafe\u0301", "café", "Cafe\u0301"),
        ("İstanbul", "istanbul", "İstanbul"),
    ],
)
def test_normalized_search_returns_exact_original_spans(source, term, expected):
    corpus = SourceCorpus([SourceDocument("id", [source])])
    citations = corpus.search(term, window=len(expected))
    assert len(citations) == 1
    citation = citations[0]
    assert citation["quote"] == expected
    assert source[citation["start"] : citation["end"]] == expected
    assert corpus.validate_answer(json.dumps(make_answer(citation))) is None


def test_search_is_ordered_filtered_and_bounds_original_excerpt_sizes():
    corpus = SourceCorpus(
        [
            SourceDocument("one", ["Petya " + "x" * 200, "Pétya"]),
            SourceDocument("two", ["tail Petya"]),
        ]
    )
    citations = corpus.search("petya", limit=2, window=15)
    assert [(item["document_id"], item["page"]) for item in citations] == [("one", 1), ("one", 2)]
    assert all(len(item["quote"]) <= 15 for item in citations)
    assert corpus.search("petya", "two", 10, 6)[0]["quote"] == " Petya"
    assert corpus.search("absent") == []
    with pytest.raises(ValueError, match="not present"):
        corpus.search("petya", "missing")


def test_search_handles_casefold_expansion_without_duplicate_citations():
    corpus = SourceCorpus([SourceDocument("id", ["ß"])])
    assert corpus.search("s", window=1) == [corpus.read("id", 1)]
    # Removing an accent must not produce a quote over the window limit.
    corpus = SourceCorpus([SourceDocument("id", ["ß\u0301"])])
    assert corpus.search("s", window=1) == []
    assert corpus.search("ss", window=2) == [corpus.read("id", 1)]


def test_search_skips_normalized_match_with_an_oversized_original_whitespace_span():
    corpus = SourceCorpus([SourceDocument("id", ["a" + " " * 500 + "b"])])
    assert corpus.search("a b", window=100) == []
    assert len(corpus.search("a b", window=502)[0]["quote"]) == 502


@pytest.mark.parametrize("limit", [True, 0, -1, 1.0, MAX_SEARCH_LIMIT + 1, None])
def test_search_rejects_invalid_result_limits(limit):
    corpus = SourceCorpus([SourceDocument("id", ["text"])])
    with pytest.raises(ValueError, match="limit"):
        corpus.search("text", limit=limit)


@pytest.mark.parametrize("window", [True, 0, -1, 1.0, MAX_SEARCH_WINDOW + 1, None])
def test_search_rejects_invalid_excerpt_windows(window):
    corpus = SourceCorpus([SourceDocument("id", ["text"])])
    with pytest.raises(ValueError, match="window"):
        corpus.search("text", window=window)


@pytest.mark.parametrize("term", ["", " \r\n", "\u0301", None, "a" * (MAX_SEARCH_TERM_CHARS + 1)])
def test_search_rejects_empty_or_unbounded_terms(term):
    corpus = SourceCorpus([SourceDocument("id", ["text"])])
    with pytest.raises(ValueError):
        corpus.search(term)


def test_valid_json_roundtrips_multiple_documents_and_explicit_unknowns():
    corpus = SourceCorpus([SourceDocument("first", ["one"]), SourceDocument("second", ["two"])])
    answer = make_answer(corpus.read("first", 1))
    answer["claims"][0]["citations"].append(corpus.read("second", 1))
    answer["unknowns"] = ["The date is not established by these excerpts."]
    raw = json.dumps(answer)
    assert corpus.parse_answer(raw) == answer
    assert corpus.validate_answer(raw) is None
    abstention = {"answer": "Insufficient evidence.", "claims": [], "unknowns": ["Date unknown"]}
    assert corpus.parse_answer(json.dumps(abstention)) == abstention


@pytest.mark.parametrize(
    "field,value",
    [
        ("document_id", "missing"),
        ("source_sha256", "0" * 64),
        ("page", True),
        ("page", 1.0),
        ("page", 2),
        ("start", False),
        ("start", 1),
        ("end", None),
        ("end", True),
        ("end", 2),
        ("end", 4),
        ("quote", "forged"),
        ("quote", 123),
    ],
)
def test_answer_rejects_fabricated_or_coerced_citation_fields(field, value):
    corpus = SourceCorpus([SourceDocument("id", ["abc"])])
    citation = corpus.read("id", 1)
    citation[field] = value
    raw = json.dumps(make_answer(citation))
    feedback = corpus.validate_answer(raw)
    assert isinstance(feedback, str)
    assert feedback == corpus.validate_answer(raw)
    with pytest.raises(ValueError, match="claims"):
        corpus.parse_answer(raw)


def test_answer_rejects_extra_missing_empty_and_oversized_citations():
    corpus = SourceCorpus([SourceDocument("id", ["a" * (MAX_READ_CHARS + 1)])])
    citation = corpus.read("id", 1, 0, 1)
    for invalid in [
        {**citation, "untrusted": "extra"},
        {key: value for key, value in citation.items() if key != "start"},
        corpus.read("id", 1, 0, 0),
        {**citation, "end": MAX_READ_CHARS + 1, "quote": "a" * (MAX_READ_CHARS + 1)},
    ]:
        assert corpus.validate_answer(json.dumps(make_answer(invalid))) is not None


@pytest.mark.parametrize(
    "raw",
    [
        '{"answer":"one","answer":"two","claims":[],"unknowns":["unknown"]}',
        '{"answer":"one","claims":[],"unknowns":[NaN]}',
        '{"answer":"one","claims":[],"unknowns":[Infinity]}',
        '```json\n{"answer":"one","claims":[],"unknowns":["unknown"]}\n```',
        '{"answer":"one","claims":[],"unknowns":["unknown"]} trailing',
        "null",
        "[]",
        '{"answer":" ","claims":[],"unknowns":["unknown"]}',
        '{"answer":"one","claims":[],"unknowns":[]}',
        '{"answer":"one","claims":{},"unknowns":["unknown"]}',
        '{"answer":"one","claims":[],"unknowns":"unknown"}',
        '{"answer":"one","claims":[],"unknowns":[""]}',
        '{"answer":"one","claims":[{"claim":"x","citations":[]}],"unknowns":[]}',
        '{"answer":"one","claims":[{"claim":"x","claim":"y","citations":[]}],"unknowns":[]}',
        '{"answer":"one","claims":[null],"unknowns":[]}',
        '{"answer":"one","claims":[{"claim":"x","citations":[null]}],"unknowns":[]}',
    ],
)
def test_answer_parser_rejects_invalid_json_and_incomplete_schema(raw):
    corpus = SourceCorpus([SourceDocument("id", ["abc"])])
    assert isinstance(corpus.validate_answer(raw), str)
    with pytest.raises(ValueError):
        corpus.parse_answer(raw)


def test_answer_parser_bounds_input_and_handles_excessive_json_nesting():
    corpus = SourceCorpus([])
    for raw in [None, " " * (MAX_ANSWER_CHARS + 1), "[" * 1500 + "]" * 1500]:
        with pytest.raises(ValueError):
            corpus.parse_answer(raw)


def test_quote_validation_does_not_claim_semantic_entailment():
    corpus = SourceCorpus([SourceDocument("id", ["The fee is ten dollars."])])
    answer = make_answer(corpus.read("id", 1), claim="The fee is a million dollars.")
    # Exact provenance is necessary, but semantic correctness remains a separate check.
    assert corpus.validate_answer(json.dumps(answer)) is None
