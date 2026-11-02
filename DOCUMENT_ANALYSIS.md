# Document analysis with checked source citations

Implementation record: November 2, 2026, GitHub release v0.6.0. The PyPI package remains v0.5.0 until a separate publication. This feature addresses long-report questions and extraction of distributed rules by retaining original page text and checking the identity and coordinates of quoted evidence.

## Python API

Install the optional PDF reader from this checkout:

```bash
pip install -e '.[documents]'
```

```python
from rlm import RLM, load_document


def main():
    report = load_document("report.pdf", document_id="annual-report")
    engine = RLM(
        model="gpt-6.1-sol",
        max_depth=2,
        max_total_calls=16,
        max_elapsed_seconds=300,
        max_output_chars=12_000,
    )
    result = engine.complete_documents(
        "Which AI products and risks are disclosed? Cite the relevant source passages.",
        [report],
    )
    print(result.answer)
    for claim in result.claims:
        print(claim["claim"], claim["citations"])
    print(result.unknowns)
    result.write_jsonl("analysis.jsonl")


if __name__ == "__main__":
    main()
```

`acomplete_documents` is the asynchronous equivalent. The existing string-context methods and their result schema remain available. `DocumentResult` contains `answer`, `claims`, `unknowns`, source metadata, and the underlying `completion` with statistics, trajectory, settings, and raw JSON answer.

Use any supported LiteLLM provider or supply the existing `completion_handler`. To use a ChatGPT-authenticated Codex CLI explicitly:

```bash
python examples/analyze_documents.py report.pdf \
  --query 'Extract the rules and exceptions with exact source citations.' \
  --codex --model gpt-6.1-sol --output analysis.jsonl
```

The command uses the current Codex login and subscription allowance. It does not fall back to API-key authentication. `--max-calls`, `--timeout`, `--call-timeout`, `--max-depth` and `--reasoning` make the experiment limits explicit. Output JSONL appends one complete successful document result per invocation.

## Source identity and search

Construct `SourceDocument(document_id, pages, title="")` directly when another parser supplies page text. Pages are immutable snapshots. The content SHA-256 includes the original UTF-8 page text and page boundaries; it is not the SHA-256 of the original PDF file. Keep a separate raw-file hash when original binary identity matters.

The model has these additional REPL functions:

- `source_manifest()` lists IDs, titles, source hashes, physical page counts and character counts.
- `search_sources(term, document_id=None, limit=10, window=200)` returns bounded source citations. Matching ignores case and combining accents and collapses whitespace. Returned quotations and offsets always refer to unchanged original text.
- `read_source(document_id, page, start=0, end=None)` returns an exact citation. Pages start at 1. Offsets are zero-based, half-open Python character positions within one extracted page. They are not UTF-8 byte positions or PDF geometry.

Search allows at most 100 results, 10,000 characters per excerpt, and a 1,000-character search term. Reads also allow at most 10,000 characters. Supply explicit ranges for larger pages and small result counts so the configured REPL output limit does not truncate observations.

Each accepted claim has at least one citation with exactly `document_id`, `source_sha256`, `page`, `start`, `end`, and `quote`. The validator rejects nonexistent sources, changed hashes, invalid coordinates, altered quotes, duplicate JSON keys, malformed schemas, and nonfinite JSON constants. Invalid finals produce deterministic feedback within the existing iteration and tree-wide budgets.

Document mode accepts either the existing final directives or a standalone JSON answer in that schema. Both paths use identical source and custom validation; final JSON is treated as data and is not executed. The Codex transport preserves explicit empty text and its reported usage so the runtime can apply its bounded empty-response repair. Missing or multiple CLI messages and host tool use remain transport errors.

All child RLMs in a document run can access the same original corpus and validate their own final JSON. Corpus access is not restricted to the `sub_context` string; delegate source IDs and scope explicitly in the subquery. Plain LM leaves return unvalidated text, so accepted parent claims still need original-source citations. Concurrent document runs have separate source state, including when they reuse one `RLM` instance.

## Meaning of a checked result

The checks establish that a quotation exists at its claimed source position. They do not establish that the quotation supports the assertion, that every assertion in `answer` appears in `claims`, or that all relevant facts and exceptions were found. The prompt requests these behaviors; semantic correctness requires independent evaluation.

`unknowns` records unresolved questions or scope. An answer with no supported claims must have a nonempty unknowns list. This is a structural abstention format, not a mechanical proof that information is absent from the entire corpus.

The result can contain code in its answer, with cited rules in its claims. This API does not automatically execute that code or certify its correctness. Evaluate generated calculations against independent examples, especially exceptions and boundary values.

Source text stays in the local corpus and REPL until the model selects passages or explicitly reads `context`. Selected text is sent through the configured model transport. There is no new hosted service or automatic remote persistence. `DocumentResult` and its JSONL export include answers and quotations even when trajectory capture is redacted; enable `capture_trajectory_content` only when full prompts and observations may also be retained. Source trajectory events contain coordinates and lengths rather than quoted text; document IDs, hashes and titles remain visible in metadata.

## PDF ingestion and companion

`load_document` accepts local UTF-8 text and PDF. Default limits are 5 MiB input, 500 PDF pages, 2,000,000 extracted characters, and 20 MiB decoded non-image PDF content and resources. `DocumentLimits` makes those limits configurable. A preflight bounds stream decoding before text extraction and counts referenced streams across pages, nested Forms and font mappings. Shared resources count once. Unused referenced non-image resources may be counted conservatively; images are not decoded by this text-only reader.

Physical PDF page boundaries are retained, including blank pages. Printed page labels are not inferred. There is no OCR, so scanned images and complex tables may be missing from extracted text. PDFs that open with an empty user password are accepted; password-required documents need an unlocked local copy.

The existing comparison companion retains its default 300,000-character profile. Explicitly enable larger local documents with:

```bash
python examples/serve_codex_comparison.py --document-profile long
```

The long profile permits 2,000,000 characters while retaining file, page, decompression and worker limits. Exact pages remain in server memory; client metadata includes page sizes and source identity. The existing side-by-side comparison still runs its original string-context experiment. Use `complete_documents` or the document CLI for citation-checked analysis.

## Evaluation

Deterministic tests cover source substitution, wrong pages and spans, strict JSON, Unicode search, root and child repair, concurrent corpus isolation, PDF encryption, extraction bounds and compatibility with existing completion APIs.

The [evaluation reports and compact evidence](experiments/publication-20261102/README.md) include the synthetic rules pilot, pinned public documents, fixed-response controls against v0.5.0, and a repeated real-document matrix. Expected answers stay outside model prompts. Public exports retain answers, selected quotations, grades, usage and source hashes while omitting full prompts and source contexts. These checks establish the tested integration behavior; they do not demonstrate a general quality advantage or that recursion is necessary.
