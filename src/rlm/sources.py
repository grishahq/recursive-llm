"""Immutable text sources and structurally checked citations.

Pages retain their original text. Page numbers start at one; span offsets are
zero-based, half-open Python string offsets, not PDF geometry or UTF-8 byte offsets.
Citation validation checks source identity and exact excerpts, not claim entailment.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

MAX_SEARCH_LIMIT = 100
MAX_SEARCH_WINDOW = 10_000
MAX_SEARCH_TERM_CHARS = 1_000
MAX_READ_CHARS = 10_000
MAX_ANSWER_CHARS = 1_000_000

_CITATION_FIELDS = {"document_id", "source_sha256", "page", "start", "end", "quote"}


def _require_integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    return value


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True, init=False)
class SourceDocument:
    """An immutable document whose SHA-256 includes original page boundaries.

    The digest hashes a version prefix, the page count, and each UTF-8 page
    preceded by its byte length. Lengths are unsigned eight-byte big-endian
    integers. Identity and title are metadata, not part of the content digest.
    Blank pages are allowed; a document must contain at least one page.
    """

    document_id: str
    pages: Tuple[str, ...]
    title: str = ""
    content_sha256: str = field(init=False)

    def __init__(self, document_id: str, pages: Sequence[str], title: str = "") -> None:
        _require_text(document_id, "document_id")
        if not isinstance(title, str):
            raise ValueError("title must be a string")
        if isinstance(pages, (str, bytes)):
            raise ValueError("pages must be a sequence of page strings")
        try:
            snapshot = tuple(pages)
        except TypeError as exc:
            raise ValueError("pages must be a sequence of page strings") from exc
        if not snapshot or any(not isinstance(page, str) for page in snapshot):
            raise ValueError("pages must contain at least one page string")
        digest = hashlib.sha256(b"recursive-llm/source-pages/v1\0")
        digest.update(len(snapshot).to_bytes(8, "big"))
        for page in snapshot:
            try:
                encoded = page.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError("pages must contain valid UTF-8 text") from exc
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        object.__setattr__(self, "document_id", document_id)
        object.__setattr__(self, "pages", snapshot)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "content_sha256", digest.hexdigest())


def _normalize_with_offsets(text: str) -> Tuple[str, List[int], List[int]]:
    """Fold accents/case/whitespace while mapping every match to original text."""
    characters: List[str] = []
    starts: List[int] = []
    ends: List[int] = []
    for index, character in enumerate(text):
        folded = unicodedata.normalize("NFD", character.casefold())
        visible = [part for part in folded if not unicodedata.category(part).startswith("M")]
        if not visible:
            # A decomposed accent belongs to the preceding original character.
            if ends:
                original_start = starts[-1]
                previous = len(ends) - 1
                while previous >= 0 and starts[previous] == original_start:
                    ends[previous] = index + 1
                    previous -= 1
            continue
        for part in visible:
            normalized = " " if part.isspace() else part
            if normalized == " " and characters and characters[-1] == " ":
                ends[-1] = index + 1
            else:
                characters.append(normalized)
                starts.append(index)
                ends.append(index + 1)
    return "".join(characters), starts, ends


def _json_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Answer JSON must not contain duplicate keys")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> Any:
    raise ValueError("Answer JSON must not contain NaN or Infinity")


@dataclass(frozen=True, init=False)
class SourceCorpus:
    """A fixed source snapshot with bounded search and an answer validator.

    Search returns at most 100 citations, each at most 10,000 original characters.
    Terms are limited to 1,000 characters. Matches are non-overlapping and ordered
    by document, page, then offset. A match whose original span exceeds ``window``
    is omitted. Normalization changes search only, never the returned quotation.
    Answer JSON is limited to 1,000,000 characters before parsing.
    """

    documents: Tuple[SourceDocument, ...]
    _by_id: Mapping[str, SourceDocument] = field(repr=False, compare=False)

    def __init__(self, documents: Sequence[SourceDocument]) -> None:
        try:
            snapshot = tuple(documents)
        except TypeError as exc:
            raise ValueError("documents must be a sequence of SourceDocument values") from exc
        by_id: Dict[str, SourceDocument] = {}
        for document in snapshot:
            if not isinstance(document, SourceDocument):
                raise ValueError("documents must contain only SourceDocument values")
            if document.document_id in by_id:
                raise ValueError("document_id values must be unique")
            by_id[document.document_id] = document
        object.__setattr__(self, "documents", snapshot)
        object.__setattr__(self, "_by_id", MappingProxyType(by_id))

    def manifest(self) -> List[Dict[str, Any]]:
        """Return detached metadata, without copying source text into the result."""
        return [
            {
                "document_id": document.document_id,
                "title": document.title,
                "source_sha256": document.content_sha256,
                "page_count": len(document.pages),
                "character_count": sum(len(page) for page in document.pages),
            }
            for document in self.documents
        ]

    @property
    def context(self) -> str:
        """Return legacy flat context with explicit document and physical-page markers."""
        parts = []
        for document in self.documents:
            metadata = json.dumps(
                {"document_id": document.document_id, "title": document.title}, ensure_ascii=False
            )
            parts.append(f"=== DOCUMENT {metadata} ===")
            for page_number, page in enumerate(document.pages, start=1):
                parts.append(f"=== PAGE {page_number} ===\n{page}")
        return "\n\n".join(parts)

    def _document(self, document_id: str) -> SourceDocument:
        _require_text(document_id, "document_id")
        if document_id not in self._by_id:
            raise ValueError("document_id is not present in this corpus")
        return self._by_id[document_id]

    def read(
        self, document_id: str, page: int, start: int = 0, end: Optional[int] = None
    ) -> Dict[str, Any]:
        """Quote at most 10,000 characters; empty spans and blank pages are readable.

        Omitted ``end`` requests the remainder of the page, subject to the same
        limit. Supply an explicit bounded end when reading a larger page.
        """
        document = self._document(document_id)
        page_number = _require_integer(page, "page", 1, len(document.pages))
        original = document.pages[page_number - 1]
        first = _require_integer(start, "start", 0, len(original))
        last = len(original) if end is None else _require_integer(end, "end", 0, len(original))
        if last < first:
            raise ValueError("end must be greater than or equal to start")
        if last - first > MAX_READ_CHARS:
            raise ValueError(f"read span must contain at most {MAX_READ_CHARS} characters")
        return {
            "document_id": document.document_id,
            "source_sha256": document.content_sha256,
            "page": page_number,
            "start": first,
            "end": last,
            "quote": original[first:last],
        }

    def search(
        self,
        term: str,
        document_id: Optional[str] = None,
        limit: int = 10,
        window: int = 200,
    ) -> List[Dict[str, Any]]:
        """Find accent-insensitive terms and return bounded original-text citations."""
        _require_text(term, "term")
        if len(term) > MAX_SEARCH_TERM_CHARS:
            raise ValueError(f"term must contain at most {MAX_SEARCH_TERM_CHARS} characters")
        result_limit = _require_integer(limit, "limit", 1, MAX_SEARCH_LIMIT)
        excerpt_limit = _require_integer(window, "window", 1, MAX_SEARCH_WINDOW)
        needle = _normalize_with_offsets(term)[0].strip()
        if not needle:
            raise ValueError("term must contain searchable characters")
        documents = self.documents if document_id is None else (self._document(document_id),)
        results: List[Dict[str, Any]] = []
        for document in documents:
            for page_number, original in enumerate(document.pages, start=1):
                normalized, starts, ends = _normalize_with_offsets(original)
                position = 0
                while True:
                    found = normalized.find(needle, position)
                    if found < 0:
                        break
                    position = found + len(needle)
                    first, last = starts[found], ends[position - 1]
                    if last - first > excerpt_limit:
                        continue
                    spare = excerpt_limit - (last - first)
                    first = max(0, first - spare // 2)
                    last = min(len(original), first + excerpt_limit)
                    first = max(0, last - excerpt_limit)
                    citation = self.read(document.document_id, page_number, first, last)
                    # Casefold expansions can expose one source character twice.
                    if not results or results[-1] != citation:
                        results.append(citation)
                    if len(results) >= result_limit:
                        return results
        return results

    def _validate_citation(self, citation: Any, location: str) -> None:
        if not isinstance(citation, dict) or set(citation) != _CITATION_FIELDS:
            raise ValueError(f"{location} must contain exactly the six citation fields")
        try:
            expected = self.read(
                citation["document_id"], citation["page"], citation["start"], citation["end"]
            )
        except ValueError as exc:
            raise ValueError(f"{location}: {exc}") from exc
        if citation["end"] is None:
            raise ValueError(f"{location}.end must be an explicit integer")
        if not isinstance(citation["source_sha256"], str) or (
            citation["source_sha256"] != expected["source_sha256"]
        ):
            raise ValueError(f"{location}.source_sha256 does not match the source")
        if not isinstance(citation["quote"], str) or citation["quote"] != expected["quote"]:
            raise ValueError(f"{location}.quote does not match the source span")
        if not citation["quote"]:
            raise ValueError(f"{location}.quote must not be empty")

    def parse_answer(self, text: str) -> Dict[str, Any]:
        """Parse strict answer JSON and validate source references, not entailment.

        The exact schema is ``{answer: str, claims: [{claim: str,
        citations: [citation]}], unknowns: [str]}``. Each claim needs an exact,
        nonempty source quotation. With no claims, at least one unknown is required.
        """
        if not isinstance(text, str) or len(text) > MAX_ANSWER_CHARS:
            raise ValueError(
                f"Answer must be a JSON string of at most {MAX_ANSWER_CHARS} characters"
            )
        try:
            result = json.loads(
                text, object_pairs_hook=_json_object, parse_constant=_reject_json_constant
            )
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("Answer must be valid JSON without surrounding prose") from exc
        if not isinstance(result, dict) or set(result) != {"answer", "claims", "unknowns"}:
            raise ValueError("Answer must contain exactly answer, claims, and unknowns")
        _require_text(result["answer"], "answer")
        if not isinstance(result["claims"], list):
            raise ValueError("claims must be a list")
        if not isinstance(result["unknowns"], list):
            raise ValueError("unknowns must be a list")
        for index, unknown in enumerate(result["unknowns"]):
            _require_text(unknown, f"unknowns[{index}]")
        if not result["claims"] and not result["unknowns"]:
            raise ValueError("An answer without claims must identify at least one unknown")
        for index, claim in enumerate(result["claims"]):
            location = f"claims[{index}]"
            if not isinstance(claim, dict) or set(claim) != {"claim", "citations"}:
                raise ValueError(f"{location} must contain exactly claim and citations")
            _require_text(claim["claim"], f"{location}.claim")
            if not isinstance(claim["citations"], list) or not claim["citations"]:
                raise ValueError(f"{location}.citations must be a nonempty list")
            for citation_index, citation in enumerate(claim["citations"]):
                self._validate_citation(citation, f"{location}.citations[{citation_index}]")
        return result

    def validate_answer(self, text: str) -> Optional[str]:
        """Return deterministic validation feedback for ``final_answer_validator``."""
        try:
            self.parse_answer(text)
        except ValueError as exc:
            return str(exc)
        return None
