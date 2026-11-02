"""Bounded local text and PDF ingestion with physical PDF page boundaries.

PDF extraction does not perform OCR or infer printed page labels. The returned
pages retain the exact text produced by the PDF parser, including empty pages.
"""

from __future__ import annotations

import hashlib
import io
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from .sources import SourceDocument


class DocumentLoadError(ValueError):
    """A local document cannot be read within the configured limits."""


class DocumentExtractionWarning(UserWarning):
    """Extracted text may omit information visible in the source document."""


@dataclass(frozen=True)
class DocumentLimits:
    """Upper bounds checked before and during local extraction."""

    max_chars: int = 2_000_000
    max_bytes: int = 5 * 1024 * 1024
    max_pages: int = 500
    max_expanded_bytes: int = 20 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in ("max_chars", "max_bytes", "max_pages", "max_expanded_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


PDF_EXTRACTION_WARNING = (
    "PDF text extraction has no OCR; scanned images and complex tables may be incomplete. "
    "Page numbers identify physical PDF pages, starting at 1, not printed page labels."
)
_TEXT_SUFFIXES = {".txt", ".md", ".csv", ".json", ".rst", ".log", ".html", ".htm"}


def _check_pdf_stream_budget(reader: Any, limits: DocumentLimits) -> None:
    """Count referenced text/resource streams once before text extraction.

    Images are not decoded by plain text extraction and are excluded. Other
    streams reachable from page contents and resources are counted conservatively,
    including unused form and font resources. Structural page/annotation
    backlinks are excluded; resource names with those spellings are traversed.
    """
    from pypdf import apply_configuration
    from pypdf.errors import LimitReachedError
    from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject, StreamObject

    pending = []
    for page in reader.pages:
        pending.append(page.get("/Contents"))
        pending.append(page.get_inherited("/Resources"))
    references = set()
    objects = set()
    expanded = 0
    while pending:
        value = pending.pop()
        if isinstance(value, IndirectObject):
            reference = (id(value.pdf), value.idnum, value.generation)
            if reference in references:
                continue
            references.add(reference)
            value = value.get_object()
        if not isinstance(value, (DictionaryObject, ArrayObject)) or id(value) in objects:
            continue
        objects.add(id(value))
        if isinstance(value, StreamObject):
            if value.get("/Subtype") == "/Image":
                continue
            # One extra byte distinguishes an exact fit from an overrun without
            # using zero, which disables pypdf's decompression limit.
            remaining = limits.max_expanded_bytes - expanded
            try:
                with apply_configuration(
                    zlib_maximum_output_length=remaining + 1,
                    lzw_maximum_output_length=remaining + 1,
                    run_length_maximum_output_length=remaining + 1,
                    array_based_stream_maximum_output_length=remaining + 1,
                ):
                    expanded += len(value.get_data())
            except LimitReachedError as exc:
                raise DocumentLoadError(
                    f"PDF expanded content exceeds {limits.max_expanded_bytes:,} bytes"
                ) from exc
            if expanded > limits.max_expanded_bytes:
                raise DocumentLoadError(
                    f"PDF expanded content exceeds {limits.max_expanded_bytes:,} bytes"
                )
        if isinstance(value, DictionaryObject):
            kind = value.get("/Type")
            if kind == "/Page":
                backlinks = {"/Parent", "/Annots"}
            elif kind == "/Pages":
                backlinks = {"/Parent"}
            elif kind == "/Annot":
                backlinks = {"/P", "/Parent"}
            else:
                backlinks = set()
            pending.extend(item for key, item in value.items() if key not in backlinks)
        else:
            pending.extend(value)


def _pdf_pages(data: bytes, limits: DocumentLimits) -> tuple[str, ...]:
    try:
        from pypdf import PdfReader, apply_configuration
    except ImportError as exc:
        raise DocumentLoadError(
            "PDF support requires pypdf >= 6.18: pip install 'recursive-llm[documents]'"
        ) from exc
    if not data.startswith(b"%PDF-"):
        raise DocumentLoadError("File does not have a PDF header")
    try:
        # pypdf's context-local configuration applies before stream decompression.
        with apply_configuration(
            zlib_maximum_output_length=limits.max_expanded_bytes,
            lzw_maximum_output_length=limits.max_expanded_bytes,
            run_length_maximum_output_length=limits.max_expanded_bytes,
            array_based_stream_maximum_output_length=limits.max_expanded_bytes,
        ):
            reader = PdfReader(io.BytesIO(data), strict=True)
            if reader.is_encrypted and not reader.decrypt(""):
                raise DocumentLoadError("PDF requires a password; provide an unlocked local copy")
            if len(reader.pages) > limits.max_pages:
                raise DocumentLoadError(f"PDF exceeds the {limits.max_pages:,}-page limit")
            _check_pdf_stream_budget(reader, limits)
            pages: list[str] = []
            chars = 0
            for page in reader.pages:
                piece = page.extract_text() or ""
                chars += len(piece)
                if chars + len(pages) > limits.max_chars:
                    raise DocumentLoadError(
                        f"Extracted PDF text exceeds {limits.max_chars:,} characters"
                    )
                pages.append(piece)
    except DocumentLoadError:
        raise
    except Exception as exc:
        raise DocumentLoadError(f"Could not extract PDF text: {exc}") from exc
    if not any(page.strip() for page in pages):
        raise DocumentLoadError("No readable text found; scanned PDFs need OCR before loading")
    warnings.warn(PDF_EXTRACTION_WARNING, DocumentExtractionWarning, stacklevel=3)
    return tuple(pages)


def read_document_bytes(
    filename: str,
    data: bytes,
    document_id: Optional[str] = None,
    limits: Optional[DocumentLimits] = None,
) -> SourceDocument:
    """Read uploaded bytes without opening a path or fetching a URL.

    UTF-8 text is one source page. PDF pages are physical page positions, with
    no guessed mapping to printed page labels. Text line endings are preserved.
    """
    limits = limits or DocumentLimits()
    if not data or len(data) > limits.max_bytes:
        raise DocumentLoadError(f"Document must contain 1 to {limits.max_bytes:,} bytes")
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        pages = _pdf_pages(data, limits)
    elif suffix in _TEXT_SUFFIXES:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentLoadError("Text documents must use UTF-8 encoding") from exc
        if "\x00" in text:
            raise DocumentLoadError("Binary content is not a UTF-8 text document")
        if not text.strip():
            raise DocumentLoadError("No readable text found in document")
        if len(text) > limits.max_chars:
            raise DocumentLoadError(f"Extracted text exceeds {limits.max_chars:,} characters")
        pages = (text,)
    else:
        raise DocumentLoadError("Supported local documents: UTF-8 text or PDF")
    identifier = document_id if document_id is not None else hashlib.sha256(data).hexdigest()[:24]
    return SourceDocument(document_id=identifier, pages=pages, title=Path(filename).name)


def load_document(
    path: Union[str, Path],
    document_id: Optional[str] = None,
    limits: Optional[DocumentLimits] = None,
) -> SourceDocument:
    """Load a local UTF-8 file or PDF within explicit extraction limits."""
    limits = limits or DocumentLimits()
    source = Path(path)
    with source.open("rb") as stream:
        data = stream.read(limits.max_bytes + 1)
    return read_document_bytes(source.name, data, document_id=document_id, limits=limits)
