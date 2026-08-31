"""Exact-graded RLM evaluation over public TXT, PDF, CSV, and HTML inputs.

The raw source files are downloaded separately and verified by SHA-256. PDF,
CSV, and HTML are converted to deterministic plain-text contexts because the
public RLM API intentionally accepts strings rather than file-format objects.

Prepare and run the suite from the repository root::

    python benchmarks/document_formats.py --prepare /tmp/rlm-document-formats
    python benchmarks/document_formats.py gpt-5-mini /tmp/rlm-document-formats
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from dotenv import load_dotenv

try:
    from .compare_same_model import Task, ValidationResult, aggregate_results, run_task
except ImportError:  # Support direct execution from the repository root.
    from compare_same_model import Task, ValidationResult, aggregate_results, run_task


SOURCE_SPECS: Mapping[str, Mapping[str, str]] = {
    "frankenstein": {
        "format": "txt",
        "filename": "frankenstein.txt",
        "sha256": "7810cd483cffcf2cc8a1d8f0d5807931e69d4f48cd14149b8c76f88af82fead3",
        "source_url": "https://www.gutenberg.org/cache/epub/84/pg84.txt",
    },
    "playbook_pdf": {
        "format": "pdf",
        "filename": "ai-rmf-playbook.pdf",
        "sha256": "65d6101d806502875aadb0fd19a75c3a9cc9a5e9461129e9398a39192d8202d2",
        "source_url": "https://airc.nist.gov/docs/AI_RMF_Playbook.pdf",
    },
    "playbook_csv": {
        "format": "csv",
        "filename": "ai-rmf-playbook.csv",
        "sha256": "3cee552201b18192e042fb94b72fe8da9395c91d36877ac7d9afb9cccb352b3a",
        "source_url": "https://airc.nist.gov/docs/playbook.csv",
    },
    "python_sqlite_html": {
        "format": "html",
        "filename": "python-sqlite3.html",
        "sha256": "cd7cdb039f49bdd5866193acf810482a16954bf7f41b31b4dcb0a4a725f51032",
        "source_url": "https://docs.python.org/3/library/sqlite3.html",
    },
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verified_bytes(path: Path, expected_sha256: str) -> bytes:
    raw = path.read_bytes()
    observed = _sha256(raw)
    if observed != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, observed {observed}"
        )
    return raw


def _download(url: str, destination: Path) -> None:
    """Download one public artifact without exposing provider credentials."""
    request = urllib.request.Request(url, headers={"User-Agent": "recursive-llm-benchmark/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())


class _HTMLTextExtractor(HTMLParser):
    """Small deterministic HTML-to-text converter for documentation pages."""

    _BLOCKS = {
        "article",
        "code",
        "dd",
        "div",
        "dt",
        "h1",
        "h2",
        "h3",
        "h4",
        "li",
        "p",
        "pre",
        "section",
        "tr",
    }
    _SKIPPED = {"nav", "script", "style"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, _attrs: List[tuple[str, str | None]]) -> None:
        if tag in self._SKIPPED:
            self._skip_depth += 1
        elif tag in self._BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIPPED and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self.parts).splitlines())
        return "\n".join(line for line in lines if line) + "\n"


def _html_to_text(raw: bytes) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(raw.decode("utf-8"))
    return parser.text()


def _csv_to_text(raw: bytes) -> str:
    """Transpose the Playbook's wide CSV into labeled section records."""
    rows = list(csv.reader(raw.decode("utf-8-sig").splitlines()))
    if len(rows) != 5 or len(rows[0]) != 73:
        raise ValueError(f"Unexpected Playbook CSV shape: {len(rows)}x{len(rows[0])}")
    labels = rows[0]
    field_names = [row[0] for row in rows[1:]]
    records = []
    for column, section in enumerate(labels[1:], start=1):
        fields = [f"{field}: {rows[index][column]}" for index, field in enumerate(field_names, 1)]
        records.append(f"===== SECTION: {section} =====\n" + "\n".join(fields))
    return "\n\n".join(records) + "\n"


def prepare_documents(directory: Path) -> Dict[str, Dict[str, Any]]:
    """Download, verify, and normalize all public benchmark inputs."""
    raw_directory = directory / "raw"
    extracted_directory = directory / "extracted"
    raw_directory.mkdir(parents=True, exist_ok=True)
    extracted_directory.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Dict[str, Any]] = {}
    for name, spec in SOURCE_SPECS.items():
        raw_path = raw_directory / spec["filename"]
        if not raw_path.exists():
            _download(spec["source_url"], raw_path)
        raw = _verified_bytes(raw_path, spec["sha256"])

        source_format = spec["format"]
        if source_format == "txt":
            context = raw.decode("utf-8-sig").replace("\r\n", "\n")
        elif source_format == "csv":
            context = _csv_to_text(raw)
        elif source_format == "html":
            context = _html_to_text(raw)
        elif source_format == "pdf":
            executable = shutil.which("pdftotext")
            if executable is None:
                raise RuntimeError("pdftotext is required to prepare the PDF benchmark")
            output_path = extracted_directory / f"{name}.txt"
            subprocess.run(
                [executable, "-layout", str(raw_path), str(output_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            context = output_path.read_text(encoding="utf-8")
        else:  # pragma: no cover - specifications are a closed constant.
            raise AssertionError(f"Unsupported source format: {source_format}")

        output_path = extracted_directory / f"{name}.txt"
        output_path.write_text(context, encoding="utf-8", newline="\n")
        manifest[name] = {
            "format": source_format,
            "source_url": spec["source_url"],
            "raw_sha256": spec["sha256"],
            "context_sha256": _sha256(context.encode("utf-8")),
            "characters": len(context),
        }

    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_documents(directory: Path) -> Dict[str, str]:
    """Load prepared contexts after re-verifying their original artifacts."""
    documents: Dict[str, str] = {}
    for name, spec in SOURCE_SPECS.items():
        _verified_bytes(directory / "raw" / spec["filename"], spec["sha256"])
        documents[name] = (directory / "extracted" / f"{name}.txt").read_text(encoding="utf-8")
    return documents


def _field(answer: str, label: str) -> str:
    match = re.search(rf"^{re.escape(label)}\s*:\s*(.+)$", answer, re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else ""


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9_.]+", " ", value.casefold()).strip()


def validate_frankenstein(answer: str) -> ValidationResult:
    fields = {
        "Letters": _normalized(_field(answer, "Letters")),
        "Chapters": _normalized(_field(answer, "Chapters")),
        "Addressee": _normalized(_field(answer, "Addressee")),
        "Observed family": _normalized(_field(answer, "Observed family")),
    }
    checks = {
        "Letters": fields["Letters"] == "4",
        "Chapters": fields["Chapters"] == "24",
        "Addressee": any(
            expected in fields["Addressee"]
            for expected in ("mrs saville", "mrs. saville", "margaret saville")
        ),
        "Observed family": "de lacey" in fields["Observed family"],
    }
    failures = tuple(
        f"incorrect or missing field {name!r}" for name, ok in checks.items() if not ok
    )
    return ValidationResult(not failures, failures)


def validate_playbook_structure(answer: str) -> ValidationResult:
    expected = {"Govern": "19", "Manage": "13", "Map": "18", "Measure": "22"}
    failures = tuple(
        f"incorrect or missing field {name!r}"
        for name, value in expected.items()
        if _normalized(_field(answer, name)) != value
    )
    return ValidationResult(not failures, failures)


def validate_sqlite_defaults(answer: str) -> ValidationResult:
    expected = {
        "timeout": "5.0",
        "isolation_level": "deferred",
        "cached_statements": "128",
        "autocommit": "sqlite3.legacy_transaction_control",
    }
    failures = tuple(
        f"incorrect or missing field {name!r}"
        for name, value in expected.items()
        if _normalized(_field(answer, name)) != value
    )
    return ValidationResult(not failures, failures)


def build_tasks(documents: Mapping[str, str], *, label: str) -> Sequence[Task]:
    def metadata(name: str) -> Dict[str, Any]:
        context = documents[name]
        spec = SOURCE_SPECS[name]
        return {
            "corpus": name,
            "format": spec["format"],
            "variant": label,
            "source_url": spec["source_url"],
            "raw_sha256": spec["sha256"],
            "context_sha256": _sha256(context.encode("utf-8")),
            "characters": len(context),
        }

    playbook_query = (
        "Count the distinct numbered subcategory headings belonging to each top-level AI RMF "
        "function. Count headings such as GOVERN 1.1 once and do not count prose mentions. Return "
        "exactly four lines labeled `Govern:`, `Manage:`, `Map:`, and `Measure:`."
    )
    return (
        Task(
            name="txt_frankenstein_structure",
            query=(
                "Use the full book text. Count the standalone Letter and Chapter headings in the "
                "book body (ignore the indented contents list), identify the addressee printed under "
                "the letters, and name the family observed by the creature. Return exactly four lines "
                "labeled `Letters:`, `Chapters:`, `Addressee:`, and `Observed family:`."
            ),
            context=documents["frankenstein"],
            validator=validate_frankenstein,
            metadata=metadata("frankenstein"),
        ),
        Task(
            name="pdf_playbook_structure",
            query=playbook_query,
            context=documents["playbook_pdf"],
            validator=validate_playbook_structure,
            metadata=metadata("playbook_pdf"),
        ),
        Task(
            name="csv_playbook_structure",
            query=playbook_query,
            context=documents["playbook_csv"],
            validator=validate_playbook_structure,
            metadata=metadata("playbook_csv"),
        ),
        Task(
            name="html_sqlite_connect_defaults",
            query=(
                "Find the documented default values in the sqlite3.connect signature for timeout, "
                "isolation_level, cached_statements, and autocommit. Return exactly four lines labeled "
                "`timeout:`, `isolation_level:`, `cached_statements:`, and `autocommit:`."
            ),
            context=documents["python_sqlite_html"],
            validator=validate_sqlite_defaults,
            metadata=metadata("python_sqlite_html"),
        ),
    )


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", help="LiteLLM model identifier")
    parser.add_argument("documents", nargs="?", type=Path, help="prepared corpus directory")
    parser.add_argument(
        "--prepare", type=Path, metavar="DIRECTORY", help="download and prepare inputs"
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--task",
        choices=(
            "txt_frankenstein_structure",
            "pdf_playbook_structure",
            "csv_playbook_structure",
            "html_sqlite_connect_defaults",
        ),
    )
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--mode", choices=("rlm", "direct"), default="rlm")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=2_000)
    parser.add_argument("--max-total-calls", type=int, default=16)
    parser.add_argument("--max-elapsed-seconds", type=float, default=180)
    parser.add_argument("--jsonl", type=Path)
    args = parser.parse_args()

    if args.prepare is not None:
        print(json.dumps(prepare_documents(args.prepare), indent=2, sort_keys=True))
        return
    if args.model is None or args.documents is None:
        parser.error("model and documents are required unless --prepare is used")
    if args.runs <= 0:
        parser.error("--runs must be greater than zero")

    load_dotenv()
    tasks = build_tasks(load_documents(args.documents), label=args.label)
    if args.task:
        tasks = tuple(task for task in tasks if task.name == args.task)

    results = []
    for task in tasks:
        for run_index in range(1, args.runs + 1):
            print(f"Running {args.model}: {task.name} ({run_index}/{args.runs})", flush=True)
            result = run_task(
                args.model,
                task,
                run_index=run_index,
                max_depth=args.max_depth,
                max_iterations=args.max_iterations,
                max_tokens=args.max_tokens,
                max_total_calls=args.max_total_calls,
                max_elapsed_seconds=args.max_elapsed_seconds,
                mode=args.mode,
            )
            results.append(result)
            print(
                f"Finished {task.name}: passed={result['passed']} "
                f"calls={result['stats']['llm_calls']} cost=${result['stats']['estimated_cost_usd']}",
                flush=True,
            )

    summary = aggregate_results(args.model, results, max_depth=args.max_depth, mode=args.mode)
    if args.jsonl:
        _write_jsonl(args.jsonl, [*results, summary])
    print(json.dumps({"summary": summary, "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
