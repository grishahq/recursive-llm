"""Exact-graded benchmark over three large public English documents.

Prepare the inputs under one directory before running this script. The commands
and pinned hashes are documented in ``BENCHMARK_RESULTS.md``. Expected names:

* ``war-and-peace.txt``
* ``911-commission-report.txt``
* ``python-3.14-docs-text.zip``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from dotenv import load_dotenv

try:
    from .compare_same_model import Task, ValidationResult, aggregate_results, run_task
    from .war_and_peace import validate_petya_final_night
except ImportError:  # Support direct execution from the repository root.
    from compare_same_model import Task, ValidationResult, aggregate_results, run_task
    from war_and_peace import validate_petya_final_night


DOCUMENT_SPECS: Mapping[str, Mapping[str, str]] = {
    "war_and_peace": {
        "filename": "war-and-peace.txt",
        "sha256": "e4bcf9042609b62c7de72a6f1b311f54c412943a9d641b7efcf79a464b5f31c8",
        "source_url": "https://www.gutenberg.org/files/2600/2600-0.txt",
    },
    "commission_report": {
        "filename": "911-commission-report.txt",
        "sha256": "33e5f373e542c58a872dde753caaf80e3c60c2b98c29c18898ae4590c9f4cfbe",
        "download_sha256": "657d41475eb3a9a5e3e87a6c7c51ac1dfbe1af7566d1abff7bf7286e7e1c0e1b",
        "source_url": ("https://www.govinfo.gov/content/pkg/GPO-911REPORT/pdf/GPO-911REPORT.pdf"),
    },
    "python_docs": {
        "filename": "python-3.14-docs-text.zip",
        "sha256": "c8ee0347f282f97e5a57f0b010cecd441464db9fe679862f51aeda0dad12ab47",
        "source_url": "https://docs.python.org/3.14/archives/python-3.14-docs-text.zip",
    },
}


def _sha256(data: bytes) -> str:
    """Return the lowercase SHA-256 digest for bytes."""
    return hashlib.sha256(data).hexdigest()


def _verified_bytes(path: Path, expected_sha256: str) -> bytes:
    """Read a file and reject bytes that differ from the answer-key input."""
    raw = path.read_bytes()
    observed = _sha256(raw)
    if observed != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, observed {observed}"
        )
    return raw


def load_documents(directory: Path) -> Dict[str, str]:
    """Load and verify every benchmark document from one directory."""
    war_spec = DOCUMENT_SPECS["war_and_peace"]
    war_raw = _verified_bytes(directory / war_spec["filename"], war_spec["sha256"])
    war = war_raw.decode("utf-8-sig").replace("\r\n", "\n")

    report_spec = DOCUMENT_SPECS["commission_report"]
    report_raw = _verified_bytes(directory / report_spec["filename"], report_spec["sha256"])
    report = report_raw.decode("utf-8").replace("\r\n", "\n")

    python_spec = DOCUMENT_SPECS["python_docs"]
    archive_path = directory / python_spec["filename"]
    _verified_bytes(archive_path, python_spec["sha256"])
    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".txt"))
        python_docs = "".join(
            f"===== FILE: {name.removeprefix('python-3.14-docs-text/')} =====\n"
            f"{archive.read(name).decode('utf-8')}\n"
            for name in names
        )

    return {
        "war_and_peace": war,
        "commission_report": report,
        "python_docs": python_docs,
    }


def _field(answer: str, label: str) -> str:
    """Return a stripped case-insensitive labeled answer field."""
    match = re.search(rf"^{re.escape(label)}\s*:\s*(.+)$", answer, re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else ""


def _compact(value: str) -> str:
    """Normalize punctuation and whitespace for exact semantic fields."""
    return re.sub(r"[^a-z0-9.]+", " ", value.casefold()).strip()


def validate_commission_facts(answer: str) -> ValidationResult:
    """Grade facts from the preface and the intelligence-reform recommendation."""
    fields = {
        "Pages reviewed": _compact(_field(answer, "Pages reviewed")),
        "Individuals interviewed": _compact(_field(answer, "Individuals interviewed")),
        "Countries": _compact(_field(answer, "Countries")),
        "Replacement": _compact(_field(answer, "Replacement")),
        "Responsibilities": _compact(_field(answer, "Responsibilities")),
        "Location": _compact(_field(answer, "Location")),
    }
    checks = {
        "Pages reviewed": "2.5 million" in fields["Pages reviewed"],
        "Individuals interviewed": bool(
            re.search(r"\b1\s*200\b", fields["Individuals interviewed"])
            or "one thousand two hundred" in fields["Individuals interviewed"]
        ),
        "Countries": fields["Countries"] in {"10", "ten", "10 countries", "ten countries"},
        "Replacement": "national intelligence director" in fields["Replacement"],
        "Responsibilities": fields["Responsibilities"] in {"2", "two", "two main areas"},
        "Location": "executive office of the president" in fields["Location"],
    }
    failures = tuple(
        f"incorrect or missing field {label!r}" for label, ok in checks.items() if not ok
    )
    return ValidationResult(not failures, failures)


def validate_python_docs_facts(answer: str) -> ValidationResult:
    """Grade facts distributed across four Python 3.14 documentation files."""
    fields = {
        "Zstandard module": _compact(_field(answer, "Zstandard module")),
        "Default pickle protocol": _compact(_field(answer, "Default pickle protocol")),
        "map parameter": _compact(_field(answer, "map parameter")),
        "Thread.join exception": _compact(_field(answer, "Thread.join exception")),
    }
    checks = {
        "Zstandard module": fields["Zstandard module"] == "compression.zstd",
        "Default pickle protocol": fields["Default pickle protocol"] == "5",
        "map parameter": fields["map parameter"] == "strict",
        "Thread.join exception": fields["Thread.join exception"] == "pythonfinalizationerror",
    }
    failures = tuple(
        f"incorrect or missing field {label!r}" for label, ok in checks.items() if not ok
    )
    return ValidationResult(not failures, failures)


def build_tasks(documents: Mapping[str, str], *, label: str) -> Sequence[Task]:
    """Build one exact-graded distributed-evidence task per document."""

    def metadata(name: str) -> Dict[str, Any]:
        document = documents[name]
        spec = DOCUMENT_SPECS[name]
        return {
            "corpus": name,
            "variant": label,
            "source_url": spec["source_url"],
            "artifact_sha256": spec["sha256"],
            **({"download_sha256": spec["download_sha256"]} if "download_sha256" in spec else {}),
            "context_sha256": _sha256(document.encode("utf-8")),
            "characters": len(document),
        }

    return (
        Task(
            name="war_and_peace_petya",
            query=(
                "Reconstruct Petya's final-night sequence using narrative evidence. State the "
                "Cossack who helped him, the service he requested, the payment he gave, the "
                "companion who entered the French camp with him, and who said `Done for!` after "
                "Petya fell. Return concise labeled fields."
            ),
            context=documents["war_and_peace"],
            validator=validate_petya_final_night,
            metadata=metadata("war_and_peace"),
        ),
        Task(
            name="commission_distributed_facts",
            query=(
                "Use only the report. From the preface, give the number of document pages reviewed, "
                "individuals interviewed, and countries. From the intelligence reform recommendation, "
                "give the title replacing the Director of Central Intelligence, its number of main "
                "responsibility areas, and its proposed location. Return exactly six lines labeled "
                "`Pages reviewed:`, `Individuals interviewed:`, `Countries:`, `Replacement:`, "
                "`Responsibilities:`, and `Location:`."
            ),
            context=documents["commission_report"],
            validator=validate_commission_facts,
            metadata=metadata("commission_report"),
        ),
        Task(
            name="python_314_distributed_facts",
            query=(
                "Use only the Python 3.14 documentation corpus. Identify the module added for "
                "Zstandard compression, the default pickle protocol in 3.14, the parameter added "
                "to map(), and the exception Thread.join() may raise during late finalization. "
                "Return exactly four lines labeled `Zstandard module:`, `Default pickle protocol:`, "
                "`map parameter:`, and `Thread.join exception:`."
            ),
            context=documents["python_docs"],
            validator=validate_python_docs_facts,
            metadata=metadata("python_docs"),
        ),
    )


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    """Write benchmark records as newline-delimited JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    """Load pinned documents, run selected tasks, and print a JSON report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="LiteLLM model identifier")
    parser.add_argument("documents", type=Path, help="directory containing the three pinned inputs")
    parser.add_argument("--runs", type=int, default=1, help="repetitions per task")
    parser.add_argument(
        "--task",
        choices=(
            "war_and_peace_petya",
            "commission_distributed_facts",
            "python_314_distributed_facts",
        ),
        help="run only one task",
    )
    parser.add_argument("--label", default="baseline", help="variant label stored in metadata")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=4_000)
    parser.add_argument("--max-total-calls", type=int, default=24)
    parser.add_argument("--max-elapsed-seconds", type=float, default=300)
    parser.add_argument("--jsonl", type=Path, help="write result and summary records")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="include full content-bearing trajectories in output",
    )
    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("--runs must be greater than zero")
    if args.max_depth < 0:
        parser.error("--max-depth has an invalid value")
    for name, value in (
        ("--max-iterations", args.max_iterations),
        ("--max-tokens", args.max_tokens),
        ("--max-total-calls", args.max_total_calls),
        ("--max-elapsed-seconds", args.max_elapsed_seconds),
    ):
        if value <= 0:
            parser.error(f"{name} has an invalid value")

    documents = load_documents(args.documents)
    tasks = build_tasks(documents, label=args.label)
    if args.task:
        tasks = tuple(task for task in tasks if task.name == args.task)

    load_dotenv()
    results: List[Dict[str, Any]] = []
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
                trace=args.trace,
            )
            results.append(result)
            print(
                f"Finished {task.name}: passed={result['passed']} "
                f"calls={result['stats']['llm_calls']} "
                f"cost=${result['stats']['estimated_cost_usd']}",
                flush=True,
            )

    summary = aggregate_results(args.model, results, max_depth=args.max_depth)
    summary["variant"] = args.label
    if args.jsonl:
        _write_jsonl(args.jsonl, [*results, summary])
    print(json.dumps({"summary": summary, "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
