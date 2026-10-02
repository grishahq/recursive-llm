"""Exact-graded real-document benchmark using Project Gutenberg eBook #2600.

Download the pinned input and run from the repository root:

    curl -L https://www.gutenberg.org/files/2600/2600-0.txt \
        -o /tmp/war-and-peace-2600-0.txt
    python benchmarks/war_and_peace.py gpt-5-mini \
        /tmp/war-and-peace-2600-0.txt --runs 1

The public-domain book is not stored in the repository. The benchmark checks
the downloaded bytes against the digest used to establish its answer key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from dotenv import load_dotenv

try:
    from .compare_same_model import (
        Task,
        ValidationResult,
        aggregate_results,
        run_task,
        validate_labeled_fields,
    )
except ImportError:  # Support direct execution from the repository root.
    from compare_same_model import (
        Task,
        ValidationResult,
        aggregate_results,
        run_task,
        validate_labeled_fields,
    )


DOCUMENT_URL = "https://www.gutenberg.org/files/2600/2600-0.txt"
DOCUMENT_SHA256 = "e4bcf9042609b62c7de72a6f1b311f54c412943a9d641b7efcf79a464b5f31c8"
GRADER_VERSION = "war-and-peace-fields-v2"
PETYA_FINAL_NIGHT_QUERY = (
    "Reconstruct Petya's final-night sequence using narrative evidence. State the "
    "Cossack who helped him, the service he requested, the payment he gave, the "
    "companion who entered the French camp with him, who said `Done for!` after "
    "Petya fell, and the words of that verdict. Return exactly six lines labeled "
    "`Cossack:`, `Service:`, `Payment:`, `Companion:`, `Verdict speaker:`, and `Verdict:`. "
    "Use only the person's name for each person field, a short affirmative action phrase "
    "for the service, amount and currency for payment, and the quoted words for the verdict. "
    "Give every field once with no other text."
)


def validate_body_chapter_count(answer: str) -> ValidationResult:
    """Require the exact count of chapter headings in the narrative body."""
    match = re.fullmatch(r"\s*total\s+chapters\s*[:=]\s*365\s*", answer, re.IGNORECASE)
    failures = () if match else ("expected only Total chapters: 365",)
    return ValidationResult(not failures, failures)


def validate_distant_fact_retrieval(answer: str) -> ValidationResult:
    """Require three facts located near the beginning and end of the novel."""
    return validate_labeled_fields(
        answer,
        {
            "illness": ("la grippe",),
            "annual cost": tuple(
                f"{amount} {currency}"
                for amount in ("40,000", "40 000", "40000", "forty thousand")
                for currency in ("rubles", "roubles")
            ),
            "horse": ("Karabakh",),
        },
    )


def validate_petya_final_night(answer: str) -> ValidationResult:
    """Require the people, service, payment, and verdict in Petya's final sequence."""
    return validate_labeled_fields(
        answer,
        {
            "cossack": ("Likhachev",),
            "service": tuple(
                f"{verb} {possessive}{weapon}"
                for verb in ("sharpen", "sharpening", "sharpened", "to sharpen")
                for possessive in ("", "the ", "his ", "Petya's ")
                for weapon in ("saber", "sabre")
            ),
            "payment": ("one ruble", "1 ruble", "one rouble", "1 rouble"),
            "companion": ("Dolokhov",),
            "verdict speaker": ("Dolokhov",),
            "verdict": ("Done for!",),
        },
    )


def build_tasks(document: str, *, document_sha256: str) -> Sequence[Task]:
    """Build exact-graded tasks over one verified copy of the document."""
    metadata: Dict[str, Any] = {
        "source_url": DOCUMENT_URL,
        "sha256": document_sha256,
        "characters": len(document),
        "grader_version": GRADER_VERSION,
    }
    return (
        Task(
            name="body_chapter_count",
            query=(
                "Analyze the full document. Count the unindented narrative-body headings whose "
                "complete line has the form `CHAPTER <Roman numeral>`. Exclude the indented table "
                "of contents and the Project Gutenberg license. Count all 15 books and both "
                "epilogues. Return exactly `Total chapters: <integer>`."
            ),
            context=document,
            validator=validate_body_chapter_count,
            metadata=metadata,
        ),
        Task(
            name="distant_fact_retrieval",
            query=(
                "Find three facts in the novel: the illness term described near the opening as "
                "a new word in St. Petersburg, the annual amount Prince Vasili says Anatole costs "
                "him, and the name Petya calls his horse late in the novel. Return exactly three "
                "lines labeled `Illness:`, `Annual cost:`, and `Horse:`. Use only the illness "
                "term, amount and currency, and horse's name in their respective fields. "
                "Give every field once with no other text."
            ),
            context=document,
            validator=validate_distant_fact_retrieval,
            metadata=metadata,
        ),
        Task(
            name="petya_final_night",
            query=PETYA_FINAL_NIGHT_QUERY,
            context=document,
            validator=validate_petya_final_night,
            metadata=metadata,
        ),
    )


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    """Write benchmark records as newline-delimited JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    """Verify the document, run the selected tasks, and print a JSON report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="LiteLLM model identifier")
    parser.add_argument("document", type=Path, help="downloaded Project Gutenberg text file")
    parser.add_argument("--runs", type=int, default=1, help="repetitions per task")
    parser.add_argument(
        "--task",
        choices=("body_chapter_count", "distant_fact_retrieval", "petya_final_night"),
        help="run only one task",
    )
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

    raw_document = args.document.read_bytes()
    digest = hashlib.sha256(raw_document).hexdigest()
    if digest != DOCUMENT_SHA256:
        parser.error(f"document SHA-256 mismatch: expected {DOCUMENT_SHA256}, observed {digest}")
    document = raw_document.decode("utf-8-sig").replace("\r\n", "\n")

    load_dotenv()
    tasks = build_tasks(document, document_sha256=digest)
    if args.task:
        tasks = tuple(task for task in tasks if task.name == args.task)

    results: List[Dict[str, Any]] = []
    for task in tasks:
        for run_index in range(1, args.runs + 1):
            print(
                f"Running {args.model}: {task.name} ({run_index}/{args.runs})",
                flush=True,
            )
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
    if args.jsonl:
        _write_jsonl(args.jsonl, [*results, summary])
    print(json.dumps({"summary": summary, "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
