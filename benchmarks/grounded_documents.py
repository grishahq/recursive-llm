"""Bounded, source-grounded document evaluation through Codex subscription login.

Prepare immutable public inputs without model calls::

    python -m benchmarks.grounded_documents --prepare

Compare a direct call, a Python/search agent, and recursive execution::

    python -m benchmarks.grounded_documents --model gpt-6.1-sol \
        --task synthetic-fee --mode direct --mode python --mode rlm

Gold answers are used only after completion. Failed attempts, raw transport
events, prompts, source identities and runtime settings are retained. CLI token
counts include transport overhead and do not measure subscription credits.
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import hashlib
import importlib.metadata
import json
import platform
import re
import time
import urllib.request
import unicodedata
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from rlm import RLM
from rlm.codex import CodexCompletion
from rlm.sources import SourceCorpus, SourceDocument
from rlm.types import Message

from .compare_same_model import ValidationResult, validate_labeled_fields

GRADER_VERSION = "grounded-document-fields-v2"
DATASET_VERSION = "grounded-documents-v1"
DEFAULT_DIRECTORY = Path(__file__).resolve().parents[1] / "experiments/grounded-documents-20261102"
SOURCE_SPECS: Mapping[str, Mapping[str, str]] = {
    "cloudflare-2024": {
        "filename": "cloudflare-10k.pdf",
        "pages_filename": "cloudflare-10k.pages.json",
        "title": "Cloudflare 2024 Form 10-K, complete 157-page report",
        "url": "https://d18rn0p25nwr6d.cloudfront.net/CIK-0001477333/2c5ce18c-bb02-4031-89cb-59ff64f22eaf.pdf",
        "raw_sha256": "ed58d484853f3a8da0dd2af65461fda3974f002cc3f609edeb749b93c22a6315",
        "content_sha256": "a28227ea404cb6df0463ee62717e01be3281623a400329a115248cf2d6e02b12",
        "extraction": "pypdf==6.18.0, extract_text() plain; physical PDF pages, blank password",
    },
    "python-sqlite-3.12": {
        "filename": "python-sqlite3-v3.12.0.rst",
        "pages_filename": "python-sqlite3-v3.12.0.pages.json",
        "title": "Python 3.12.0 sqlite3 documentation source",
        "url": "https://raw.githubusercontent.com/python/cpython/v3.12.0/Doc/library/sqlite3.rst",
        "raw_sha256": "84b20c22a0201ccaa0ff584a224d743cc1eb6d0dc0409210ab8f11b910ea51ca",
        "content_sha256": "adbfbad0c922070862984f00ba4e1011bc9c0c343767cbcf8575fb5b6a2d900d",
        "extraction": "UTF-8 decode, one text page, no normalization",
    },
}


@dataclass(frozen=True)
class GoldEvidence:
    document_id: str
    page: int
    text: str


@dataclass(frozen=True)
class GroundedTask:
    name: str
    question: str
    documents: Tuple[SourceDocument, ...]
    expected: Mapping[str, Tuple[str, ...]]
    evidence: Mapping[str, Tuple[GoldEvidence, ...]] = field(default_factory=dict)
    unknowns: Tuple[str, ...] = ()
    answer_kind: str = "fields"

    @property
    def query(self) -> str:
        fields = ", ".join(self.expected)
        if self.answer_kind == "python":
            return (
                self.question + "\nThe answer string must contain only a Python function named "
                "application_quote(application_count, approved_nonprofit, expedited), without "
                "Markdown fences or imports. Return a tuple (total_fee_eur, review_business_days). "
                "application_count is a positive integer; the other inputs are booleans. "
                "The total fee is the per-application fee multiplied by application_count. "
                "Use only the policy rules and respect its exception. "
                "Create exactly one cited factual claim for each of these source facts, "
                f"using 'field: value' claim strings: {fields}. "
                "Use numeric values, preserve exact source citations, and return the required "
                "JSON object with answer, claims and unknowns."
            )
        return (
            self.question
            + "\nFor deterministic evaluation, put only these labeled fields in the answer "
            + f"string, one field per line: {fields}. Use 'field: value'. "
            + "For each supported field, create exactly one claim whose claim text is the "
            + "identical 'field: value' line and cite the evidence needed for that value. "
            + "For an unsupported requested value, use 'not specified' in its answer field, "
            + "do not create a factual claim for it, and put the field name in unknowns. "
            + "Do not infer an absent fee to be zero. Preserve names and accents from the source. "
            + "Do not use outside knowledge. Return the required JSON object."
        )


def synthetic_documents() -> Tuple[SourceDocument, ...]:
    """A fixed fictional policy with distant rules, an exception and Unicode names."""
    pages = []
    for page in range(1, 25):
        records = [
            f"Archive entry {page:02d}-{row:02d}: warehouse zone {row % 7}; "
            f"inventory reference AUR-{page:02d}{row:02d}; reconciliation is pending.\n"
            for row in range(22)
        ]
        pages.append(f"AURORA application policy. Physical page {page}.\n" + "".join(records))
    pages[2] += "Rule F1. The standard application fee is EUR 120.\n"
    pages[4] += (
        "Rule T0. Standard review takes 20 business days.\n"
        "Rule T1. An expedited application is normally reviewed in 5 business days.\n"
    )
    pages[13] += "Rule F2. An approved nonprofit pays 25% of the standard application fee in F1.\n"
    pages[16] += "The review owner for Aurora is Pétya Sokolova.\n"
    pages[21] += (
        "Exception T2. Approved nonprofit applications take 14 business days, "
        "including expedited applications; this overrides T1.\n"
    )
    pages[23] += (
        "This is the complete fictional policy snapshot. It defines application fees "
        "and review times. It does not specify a renewal fee.\n"
    )
    other = SourceDocument(
        "borealis-policy",
        (
            "BOREALIS policy. The standard application fee is EUR 900. "
            "Its review owner is Petya Ivanov. These rules do not apply to Aurora.\n",
        ),
        "Borealis policy, a different program",
    )
    return (
        SourceDocument("aurora-policy", tuple(pages), "Aurora fictional application policy"),
        other,
    )


def synthetic_tasks() -> Tuple[GroundedTask, ...]:
    documents = synthetic_documents()
    return (
        GroundedTask(
            "synthetic-fee",
            "Under the Aurora policy, what application fee in EUR does an approved "
            "nonprofit pay? Compute it from the stated standard fee and nonprofit rule.",
            documents,
            {"approved_nonprofit_fee_eur": ("30",)},
            {
                "approved_nonprofit_fee_eur": (
                    GoldEvidence("aurora-policy", 3, "The standard application fee is EUR 120."),
                    GoldEvidence("aurora-policy", 14, "An approved nonprofit pays 25%"),
                )
            },
        ),
        GroundedTask(
            "synthetic-exception",
            "How many business days does Aurora require to review an expedited "
            "application from an approved nonprofit? Account for the exception.",
            documents,
            {"expedited_nonprofit_review_business_days": ("14",)},
            {
                "expedited_nonprofit_review_business_days": (
                    GoldEvidence(
                        "aurora-policy", 22, "Approved nonprofit applications take 14 business days"
                    ),
                    GoldEvidence("aurora-policy", 22, "this overrides T1"),
                )
            },
        ),
        GroundedTask(
            "synthetic-unicode",
            "Find Petya, the review owner for Aurora, and give the full name with "
            "the original spelling. Do not confuse Aurora with Borealis.",
            documents,
            {"review_owner": ("Pétya Sokolova",)},
            {"review_owner": (GoldEvidence("aurora-policy", 17, "Pétya Sokolova"),)},
        ),
        GroundedTask(
            "synthetic-absence",
            "What renewal fee in EUR does the Aurora policy specify? Report an "
            "unknown when this policy gives no amount.",
            documents,
            {"renewal_fee_eur": ("not specified",)},
            unknowns=("renewal_fee_eur",),
        ),
        GroundedTask(
            "synthetic-code",
            "Compile the distributed Aurora application-fee and review-time rules into "
            "a reusable Python function. It must handle normal and expedited applications "
            "from approved nonprofits and other applicants. Account for every fee rule "
            "and the review-time override.",
            documents,
            {
                "standard_fee_eur": ("120",),
                "nonprofit_fee_percent": ("25",),
                "standard_review_days": ("20",),
                "expedited_review_days": ("5",),
                "nonprofit_review_days": ("14",),
            },
            {
                "standard_fee_eur": (
                    GoldEvidence("aurora-policy", 3, "The standard application fee is EUR 120."),
                ),
                "nonprofit_fee_percent": (
                    GoldEvidence("aurora-policy", 14, "An approved nonprofit pays 25%"),
                ),
                "standard_review_days": (
                    GoldEvidence("aurora-policy", 5, "Standard review takes 20 business days."),
                ),
                "expedited_review_days": (
                    GoldEvidence("aurora-policy", 5, "normally reviewed in 5 business days."),
                ),
                "nonprofit_review_days": (
                    GoldEvidence(
                        "aurora-policy", 22, "Approved nonprofit applications take 14 business days"
                    ),
                    GoldEvidence("aurora-policy", 22, "this overrides T1"),
                ),
            },
            answer_kind="python",
        ),
    )


def _verified_raw(path: Path, expected: str) -> bytes:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError(f"Raw source SHA-256 mismatch for {path.name}: {actual}")
    return raw


def prepare_sources(directory: Path) -> List[Dict[str, Any]]:
    """Pin original bytes and extraction, failing closed on source changes."""
    source_directory = directory / "sources"
    source_directory.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    for document_id, spec in SOURCE_SPECS.items():
        path = source_directory / spec["filename"]
        if not path.exists():
            request = urllib.request.Request(
                spec["url"], headers={"User-Agent": "recursive-llm-evaluation/1"}
            )
            with urllib.request.urlopen(request, timeout=45) as response:
                path.write_bytes(response.read())
        raw = _verified_raw(path, spec["raw_sha256"])
        if path.suffix == ".pdf":
            if importlib.metadata.version("pypdf") != "6.18.0":
                raise ValueError("Reproducible extraction requires pypdf==6.18.0")
            from pypdf import PdfReader

            reader = PdfReader(path)
            if reader.is_encrypted and not reader.decrypt(""):
                raise ValueError("Pinned public PDF cannot be opened with an empty password")
            pages = tuple(page.extract_text() or "" for page in reader.pages)
        else:
            pages = (raw.decode("utf-8"),)
        document = SourceDocument(document_id, pages, spec["title"])
        if document.content_sha256 != spec["content_sha256"]:
            raise ValueError(f"Pinned page extraction changed for {document_id}")
        (source_directory / spec["pages_filename"]).write_text(
            json.dumps(pages, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest.append(
            {**dict(spec), **SourceCorpus((document,)).manifest()[0], "raw_bytes": len(raw)}
        )
    for document in synthetic_documents():
        manifest.append({**SourceCorpus((document,)).manifest()[0], "generator": DATASET_VERSION})
    (directory / "source-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tasks = synthetic_tasks() + public_tasks(load_public_documents(directory))
    gold = []
    for task in tasks:
        corpus = SourceCorpus(task.documents)
        by_id = {document.document_id: document for document in task.documents}
        evidence = {}
        for label, required in task.evidence.items():
            spans = []
            for item in required:
                original = by_id[item.document_id].pages[item.page - 1]
                first = original.index(item.text)
                spans.append(
                    corpus.read(item.document_id, item.page, first, first + len(item.text))
                )
            evidence[label] = spans
        gold.append(
            {
                "task": task.name,
                "expected_fields": task.expected,
                "answer_kind": task.answer_kind,
                "unknowns": task.unknowns,
                "evidence": evidence,
            }
        )
    (directory / "gold.json").write_text(
        json.dumps(
            {"grader_version": GRADER_VERSION, "model_feedback": False, "tasks": gold},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def load_public_documents(directory: Path) -> Dict[str, SourceDocument]:
    documents = {}
    for document_id, spec in SOURCE_SPECS.items():
        _verified_raw(directory / "sources" / spec["filename"], spec["raw_sha256"])
        pages = json.loads(
            (directory / "sources" / spec["pages_filename"]).read_text(encoding="utf-8")
        )
        document = SourceDocument(document_id, pages, spec["title"])
        if document.content_sha256 != spec["content_sha256"]:
            raise ValueError(f"Prepared source pages changed for {document_id}; run --prepare")
        documents[document_id] = document
    return documents


def public_tasks(documents: Mapping[str, SourceDocument]) -> Tuple[GroundedTask, ...]:
    cloudflare = (documents["cloudflare-2024"],)
    return (
        GroundedTask(
            "cloudflare-year",
            "For which fiscal year was this Cloudflare Form 10-K filed? Give the four-digit year.",
            cloudflare,
            {"report_year": ("2024",)},
            {
                "report_year": (
                    GoldEvidence(
                        "cloudflare-2024", 1, "For the fiscal year ended December 31, 2024"
                    ),
                )
            },
        ),
        GroundedTask(
            "cloudflare-products",
            "According to the Developer-based Solutions AI product descriptions, "
            "what compute powers Workers AI and what data does Vectorize store and query? "
            "Use short noun phrases for workers_ai_compute and vectorize_data.",
            cloudflare,
            {"workers_ai_compute": ("serverless GPUs",), "vectorize_data": ("embeddings",)},
            {
                "workers_ai_compute": (
                    GoldEvidence("cloudflare-2024", 12, "powered by serverless GPUs"),
                ),
                "vectorize_data": (
                    GoldEvidence("cloudflare-2024", 12, "store and query embeddings"),
                ),
            },
        ),
        GroundedTask(
            "cloudflare-gateway",
            "List the four AI Gateway features named in the Developer-based Solutions "
            "AI product description. Use a comma-separated list in the report's order, "
            "without 'and' or trailing punctuation.",
            cloudflare,
            {"ai_gateway_features": ("observability, caching, security, routing",)},
            {
                "ai_gateway_features": (
                    GoldEvidence(
                        "cloudflare-2024", 12, "observability, caching, security, and routing"
                    ),
                )
            },
        ),
        GroundedTask(
            "cloudflare-ai-product-and-risks",
            "Across Cloudflare's Developer-based Solutions and its risk factors, "
            "identify the compute powering Workers AI; the two dimensions of cyber "
            "attacks that artificial intelligence or machine learning may increase; "
            "and the information that employee or contractor errors using AI or machine "
            "learning could expose externally. For workers_ai_compute use the short "
            "noun phrase. For ai_attack_dimensions use comma-separated nouns in source "
            "order, without 'and' or punctuation. For ai_information_exposure use the "
            "complete noun phrase describing the exposed information.",
            cloudflare,
            {
                "workers_ai_compute": ("serverless GPUs",),
                "ai_attack_dimensions": ("prevalence, intensity",),
                "ai_information_exposure": ("proprietary or other confidential information",),
            },
            {
                "workers_ai_compute": (
                    GoldEvidence("cloudflare-2024", 12, "powered by serverless GPUs"),
                ),
                "ai_attack_dimensions": (
                    GoldEvidence(
                        "cloudflare-2024",
                        37,
                        "increase the prevalence or intensity of cyber attacks.",
                    ),
                ),
                "ai_information_exposure": (
                    GoldEvidence(
                        "cloudflare-2024",
                        59,
                        "proprietary or\nother confidential information being exposed externally.",
                    ),
                ),
            },
        ),
        GroundedTask(
            "sqlite-defaults",
            "In the Python 3.12.0 sqlite3.connect signature, what are the default timeout "
            "in seconds and cached_statements count? Use timeout_seconds and cached_statements.",
            (documents["python-sqlite-3.12"],),
            {"timeout_seconds": ("5.0",), "cached_statements": ("128",)},
            {
                "timeout_seconds": (GoldEvidence("python-sqlite-3.12", 1, "timeout=5.0"),),
                "cached_statements": (
                    GoldEvidence("python-sqlite-3.12", 1, "cached_statements=128"),
                ),
            },
        ),
    )


def _claim_field(text: str) -> Optional[Tuple[str, str]]:
    match = re.fullmatch(r"\s*([^:=\n]+?)\s*[:=]\s*(\S[^\n]*?)\s*", text)
    if match is None:
        return None
    return " ".join(match.group(1).casefold().split()), match.group(2)


def _validate_fields(answer: str, expected: Mapping[str, Sequence[str]]) -> ValidationResult:
    base = validate_labeled_fields(answer, expected)
    failures = list(base.failures)
    # Preserve accents and exact finite aliases, including names found by folded search.
    for line in answer.splitlines():
        pair = _claim_field(line)
        if pair is None or pair[0] not in expected:
            continue
        label, value = pair
        normalized = " ".join(unicodedata.normalize("NFC", value).casefold().split())
        aliases = {
            " ".join(unicodedata.normalize("NFC", item).casefold().split())
            for item in expected[label]
        }
        if normalized not in aliases:
            failures.append(f"incorrect exact value for {label!r}")
    return ValidationResult(not failures, tuple(failures))


def _strict_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(_value: str) -> None:
    raise ValueError("Non-finite JSON value")


def _parse_shape(raw: str) -> Dict[str, Any]:
    """Check output shape while keeping factual and source errors separable."""
    value = json.loads(raw, object_pairs_hook=_strict_object, parse_constant=_invalid_constant)
    if not isinstance(value, dict) or set(value) != {"answer", "claims", "unknowns"}:
        raise ValueError("Answer must contain exactly answer, claims, unknowns")
    if not isinstance(value["answer"], str) or not value["answer"].strip():
        raise ValueError("Answer must be nonempty text")
    if not isinstance(value["unknowns"], list) or any(
        not isinstance(item, str) for item in value["unknowns"]
    ):
        raise ValueError("Unknowns must be strings")
    if not isinstance(value["claims"], list):
        raise ValueError("Claims must be a list")
    for claim in value["claims"]:
        if not isinstance(claim, dict) or set(claim) != {"claim", "citations"}:
            raise ValueError("Each claim must contain exactly claim and citations")
        if not isinstance(claim["claim"], str) or not isinstance(claim["citations"], list):
            raise ValueError("Invalid claim shape")
        if any(not isinstance(citation, dict) for citation in claim["citations"]):
            raise ValueError("Citations must be objects")
    return value


def grade_python_function(answer: str) -> Dict[str, Any]:
    """Execute generated code in the existing restricted, time-bounded worker.

    Numeric examples and expected outputs are never supplied in model prompts.
    All flag combinations are crossed with four quantities to avoid correlated
    inputs rewarding a function that ignores its boolean arguments. This is a
    toy policy check, not formal verification.
    """
    cases = tuple(
        (count, nonprofit, expedited)
        for count in (1, 3, 7, 13)
        for nonprofit in (False, True)
        for expedited in (False, True)
    )
    expected = tuple(
        ((30 if nonprofit else 120) * count, 14 if nonprofit else (5 if expedited else 20))
        for count, nonprofit, expedited in cases
    )
    try:
        tree = ast.parse(answer)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise ValueError("Code answer must contain exactly one function")
        function = tree.body[0]
        arguments = function.args
        if (
            function.name != "application_quote"
            or function.decorator_list
            or [argument.arg for argument in arguments.args]
            != ["application_count", "approved_nonprofit", "expedited"]
            or arguments.posonlyargs
            or arguments.kwonlyargs
            or arguments.vararg
            or arguments.kwarg
            or arguments.defaults
        ):
            raise ValueError("Unexpected function signature or decorators")
        from rlm.repl import REPLExecutor

        executor = REPLExecutor(timeout=2)
        environment: Dict[str, Any] = {}
        try:
            executor.execute(
                answer
                + "\nbenchmark_outputs = [application_quote(case[0], case[1], case[2]) for case in benchmark_inputs]",
                {"benchmark_inputs": cases, **environment},
            )
            found, outputs = executor.get_variable("benchmark_outputs")
        finally:
            executor.close()
        if not found or not isinstance(outputs, (list, tuple)) or len(outputs) != len(expected):
            raise ValueError("Function did not return one result per held-out input")
        failures = []
        for index, (actual, wanted) in enumerate(zip(outputs, expected)):
            if (
                not isinstance(actual, (list, tuple))
                or len(actual) != 2
                or any(type(item) not in (int, float) for item in actual)
                or tuple(actual) != wanted
            ):
                failures.append(f"held-out input {index + 1} returned an incorrect quote")
        return {
            "passed": not failures,
            "failures": failures,
            "tested_inputs": len(cases),
            "observed_outputs": outputs,
        }
    except Exception as exc:
        return {
            "passed": False,
            "failures": [f"{type(exc).__name__}: {exc}"],
            "tested_inputs": len(cases),
        }


def grade_answer(task: GroundedTask, raw: str) -> Dict[str, Any]:
    """Grade factual fields and evidence independently, with no model feedback.

    Exact field aliases bound the answer value. Gold source snippets bind each
    claim's evidence; a real quotation from an unrelated location cannot pass.
    This deliberately grades a finite contract, not arbitrary paraphrase entailment.
    """
    corpus = SourceCorpus(task.documents)
    failures: List[str] = []
    provenance_error = corpus.validate_answer(raw)
    try:
        parsed = _parse_shape(raw)
    except (TypeError, ValueError, RecursionError) as exc:
        return {
            "passed": False,
            "factual_passed": False,
            "provenance_passed": False,
            "failures": [str(exc)],
            "grader_version": GRADER_VERSION,
        }
    if task.answer_kind == "python":
        content_grade = grade_python_function(parsed["answer"])
    else:
        fields = _validate_fields(parsed["answer"], task.expected)
        content_grade = {"passed": fields.passed, "failures": list(fields.failures)}
    failures.extend(content_grade["failures"])
    if provenance_error:
        failures.append(provenance_error)
    if tuple(sorted(parsed["unknowns"])) != tuple(sorted(task.unknowns)):
        failures.append("unknown fields differ from gold")
    observed: Dict[str, Dict[str, Any]] = {}
    for claim in parsed["claims"]:
        pair = _claim_field(claim["claim"])
        if pair is None:
            failures.append("claim must contain exactly one labeled field")
            continue
        label, _value = pair
        if label in observed:
            failures.append(f"duplicate claim for {label!r}")
        observed[label] = claim
    if set(observed) != set(task.evidence):
        failures.append("claim fields differ from supported gold fields")
    supported_fields = 0
    for label, required in task.evidence.items():
        claim = observed.get(label)
        if claim is None:
            continue
        value_check = _validate_fields(claim["claim"], {label: task.expected[label]})
        if not value_check.passed:
            failures.append(f"incorrect claim value for {label!r}")
        covered = all(
            any(
                citation.get("document_id") == evidence.document_id
                and citation.get("page") == evidence.page
                and isinstance(citation.get("quote"), str)
                and evidence.text in citation.get("quote", "")
                for citation in claim["citations"]
            )
            for evidence in required
        )
        if not covered:
            failures.append(f"claim evidence does not cover gold source for {label!r}")
        if value_check.passed and covered and not provenance_error:
            supported_fields += 1
    return {
        "passed": not failures,
        "factual_passed": content_grade["passed"]
        and tuple(sorted(parsed["unknowns"])) == tuple(sorted(task.unknowns)),
        "provenance_passed": provenance_error is None,
        "supported_gold_fields": supported_fields,
        "required_gold_fields": len(task.evidence),
        "failures": failures,
        "answer_grade": content_grade,
        "grader_version": GRADER_VERSION,
    }


DIRECT_INSTRUCTIONS = (
    "Answer using only the supplied source snapshot. Return exactly one JSON object with "
    "answer (string), claims (list of objects with claim string and citations list), and "
    "unknowns (list of field-name strings). Each citation must have exactly document_id, "
    "source_sha256, page, start, end, quote. Physical pages start at 1; start/end are "
    "zero-based half-open Python character offsets within that original page. "
    "The source is supplied as exact precomputed segments. You may copy a whole segment "
    "with its coordinates as a citation, using document_id and source_sha256 from its "
    "document header. No tools are available. Do not invent missing values or quotations."
)


def direct_sources(
    documents: Sequence[SourceDocument], segment_chars: int = 1200
) -> List[Dict[str, Any]]:
    """Provide exact precomputed segments so direct calls need no offset arithmetic.

    All source characters remain present. Segments usually end at a newline;
    direct-mode input overhead is recorded as part of that mode's actual usage.
    """
    result = []
    for document in documents:
        segments = []
        for page, original in enumerate(document.pages, 1):
            start = 0
            while start < len(original):
                end = min(start + segment_chars, len(original))
                if end < len(original):
                    newline = original.rfind("\n", start + segment_chars // 2, end)
                    if newline >= 0:
                        end = newline + 1
                segments.append(
                    {"page": page, "start": start, "end": end, "quote": original[start:end]}
                )
                start = end
        result.append(
            {
                "document_id": document.document_id,
                "source_sha256": document.content_sha256,
                "title": document.title,
                "segments": segments,
            }
        )
    return result


def _transport_usage(calls: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    usage: Dict[str, Any] = {
        "llm_calls": len(calls),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "usage_calls": 0,
        "unknown_usage_calls": 0,
        "failed_calls_with_known_usage": 0,
        "estimated_cost_usd": None,
        "billing_mode": "subscription",
    }
    for call in calls:
        response = call.get("response", {})
        values = response.get("usage")
        if not values:
            # A contract failure can occur after the CLI has reported token usage.
            completed = [
                event
                for event in call.get("events", [])
                if isinstance(event, dict) and event.get("type") == "turn.completed"
            ]
            if len(completed) == 1:
                observed = completed[0].get("usage")
                keys = ("input_tokens", "cached_input_tokens", "output_tokens")
                if (
                    isinstance(observed, dict)
                    and all(type(observed.get(key)) is int and observed[key] >= 0 for key in keys)
                    and observed["cached_input_tokens"] <= observed["input_tokens"]
                ):
                    values = {
                        "prompt_tokens": observed["input_tokens"],
                        "completion_tokens": observed["output_tokens"],
                        "total_tokens": observed["input_tokens"] + observed["output_tokens"],
                        "prompt_tokens_details": {"cached_tokens": observed["cached_input_tokens"]},
                    }
        if values:
            usage["usage_calls"] += 1
            usage["failed_calls_with_known_usage"] += call.get("status") == "failed"
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[key] += values.get(key, 0)
            usage["cached_tokens"] += values.get("prompt_tokens_details", {}).get(
                "cached_tokens", 0
            )
        else:
            usage["unknown_usage_calls"] += 1
    return usage


async def run_one(task: GroundedTask, mode: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Execute one bounded attempt; retain failures without switching transport."""
    transport = CodexCompletion(timeout=args.call_timeout, reasoning_effort=args.reasoning_effort)
    events: List[Dict[str, Any]] = []
    runtime = None
    raw = None
    record: Dict[str, Any] = {
        "dataset_version": DATASET_VERSION,
        "grader_version": GRADER_VERSION,
        "task": task.name,
        "mode": mode,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "query": task.query,
        "sources": SourceCorpus(task.documents).manifest(),
        "config": {
            "max_total_calls": 1 if mode == "direct" else args.max_calls,
            "max_iterations": args.max_iterations,
            "max_elapsed_seconds": args.timeout,
            "call_timeout": args.call_timeout,
            "max_total_tokens": args.max_tokens,
            "max_output_chars": args.max_output_chars,
            "max_depth": {"direct": None, "python": 0, "rlm": 2}[mode],
            "token_budget_enforcement": (
                "not available through text-only CLI; usage observed after the single call"
                if mode == "direct"
                else "tree-wide reported usage budget"
            ),
        },
        "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    started = time.monotonic()
    try:
        if mode == "direct":
            messages: List[Message] = [
                {"role": "system", "content": DIRECT_INSTRUCTIONS},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"question": task.query, "documents": direct_sources(task.documents)},
                        ensure_ascii=False,
                    ),
                },
            ]
            response = await asyncio.wait_for(
                transport(model=args.model, messages=messages), args.timeout
            )
            raw = response["choices"][0]["message"]["content"]
            record["direct_input_format"] = (
                "complete original pages in exact 1200-character segments"
            )
        else:
            runtime = RLM(
                model=args.model,
                completion_handler=transport,
                max_depth=0 if mode == "python" else 2,
                max_iterations=args.max_iterations,
                max_total_calls=args.max_calls,
                max_total_tokens=args.max_tokens,
                max_elapsed_seconds=args.timeout,
                max_output_chars=args.max_output_chars,
                capture_trajectory_content=True,
                event_handler=lambda event: events.append(event.to_dict()),
                max_concurrent_subcalls=2,
                max_retries=0,
            )
            result = await asyncio.wait_for(
                runtime.acomplete_documents(task.query, task.documents), args.timeout + 5
            )
            record["result"] = result.to_dict()
            raw = result.completion.answer
        record["status"] = "completed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        record["elapsed_seconds"] = round(time.monotonic() - started, 6)
        record["raw_answer"] = raw
        record["transport_calls"] = transport.calls
        record["transport_usage"] = _transport_usage(transport.calls)
        record["runtime_stats"] = runtime.stats if runtime is not None else None
        record["trajectory"] = events
        record["grade"] = (
            grade_answer(task, raw)
            if raw is not None
            else {
                "passed": False,
                "factual_passed": False,
                "provenance_passed": False,
                "failures": ["no completed answer"],
                "grader_version": GRADER_VERSION,
            }
        )
        if getattr(args, "force_child", False):
            child_calls = (record["runtime_stats"] or {}).get("recursive_calls", 0)
            record["integration_checks"] = {
                "forced_child": True,
                "child_calls": child_calls,
                "child_used": child_calls > 0,
                "note": "Explicit integration exercise; excluded from the performance comparison.",
            }
            if not child_calls:
                record["grade"]["passed"] = False
                record["grade"]["failures"].append("forced child call was not observed")
    return record


def summarize(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, Any]] = {}
    for record in records:
        group = groups.setdefault(
            record["mode"],
            {
                "attempts": 0,
                "completed": 0,
                "exact_passes": 0,
                "factual_passes": 0,
                "provenance_passes": 0,
                "total_tokens": 0,
                "elapsed_seconds": 0.0,
                "calls": 0,
            },
        )
        group["attempts"] += 1
        group["completed"] += record["status"] == "completed"
        for target, source in (
            ("exact_passes", "passed"),
            ("factual_passes", "factual_passed"),
            ("provenance_passes", "provenance_passed"),
        ):
            group[target] += bool(record["grade"][source])
        group["total_tokens"] += record["transport_usage"]["total_tokens"]
        group["calls"] += record["transport_usage"]["llm_calls"]
        group["elapsed_seconds"] += record["elapsed_seconds"]
    return {
        "groups": groups,
        "attempt_count": len(records),
        "limitations": [
            "Small engineering pilot, not a population estimate or paper reproduction.",
            "Codex subscription token usage is not subscription credits or a dollar price.",
            "Failed calls without usage can make observed tokens a lower bound.",
            "Direct input includes exact source segments to avoid manual offset arithmetic.",
            "No tail-latency estimate is made from a small number of repetitions.",
            "RLM mode permits children; a trajectory must confirm whether children were used.",
        ],
    }


async def run_suite(tasks: Sequence[GroundedTask], args: argparse.Namespace) -> Dict[str, Any]:
    modes = args.mode or ["direct", "python", "rlm"]
    for repeat in range(1, args.repeats + 1):
        for task in tasks:
            for mode in modes:
                path = args.output / f"{task.name}-{mode}-{repeat}.json"
                if path.exists():
                    raise ValueError(f"Refusing to overwrite an earlier attempt: {path}")
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    snapshot = args.output / "runtime-snapshot"
    files = list((root / "src/rlm").glob("*.py")) + [
        Path(__file__).resolve(),
        Path(__file__).with_name("compare_same_model.py").resolve(),
    ]
    code_manifest: Dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "note": "Source files copied after module import, before model calls. Keep the workspace stable during a suite.",
        "files": {},
    }
    for path in files:
        relative = path.relative_to(root)
        data = path.read_bytes()
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        code_manifest["files"][str(relative)] = hashlib.sha256(data).hexdigest()
    for package in ("litellm", "RestrictedPython", "pypdf"):
        try:
            code_manifest[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            code_manifest[package] = None
    (args.output / "runtime-manifest.json").write_text(
        json.dumps(code_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    login = CodexCompletion(timeout=args.call_timeout, reasoning_effort=args.reasoning_effort)
    await login.check_login()
    records = []
    for repeat in range(1, args.repeats + 1):
        for task in tasks:
            # Rotate order across repeats to reduce a fixed-order cache advantage.
            shift = (repeat - 1) % len(modes)
            for mode in modes[shift:] + modes[:shift]:
                record = await run_one(task, mode, args)
                record["repeat"] = repeat
                path = args.output / f"{task.name}-{mode}-{repeat}.json"
                path.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                records.append(record)
                print(
                    json.dumps(
                        {
                            "task": task.name,
                            "mode": mode,
                            "repeat": repeat,
                            "status": record["status"],
                            "grade": record["grade"],
                            "tokens": record["transport_usage"]["total_tokens"],
                            "seconds": record["elapsed_seconds"],
                            "artifact": str(path),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    summary = summarize(records)
    summary.update(
        {
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "dataset_version": DATASET_VERSION,
            "grader_version": GRADER_VERSION,
        }
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare", action="store_true", help="Download and pin sources; no model calls"
    )
    parser.add_argument("--list", action="store_true", help="List task names; no model calls")
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument(
        "--reasoning-effort", choices=("low", "medium", "high", "xhigh", "max"), default="low"
    )
    parser.add_argument("--mode", action="append", choices=("direct", "python", "rlm"))
    parser.add_argument("--task", action="append")
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument(
        "--force-child",
        action="store_true",
        help="Separate RLM-only source-inheritance integration exercise",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-calls", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=350000)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--call-timeout", type=float, default=120)
    parser.add_argument("--max-output-chars", type=int, default=16000)
    args = parser.parse_args()
    if args.mode and len(args.mode) != len(set(args.mode)):
        parser.error("--mode values must be unique")
    if args.force_child:
        if args.mode is not None and args.mode != ["rlm"]:
            parser.error("--force-child supports only --mode rlm")
        args.mode = ["rlm"]
    for name in (
        "repeats",
        "max_calls",
        "max_iterations",
        "max_tokens",
        "timeout",
        "call_timeout",
        "max_output_chars",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.prepare:
        print(json.dumps(prepare_sources(args.directory), ensure_ascii=False, indent=2))
        return
    tasks = list(synthetic_tasks())
    if not args.synthetic_only:
        tasks.extend(public_tasks(load_public_documents(args.directory)))
    if args.task:
        unknown = set(args.task) - {task.name for task in tasks}
        if unknown:
            parser.error(f"Unknown tasks: {sorted(unknown)}")
        tasks = [task for task in tasks if task.name in args.task]
    if args.list:
        for task in tasks:
            print(task.name)
        return
    if args.force_child:
        tasks = [
            replace(
                task,
                name=task.name + "-forced-child",
                question=(
                    "This is an explicit child-source integration check, not a performance baseline. "
                    "Call recursive_llm exactly once to ask a child RLM to solve this question using "
                    "source_manifest, search_sources and read_source. Pass a short sub_context telling "
                    "the child to inspect the inherited documents. Do not send the entire source as "
                    "the sub_context. Ask the child for the required grounded JSON, verify its cited "
                    "evidence, and return the answer. The child must read the inherited source.\n"
                    + task.question
                ),
            )
            for task in tasks
        ]
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output = args.directory / "runs" / stamp
    summary = asyncio.run(run_suite(tasks, args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
