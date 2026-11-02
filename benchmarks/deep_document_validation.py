"""Frozen real-document questions and differential reliability evidence.

Offline preparation and fixed-response controls make no model calls. Live runs
use the current runtime through Codex subscription login, retain every attempt,
and compare depth zero with depth two on identical source snapshots. They do not
measure an LLM-quality gain over the old release, which lacks this document API.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib
import importlib.util
import json
import statistics
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from rlm import RLM
from rlm.codex import CodexCompletion, parse_events
from rlm.document_io import load_document
from rlm.sources import SourceCorpus, SourceDocument

from .grounded_documents import (
    DEFAULT_DIRECTORY as ORIGINAL_DIRECTORY,
    GoldEvidence,
    GroundedTask,
    _claim_field,
    _parse_shape,
    _transport_usage,
    _validate_fields,
    load_public_documents,
    run_one,
)

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "experiments/deep-documents-20261102"
HEAD_REVISION = "3c255c070060837343e594f60523e02b5369b87d"
DATASET_VERSION = "deep-documents-v2"
GRADER_VERSION = "deep-document-fields-v1"
EvidenceVariants = Tuple[Tuple[GoldEvidence, ...], ...]


@dataclass(frozen=True)
class DeepTask:
    task: GroundedTask
    alternatives: Mapping[str, EvidenceVariants]


def _variants(document_id: str, pages: Sequence[int], text: str) -> EvidenceVariants:
    return tuple((GoldEvidence(document_id, page, text),) for page in pages)


def build_tasks(documents: Mapping[str, SourceDocument]) -> Tuple[DeepTask, ...]:
    cloud = (documents["cloudflare-2024"],)
    sqlite = (documents["python-sqlite-3.12"],)
    result = []

    def add(
        name: str,
        question: str,
        source: Tuple[SourceDocument, ...],
        expected: Mapping[str, Tuple[str, ...]],
        alternatives: Mapping[str, EvidenceVariants],
    ) -> None:
        task = GroundedTask(
            name,
            question,
            source,
            expected,
            {field: variants[0] for field, variants in alternatives.items()},
        )
        result.append(DeepTask(task, alternatives))

    add(
        "cloud-workforce",
        "At December 31, 2024, give Cloudflare's total full-time "
        "employee count, the count outside the United States, and compute the count "
        "inside the United States by subtraction. Use employees_total, employees_outside_us, "
        "employees_inside_us. Give integer counts without commas.",
        cloud,
        {
            "employees_total": ("4263",),
            "employees_outside_us": ("1896",),
            "employees_inside_us": ("2367",),
        },
        {
            field: _variants(
                "cloudflare-2024",
                (17,),
                "4,263 full-time employees, including 1,896 employees located outside of the United States",
            )
            for field in ("employees_total", "employees_outside_us", "employees_inside_us")
        },
    )
    add(
        "cloud-patents",
        "At December 31, 2024, how many issued patents and pending patent "
        "applications did Cloudflare have, and what first and last expiry years are stated "
        "for its issued patents? Use issued_patents, pending_patents, first_expiry_year, "
        "last_expiry_year. Give integers only.",
        cloud,
        {
            "issued_patents": ("334",),
            "pending_patents": ("72",),
            "first_expiry_year": ("2030",),
            "last_expiry_year": ("2044",),
        },
        {
            "issued_patents": _variants(
                "cloudflare-2024", (18, 59), "334 issued patents and 72 pending patent"
            ),
            "pending_patents": _variants(
                "cloudflare-2024", (18, 59), "334 issued patents and 72 pending patent"
            ),
            "first_expiry_year": _variants(
                "cloudflare-2024", (18,), "scheduled to expire between 2030 and 2044"
            ),
            "last_expiry_year": _variants(
                "cloudflare-2024", (18,), "scheduled to expire between 2030 and 2044"
            ),
        },
    )
    totals = (
        (GoldEvidence("cloudflare-2024", 78, "Paying customers 237,714 189,791 162,086"),),
        (
            GoldEvidence(
                "cloudflare-2024",
                80,
                "The number of paying customers was 237,714, 189,791, and 162,086",
            ),
        ),
    )
    large = (
        (
            GoldEvidence(
                "cloudflare-2024",
                78,
                "Paying customers (> $100,000 Annualized Revenue) 3,497 2,756 2,042",
            ),
        ),
        (
            GoldEvidence(
                "cloudflare-2024",
                81,
                "Annualized Revenue greater than $100,000 was 3,497, 2,756, and 2,042",
            ),
        ),
    )
    add(
        "cloud-customer-counts",
        "Give total paying customers and customers above $100,000 "
        "Annualized Revenue at December 31 in 2024 and 2023. Use paying_2024, paying_2023, "
        "large_2024, large_2023. Use exact counts rather than the rounded overview, without commas.",
        cloud,
        {
            "paying_2024": ("237714",),
            "paying_2023": ("189791",),
            "large_2024": ("3497",),
            "large_2023": ("2756",),
        },
        {"paying_2024": totals, "paying_2023": totals, "large_2024": large, "large_2023": large},
    )
    add(
        "cloud-free-cash-flow",
        "For 2024, give net cash provided by operating activities, "
        "purchases of property and equipment, capitalized internal-use software, and free "
        "cash flow. Check the stated subtraction. Use operating_cash, capex, software, "
        "free_cash_flow. Report USD thousands as nonnegative integers without commas; "
        "capex and software are magnitudes of the amounts subtracted.",
        cloud,
        {
            "operating_cash": ("380429",),
            "capex": ("185037",),
            "software": ("28477",),
            "free_cash_flow": ("166915",),
        },
        {
            "operating_cash": _variants(
                "cloudflare-2024",
                (78, 80, 89),
                "Net cash provided by operating activities $ 380,429",
            ),
            "capex": _variants(
                "cloudflare-2024", (80, 101), "Purchases of property and equipment (185,037)"
            ),
            "software": _variants(
                "cloudflare-2024", (80, 101), "Capitalized internal-use software (28,477)"
            ),
            "free_cash_flow": _variants("cloudflare-2024", (78, 80), "Free cash flow $ 166,915"),
        },
    )
    add(
        "cloud-concentration",
        "For 2024, did any single customer account for more than "
        "10% of revenue, and did any customer represent 10% or more of net accounts "
        "receivable? Use revenue_over_10pct and receivables_at_least_10pct, each exactly yes or no.",
        cloud,
        {"revenue_over_10pct": ("no",), "receivables_at_least_10pct": ("no",)},
        {
            "revenue_over_10pct": (
                (
                    GoldEvidence(
                        "cloudflare-2024",
                        13,
                        "No single customer accounted for more than 10% of our revenue",
                    ),
                ),
                (
                    GoldEvidence(
                        "cloudflare-2024",
                        104,
                        "no customer accounted for more than\n10% of the Company’s revenue",
                    ),
                ),
            ),
            "receivables_at_least_10pct": _variants(
                "cloudflare-2024",
                (104,),
                "No customer represented 10% or more of accounts receivable, net as of December 31, 2024 and 2023.",
            ),
        },
    )
    add(
        "cloud-retention-terms",
        "Give the dollar-based net retention rates for the three "
        "months ended December 31, 2024 and 2023, plus the minimum and maximum typical "
        "contracted-customer subscription/support term in years. Use retention_2024_pct, "
        "retention_2023_pct, contract_min_years, contract_max_years. Give numbers without percent signs.",
        cloud,
        {
            "retention_2024_pct": ("111",),
            "retention_2023_pct": ("115",),
            "contract_min_years": ("1",),
            "contract_max_years": ("3",),
        },
        {
            "retention_2024_pct": _variants(
                "cloudflare-2024", (81,), "2024, 2023, and 2022 were 111%, 115%"
            ),
            "retention_2023_pct": _variants(
                "cloudflare-2024", (81,), "2024, 2023, and 2022 were 111%, 115%"
            ),
            "contract_min_years": _variants(
                "cloudflare-2024", (81,), "typically range from one to three years"
            ),
            "contract_max_years": _variants(
                "cloudflare-2024", (81,), "typically range from one to three years"
            ),
        },
    )
    add(
        "cloud-regional-revenue",
        "For 2024, give revenue from the United States, Europe "
        "Middle East and Africa, Asia Pacific, and Other, based on customer billing "
        "addresses. Use us_revenue, emea_revenue, apac_revenue, other_revenue. Report USD "
        "thousands as integers without commas.",
        cloud,
        {
            "us_revenue": ("849500",),
            "emea_revenue": ("466499",),
            "apac_revenue": ("223234",),
            "other_revenue": ("130393",),
        },
        {
            "us_revenue": _variants("cloudflare-2024", (78,), "United States $ 849,500 51 %"),
            "emea_revenue": _variants(
                "cloudflare-2024", (78,), "Europe, Middle East, and Africa 466,499 28 %"
            ),
            "apac_revenue": _variants("cloudflare-2024", (78,), "Asia Pacific 223,234 13 %"),
            "other_revenue": _variants("cloudflare-2024", (78,), "Other 130,393 8 %"),
        },
    )
    add(
        "cloud-distributed-profile",
        "Prepare a small Cloudflare profile combining its "
        "staff, voting structure, and advertising notes: employees located outside the "
        "United States at December 31, 2024; votes per Class B share; and advertising "
        "expense in 2024. Use employees_outside_us, class_b_votes, advertising_usd_millions. "
        "Give numbers only, with no commas or unit symbols.",
        cloud,
        {
            "employees_outside_us": ("1896",),
            "class_b_votes": ("10",),
            "advertising_usd_millions": ("78.6",),
        },
        {
            "employees_outside_us": _variants(
                "cloudflare-2024", (17,), "1,896 employees located outside of the United States"
            ),
            "class_b_votes": (
                (
                    GoldEvidence(
                        "cloudflare-2024", 62, "Class B common stock has 10 votes per share"
                    ),
                ),
                (GoldEvidence("cloudflare-2024", 124, "entitled to 10 votes per share"),),
            ),
            "advertising_usd_millions": _variants(
                "cloudflare-2024",
                (106,),
                "Advertising expense for\nthe years ended December 31, 2024, 2023, and 2022 was $78.6 million",
            ),
        },
    )
    add(
        "sqlite-thread-contract",
        "In Python 3.12.0 sqlite3.connect, give the default "
        "check_same_thread value and the exception raised when that default connection "
        "is used from a different thread. Use check_same_thread and other_thread_exception. "
        "Use the Python boolean spelling and the bare exception class name.",
        sqlite,
        {"check_same_thread": ("True",), "other_thread_exception": ("ProgrammingError",)},
        {
            field: _variants(
                "python-sqlite-3.12",
                (1,),
                "If ``True`` (default), :exc:`ProgrammingError` will be raised\n       if the database connection is used by a thread\n       other than the one that created it.",
            )
            for field in ("check_same_thread", "other_thread_exception")
        },
    )
    add(
        "sqlite-row-factory",
        "For Connection.row_factory, give its default value, "
        "the resulting Python row type, and whether assigning a new factory changes "
        "already-existing cursors. Use row_factory_default, row_type, affects_existing. "
        "Use None, the bare type name, and yes/no respectively.",
        sqlite,
        {"row_factory_default": ("None",), "row_type": ("tuple",), "affects_existing": ("no",)},
        {
            "row_factory_default": _variants("python-sqlite-3.12", (1,), "Is ``None`` by default"),
            "row_type": _variants(
                "python-sqlite-3.12", (1,), "meaning each row is returned as a :class:`tuple`"
            ),
            "affects_existing": _variants(
                "python-sqlite-3.12",
                (1,),
                "Assigning to this attribute does not affect the :attr:`!row_factory`\n      of existing cursors",
            ),
        },
    )
    add(
        "sqlite-parameter-containers",
        "Which Python container type is required for "
        "qmark placeholders and which for named placeholders? What exception is raised "
        "when the qmark parameter count differs from the placeholder count? Use "
        "qmark_container, named_container, count_mismatch_exception. Use sequence, the "
        "bare concrete container type name, and the bare exception class name.",
        sqlite,
        {
            "qmark_container": ("sequence",),
            "named_container": ("dict",),
            "count_mismatch_exception": ("ProgrammingError",),
        },
        {
            "qmark_container": _variants(
                "python-sqlite-3.12",
                (1,),
                "For the qmark style, *parameters* must be a\n:term:`sequence`",
            ),
            "named_container": _variants(
                "python-sqlite-3.12",
                (1,),
                "For the named style, *parameters* must be\nan instance of a :class:`dict`",
            ),
            "count_mismatch_exception": _variants(
                "python-sqlite-3.12",
                (1,),
                "whose length must match the number of placeholders,\nor a :exc:`ProgrammingError` is raised",
            ),
        },
    )
    add(
        "sqlite-lastrowid",
        "For Cursor.lastrowid, give its initial value, the method "
        "that updates it after successful INSERT or REPLACE, and whether executemany "
        "updates it. Use initial_lastrowid, updating_method, executemany_updates. Use "
        "None, the bare method name, and yes/no respectively.",
        sqlite,
        {
            "initial_lastrowid": ("None",),
            "updating_method": ("execute",),
            "executemany_updates": ("no",),
        },
        {
            "initial_lastrowid": _variants(
                "python-sqlite-3.12", (1,), "The initial value of\n      ``lastrowid`` is ``None``"
            ),
            "updating_method": _variants(
                "python-sqlite-3.12",
                (1,),
                "only updated after successful ``INSERT`` or ``REPLACE`` statements\n      using the :meth:`execute` method",
            ),
            "executemany_updates": _variants(
                "python-sqlite-3.12",
                (1,),
                "after\n      :meth:`executemany` or :meth:`executescript`, or if the insertion failed,\n      the value of ``lastrowid`` is left unchanged",
            ),
        },
    )
    # Independent source review, completed before live inference, identified
    # equivalent locations and narrower sufficient quotations for these fields.
    reviewed = []
    for item in result:
        evidence = dict(item.alternatives)
        name = item.task.name
        if name == "cloud-workforce":
            evidence["employees_total"] = _variants(
                "cloudflare-2024", (17,), "4,263 full-time employees"
            )
            evidence["employees_outside_us"] = _variants(
                "cloudflare-2024", (17,), "1,896 employees located outside of the United States"
            )
            evidence["employees_inside_us"] = (
                (
                    GoldEvidence("cloudflare-2024", 17, "4,263 full-time employees"),
                    GoldEvidence(
                        "cloudflare-2024",
                        17,
                        "1,896 employees located outside of the United States",
                    ),
                ),
            )
        elif name == "cloud-patents":
            evidence["issued_patents"] = _variants(
                "cloudflare-2024", (18, 59), "334 issued patents"
            )
            evidence["pending_patents"] = _variants(
                "cloudflare-2024", (18, 59), "72 pending patent"
            )
        elif name == "cloud-customer-counts":
            alternative = _variants(
                "cloudflare-2024",
                (13,),
                "2,756 as of\nDecember 31, 2023 to 3,497 as of December 31, 2024",
            )
            evidence["large_2024"] += alternative
            evidence["large_2023"] += alternative
        elif name == "cloud-free-cash-flow":
            evidence["operating_cash"] += _variants(
                "cloudflare-2024", (101,), "Net cash provided by operating activities 380,429"
            )
        elif name == "cloud-retention-terms":
            evidence["retention_2024_pct"] += _variants(
                "cloudflare-2024", (85,), "dollar-based net retention rate of 111%"
            )
            for field in ("contract_min_years", "contract_max_years"):
                evidence[field] = _variants(
                    "cloudflare-2024", (33, 74, 81, 105), "typically range from one to three years"
                )
        elif name == "cloud-regional-revenue":
            for field, variants in evidence.items():
                evidence[field] = variants + _variants(
                    "cloudflare-2024", (112,), variants[0][0].text
                )
        elif name == "sqlite-thread-contract":
            evidence["check_same_thread"] += _variants(
                "python-sqlite-3.12", (1,), "check_same_thread=True"
            )
        reviewed.append(
            DeepTask(
                replace(
                    item.task, evidence={field: variants[0] for field, variants in evidence.items()}
                ),
                evidence,
            )
        )
    return tuple(reviewed)


def grade(task: DeepTask, raw: str) -> Dict[str, Any]:
    corpus = SourceCorpus(task.task.documents)
    source_error = corpus.validate_answer(raw)
    try:
        parsed = _parse_shape(raw)
    except (ValueError, TypeError, RecursionError) as exc:
        return {
            "passed": False,
            "field_contract_passed": False,
            "citation_structure_passed": False,
            "evidence_coverage_passed": False,
            "failures": [str(exc)],
            "grader_version": GRADER_VERSION,
        }
    failures = list(_validate_fields(parsed["answer"], task.task.expected).failures)
    answer_passed = not failures
    if source_error:
        failures.append(source_error)
    if parsed["unknowns"]:
        failures.append("Unexpected unknown for a gold-supported requested field")
    claims = {}
    for claim in parsed["claims"]:
        pair = _claim_field(claim["claim"])
        if pair is None or pair[0] in claims:
            failures.append("Malformed or duplicate claim field")
            continue
        claims[pair[0]] = claim
    if set(claims) != set(task.task.expected):
        failures.append("Claim fields differ from the requested supported fields")
    covered_fields = 0
    for field, variants in task.alternatives.items():
        if field not in claims:
            continue
        claim = claims[field]
        if not _validate_fields(claim["claim"], {field: task.task.expected[field]}).passed:
            failures.append(f"Incorrect claim value for {field}")
        covered = any(
            all(
                any(
                    citation.get("document_id") == item.document_id
                    and citation.get("page") == item.page
                    and isinstance(citation.get("quote"), str)
                    and item.text in citation["quote"]
                    for citation in claim["citations"]
                )
                for item in variant
            )
            for variant in variants
        )
        if not covered:
            failures.append(f"Gold source evidence not covered for {field}")
        else:
            covered_fields += 1
    return {
        "passed": not failures,
        "field_contract_passed": answer_passed,
        "citation_structure_passed": source_error is None,
        "evidence_coverage_passed": covered_fields == len(task.alternatives),
        "supported_fields": covered_fields,
        "required_fields": len(task.alternatives),
        "failures": failures,
        "grader_version": GRADER_VERSION,
    }


def oracle(task: DeepTask, alternative_index: int = 0) -> str:
    """Produce a finite gold control without using a model."""
    corpus = SourceCorpus(task.task.documents)
    documents = {document.document_id: document for document in task.task.documents}
    claims = []
    for field, variants in task.alternatives.items():
        variant = variants[min(alternative_index, len(variants) - 1)]
        citations = []
        for item in variant:
            page = documents[item.document_id].pages[item.page - 1]
            first = page.index(item.text)
            citations.append(
                corpus.read(item.document_id, item.page, first, first + len(item.text))
            )
        claims.append({"claim": f"{field}: {task.task.expected[field][0]}", "citations": citations})
    return json.dumps(
        {
            "answer": "\n".join(
                f"{key}: {values[0]}" for key, values in task.task.expected.items()
            ),
            "claims": claims,
            "unknowns": [],
        },
        ensure_ascii=False,
    )


def prepare(directory: Path = DIRECTORY) -> Dict[str, Any]:
    documents = load_public_documents(ORIGINAL_DIRECTORY)
    tasks = build_tasks(documents)
    gold = []
    for task in tasks:
        # Every allowed location is verified before any live output is observed.
        for index in range(max(len(variants) for variants in task.alternatives.values())):
            control = oracle(task, index)
            result = grade(task, control)
            if not result["passed"]:
                raise ValueError(f"Invalid gold control for {task.task.name}: {result}")
        gold.append(
            {
                "name": task.task.name,
                "query": task.task.query,
                "sources": SourceCorpus(task.task.documents).manifest(),
                "expected": task.task.expected,
                "evidence_alternatives": {
                    field: [[item.__dict__ for item in variant] for variant in variants]
                    for field, variants in task.alternatives.items()
                },
                "oracle": json.loads(oracle(task)),
            }
        )
    payload = {
        "dataset_version": DATASET_VERSION,
        "grader_version": GRADER_VERSION,
        "note": "Gold frozen before live inference; 12 questions, two source snapshots, overlapping facts are not independent samples.",
        "tasks": gold,
    }
    directory.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path = directory / "gold.json"
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise ValueError("Gold changed; use a new dataset version and output directory")
    path.write_text(encoded, encoding="utf-8")
    manifest = {
        "gold_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "dataset_version": DATASET_VERSION,
        "grader_version": GRADER_VERSION,
        "tasks": [task.task.name for task in tasks],
    }
    (directory / "dataset-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def snapshot_runtime(output: Path) -> Dict[str, Any]:
    """Keep code identities for this stage; no file in the checkout is mutated."""
    files = list((ROOT / "src/rlm").glob("*.py")) + [
        Path(__file__).resolve(),
        ROOT / "benchmarks/grounded_documents.py",
        ROOT / "benchmarks/compare_same_model.py",
        ROOT / "examples/serve_codex_comparison.py",
    ]
    manifest: Dict[str, Any] = {"captured_at": datetime.now(timezone.utc).isoformat(), "files": {}}
    for path in files:
        relative = path.relative_to(ROOT)
        raw = path.read_bytes()
        target = output / "runtime-snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        manifest["files"][str(relative)] = hashlib.sha256(raw).hexdigest()
    (output / "runtime-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def snapshot_head(directory: Path = DIRECTORY) -> Any:
    """Import byte-identical HEAD files under a filesystem-backed package alias.

    The alias is needed so current and old workers can coexist in one process.
    The package files themselves are unmodified; spawned workers inherit its path.
    """
    target = directory / "head-runtime"
    package = target / "rlm_head_3c255c0"
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", HEAD_REVISION, "src/rlm"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    manifest: Dict[str, Any] = {"commit": HEAD_REVISION, "package_alias": package.name, "files": {}}
    for name in names + ["examples/serve_codex_comparison.py", "examples/document_presets.json"]:
        if not (name.endswith(".py") or name.endswith(".json")):
            continue
        raw = subprocess.check_output(["git", "show", f"{HEAD_REVISION}:{name}"], cwd=ROOT)
        path = package / Path(name).name if name.startswith("src/rlm/") else target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != raw:
            raise ValueError(f"Head snapshot was modified: {path}")
        path.write_bytes(raw)
        manifest["files"][name] = hashlib.sha256(raw).hexdigest()
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    sys.path.insert(0, str(target))
    return importlib.import_module(package.name)


def _response(text: str) -> Dict[str, Any]:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 12, "total_tokens": 112},
        "billing_mode": "subscription",
    }


def fixture_task() -> DeepTask:
    document = SourceDocument(
        "fixture", ("value: 7\nunrelated: 99", "value: 12"), "Fixed evidence control"
    )
    evidence = _variants("fixture", (1,), "value: 7")
    return DeepTask(
        GroundedTask(
            "fixture",
            "Return value on page 1",
            (document,),
            {"value": ("7",)},
            {"value": evidence[0]},
        ),
        {"value": evidence},
    )


def adversarial_fixtures() -> List[Dict[str, Any]]:
    task = fixture_task()
    valid = json.loads(oracle(task))
    result: List[Dict[str, Any]] = [
        {
            "name": "valid_directive",
            "replies": [f"FINAL({oracle(task)!r})"],
            "expected_current": "accepted",
        }
    ]
    for name, changes in (
        ("forged_quote", {"quote": "value: 9"}),
        ("forged_hash", {"source_sha256": "0" * 64}),
        ("wrong_page", {"page": 2}),
        ("shifted_offset", {"start": 1}),
        ("boolean_page", {"page": True}),
        ("unknown_document", {"document_id": "missing"}),
        (
            "stale_same_id",
            {"source_sha256": SourceDocument("fixture", ("value: 8",)).content_sha256},
        ),
    ):
        forged = copy.deepcopy(valid)
        forged["claims"][0]["citations"][0].update(changes)
        result.append(
            {
                "name": name,
                "replies": [f"FINAL({json.dumps(forged)!r})", f"FINAL({oracle(task)!r})"],
                "expected_current": "reject_then_repair",
            }
        )
    result.append(
        {"name": "native_json", "replies": [oracle(task)], "expected_current": "accepted"}
    )
    wrong = copy.deepcopy(valid)
    wrong["answer"] = "value: 9"
    wrong["claims"][0]["claim"] = "value: 9"
    result.append(
        {
            "name": "wrong_claim_real_quote",
            "replies": [f"FINAL({json.dumps(wrong)!r})"],
            "expected_current": "accepted_but_semantically_wrong",
            "negative_control": True,
        }
    )
    return result


async def _fixed_attempt(variant: str, legacy: Any, case: Mapping[str, Any]) -> Dict[str, Any]:
    task = fixture_task()
    calls: List[Dict[str, Any]] = []
    replies = case["replies"]

    async def completion(**kwargs: Any) -> Dict[str, Any]:
        if len(calls) >= len(replies):
            raise AssertionError("Fixed responses exhausted; no provider fallback")
        calls.append(copy.deepcopy(kwargs))
        return _response(replies[len(calls) - 1])

    kwargs: Dict[str, Any] = {
        "model": "fixed-fixture",
        "completion_handler": completion,
        "max_depth": 0,
        "max_iterations": len(replies),
        "max_total_calls": len(replies),
        "max_elapsed_seconds": 15,
        "capture_trajectory_content": True,
    }
    if variant == "head-with-source-validator":
        kwargs["final_answer_validator"] = SourceCorpus(task.task.documents).validate_answer
    engine = RLM(**kwargs) if variant == "current-document-api" else legacy.RLM(**kwargs)
    record = {
        "variant": variant,
        "fixture": case["name"],
        "live_model_calls": 0,
        "fixed_replies": replies,
        "expected_current": case["expected_current"],
    }
    try:
        if variant == "current-document-api":
            completed = await engine.acomplete_documents(task.task.query, task.task.documents)
            raw = completed.completion.answer
        else:
            completed = await engine.acomplete_result(
                task.task.query, SourceCorpus(task.task.documents).context
            )
            raw = completed.answer
        record.update({"status": "accepted", "raw_answer": raw, "grade": grade(task, raw)})
    except Exception as exc:
        record.update(
            {
                "status": "rejected_or_failed",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
    record["calls"] = calls
    record["runtime_stats"] = engine.stats
    record["trajectory"] = [event.to_dict() for event in engine.trajectory]
    return record


def _cli_events(text: str) -> List[Dict[str, Any]]:
    return [
        {"type": "item.completed", "item": {"type": "agent_message", "text": text}},
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 100, "cached_input_tokens": 0, "output_tokens": 12},
        },
    ]


async def _empty_cli_control(legacy: Any, current: bool) -> Dict[str, Any]:
    parser = (
        parse_events
        if current
        else importlib.import_module(legacy.__name__ + ".codex").parse_events
    )
    replies = ("", 'FINAL("recovered")')
    calls: List[Dict[str, Any]] = []

    async def completion(**kwargs: Any) -> Dict[str, Any]:
        position = len(calls)
        calls.append(
            {
                "messages": copy.deepcopy(kwargs["messages"]),
                "events": _cli_events(replies[position]),
            }
        )
        return parser(calls[-1]["events"])

    kind = "current" if current else "head"
    engine = (RLM if current else legacy.RLM)(
        model="fixed-fixture",
        completion_handler=completion,
        max_depth=0,
        max_iterations=2,
        max_total_calls=2,
        max_elapsed_seconds=10,
    )
    record = {"variant": kind, "fixture": "empty_cli_then_valid", "live_model_calls": 0}
    try:
        completed = await engine.acomplete_result("Return recovered", "source")
        record.update({"status": "accepted", "answer": completed.answer})
    except Exception as exc:
        record.update(
            {
                "status": "rejected_or_failed",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
    record["calls"] = calls
    record["runtime_stats"] = engine.stats
    return record


def _import_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot import comparison module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def loader_controls(directory: Path) -> List[Dict[str, Any]]:
    path = ORIGINAL_DIRECTORY / "sources/cloudflare-10k.pdf"
    raw = path.read_bytes()
    old = _import_file(
        "head_document_companion", directory / "head-runtime/examples/serve_codex_comparison.py"
    )
    current = _import_file(
        "current_document_companion", ROOT / "examples/serve_codex_comparison.py"
    )
    records = []
    for name, callback in (
        ("head_companion_default", lambda: old._extract_text(path.name, raw)),
        ("current_companion_standard", lambda: current._extract_text(path.name, raw)),
        (
            "current_companion_long",
            lambda: current._extract_text(path.name, raw, current.DOCUMENT_PROFILES["long"]),
        ),
        ("current_library_default", lambda: load_document(path, document_id="cloudflare-2024")),
    ):
        record = {
            "variant": name,
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "note": "Profile capability and extraction checks, not LLM quality scores.",
            "live_model_calls": 0,
        }
        started = time.monotonic()
        try:
            with warnings.catch_warnings(record=True) as caught:
                result: Any = callback()
            pages = result.pages if isinstance(result, SourceDocument) else result.get("pages")
            record.update(
                {
                    "status": "accepted",
                    "characters": sum(map(len, pages)) if pages else len(result["text"]),
                    "page_count": len(pages) if pages else None,
                    "warnings": [str(item.message) for item in caught],
                    "content_sha256": (
                        SourceDocument("cloudflare-2024", pages).content_sha256 if pages else None
                    ),
                }
            )
        except Exception as exc:
            record.update(
                {"status": "rejected", "error": {"type": type(exc).__name__, "message": str(exc)}}
            )
        record["elapsed_seconds"] = time.monotonic() - started
        records.append(record)
    return records


def search_controls() -> List[Dict[str, Any]]:
    samples = (
        ("Pétya", "Petya"),
        ("Pe\u0301tya", "PETYA"),
        ("Straße\u0301", "STRASSE"),
        ("alpha\t\r\n beta", "ALPHA BETA"),
    )
    records = []
    for original, term in samples:
        document = SourceDocument("unicode", ("prefix " + original + " suffix",))
        corpus = SourceCorpus((document,))
        matches = corpus.search(term, window=len(original))
        records.append(
            {
                "original": original,
                "term": term,
                "head_literal_find": document.pages[0].find(term),
                "current_matches": matches,
                "correct_original_span": bool(matches) and matches[0]["quote"] == original,
                "note": "Built-in helper versus literal search recipe. Legacy users could implement custom normalization.",
            }
        )
    return records


async def offline_controls(output: Path, directory: Path = DIRECTORY) -> Dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    snapshot_runtime(output)
    legacy = snapshot_head(directory)
    records = []
    for case in adversarial_fixtures():
        for variant in ("head-default", "head-with-source-validator", "current-document-api"):
            record = await _fixed_attempt(variant, legacy, case)
            records.append(record)
            (output / f"{case['name']}-{variant}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    empty = [await _empty_cli_control(legacy, current) for current in (False, True)]
    summary = {
        "kind": "OFFLINE_FIXED_RESPONSE_CONTROLS",
        "live_model_calls": 0,
        "baseline_commit": HEAD_REVISION,
        "records": records,
        "empty_cli": empty,
        "loader": loader_controls(directory),
        "search": search_controls(),
        "limitations": [
            "Fixed response replay tests runtime behavior, not model quality.",
            "The old release already supports a custom final validator; that control is included.",
            "A valid real quotation can support a wrong claim; the negative control must remain visible.",
            "Missing old APIs and different upload profiles are capability differences, not accuracy gains.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def summarize_live(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, Any]] = {}
    for record in records:
        group = groups.setdefault(
            record["mode"],
            {
                "attempts": 0,
                "completed": 0,
                "strict_passes": 0,
                "field_contract_passes": 0,
                "citation_structure_passes": 0,
                "evidence_coverage_passes": 0,
                "model_calls": 0,
                "recursive_model_calls": 0,
                "total_tokens": 0,
                "unknown_usage_calls": 0,
                "elapsed_values": [],
            },
        )
        group["attempts"] += 1
        group["completed"] += record["status"] == "completed"
        for target, key in (
            ("strict_passes", "passed"),
            ("field_contract_passes", "field_contract_passed"),
            ("citation_structure_passes", "citation_structure_passed"),
            ("evidence_coverage_passes", "evidence_coverage_passed"),
        ):
            group[target] += bool(record["grade"][key])
        usage = record["transport_usage"]
        group["model_calls"] += usage["llm_calls"]
        group["total_tokens"] += usage["total_tokens"]
        group["unknown_usage_calls"] += usage.get("unknown_usage_calls", 0)
        group["recursive_model_calls"] += (record.get("runtime_stats") or {}).get(
            "recursive_calls", 0
        )
        group["elapsed_values"].append(record["elapsed_seconds"])
    for group in groups.values():
        values = group.pop("elapsed_values")
        group["elapsed_seconds_sum"] = sum(values)
        group["elapsed_seconds_median"] = statistics.median(values)
        group["elapsed_seconds_min"] = min(values)
        group["elapsed_seconds_max"] = max(values)
    return {
        "dataset_version": DATASET_VERSION,
        "grader_version": GRADER_VERSION,
        "groups": groups,
        "attempts": len(records),
        "limitations": [
            "Current depth-zero versus depth-two workflow comparison, not old-versus-new LLM quality.",
            "Two repeated attempts per question do not establish tail reliability or p99 latency.",
            "Question facts overlap; these are not independent population samples.",
            "Strict field aliases can reject a correct expanded phrase; inspect failures before making factual-error claims.",
            "Citation validity and gold-anchor coverage do not prove arbitrary claim entailment.",
            "CLI tokens measure observed model usage, not subscription credits or dollar savings.",
        ],
    }


async def live_suite(args: argparse.Namespace) -> Dict[str, Any]:
    manifest = prepare(args.directory)
    tasks = build_tasks(load_public_documents(ORIGINAL_DIRECTORY))
    if args.task:
        unknown = set(args.task) - {item.task.name for item in tasks}
        if unknown:
            raise ValueError(f"Unknown tasks: {sorted(unknown)}")
        tasks = tuple(task for task in tasks if task.task.name in args.task)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    snapshot_runtime(args.output)
    login = CodexCompletion(timeout=args.call_timeout, reasoning_effort=args.reasoning_effort)
    await login.check_login()
    planned = []
    first_repeat = getattr(args, "start_repeat", 1)
    for repeat in range(first_repeat, first_repeat + args.repeats):
        order = ("python", "rlm") if repeat % 2 else ("rlm", "python")
        for item in tasks:
            for mode in order:
                filename = f"{item.task.name}-{mode}-{repeat}.json"
                if (args.output / filename).exists():
                    raise ValueError(f"Refusing to overwrite {filename}")
                planned.append((item, mode, repeat, filename))
    (args.output / "plan.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "gold_sha256": manifest["gold_sha256"],
                "dataset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "planned_attempts": [
                    {"task": task.task.name, "mode": mode, "repeat": repeat, "file": name}
                    for task, mode, repeat, name in planned
                ],
                "max_parallel": args.parallel,
                "max_calls_per_attempt": args.max_calls,
                "max_tokens_per_attempt": args.max_tokens,
                "timeout_per_attempt": args.timeout,
                "circuit_breaker": "Do not start more attempts after two consecutive completed attempts terminate in HTTP 429.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    queue: asyncio.Queue[Tuple[DeepTask, str, int, str]] = asyncio.Queue()
    for entry in planned:
        queue.put_nowait(entry)
    state = {"consecutive_429": 0, "stopped": False}
    records = []
    skipped = []

    async def worker() -> None:
        while not queue.empty():
            task, mode, repeat, filename = queue.get_nowait()
            if state["stopped"]:
                skipped.append(
                    {
                        "task": task.task.name,
                        "mode": mode,
                        "repeat": repeat,
                        "reason": "HTTP 429 circuit breaker; no model call made",
                    }
                )
                continue
            record = await run_one(task.task, mode, args)
            record["initial_grounded_grader"] = record["grade"]
            raw = record["raw_answer"]
            record["grade"] = (
                grade(task, raw)
                if raw is not None
                else {
                    "passed": False,
                    "field_contract_passed": False,
                    "citation_structure_passed": False,
                    "evidence_coverage_passed": False,
                    "failures": ["No accepted answer"],
                    "grader_version": GRADER_VERSION,
                }
            )
            record["dataset_version"] = DATASET_VERSION
            record["grader_version"] = GRADER_VERSION
            record["repeat"] = repeat
            record["gold_sha256"] = manifest["gold_sha256"]
            record["transport_usage"] = _transport_usage(record["transport_calls"])
            if record.get("error") and "429" in record["error"]["message"]:
                state["consecutive_429"] += 1
            else:
                state["consecutive_429"] = 0
            if state["consecutive_429"] >= 2:
                state["stopped"] = True
            (args.output / filename).write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            records.append(record)
            summary = summarize_live(records)
            summary.update({"skipped": skipped, "circuit_breaker_triggered": state["stopped"]})
            (args.output / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "task": task.task.name,
                        "mode": mode,
                        "repeat": repeat,
                        "status": record["status"],
                        "grade": record["grade"],
                        "tokens": record["transport_usage"]["total_tokens"],
                        "seconds": record["elapsed_seconds"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    await asyncio.gather(*(worker() for _ in range(args.parallel)))
    summary = summarize_live(records)
    summary.update({"skipped": skipped, "circuit_breaker_triggered": state["stopped"]})
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--directory", type=Path, default=DIRECTORY)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task", action="append")
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument(
        "--reasoning-effort", choices=("low", "medium", "high", "xhigh", "max"), default="low"
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--start-repeat", type=int, default=1)
    parser.add_argument("--parallel", type=int, choices=(1, 2), default=2)
    parser.add_argument("--max-calls", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=150000)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--call-timeout", type=float, default=150)
    parser.add_argument("--max-output-chars", type=int, default=16000)
    args = parser.parse_args()
    if sum((args.prepare, args.offline, args.live)) != 1:
        parser.error("Choose exactly one of --prepare, --offline, --live")
    for name in (
        "repeats",
        "start_repeat",
        "max_calls",
        "max_iterations",
        "max_tokens",
        "timeout",
        "call_timeout",
        "max_output_chars",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.prepare:
        print(json.dumps(prepare(args.directory), indent=2))
        return
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output = args.directory / ("live" if args.live else "offline") / stamp
    if args.offline:
        summary = asyncio.run(offline_controls(args.output, args.directory))
        print(
            json.dumps(
                {
                    "kind": summary["kind"],
                    "live_model_calls": 0,
                    "control_count": len(summary["records"]),
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(asyncio.run(live_suite(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
