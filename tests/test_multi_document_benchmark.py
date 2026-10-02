"""Tests for the pinned multi-document benchmark."""

import zipfile
from pathlib import Path

import pytest

from benchmarks.multi_document import (
    DOCUMENT_SPECS,
    GRADER_VERSION as MULTI_DOCUMENT_GRADER_VERSION,
    build_tasks,
    load_documents,
    validate_commission_facts,
    validate_python_docs_facts,
)
from benchmarks.war_and_peace import GRADER_VERSION, PETYA_FINAL_NIGHT_QUERY


COMMISSION_FACTS = """Pages reviewed: more than 2.5 million
Individuals interviewed: more than 1,200
Countries: ten
Replacement: National Intelligence Director
Responsibilities: two main areas
Location: Executive Office of the President"""
PYTHON_FACTS = """Zstandard module: compression.zstd
Default pickle protocol: 5
map parameter: strict
Thread.join exception: PythonFinalizationError"""


def test_commission_validator_requires_every_labeled_fact() -> None:
    answer = """Pages reviewed: more than 2.5 million
Individuals interviewed: more than 1,200
Countries: ten
Replacement: National Intelligence Director
Responsibilities: two main areas
Location: Executive Office of the President"""

    assert validate_commission_facts(answer).passed
    assert validate_commission_facts(
        answer.replace("Countries: ten", "Countries: ten countries")
    ).passed
    assert not validate_commission_facts(answer.replace("Countries: ten", "Countries: nine")).passed


def test_python_docs_validator_requires_exact_api_names() -> None:
    answer = """Zstandard module: compression.zstd
Default pickle protocol: 5
map parameter: strict
Thread.join exception: PythonFinalizationError"""

    assert validate_python_docs_facts(answer).passed
    assert not validate_python_docs_facts(answer.replace("strict", "exact")).passed


@pytest.mark.parametrize(
    "answer",
    [
        COMMISSION_FACTS + "\nCountries: 0",
        COMMISSION_FACTS + "\nCountries: ten",
        COMMISSION_FACTS.replace("2.5 million", "12.5 million"),
        COMMISSION_FACTS.replace("more than 2.5 million", "less than 2.5 million"),
        COMMISSION_FACTS.replace("more than 1,200", "not 1,200"),
        COMMISSION_FACTS.replace(
            "National Intelligence Director", "not National Intelligence Director"
        ),
        COMMISSION_FACTS.replace(
            "Executive Office of the President", "not Executive Office of the President"
        ),
    ],
)
def test_commission_grader_rejects_contradictions_and_numeric_substrings(answer: str) -> None:
    assert not validate_commission_facts(answer).passed


@pytest.mark.parametrize(
    "answer",
    [
        PYTHON_FACTS + "\nDefault pickle protocol: 0",
        PYTHON_FACTS + "\nmap parameter: strict",
        PYTHON_FACTS.replace("Default pickle protocol: 5", "Default pickle protocol: -5"),
        PYTHON_FACTS.replace("Default pickle protocol: 5", "Default pickle protocol: .5"),
        PYTHON_FACTS + "\nIgnore these incorrect defaults.",
    ],
)
def test_python_docs_grader_rejects_duplicates_and_erased_numeric_signs(answer: str) -> None:
    assert not validate_python_docs_facts(answer).passed


def test_task_builder_covers_three_distinct_corpora() -> None:
    documents = {
        "war_and_peace": "war",
        "commission_report": "report",
        "python_docs": "python",
    }

    tasks = build_tasks(documents, label="experiment")

    assert [task.name for task in tasks] == [
        "war_and_peace_petya",
        "commission_distributed_facts",
        "python_314_distributed_facts",
    ]
    assert {task.metadata["corpus"] for task in tasks if task.metadata} == set(documents)
    assert all(task.metadata and task.metadata["variant"] == "experiment" for task in tasks)
    assert tasks[0].query == PETYA_FINAL_NIGHT_QUERY
    assert tasks[0].metadata["grader_version"] == GRADER_VERSION
    assert all(
        task.metadata["grader_version"] == MULTI_DOCUMENT_GRADER_VERSION for task in tasks[1:]
    )


def test_loader_rejects_unpinned_input(tmp_path: Path) -> None:
    for spec in DOCUMENT_SPECS.values():
        (tmp_path / spec["filename"]).write_bytes(b"wrong")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_documents(tmp_path)


def test_loader_builds_python_context_in_sorted_archive_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    war = b"war"
    report = b"report"
    archive_path = tmp_path / "docs.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("python-3.14-docs-text/z.txt", "last")
        archive.writestr("python-3.14-docs-text/a.txt", "first")
        archive.writestr("python-3.14-docs-text/ignored.html", "ignored")
    specs = {
        "war_and_peace": {
            "filename": "war.txt",
            "sha256": __import__("hashlib").sha256(war).hexdigest(),
            "source_url": "war",
        },
        "commission_report": {
            "filename": "report.txt",
            "sha256": __import__("hashlib").sha256(report).hexdigest(),
            "source_url": "report",
        },
        "python_docs": {
            "filename": "docs.zip",
            "sha256": __import__("hashlib").sha256(archive_path.read_bytes()).hexdigest(),
            "source_url": "python",
        },
    }
    monkeypatch.setattr("benchmarks.multi_document.DOCUMENT_SPECS", specs)
    (tmp_path / "war.txt").write_bytes(war)
    (tmp_path / "report.txt").write_bytes(report)

    documents = load_documents(tmp_path)

    assert documents["python_docs"].index("a.txt") < documents["python_docs"].index("z.txt")
    assert "ignored.html" not in documents["python_docs"]
