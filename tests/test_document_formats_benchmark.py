"""Tests for the cross-format public-document benchmark."""

import csv
import io

import pytest

from benchmarks.document_formats import (
    GRADER_VERSION,
    SOURCE_SPECS,
    _csv_to_text,
    _html_to_text,
    build_tasks,
    validate_frankenstein,
    validate_playbook_structure,
    validate_sqlite_defaults,
)


FRANKENSTEIN_FACTS = "Letters: 4\nChapters: 24\nAddressee: Mrs. Saville\nObserved family: De Lacey"
PLAYBOOK_COUNTS = "Govern: 19\nManage: 13\nMap: 18\nMeasure: 22"
SQLITE_DEFAULTS = (
    "timeout: 5.0\nisolation_level: DEFERRED\n"
    "cached_statements: 128\nautocommit: sqlite3.LEGACY_TRANSACTION_CONTROL"
)


def test_playbook_csv_is_transposed_into_labeled_sections() -> None:
    output = io.StringIO()
    writer = csv.writer(output)
    headers = [""] + [f"SECTION {index}" for index in range(72)]
    writer.writerow(headers)
    for label in ("section_about", "section_actions", "section_doc", "section_ref"):
        writer.writerow([label] + [f"{label}-{index}" for index in range(72)])

    text = _csv_to_text(output.getvalue().encode())

    assert text.count("===== SECTION:") == 72
    assert "===== SECTION: SECTION 0 =====" in text
    assert "section_actions: section_actions-71" in text


def test_html_extraction_drops_scripts_and_keeps_documentation() -> None:
    raw = b"<html><nav>menu</nav><h1>Title</h1><p>Useful <code>value</code></p><script>secret</script></html>"

    text = _html_to_text(raw)

    assert "Title" in text
    assert "Useful" in text
    assert "value" in text
    assert "menu" not in text
    assert "secret" not in text


def test_exact_document_graders() -> None:
    assert validate_frankenstein(
        "Letters: 4\nChapters: 24\nAddressee: Mrs. Saville\nObserved family: De Lacey"
    ).passed
    assert validate_playbook_structure("Govern: 19\nManage: 13\nMap: 18\nMeasure: 22").passed
    assert validate_sqlite_defaults(
        "timeout: 5.0\nisolation_level: DEFERRED\n"
        "cached_statements: 128\nautocommit: sqlite3.LEGACY_TRANSACTION_CONTROL"
    ).passed


def test_document_graders_reject_near_misses() -> None:
    assert not validate_frankenstein(
        "Letters: 4\nChapters: 23\nAddressee: Mrs. Saville\nObserved family: De Lacey"
    ).passed
    assert not validate_playbook_structure("Govern: 19\nManage: 13\nMap: 18\nMeasure: 21").passed
    assert not validate_sqlite_defaults(
        "timeout: 5\nisolation_level: DEFERRED\n"
        "cached_statements: 128\nautocommit: sqlite3.LEGACY_TRANSACTION_CONTROL"
    ).passed


@pytest.mark.parametrize(
    "validator,answer",
    [
        (validate_frankenstein, FRANKENSTEIN_FACTS + "\nLetters: 0"),
        (validate_frankenstein, FRANKENSTEIN_FACTS.replace("Letters: 4", "Letters: -4")),
        (validate_frankenstein, FRANKENSTEIN_FACTS.replace("Letters: 4", "Letters: .4")),
        (validate_frankenstein, FRANKENSTEIN_FACTS.replace("Mrs. Saville", "not Mrs. Saville")),
        (validate_frankenstein, FRANKENSTEIN_FACTS.replace("De Lacey", "not De Lacey")),
        (validate_playbook_structure, PLAYBOOK_COUNTS + "\nGovern: 0"),
        (validate_playbook_structure, PLAYBOOK_COUNTS + "\nGovern: 19"),
        (validate_playbook_structure, PLAYBOOK_COUNTS.replace("19", "-19")),
        (validate_playbook_structure, PLAYBOOK_COUNTS.replace("19", "19.5")),
        (validate_sqlite_defaults, SQLITE_DEFAULTS + "\ntimeout: 0"),
        (validate_sqlite_defaults, SQLITE_DEFAULTS.replace("5.0", "-5.0")),
        (validate_sqlite_defaults, SQLITE_DEFAULTS + "\nActually these values are wrong."),
    ],
)
def test_document_graders_reject_duplicates_negations_and_signed_counts(validator, answer) -> None:
    assert not validator(answer).passed


def test_document_tasks_record_grader_version() -> None:
    tasks = build_tasks({name: "context" for name in SOURCE_SPECS}, label="test")

    assert all(task.metadata["grader_version"] == GRADER_VERSION for task in tasks)
