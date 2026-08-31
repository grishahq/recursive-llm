"""Tests for the cross-format public-document benchmark."""

import csv
import io

from benchmarks.document_formats import (
    _csv_to_text,
    _html_to_text,
    validate_frankenstein,
    validate_playbook_structure,
    validate_sqlite_defaults,
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
