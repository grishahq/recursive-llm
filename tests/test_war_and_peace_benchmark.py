"""Tests for the exact-graded real-document benchmark."""

from benchmarks.war_and_peace import (
    DOCUMENT_SHA256,
    build_tasks,
    validate_body_chapter_count,
    validate_distant_fact_retrieval,
    validate_petya_final_night,
)


def test_body_chapter_count_validator_requires_labeled_exact_value() -> None:
    assert validate_body_chapter_count("Total chapters: 365").passed
    assert not validate_body_chapter_count("There are 365 chapters.").passed
    assert not validate_body_chapter_count("Total chapters: 364").passed


def test_distant_fact_validator_accepts_accents_and_number_formats() -> None:
    numeric = "Illness: la grippe\nAnnual cost: 40,000 rubles\nHorse: Karabákh"
    written = "Illness: la grippe\nAnnual cost: forty thousand rubles\nHorse: Karabakh"

    assert validate_distant_fact_retrieval(numeric).passed
    assert validate_distant_fact_retrieval(written).passed
    assert not validate_distant_fact_retrieval("Illness: grippe").passed


def test_petya_validator_accepts_expected_sequence() -> None:
    answer = (
        "Cossack: Likhachëv\n"
        "Service: sharpen Petya's saber\n"
        "Payment: 1 ruble\n"
        "Companion: Dólokhov\n"
        "Verdict: Done for"
    )

    assert validate_petya_final_night(answer).passed
    assert not validate_petya_final_night("Likhachev sharpened a saber.").passed


def test_document_tasks_share_reproduction_metadata() -> None:
    tasks = build_tasks("document", document_sha256=DOCUMENT_SHA256)

    assert [task.name for task in tasks] == [
        "body_chapter_count",
        "distant_fact_retrieval",
        "petya_final_night",
    ]
    assert all(task.metadata and task.metadata["sha256"] == DOCUMENT_SHA256 for task in tasks)
    assert all(task.metadata and task.metadata["characters"] == 8 for task in tasks)
