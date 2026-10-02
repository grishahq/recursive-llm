"""Tests for the exact-graded real-document benchmark."""

import pytest

from benchmarks.war_and_peace import (
    DOCUMENT_SHA256,
    GRADER_VERSION,
    PETYA_FINAL_NIGHT_QUERY,
    build_tasks,
    validate_body_chapter_count,
    validate_distant_fact_retrieval,
    validate_petya_final_night,
)


DISTANT_FACTS = "Illness: la grippe\nAnnual cost: 40,000 rubles\nHorse: Karabákh"
PETYA_FACTS = (
    "Cossack: Likhachëv\n"
    "Service: sharpen Petya's saber\n"
    "Payment: 1 ruble\n"
    "Companion: Dólokhov\n"
    "Verdict speaker: Dólokhov\n"
    "Verdict: Done for"
)


def test_body_chapter_count_validator_requires_labeled_exact_value() -> None:
    assert validate_body_chapter_count("Total chapters: 365").passed
    assert not validate_body_chapter_count("There are 365 chapters.").passed
    assert not validate_body_chapter_count("Total chapters: 364").passed


@pytest.mark.parametrize(
    "answer",
    [
        "Total chapters: 365.5",
        "Total chapters: 365e1",
        "Total chapters: 3650",
        "Total chapters: 365\nTotal chapters: 0",
        "Total chapters: 365\nTotal chapters: 365",
        "Not Total chapters: 365",
        "Total chapters: 365 but actually 0",
    ],
)
def test_chapter_count_rejects_numeric_prefixes_and_contradictions(answer: str) -> None:
    assert not validate_body_chapter_count(answer).passed


def test_distant_fact_validator_accepts_accents_and_number_formats() -> None:
    numeric = "Illness: la grippe\nAnnual cost: 40,000 rubles\nHorse: Karabákh"
    written = "Illness: la grippe\nAnnual cost: forty thousand rubles\nHorse: Karabakh"

    assert validate_distant_fact_retrieval(numeric).passed
    assert validate_distant_fact_retrieval(written).passed
    assert not validate_distant_fact_retrieval("Illness: grippe").passed


def test_petya_validator_accepts_expected_sequence() -> None:
    assert validate_petya_final_night(PETYA_FACTS).passed
    assert not validate_petya_final_night("Likhachev sharpened a saber.").passed


@pytest.mark.parametrize(
    "answer",
    [
        "Illness: Karabakh\nAnnual cost: la grippe\nHorse: 40,000 rubles",
        DISTANT_FACTS.replace("Illness: la grippe", "Illness: not la grippe"),
        DISTANT_FACTS.replace("40,000", "-40,000"),
        DISTANT_FACTS.replace("40,000", "40,000.5"),
        DISTANT_FACTS + "\nHorse: not Karabakh",
        DISTANT_FACTS + "\nHorse: Karabakh",
        DISTANT_FACTS + "\nActually those facts are wrong.",
        DISTANT_FACTS.replace("Karabákh", "Karabakh or another horse"),
        "la grippe, Karabakh, forty thousand rubles",
    ],
)
def test_distant_facts_reject_swaps_negations_duplicates_and_extra_prose(answer: str) -> None:
    assert not validate_distant_fact_retrieval(answer).passed


@pytest.mark.parametrize(
    "answer",
    [
        PETYA_FACTS.replace("Cossack: Likhachëv", "Cossack: Dólokhov").replace(
            "Companion: Dólokhov", "Companion: Likhachëv"
        ),
        PETYA_FACTS.replace("sharpen Petya's saber", "did not sharpen Petya's saber"),
        PETYA_FACTS.replace("Payment: 1 ruble", "Payment: not one ruble"),
        PETYA_FACTS.replace("Payment: 1 ruble", "Payment: -1 ruble"),
        PETYA_FACTS.replace("Verdict speaker: Dólokhov", "Verdict speaker: Likhachev"),
        PETYA_FACTS.replace("Verdict speaker: Dólokhov\n", ""),
        PETYA_FACTS + "\nCompanion: Denisov",
        PETYA_FACTS + "\nCompanion: Dólokhov",
        "Likhachev did not sharpen saber. He did not get one ruble. Dolokhov did not say Done for.",
    ],
)
def test_petya_facts_bind_people_actions_and_verdict_to_unique_fields(answer: str) -> None:
    assert not validate_petya_final_night(answer).passed


def test_petya_grader_accepts_simple_typography_and_spelling_variants() -> None:
    answer = PETYA_FACTS.replace("sharpen Petya's saber", "sharpening Pétya’s sabre").replace(
        "Payment: 1 ruble", "Payment: one rouble"
    )
    answer = answer.replace("Verdict: Done for", "Verdict: “Done for!”")

    assert validate_petya_final_night(answer).passed


def test_document_tasks_share_reproduction_metadata() -> None:
    tasks = build_tasks("document", document_sha256=DOCUMENT_SHA256)

    assert [task.name for task in tasks] == [
        "body_chapter_count",
        "distant_fact_retrieval",
        "petya_final_night",
    ]
    assert all(task.metadata and task.metadata["sha256"] == DOCUMENT_SHA256 for task in tasks)
    assert all(task.metadata and task.metadata["characters"] == 8 for task in tasks)
    assert all(
        task.metadata and task.metadata["grader_version"] == GRADER_VERSION for task in tasks
    )
    assert tasks[-1].query == PETYA_FINAL_NIGHT_QUERY
    assert "`Verdict speaker:`" in tasks[-1].query
    assert "Likhachev" not in tasks[-1].query
    assert "Dolokhov" not in tasks[-1].query
    assert "one ruble" not in tasks[-1].query
