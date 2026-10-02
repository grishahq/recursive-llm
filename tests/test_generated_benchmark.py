"""Tests for reproducible long-context benchmark generation."""

import re

import pytest

from benchmarks.generated_long_context import generate_long_context


def test_generated_context_is_reproducible_and_meets_requested_size() -> None:
    """The same seed and size must reproduce byte-identical benchmark input."""
    first = generate_long_context(target_chars=10_000, seed=7)
    second = generate_long_context(target_chars=10_000, seed=7)
    different = generate_long_context(target_chars=10_000, seed=8)

    assert first.context == second.context
    assert first.sha256 == second.sha256
    assert first.truth == second.truth
    assert first.actual_chars >= 10_000
    assert different.sha256 != first.sha256


def test_ground_truth_matches_an_independent_context_scan() -> None:
    """The answer key must be derivable from only the emitted records."""
    generated = generate_long_context(target_chars=20_000, seed=11)
    matches = []
    for line in generated.context.splitlines():
        if "region=EMEA" not in line or "status=SETTLED" not in line:
            continue
        transaction_id = re.match(r"TX-\d+", line).group(0)  # type: ignore[union-attr]
        amount = int(re.search(r"amount_cents=(\d+)", line).group(1))  # type: ignore[union-attr]
        matches.append((transaction_id, amount))

    expected_max_id, expected_max_amount = max(matches, key=lambda item: (item[1], item[0]))
    assert generated.truth.count == len(matches)
    assert generated.truth.total_amount_cents == sum(amount for _id, amount in matches)
    assert generated.truth.max_transaction_id == expected_max_id
    assert generated.truth.max_amount_cents == expected_max_amount


def test_generated_answer_validator_requires_exact_labeled_fields() -> None:
    """Ground-truth values elsewhere in prose cannot replace the output contract."""
    generated = generate_long_context(target_chars=5_000, seed=9)
    truth = generated.truth
    valid = (
        f"count={truth.count} total_amount_cents={truth.total_amount_cents} "
        f"max_transaction_id={truth.max_transaction_id} "
        f"max_amount_cents={truth.max_amount_cents}"
    )

    assert generated.validate(valid) == ()
    assert generated.validate(valid.replace("count=", "records="))
    assert generated.validate(valid.replace(str(truth.total_amount_cents), "0"))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda answer: answer + " count=0",
        lambda answer: answer + "\nCOUNT=0",
        lambda answer: answer + " " + answer,
        lambda answer: answer.replace("count=", "count=0 count="),
        lambda answer: answer.replace("count=", "count=-"),
        lambda answer: re.sub(r"(count=\d+)", r"\1.5", answer),
        lambda answer: re.sub(r"(count=\d+)", r"\1e1", answer),
        lambda answer: answer + " extra=0",
        lambda answer: "Not " + answer,
        lambda answer: answer + "\nThe count above is incorrect.",
    ],
)
def test_generated_grader_rejects_ambiguous_or_partial_field_values(mutation) -> None:
    """No correct prefix, duplicate or contradictory prose can earn an exact pass."""
    generated = generate_long_context(target_chars=5_000, seed=9)
    valid = " ".join(f"{key}={value}" for key, value in vars(generated.truth).items())

    assert generated.validate(mutation(valid))


def test_generated_grader_allows_whitespace_and_field_order_variants() -> None:
    generated = generate_long_context(target_chars=5_000, seed=9)
    answer = "\n".join(
        f" {key.upper()} = {value} " for key, value in reversed(vars(generated.truth).items())
    )

    assert generated.validate(answer) == ()


@pytest.mark.parametrize("target_chars", [0, -1])
def test_invalid_generated_context_size_is_rejected(target_chars: int) -> None:
    """The generator should reject empty or negative benchmark sizes."""
    with pytest.raises(ValueError, match="target_chars"):
        generate_long_context(target_chars=target_chars)
