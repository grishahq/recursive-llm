"""Deterministic long-context benchmark generation with exact ground truth."""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Tuple


GRADER_VERSION = "transaction-fields-v2"


@dataclass(frozen=True)
class RollupTruth:
    """Exact answer key for one generated transaction corpus."""

    count: int
    total_amount_cents: int
    max_transaction_id: str
    max_amount_cents: int


@dataclass(frozen=True)
class GeneratedLongContext:
    """A reproducible corpus, query, answer key, and content identity."""

    query: str
    context: str
    truth: RollupTruth
    seed: int
    target_chars: int
    sha256: str

    @property
    def actual_chars(self) -> int:
        """Return the generated context length."""
        return len(self.context)

    def validate(self, answer: str) -> Tuple[str, ...]:
        """Require exactly one complete value for each requested field, with no prose."""
        expected = {
            "count": str(self.truth.count),
            "total_amount_cents": str(self.truth.total_amount_cents),
            "max_transaction_id": self.truth.max_transaction_id,
            "max_amount_cents": str(self.truth.max_amount_cents),
        }
        failures = []
        fields = {}
        end = 0
        for match in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^\s=]+)", answer):
            if answer[end : match.start()].strip():
                failures.append("answer contains text outside the requested fields")
            end = match.end()
            label, observed = match.group(1).casefold(), match.group(2)
            if label not in expected:
                failures.append(f"unexpected field {label!r}")
            if label in fields:
                failures.append(f"duplicate field {label!r}")
            fields[label] = observed
        if answer[end:].strip():
            failures.append("answer contains text outside the requested fields")
        for label, expected_value in expected.items():
            observed = fields.get(label)
            if observed != expected_value:
                failures.append(f"{label} differs: expected={expected_value}, observed={observed}")
        return tuple(failures)


def generate_long_context(*, target_chars: int = 100_000, seed: int = 2026) -> GeneratedLongContext:
    """Generate transaction records until the requested minimum size is reached."""
    if target_chars <= 0:
        raise ValueError("target_chars must be greater than zero")

    rng = random.Random(seed)
    regions = ("APAC", "EMEA", "LATAM", "NA")
    statuses = ("CANCELLED", "PENDING", "REFUNDED", "SETTLED")
    channels = ("api", "batch", "mobile", "web")
    lines = []
    qualifying = []
    current_chars = 0
    index = 1
    while current_chars < target_chars:
        transaction_id = f"TX-{index:07d}"
        region = rng.choice(regions)
        status = rng.choice(statuses)
        amount_cents = rng.randint(100, 999_999)
        account_id = f"ACCT-{rng.randint(1, 9999):04d}"
        channel = rng.choice(channels)
        checksum = rng.getrandbits(32)
        line = (
            f"{transaction_id} account={account_id} region={region} status={status} "
            f"amount_cents={amount_cents} channel={channel} checksum={checksum:08x}\n"
        )
        lines.append(line)
        current_chars += len(line)
        if region == "EMEA" and status == "SETTLED":
            qualifying.append((transaction_id, amount_cents))
        index += 1

    if not qualifying:
        raise AssertionError("generated corpus unexpectedly has no qualifying records")
    max_transaction_id, max_amount_cents = max(qualifying, key=lambda item: (item[1], item[0]))
    truth = RollupTruth(
        count=len(qualifying),
        total_amount_cents=sum(amount for _transaction_id, amount in qualifying),
        max_transaction_id=max_transaction_id,
        max_amount_cents=max_amount_cents,
    )
    context = "".join(lines)
    query = (
        "Inspect all transaction records. For records where region=EMEA and status=SETTLED, "
        "return the exact count, sum of amount_cents, and the transaction with the largest "
        "amount_cents. Break ties by the lexicographically largest transaction ID. Return "
        "exactly these labeled fields, each once, with no other text: count=<integer> "
        "total_amount_cents=<integer> max_transaction_id=<ID> max_amount_cents=<integer>."
    )
    return GeneratedLongContext(
        query=query,
        context=context,
        truth=truth,
        seed=seed,
        target_chars=target_chars,
        sha256=hashlib.sha256(context.encode("utf-8")).hexdigest(),
    )
