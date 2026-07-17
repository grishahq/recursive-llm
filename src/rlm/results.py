"""Structured completion and trajectory results."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Tuple, Union


RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrajectoryEvent:
    """One ordered event from an RLM completion tree."""

    sequence: int
    kind: str
    elapsed_seconds: float
    depth: int
    node_id: str
    parent_id: str
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Return a detached JSON-serializable representation."""
        return {
            "sequence": self.sequence,
            "kind": self.kind,
            "elapsed_seconds": self.elapsed_seconds,
            "depth": self.depth,
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "data": deepcopy(self.data),
        }


@dataclass(frozen=True)
class CompletionResult:
    """Answer, exact per-run usage, and the full recursion trajectory."""

    answer: str
    stats: Dict[str, Any]
    trajectory: Tuple[TrajectoryEvent, ...]
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a detached JSON-serializable representation."""
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "termination_reason": "completed",
            "answer": self.answer,
            "stats": deepcopy(self.stats),
            "config": deepcopy(self.config),
            "trajectory": [event.to_dict() for event in self.trajectory],
        }

    def write_jsonl(
        self,
        path: Union[str, PathLike[str]],
        *,
        append: bool = True,
    ) -> None:
        """Write this completed run as one versioned UTF-8 JSONL record."""
        mode = "a" if append else "w"
        record = json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))
        with Path(path).open(mode, encoding="utf-8", newline="\n") as stream:
            stream.write(record + "\n")
