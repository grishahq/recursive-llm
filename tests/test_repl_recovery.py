"""A lost worker cannot publish an obsolete parent-side answer."""

from unittest.mock import patch

import pytest

from rlm import RLM
from rlm.repl import REPLExecutor


@pytest.mark.asyncio
async def test_deleted_answer_stays_deleted_after_worker_timeout():
    steps = iter(
        [
            "result = 'stale answer'",
            "del result",
            "while True: pass",
            "FINAL_VAR(result)",
            'FINAL("recovered answer")',
        ]
    )

    async def complete(**_kwargs):
        return {"choices": [{"message": {"content": next(steps)}}]}

    class ShortExecutionREPL(REPLExecutor):
        def execute(self, code, env, *, timeout=None):
            if code == "while True: pass":
                timeout = 0.1  # Only shorten the already-running worker's local step.
            return super().execute(code, env, timeout=timeout)

    rlm = RLM(
        model="test",
        completion_handler=complete,
        repl_timeout=30,
        max_iterations=5,
        capture_trajectory_content=True,
    )
    with patch("rlm.core.REPLExecutor", ShortExecutionREPL):
        result = await rlm.acomplete_result("question", "source")

    assert result.answer == "recovered answer"
    finals = [event.data for event in result.trajectory if event.kind == "final_answer"]
    assert finals == [{"method": "directive", "answer": "recovered answer"}]
