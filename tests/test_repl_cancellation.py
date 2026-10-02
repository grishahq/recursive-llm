"""Cancellation stops real workers and their child calls, without provider requests."""

import asyncio
import multiprocessing
import os
import signal
import threading
import time
from multiprocessing.process import BaseProcess
from unittest.mock import patch

import pytest

from rlm.repl import REPLError, REPLExecutor, _worker_main


def response(text):
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "billing_mode": "subscription",
    }


@pytest.mark.asyncio
async def test_cancel_interrupts_running_python_without_waiting_for_repl_timeout():
    from rlm import RLM

    executing = threading.Event()
    workers = []

    class ObservedREPL(REPLExecutor):
        def _wait_for_message(self, budget):
            workers.append(self._process)
            executing.set()
            return super()._wait_for_message(budget)

    async def complete(**_kwargs):
        return response("while True: pass")

    # Startup imports can exceed five seconds under subprocess coverage. Start the
    # cancellation clock only once Python execution is ready; the assertion below
    # still requires cancellation well before the local execution timeout.
    rlm = RLM(model="test", completion_handler=complete, repl_timeout=30)
    with patch("rlm.core.REPLExecutor", ObservedREPL):
        task = asyncio.create_task(rlm.acomplete_result("question", "context"))
        try:
            assert await asyncio.to_thread(executing.wait, 15), "REPL did not start"
            started = time.monotonic()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
            assert time.monotonic() - started < 1.5
            assert workers and all(not worker.is_alive() for worker in workers)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def test_abort_during_worker_startup_cannot_leave_a_new_process_running(monkeypatch):
    repl = REPLExecutor(timeout=8)
    workers = []
    original_start = BaseProcess.start

    def start_then_abort(process):
        is_repl_worker = getattr(process, "_target", None) is _worker_main
        original_start(process)
        if is_repl_worker:
            workers.append(process)
            # The executor has not published its process reference yet.
            repl.abort()

    monkeypatch.setattr(BaseProcess, "start", start_then_abort)
    try:
        with pytest.raises(REPLError, match="aborted"):
            repl.execute("while True: pass", {})
        assert workers and all(not worker.is_alive() for worker in workers)
        with pytest.raises(REPLError, match="aborted"):
            repl.execute("1 + 1", {})
        assert len(workers) == 1
    finally:
        repl.close()


def _ignore_termination(ready):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    while True:
        time.sleep(1)


@pytest.mark.skipif(os.name != "posix", reason="SIGTERM handlers require POSIX")
@pytest.mark.parametrize("cleanup", ["close", "_terminate_timed_out_worker"])
def test_cleanup_kills_a_worker_that_ignores_sigterm(cleanup):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    worker = context.Process(target=_ignore_termination, args=(ready,), daemon=True)
    worker.start()
    repl = REPLExecutor()
    repl._process = worker
    try:
        assert ready.wait(10), "Test worker did not start"
        getattr(repl, cleanup)()
        assert not worker.is_alive()
        assert worker.exitcode == -signal.SIGKILL
    finally:
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=2)
        repl.close()


@pytest.mark.asyncio
async def test_cancel_stops_child_callback_without_cancelling_an_independent_run():
    from rlm import RLM

    child_started = asyncio.Event()
    child_cancelled = asyncio.Event()
    healthy_started = asyncio.Event()
    healthy_release = asyncio.Event()

    async def complete(*, messages, **_kwargs):
        question = messages[-1]["content"]
        if question == "cancel this run":
            return response('print(recursive_llm("blocked child", context))')
        if question == "keep this run":
            healthy_started.set()
            await healthy_release.wait()
            return response('FINAL("healthy answer")')
        assert "blocked child" in question
        child_started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            child_cancelled.set()
            raise
        raise AssertionError("The child call was not cancelled")

    rlm = RLM(model="test", completion_handler=complete, max_depth=1, repl_timeout=30)
    cancelled = asyncio.create_task(rlm.acomplete_result("cancel this run", "document A"))
    healthy = asyncio.create_task(rlm.acomplete_result("keep this run", "document B"))
    try:
        # Readiness is independent of the cancellation latency checked afterward.
        await asyncio.wait_for(child_started.wait(), timeout=15)
        await asyncio.wait_for(healthy_started.wait(), timeout=1)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(cancelled, timeout=2)
        await asyncio.wait_for(child_cancelled.wait(), timeout=1)
        assert not healthy.done()
        healthy_release.set()
        result = await asyncio.wait_for(healthy, timeout=2)
        assert result.answer == "healthy answer"
        assert result.stats["llm_calls"] == 1
    finally:
        healthy_release.set()
        cancelled.cancel()
        healthy.cancel()
        await asyncio.gather(cancelled, healthy, return_exceptions=True)
