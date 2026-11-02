"""Local source work obeys hard worker timeouts without charging LM callback waits."""

import json
import time

import pytest

from rlm.repl import REPLExecutor, REPLTimeoutError
from rlm.sources import SourceDocument


def test_source_helpers_are_local_runtime_helpers_with_redacted_events():
    document = SourceDocument("report", ("Pétya\r\narrived.",))
    env = {"_source_documents": (document,)}
    events = []
    with REPLExecutor(source_event_handler=lambda kind, data: events.append((kind, data))) as repl:
        output = repl.execute(
            "metadata = source_manifest()\n"
            "hits = search_sources('PETYA', limit=1, window=5)\n"
            "citation = read_source('report', 1, 0, 5)\n"
            "print(citation['quote'])",
            env,
        )
        assert output == "Pétya"
        assert repl._callbacks == {}
        assert env["metadata"][0]["source_sha256"] == document.content_sha256
        assert env["hits"] == [env["citation"]]
        assert all(name not in env for name in ("read_source", "search_sources", "source_manifest"))
        variables = repl.execute("SHOW_VARS()", env)
        assert "_source_documents" not in variables and "read_source" not in variables
    assert [kind for kind, _ in events] == ["source_search", "source_read"]
    assert events[0][1] == {"document_id": None, "matches": 1, "term_chars": 5}
    assert events[1][1] == {
        "document_id": "report",
        "source_sha256": document.content_sha256,
        "page": 1,
        "start": 0,
        "end": 5,
        "quote_chars": 5,
    }
    serialized = json.dumps(events)
    assert "Pétya" not in serialized and "PETYA" not in serialized


def test_source_event_handler_failure_does_not_interrupt_execution():
    def failing_handler(_kind, _data):
        raise RuntimeError("User telemetry handler failed")

    with REPLExecutor(source_event_handler=failing_handler) as repl:
        output = repl.execute(
            "read_source('report', 1)['quote']",
            {"_source_documents": (SourceDocument("report", ("value=7",)),)},
        )
        assert output == "value=7"


@pytest.mark.parametrize("single_heavy_search", [False, True])
def test_search_cpu_is_interrupted_by_the_hard_local_step_timeout(single_heavy_search):
    # Warm the real worker before starting the timeout clock. Coverage imports
    # and process startup have a separate allowance and are not this regression.
    size = 4_000_000 if single_heavy_search else 2_000
    env = {"_source_documents": (SourceDocument("large", ("x" * size,)),)}
    with REPLExecutor(timeout=5) as repl:
        repl.execute("ready = True", env)
        worker = repl._process
        repl.timeout = 0.03
        code = "search_sources('missing')"
        if not single_heavy_search:
            code = "for i in range(1000):\n    hits = search_sources('missing')"
        started = time.monotonic()
        with pytest.raises(REPLTimeoutError, match="timed out"):
            repl.execute(code, env)
        assert time.monotonic() - started < 1.5
        assert worker is not None and not worker.is_alive()


def test_model_callback_wait_is_still_exempt_with_worker_source_helpers():
    def llm_query():
        time.sleep(0.1)
        return "model completed"

    env = {
        "_source_documents": (SourceDocument("report", ("source",)),),
        "llm_query": llm_query,
    }
    with REPLExecutor(timeout=5) as repl:
        repl.execute("ready = True", env)
        repl.timeout = 0.03
        assert repl.execute("llm_query()", env) == "model completed"


def test_source_helpers_survive_worker_restart_without_snapshotting_the_corpus():
    env = {"_source_documents": (SourceDocument("report", ("original",)),)}
    with REPLExecutor() as repl:
        repl.execute("quote = read_source('report', 1)['quote']", env)
        repl.close()
        assert repl.execute("read_source('report', 1)['quote'] == quote", env) == "True"
        assert "source_manifest" not in env and "search_sources" not in env
