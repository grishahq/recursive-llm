"""Text-only Codex transport, usage, cancellation, and recursion integration tests."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from rlm import RLM
from rlm.codex import CodexCompletion, CodexError, _environment, parse_events


@pytest.fixture(autouse=True)
def portable_fake_cli(monkeypatch):
    """Run fake CLI scripts with Python on POSIX and Windows alike."""
    create_process = asyncio.create_subprocess_exec

    async def create(*command, **kwargs):
        if command and Path(command[0]).name == "fake-codex":
            command = (sys.executable, *command)
        return await create_process(*command, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)


def events(answer="hello", usage=None):
    return [
        {"type": "thread.started", "thread_id": "test"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": answer}},
        {
            "type": "turn.completed",
            "usage": usage
            or {
                "input_tokens": 100,
                "cached_input_tokens": 40,
                "output_tokens": 12,
            },
        },
    ]


def test_usage_keeps_cached_input_and_is_unpriced():
    response = parse_events(events())
    assert response["usage"]["total_tokens"] == 112
    assert response["usage"]["prompt_tokens_details"]["cached_tokens"] == 40
    with patch("rlm.core.litellm.completion_cost") as cost:
        assert RLM._get_response_cost(response) is None
        cost.assert_not_called()


@pytest.mark.parametrize(
    "item_type",
    [
        "command_execution",
        "mcp_tool_call",
        "web_search",
        "file_change",
        "todo_list",
        "unknown",
    ],
)
def test_rejects_every_non_text_item(item_type):
    trace = [{"type": "item.started", "item": {"type": item_type}}] + events()
    with pytest.raises(CodexError, match="Text-only contract"):
        parse_events(trace)


@pytest.mark.parametrize(
    "trace",
    [
        [],
        events() + events(),
        events(answer=""),
        [{"type": "turn.failed", "error": {"message": "failed"}}],
        events(usage={"input_tokens": 1, "cached_input_tokens": 5, "output_tokens": 1}),
        events(usage={"input_tokens": "100", "cached_input_tokens": 0, "output_tokens": 1}),
    ],
)
def test_rejects_missing_or_invalid_completion(trace):
    with pytest.raises(CodexError):
        parse_events(trace)


@pytest.mark.parametrize(
    "event",
    [
        {"type": "unknown.tool"},
        {"type": "item.completed", "item": []},
        {"type": "item.completed", "item": {"type": "error", "message": "startup failed"}},
    ],
)
def test_rejects_unknown_events_and_startup_diagnostics(event):
    with pytest.raises(CodexError):
        parse_events([event] + events())


def test_command_isolated_and_chatgpt_only(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "test-url")
    monkeypatch.setenv("CODEX_API_KEY", "test-key")
    monkeypatch.setenv("CODEX_THREAD_ID", "parent")
    monkeypatch.setenv("CODEX_HOME", "/existing/credentials")
    env = _environment()
    assert env["CODEX_HOME"] == "/existing/credentials"
    assert (
        not {"OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_API_KEY", "CODEX_THREAD_ID"} & env.keys()
    )
    command = CodexCompletion()._command("gpt-5.6-luna", Path("/empty"), Path("/prompt"))
    for value in [
        "--ignore-user-config",
        "--ephemeral",
        'forced_login_method="chatgpt"',
        'web_search="disabled"',
        "shell_tool",
        "plugins",
        "apps",
    ]:
        assert value in command


def fake_executable(tmp_path, source):
    executable = tmp_path / "fake-codex"
    executable.write_text(f"#!{sys.executable}\n" + source)
    executable.chmod(0o700)
    return str(executable)


@pytest.mark.asyncio
async def test_subprocess_records_real_stream_and_stdin(tmp_path):
    source = (
        "import json,sys\n"
        "payload=json.load(sys.stdin)\n"
        f"trace={events()!r}\n"
        "trace[2]['item']['text']=payload[-1]['content']\n"
        "for item in trace: print(json.dumps(item),flush=True)\n"
    )
    transport = CodexCompletion(executable=fake_executable(tmp_path, source))
    response = await transport(model="gpt-5.6-luna", messages=[{"role": "user", "content": "hi"}])
    assert response["choices"][0]["message"]["content"] == "hi"
    assert transport.calls[0]["status"] == "completed"
    assert len(transport.calls[0]["events"]) == 4
    assert transport.calls[0]["elapsed_seconds"] > 0


@pytest.mark.asyncio
async def test_subprocess_timeout_and_tool_event_kill_child(tmp_path, monkeypatch):
    create_process = asyncio.create_subprocess_exec

    async def create_ready_process(*command, **kwargs):
        process = await create_process(*command, **kwargs)
        try:
            # Coverage initialization happens before the fake CLI starts. Keep that
            # startup deadline separate from the short exchange timeout under test.
            ready = await asyncio.wait_for(process.stderr.readline(), timeout=20)
            assert ready == b"FAKE_CLI_READY\n", "Fake CLI failed before becoming ready"
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_ready_process)
    for source, exception in [
        ("import time\ntime.sleep(20)\n", asyncio.TimeoutError),
        (
            "import json,time\nprint(json.dumps({'type':'item.started',"
            "'item':{'type':'command_execution'}}),flush=True)\ntime.sleep(20)\n",
            CodexError,
        ),
    ]:
        transport = CodexCompletion(
            executable=fake_executable(
                tmp_path,
                "import sys\nprint('FAKE_CLI_READY', file=sys.stderr, flush=True)\n" + source,
            ),
            timeout=5 if exception is CodexError else 0.2,
        )
        with pytest.raises(exception):
            await transport(model="gpt-5.6-luna", messages=[])
        assert transport.calls[0]["status"] == "failed"
        assert transport.calls[0]["returncode"] != 0
        if exception is CodexError:
            assert transport.calls[0]["events"][0]["item"]["type"] == "command_execution"


@pytest.mark.asyncio
async def test_login_requires_chatgpt(tmp_path):
    for status, accepted in [
        ("Logged in using ChatGPT", True),
        ("Logged in using an API key", False),
    ]:
        transport = CodexCompletion(executable=fake_executable(tmp_path, f"print({status!r})\n"))
        if accepted:
            assert await transport.check_login() == "ChatGPT"
        else:
            with pytest.raises(CodexError, match="codex login"):
                await transport.check_login()


@pytest.mark.asyncio
async def test_custom_handler_reaches_child_rlm_and_leaf():
    handler = AsyncMock(
        side_effect=[
            parse_events(events('print(recursive_llm("child", context))')),
            parse_events(events('print(recursive_llm("leaf", context))')),
            parse_events(events("leaf evidence")),
            parse_events(events('FINAL("child evidence")')),
            parse_events(events('FINAL("root answer")')),
        ]
    )
    rlm = RLM(
        model="gpt-5.6-luna",
        completion_handler=handler,
        max_depth=2,
        max_total_calls=12,
        # This checks dispatch and accounting, not wall-clock budgets. Worker
        # initialization under coverage must not consume an artificial run budget.
        repl_timeout=30,
        capture_trajectory_content=True,
    )
    with patch("rlm.core.litellm.acompletion") as api:
        result = await asyncio.wait_for(rlm.acomplete_result("question", "document"), timeout=60)
        api.assert_not_called()
    assert result.answer == "root answer"
    assert not [
        event.to_dict()
        for event in result.trajectory
        if event.kind == "repl_step" and event.data["status"] != "ok"
    ]
    assert result.stats["llm_calls"] == 5
    assert result.stats["leaf_calls"] == 1
    assert result.stats["total_tokens"] == 560
    assert result.stats["estimated_cost_usd"] is None


@pytest.mark.asyncio
async def test_rejects_api_options_without_starting_process():
    transport = CodexCompletion()
    with pytest.raises(ValueError, match="API options"):
        await transport(model="gpt-5.6-luna", messages=[], api_key="invalid")
    assert not transport.calls


@pytest.mark.asyncio
async def test_capture_custom_input_is_ungraded_and_preserves_attempt(tmp_path):
    from examples.capture_codex_comparison import capture_comparison

    handlers = []
    for answers in [["Paris"], ["print(context)", 'FINAL("Paris")']]:
        handler = AsyncMock(side_effect=[parse_events(events(answer)) for answer in answers])
        handler.check_login = AsyncMock(return_value="ChatGPT")
        handler.calls = []
        handlers.append(handler)
    output = tmp_path / "custom.json"
    with patch("examples.capture_codex_comparison.CodexCompletion", side_effect=handlers):
        record = await capture_comparison(
            context="The capital is Paris.", query="Capital?", output=output
        )
    assert record["rlm"]["result"] == record["direct"]["result"] == "Paris"
    assert record["rlm"]["passed"] is record["direct"]["passed"] is None
    assert record["rlm"]["usage_complete"]
    assert output.with_suffix(".raw.json").exists()
    with pytest.raises(FileExistsError, match="preserve"):
        await capture_comparison(context="Source", query="Question", output=output)


@pytest.mark.asyncio
async def test_capture_partial_usage_cannot_be_presented_as_total(tmp_path):
    from examples.capture_codex_comparison import capture_comparison

    direct = AsyncMock(return_value=parse_events(events("direct answer")))
    direct.check_login = AsyncMock(return_value="ChatGPT")
    direct.calls = []
    recursive = AsyncMock(
        side_effect=[parse_events(events("print(context)")), asyncio.TimeoutError()]
    )
    recursive.calls = []
    with patch(
        "examples.capture_codex_comparison.CodexCompletion", side_effect=[direct, recursive]
    ):
        record = await capture_comparison(
            context="Source", query="Question", output=tmp_path / "partial.json"
        )
    assert record["rlm"]["tokens"] is None
    assert record["rlm"]["observed_tokens"] == 112
    assert not record["rlm"]["usage_complete"]
    assert "error" in record["rlm"]


@pytest.mark.asyncio
async def test_capture_validates_custom_pair_before_login(tmp_path):
    from examples.capture_codex_comparison import capture_comparison

    with pytest.raises(ValueError, match="together"):
        await capture_comparison(context="Source", output=tmp_path / "invalid.json")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options,model,effort,depth",
    [
        ({}, "gpt-5.6-luna", "medium", 1),
        (
            {"model": "gpt-6.1-sol", "reasoning_effort": "low", "max_depth": 0},
            "gpt-6.1-sol",
            "low",
            0,
        ),
        (
            {"model": "gpt-6.1-sol", "reasoning_effort": "medium", "max_depth": 2},
            "gpt-6.1-sol",
            "medium",
            2,
        ),
    ],
)
async def test_capture_configuration_matches_both_transports_and_recordings(
    tmp_path, options, model, effort, depth
):
    from examples.capture_codex_comparison import capture_comparison

    direct = AsyncMock(return_value=parse_events(events("Paris")))
    direct.check_login = AsyncMock(return_value="ChatGPT")
    direct.calls = []
    recursive = AsyncMock(return_value=parse_events(events('FINAL("Paris")')))
    recursive.calls = []
    progress = []
    output = tmp_path / "configured.json"
    with patch(
        "examples.capture_codex_comparison.CodexCompletion", side_effect=[direct, recursive]
    ) as factory:
        record = await capture_comparison(
            context="The capital is Paris.",
            query="Capital?",
            output=output,
            progress_handler=progress.append,
            **options,
        )
    assert len(factory.call_args_list) == 2
    assert all(call.kwargs == {"reasoning_effort": effort} for call in factory.call_args_list)
    assert direct.call_args.kwargs["model"] == recursive.call_args.kwargs["model"] == model
    raw = json.loads(output.with_suffix(".raw.json").read_text())
    for value in [record, raw, *(item["comparison"] for item in progress)]:
        assert value["model"] == model
        assert value["reasoning_effort"] == effort
        assert value["max_depth"] == depth
    assert raw["methodology"]["rlm_max_depth"] == depth
    assert raw["rlm"]["result"]["config"]["max_depth"] == depth
    assert f"{effort} reasoning" in record["sample_note"]
    assert f"RLM max depth {depth}" in record["sample_note"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [{"model": " "}, {"max_depth": -1}, {"max_depth": True}, {"reasoning_effort": "instant"}],
)
async def test_capture_rejects_invalid_configuration_before_login(tmp_path, options):
    from examples.capture_codex_comparison import capture_comparison

    with patch.object(CodexCompletion, "check_login") as login:
        with pytest.raises(ValueError):
            await capture_comparison(output=tmp_path / "invalid.json", **options)
        login.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_capture_cli_forwards_configuration(tmp_path, monkeypatch, capsys):
    from examples.capture_codex_comparison import main

    output = tmp_path / "cli.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capture_codex_comparison.py",
            "--model",
            "gpt-6.1-sol",
            "--reasoning-effort",
            "low",
            "--max-depth",
            "0",
            "--output",
            str(output),
        ],
    )
    lane = {"tokens": 1, "seconds": 0, "calls": 1, "passed": None}
    with patch(
        "examples.capture_codex_comparison.capture_comparison",
        new_callable=AsyncMock,
        return_value={"rlm": lane, "direct": lane},
    ) as capture:
        main()
    assert capture.call_args.kwargs["model"] == "gpt-6.1-sol"
    assert capture.call_args.kwargs["reasoning_effort"] == "low"
    assert capture.call_args.kwargs["max_depth"] == 0
    assert capture.call_args.kwargs["output"] == output
    assert json.loads(capsys.readouterr().out)["direct"]["calls"] == 1
