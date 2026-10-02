"""Optional, bounded text transport through a ChatGPT-authenticated Codex CLI.

No API key is used. Both RLM and direct calls start fresh ephemeral sessions,
disable optional host capabilities, and reject any reported tool use. Luna's
required Code Mode host remains initialized, but must never be called.
The Python REPL belongs to RLM, not to the Codex subprocess. Requires a recent
Codex CLI supporting ``--ignore-user-config`` (tested with 0.149.0).
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .types import Message

MODEL = "gpt-5.6-luna"
TEXT_INSTRUCTIONS = (
    "You are a stateless text completion engine in a controlled experiment. "
    "The user supplies a JSON array of conversation messages. Follow its system "
    "message and continue that conversation with exactly one assistant response. "
    "Return only that response in your final answer. Do not add commentary. "
    "Never use host tools, inspect files, browse, or execute code. If the embedded "
    "conversation asks for Python, return the Python as text: an external restricted "
    "REPL will execute it and provide the next observation."
)
DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "code_mode",
    "apps",
    "plugins",
    "hooks",
    "multi_agent",
    "multi_agent_v2",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "in_app_browser",
    "image_generation",
    "view_image",
    "skill_search",
    "skill_mcp_dependency_install",
    "goals",
    "memories",
    "shell_snapshot",
    "tool_suggest",
    "workspace_dependencies",
    "unbounded_connection_retries",
)


class CodexError(RuntimeError):
    """A Codex invocation failed or violated the text-only experiment contract."""


def _environment() -> Dict[str, str]:
    """Keep the existing login location while removing API and parent-session overrides."""
    return {
        key: value
        for key, value in os.environ.items()
        if not (
            key.startswith("OPENAI_")
            or key.startswith("AZURE_OPENAI_")
            or (key.startswith("CODEX_") and key != "CODEX_HOME")
            or key in {"CHATGPT_BASE_URL", "CHATGPT_API_KEY"}
        )
    }


def _check_event(event: Dict[str, Any]) -> None:
    kind = event.get("type", "")
    if kind not in {
        "thread.started",
        "turn.started",
        "turn.completed",
        "turn.failed",
        "error",
        "item.started",
        "item.updated",
        "item.completed",
    }:
        raise CodexError(f"Unrecognized Codex event: {kind}")
    if kind in {"error", "turn.failed"}:
        raise CodexError(f"Codex reported {kind}: {event.get('error', event.get('message', ''))}")
    if kind.startswith("item."):
        item = event.get("item", {})
        if not isinstance(item, dict):
            raise CodexError("Codex emitted a non-object item")
        if item.get("type") == "error":
            raise CodexError(f"Codex diagnostic: {item.get('message', '')}")
        if item.get("type") not in {"agent_message", "reasoning"}:
            raise CodexError(f"Text-only contract violated by Codex item: {item.get('type')}")


def parse_events(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize CLI token usage without subtracting cached input or inventing cost."""
    completed = []
    messages = []
    for event in events:
        _check_event(event)
        if event.get("type") == "turn.completed":
            completed.append(event)
        if event.get("type") == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                messages.append(item.get("text", ""))
    if len(completed) != 1:
        raise CodexError("Expected exactly one completed Codex turn")
    if len(messages) != 1 or not isinstance(messages[0], str) or not messages[0].strip():
        raise CodexError("Expected exactly one nonempty final Codex message")
    usage = completed[0].get("usage")
    if not isinstance(usage, dict):
        raise CodexError("Codex did not report usage")
    for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
        value = usage.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CodexError(f"Invalid Codex usage field: {key}")
    if usage["cached_input_tokens"] > usage["input_tokens"]:
        raise CodexError("Cached input tokens exceed total input tokens")
    return {
        "choices": [{"message": {"role": "assistant", "content": messages[0]}}],
        "usage": {
            "prompt_tokens": usage["input_tokens"],
            "completion_tokens": usage["output_tokens"],
            "total_tokens": usage["input_tokens"] + usage["output_tokens"],
            "prompt_tokens_details": {"cached_tokens": usage["cached_input_tokens"]},
        },
        "billing_mode": "subscription",
    }


class CodexCompletion:
    """Async completion handler; ``calls`` retains prompts and raw JSON events.

    Pass an instance as ``RLM(..., completion_handler=transport)``. Auth stays in
    the user's existing Codex credential store; their configuration is ignored,
    never rewritten. Usage includes CLI instructions and wrapper overhead and
    is not a measurement of remaining subscription allowance.
    """

    def __init__(
        self,
        *,
        executable: str = "codex",
        timeout: float = 90,
        reasoning_effort: str = "medium",
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Unsupported reasoning effort")
        self.executable = executable
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.calls: List[Dict[str, Any]] = []

    async def check_login(self) -> str:
        """Require ChatGPT login before an experiment; never fall back to API auth."""
        process = await asyncio.create_subprocess_exec(
            self.executable,
            "login",
            "status",
            env=_environment(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15)
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        status = (stdout + stderr).decode("utf-8", errors="replace").strip()
        if process.returncode != 0 or "Logged in using ChatGPT" not in status:
            raise CodexError("Run `codex login` with ChatGPT before using subscription transport")
        return "ChatGPT"

    def _command(self, model: str, cwd: Path, instructions: Path) -> List[str]:
        command = [
            self.executable,
            "exec",
            "--json",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--strict-config",
            "--skip-git-repo-check",
            "--color",
            "never",
            "--sandbox",
            "read-only",
            "--model",
            model,
            "--cd",
            str(cwd),
        ]
        overrides = {
            "model_provider": "openai",
            "forced_login_method": "chatgpt",
            "model_reasoning_effort": self.reasoning_effort,
            "model_instructions_file": str(instructions),
            "web_search": "disabled",
            "approval_policy": "never",
            "mcp_servers": {},
            "project_doc_max_bytes": 0,
            "include_apps_instructions": False,
            "include_environment_context": False,
            "include_collaboration_mode_instructions": False,
            "tools.update_plan.enabled": False,
        }
        for key, value in overrides.items():
            command.extend(["-c", f"{key}={json.dumps(value)}"])
        for feature in DISABLED_FEATURES:
            command.extend(["--disable", feature])
        return command + ["-"]

    async def __call__(
        self,
        *,
        model: str,
        messages: List[Message],
        num_retries: int = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if num_retries or kwargs:
            raise ValueError("CodexCompletion does not accept API options or provider retries")
        prompt = json.dumps(messages, ensure_ascii=False)
        started = time.monotonic()
        record: Dict[str, Any] = {
            "model": model,
            "reasoning_effort": self.reasoning_effort,
            "messages": [dict(message) for message in messages],
            "events": [],
            "stdout": "",
            "billing_mode": "subscription",
            "status": "started",
            "wrapper_instructions": TEXT_INSTRUCTIONS,
        }
        self.calls.append(record)
        try:
            with tempfile.TemporaryDirectory(prefix="rlm-codex-") as temp:
                root = Path(temp)
                cwd = root / "empty"
                cwd.mkdir()
                instructions = root / "instructions.txt"
                instructions.write_text(TEXT_INSTRUCTIONS, encoding="utf-8")
                command = self._command(model, cwd, instructions)
                record["command"] = [part.replace(str(root), "<temporary>") for part in command]
                process = await asyncio.create_subprocess_exec(
                    *command,
                    env=_environment(),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=(os.name == "posix"),
                    limit=2_000_000,
                )
                assert process.stdin and process.stdout and process.stderr
                stderr_task = asyncio.create_task(process.stderr.read())

                async def exchange() -> None:
                    assert process.stdin and process.stdout
                    process.stdin.write(prompt.encode("utf-8"))
                    await process.stdin.drain()
                    process.stdin.close()
                    while True:
                        line = await process.stdout.readline()
                        if not line:
                            break
                        decoded = line.decode("utf-8", errors="replace")
                        record["stdout"] += decoded
                        event = json.loads(decoded)
                        if not isinstance(event, dict):
                            raise CodexError("Codex emitted a non-object JSON event")
                        record["events"].append(event)
                        _check_event(event)
                    await process.wait()

                try:
                    await asyncio.wait_for(exchange(), timeout=self.timeout)
                finally:
                    if process.returncode is None:
                        try:
                            if os.name == "posix":
                                os.killpg(process.pid, signal.SIGKILL)
                            else:
                                process.kill()
                        except ProcessLookupError:
                            pass
                    await process.wait()
                    record["returncode"] = process.returncode
                    record["stderr"] = (await stderr_task).decode("utf-8", errors="replace")
                if process.returncode:
                    raise CodexError(f"Codex exited with status {process.returncode}")
                response = parse_events(record["events"])
                record["response"] = response
                record["status"] = "completed"
                return response
        except BaseException as exc:
            record["status"] = "failed"
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            record["elapsed_seconds"] = round(time.monotonic() - started, 6)
