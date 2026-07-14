"""Persistent restricted Python executor with a hard per-step timeout."""

from __future__ import annotations

import ast
import importlib
import multiprocessing
import pickle
import threading
import time
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Set, Tuple, cast

from RestrictedPython import (
    compile_restricted_exec,
    limited_builtins,
    safe_globals,
    utility_builtins,
)
from RestrictedPython.Guards import (
    full_write_guard,
    guarded_iter_unpack_sequence,
    safer_getattr,
)
from RestrictedPython.PrintCollector import PrintCollector


class REPLError(Exception):
    """Error during REPL execution."""


class REPLTimeoutError(REPLError):
    """A REPL step exceeded its local execution budget."""


_RESULT_NAME = "rlm_internal_last_result"
_RESERVED_NAMES = {
    "context",
    "query",
    "answer",
    "SHOW_VARS",
    "_print",
    _RESULT_NAME,
}


def _is_picklable(value: Any) -> bool:
    """Return whether a value can cross the worker process boundary."""
    try:
        pickle.dumps(value)
    except (pickle.PickleError, TypeError, AttributeError):
        return False
    return True


def _extract_code(text: str) -> str:
    """Extract the first Markdown code block when one is present."""
    for marker in ("```python", "```"):
        if marker in text:
            start = text.find(marker) + len(marker)
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()
    return text


def _build_globals() -> Dict[str, Any]:
    """Build the fixed global namespace used by the restricted worker."""
    restricted_globals: Dict[str, Any] = safe_globals.copy()
    restricted_globals.update(limited_builtins)
    restricted_globals.update(utility_builtins)
    allowed_imports = {"collections", "datetime", "json", "math", "re"}

    def safe_import(
        name: str,
        _globals: Any = None,
        _locals: Any = None,
        _fromlist: Any = (),
        level: int = 0,
    ) -> ModuleType:
        if level != 0 or name not in allowed_imports:
            raise ImportError(f"Import of {name!r} is not allowed")
        return importlib.import_module(name)

    restricted_builtins = dict(restricted_globals.get("__builtins__", {}))
    restricted_builtins["__import__"] = safe_import
    restricted_globals["__builtins__"] = restricted_builtins
    restricted_globals.update(
        {
            "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
            "_getattr_": safer_getattr,
            "_getitem_": lambda obj, index: obj[index],
            "_getiter_": iter,
            "_print_": PrintCollector,
            "_write_": full_write_guard,
            "__import__": safe_import,
            "_import_": safe_import,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "frozenset": frozenset,
            "bytes": bytes,
            "bytearray": bytearray,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "reversed": reversed,
            "iter": iter,
            "next": next,
            "sorted": sorted,
            "sum": sum,
            "min": min,
            "max": max,
            "any": any,
            "all": all,
            "abs": abs,
            "round": round,
            "pow": pow,
            "divmod": divmod,
            "chr": chr,
            "ord": ord,
            "hex": hex,
            "oct": oct,
            "bin": bin,
            "repr": repr,
            "ascii": ascii,
            "format": format,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "callable": callable,
            "type": type,
            "hasattr": hasattr,
            "True": True,
            "False": False,
            "None": None,
        }
    )

    import json
    import math
    import re
    from collections import Counter, defaultdict
    from datetime import datetime, timedelta

    restricted_globals.update(
        {
            "re": re,
            "json": json,
            "math": math,
            "datetime": datetime,
            "timedelta": timedelta,
            "Counter": Counter,
            "defaultdict": defaultdict,
        }
    )
    return restricted_globals


def _make_callback_proxy(connection: Connection, name: str) -> Callable[..., Any]:
    """Create a worker-side proxy for a callback owned by the parent."""

    def proxy(*args: Any, **kwargs: Any) -> Any:
        connection.send(
            {
                "type": "callback",
                "name": name,
                "args": args,
                "kwargs": kwargs,
            }
        )
        message = connection.recv()
        if message.get("type") != "callback_result":
            raise RuntimeError("Invalid callback response from the parent process")
        if message.get("error") is not None:
            raise RuntimeError(str(message["error"]))
        return message.get("value")

    return proxy


def _snapshot_environment(env: Dict[str, Any], callback_names: Set[str]) -> Dict[str, Any]:
    """Copy ordinary user variables back to the parent process."""
    snapshot: Dict[str, Any] = {}
    excluded = _RESERVED_NAMES | callback_names
    for name, value in env.items():
        if name.startswith("_") or name in excluded or isinstance(value, ModuleType):
            continue
        if _is_picklable(value):
            snapshot[name] = value
    return snapshot


def _execute_code(code: str, env: Dict[str, Any], globals_: Dict[str, Any]) -> str:
    """Execute one step and evaluate its final expression exactly once."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise REPLError(f"Compilation error: {exc.msg}") from exc

    env.pop(_RESULT_NAME, None)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        final_expression = tree.body[-1]
        tree.body[-1] = ast.copy_location(
            ast.Assign(
                targets=[ast.Name(id=_RESULT_NAME, ctx=ast.Store())],
                value=final_expression.value,
            ),
            final_expression,
        )
        ast.fix_missing_locations(tree)

    compiled = compile_restricted_exec(ast.unparse(tree))
    if compiled.errors:
        raise REPLError(f"Compilation error: {', '.join(compiled.errors)}")
    if compiled.code is None:
        raise REPLError("Compilation did not produce executable code")

    exec(compiled.code, globals_, env)

    output = ""
    collector = env.get("_print")
    if collector is not None and hasattr(collector, "txt"):
        output = "".join(collector.txt)

    result = env.pop(_RESULT_NAME, None)
    if result is not None:
        output += f"{result}\n"
    return output.strip() or "Code executed successfully (no output)"


def _worker_main(
    connection: Connection,
    initial_env: Dict[str, Any],
    callback_names: Set[str],
    max_output_chars: int,
) -> None:
    """Serve execution requests while preserving state between REPL steps."""
    globals_ = _build_globals()
    env = dict(initial_env)
    env.setdefault("answer", {"content": "", "ready": False})
    for name in callback_names:
        env[name] = _make_callback_proxy(connection, name)

    def show_vars() -> Dict[str, str]:
        return {
            name: type(value).__name__
            for name, value in env.items()
            if not name.startswith("_") and name not in callback_names
        }

    env["SHOW_VARS"] = show_vars
    connection.send({"type": "ready"})

    while True:
        try:
            message = connection.recv()
        except EOFError:
            break

        command = message.get("command")
        if command == "close":
            break
        if command == "get_variable":
            name = str(message.get("name", ""))
            found = name in env and _is_picklable(env[name])
            connection.send(
                {
                    "type": "variable",
                    "found": found,
                    "value": env.get(name) if found else None,
                }
            )
            continue
        if command != "execute":
            connection.send({"type": "error", "error": "Unknown REPL command"})
            continue

        env.pop("_print", None)
        try:
            output = _execute_code(str(message.get("code", "")), env, globals_)
            if len(output) > max_output_chars:
                output = (
                    f"{output[:max_output_chars]}\n\n"
                    f"[Output truncated: {len(output)} chars total, "
                    f"showing first {max_output_chars}]"
                )
            answer = env.get("answer")
            final_answer: Optional[str] = None
            if isinstance(answer, dict) and answer.get("ready"):
                final_answer = str(answer.get("content", ""))
            connection.send(
                {
                    "type": "result",
                    "output": output,
                    "snapshot": _snapshot_environment(env, callback_names),
                    "final_answer": final_answer,
                }
            )
        except BaseException as exc:
            connection.send(
                {
                    "type": "error",
                    "error": str(exc),
                    "snapshot": _snapshot_environment(env, callback_names),
                }
            )

    connection.close()


class REPLExecutor:
    """Persistent restricted Python executor backed by an isolated worker process."""

    def __init__(self, timeout: float = 5, max_output_chars: int = 2000):
        """Initialize the executor with a hard local-code timeout."""
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if max_output_chars <= 0:
            raise ValueError("max_output_chars must be greater than zero")
        self.timeout = float(timeout)
        self.max_output_chars = max_output_chars
        self._process: Optional[BaseProcess] = None
        self._connection: Optional[Connection] = None
        self._callbacks: Dict[str, Callable[..., Any]] = {}
        self._final_answer: Optional[str] = None
        self._lock = threading.RLock()

    def execute(self, code: str, env: Dict[str, Any]) -> str:
        """Execute one restricted step, preserving variables for later steps."""
        code = _extract_code(code)
        if not code.strip():
            return "No code to execute"

        with self._lock:
            self._ensure_worker(env)
            connection = self._require_connection()
            connection.send({"command": "execute", "code": code})
            message = self._wait_for_message(self.timeout)
            env.update(message.get("snapshot", {}))

            if message.get("type") == "error":
                raise REPLError(f"Execution error: {message.get('error', 'unknown error')}")
            if message.get("type") != "result":
                raise REPLError("Invalid response from the REPL worker")

            self._final_answer = message.get("final_answer")
            return str(message["output"])

    def get_variable(self, name: str) -> Tuple[bool, Any]:
        """Read a variable from the persistent worker namespace."""
        with self._lock:
            connection = self._require_connection()
            connection.send({"command": "get_variable", "name": name})
            message = self._wait_for_message(self.timeout)
            if message.get("type") != "variable":
                raise REPLError("Invalid variable response from the REPL worker")
            return bool(message.get("found")), message.get("value")

    def pop_final_answer(self) -> Optional[str]:
        """Return and clear an answer published through the REPL answer object."""
        answer = self._final_answer
        self._final_answer = None
        return answer

    def close(self) -> None:
        """Stop the worker process and release its communication channel."""
        with self._lock:
            connection = self._connection
            process = self._process
            self._connection = None
            self._process = None
            if connection is not None:
                try:
                    connection.send({"command": "close"})
                except (BrokenPipeError, EOFError, OSError):
                    pass
                connection.close()
            if process is not None:
                process.join(timeout=0.2)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1)

    def _ensure_worker(self, env: Dict[str, Any]) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self.close()

        callbacks = {name: value for name, value in env.items() if callable(value)}
        initial_env = {
            name: value
            for name, value in env.items()
            if name not in callbacks and not isinstance(value, ModuleType) and _is_picklable(value)
        }
        parent_connection, child_connection = multiprocessing.get_context("spawn").Pipe()
        process = multiprocessing.get_context("spawn").Process(
            target=_worker_main,
            args=(child_connection, initial_env, set(callbacks), self.max_output_chars),
            daemon=True,
        )
        process.start()
        child_connection.close()
        self._callbacks = callbacks
        self._connection = parent_connection
        self._process = process
        if not parent_connection.poll(10):
            self._terminate_timed_out_worker()
            raise REPLError("REPL worker did not start within 10 seconds")
        try:
            ready_message = parent_connection.recv()
        except (EOFError, OSError) as exc:
            self._terminate_timed_out_worker()
            raise REPLError("REPL worker exited during startup") from exc
        if ready_message.get("type") != "ready":
            self._terminate_timed_out_worker()
            raise REPLError("REPL worker returned an invalid startup response")

    def _wait_for_message(self, budget: float) -> Dict[str, Any]:
        connection = self._require_connection()
        remaining = budget
        while remaining > 0:
            started = time.monotonic()
            if not connection.poll(remaining):
                self._terminate_timed_out_worker()
                raise REPLTimeoutError(f"Execution timed out after {budget:g} seconds")
            remaining -= time.monotonic() - started
            try:
                message = connection.recv()
            except (EOFError, OSError) as exc:
                self.close()
                raise REPLError("REPL worker exited unexpectedly") from exc

            if message.get("type") != "callback":
                return cast(Dict[str, Any], message)

            name = str(message.get("name", ""))
            callback = self._callbacks.get(name)
            if callback is None:
                connection.send(
                    {"type": "callback_result", "error": f"Unknown callback: {name}"}
                )
                continue
            try:
                value = callback(*message.get("args", ()), **message.get("kwargs", {}))
                if not _is_picklable(value):
                    raise TypeError(f"Callback {name} returned a non-serializable value")
                connection.send({"type": "callback_result", "value": value, "error": None})
            except BaseException as exc:
                connection.send({"type": "callback_result", "value": None, "error": str(exc)})

        self._terminate_timed_out_worker()
        raise REPLTimeoutError(f"Execution timed out after {budget:g} seconds")

    def _terminate_timed_out_worker(self) -> None:
        process = self._process
        connection = self._connection
        self._process = None
        self._connection = None
        if connection is not None:
            connection.close()
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=1)

    def _require_connection(self) -> Connection:
        if self._connection is None:
            raise REPLError("REPL worker is not running")
        return self._connection

    def __enter__(self) -> "REPLExecutor":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
