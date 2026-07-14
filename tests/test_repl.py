"""Tests for REPL executor."""

import pytest
import re
import time

from rlm.repl import (
    REPLError,
    REPLExecutor,
    REPLTimeoutError,
    _extract_code,
    _is_picklable,
)


@pytest.fixture
def repl():
    """Create REPL executor."""
    executor = REPLExecutor()
    yield executor
    executor.close()


def test_simple_expression(repl):
    """Test simple expression."""
    env = {}
    repl.execute("x = 5 + 3", env)
    assert env["x"] == 8


def test_string_operations(repl):
    """Test string operations on context."""
    env = {"context": "Hello World"}
    repl.execute("result = context[:5]", env)
    assert env["result"] == "Hello"


def test_regex_operations(repl):
    """Test regex operations."""
    env = {"context": "The year is 2025", "re": re}
    repl.execute("matches = re.findall(r'\\d+', context)", env)
    assert env["matches"] == ["2025"]


def test_print_output(repl):
    """Test capturing print output."""
    env = {}
    result = repl.execute("print('Hello')", env)
    assert "Hello" in result


def test_multiline_code(repl):
    """Test multiline code."""
    code = """
x = 10
y = 20
z = x + y
print(z)
"""
    env = {}
    result = repl.execute(code, env)
    assert "30" in result


def test_code_block_extraction(repl):
    """Test extracting code from markdown blocks."""
    text = """
Here's some code:
```python
x = 5
print(x)
```
"""
    env = {}
    repl.execute(text, env)
    assert env["x"] == 5


def test_list_operations(repl):
    """Test list operations."""
    env = {}
    repl.execute("items = [1, 2, 3, 4, 5]", env)
    assert env["items"] == [1, 2, 3, 4, 5]


def test_forbidden_import(repl):
    """Test that arbitrary imports are forbidden."""
    env = {}
    with pytest.raises(REPLError):
        repl.execute("import os", env)


def test_whitelisted_helper_imports(repl):
    """Test compatibility imports for the already exposed safe helpers."""
    env = {}
    result = repl.execute("import re, json; json.dumps(re.findall(r'\\d+', 'a12'))", env)

    assert result == '["12"]'


def test_safe_builtins(repl):
    """Test safe built-in functions."""
    env = {}
    repl.execute("result = len([1, 2, 3])", env)
    assert env["result"] == 3


def test_comprehension(repl):
    """Test list comprehension."""
    env = {"context": "Hello World"}
    repl.execute("chars = [c for c in context if c.isupper()]", env)
    assert env["chars"] == ["H", "W"]


def test_empty_code(repl):
    """Test empty code."""
    env = {}
    result = repl.execute("", env)
    assert "No code" in result


def test_syntax_error(repl):
    """Test syntax error handling."""
    env = {}
    with pytest.raises(REPLError):
        repl.execute("x = ", env)


def test_runtime_error(repl):
    """Test runtime error handling."""
    env = {}
    with pytest.raises(REPLError):
        repl.execute("x = 1 / 0", env)


def test_final_expression_runs_once(repl):
    """Test that a stateful final expression is not evaluated twice."""
    env = {"items": [1, 2, 3]}

    result = repl.execute("items.pop()", env)

    assert result == "3"
    assert env["items"] == [1, 2]


def test_callback_runs_once(repl):
    """Test that a final callback expression has one side effect."""
    calls = []

    def callback(value):
        calls.append(value)
        return value.upper()

    env = {"callback": callback}
    result = repl.execute("callback('one')", env)

    assert result == "ONE"
    assert calls == ["one"]


def test_print_output_does_not_leak_between_steps(repl):
    """Test that every step receives a fresh print collector."""
    env = {}

    assert repl.execute("print('first')", env) == "first"
    assert repl.execute("x = 1", env) == "Code executed successfully (no output)"


def test_answer_object_publishes_final_result(repl):
    """Test the paper-compatible mutable answer object."""
    env = {"answer": {"content": "", "ready": False}}

    repl.execute("answer['content'] = 'done'; answer['ready'] = True", env)

    assert repl.pop_final_answer() == "done"
    assert repl.pop_final_answer() is None


def test_hard_timeout_terminates_and_recovers():
    """Test that non-terminating local code is killed and the executor restarts."""
    executor = REPLExecutor(timeout=0.5)
    try:
        executor.execute("1", {})
        started = time.monotonic()
        with pytest.raises(REPLTimeoutError, match="timed out"):
            executor.execute("while True: pass", {})
        assert time.monotonic() - started < 2
        assert executor.execute("6 * 7", {}) == "42"
    finally:
        executor.close()


def test_callback_time_is_not_charged_to_local_code_timeout():
    """Test that provider latency does not consume the Python execution budget."""
    executor = REPLExecutor(timeout=0.5)

    def slow_callback():
        time.sleep(0.7)
        return "finished"

    try:
        assert executor.execute("slow_callback()", {"slow_callback": slow_callback}) == "finished"
    finally:
        executor.close()


def test_worker_variable_lookup_and_show_vars(repl):
    """Test explicit access to persistent worker state."""
    repl.execute("meaning = 42", {})

    assert repl.get_variable("meaning") == (True, 42)
    assert repl.get_variable("missing") == (False, None)
    assert "meaning" in repl.execute("SHOW_VARS()", {})


def test_callback_errors_and_non_serializable_results(repl):
    """Test that callback boundary failures become ordinary REPL errors."""

    def raises():
        raise RuntimeError("callback failed")

    def returns_lambda():
        return lambda: None

    with pytest.raises(REPLError, match="callback failed"):
        repl.execute("raises()", {"raises": raises, "returns_lambda": returns_lambda})
    with pytest.raises(REPLError, match="non-serializable"):
        repl.execute("returns_lambda()", {"raises": raises, "returns_lambda": returns_lambda})


def test_output_truncation():
    """Test the bounded observation returned to the model."""
    executor = REPLExecutor(max_output_chars=10)
    try:
        result = executor.execute("'x' * 50", {})
        assert result.startswith("xxxxxxxxxx")
        assert "Output truncated: 50 chars total" in result
    finally:
        executor.close()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"timeout": 0}, "timeout"), ({"max_output_chars": 0}, "max_output_chars")],
)
def test_invalid_executor_configuration(kwargs, message):
    """Test executor configuration validation."""
    with pytest.raises(ValueError, match=message):
        REPLExecutor(**kwargs)


def test_executor_requires_a_running_worker_for_variable_access():
    """Test the explicit precondition for worker-only operations."""
    executor = REPLExecutor()
    with pytest.raises(REPLError, match="not running"):
        executor.get_variable("missing")


def test_helper_edge_cases():
    """Test serialization and incomplete Markdown extraction edge cases."""
    assert not _is_picklable(lambda: None)
    assert _extract_code("```\nvalue = 1\n```") == "value = 1"
    assert _extract_code("```python\nvalue = 1") == "```python\nvalue = 1"
