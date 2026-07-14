"""Parse standalone FINAL() and FINAL_VAR() protocol directives."""

import ast
import re
from typing import Optional, Dict, Any


def extract_final(response: str) -> Optional[str]:
    """
    Extract answer from FINAL() statement.

    Args:
        response: LLM response text

    Returns:
        Extracted answer or None if not found
    """
    match = re.fullmatch(r"\s*FINAL\s*\((.*)\)\s*", response, re.DOTALL)
    if not match:
        return None

    try:
        value = ast.literal_eval(match.group(1).strip())
    except (SyntaxError, ValueError):
        return None

    if isinstance(value, str):
        return value.strip()

    return None


def extract_final_var(response: str, env: Dict[str, Any]) -> Optional[str]:
    """
    Extract answer from FINAL_VAR() statement.

    Args:
        response: LLM response text
        env: REPL environment with variables

    Returns:
        Variable value as string or None if not found
    """
    # Look for FINAL_VAR(var_name)
    match = re.fullmatch(r"\s*FINAL_VAR\s*\(\s*(\w+)\s*\)\s*", response)
    if not match:
        return None

    var_name = match.group(1)

    # Get variable from environment
    if var_name in env:
        value = env[var_name]
        return str(value)

    return None


def extract_final_var_name(response: str) -> Optional[str]:
    """Return the variable name from a standalone FINAL_VAR directive."""
    match = re.fullmatch(r"\s*FINAL_VAR\s*\(\s*(\w+)\s*\)\s*", response)
    return match.group(1) if match else None


def is_final(response: str) -> bool:
    """
    Check if response is a standalone FINAL() or FINAL_VAR() directive.

    Args:
        response: LLM response text

    Returns:
        True if response contains final statement
    """
    return bool(
        re.fullmatch(r"\s*FINAL\s*\(.*\)\s*", response, re.DOTALL)
        or re.fullmatch(r"\s*FINAL_VAR\s*\(\s*\w+\s*\)\s*", response)
    )


def parse_response(response: str, env: Dict[str, Any]) -> Optional[str]:
    """
    Parse response for any final statement.

    Args:
        response: LLM response text
        env: REPL environment

    Returns:
        Final answer or None
    """
    # Try FINAL() first
    answer = extract_final(response)
    if answer is not None:
        return answer

    # Try FINAL_VAR()
    answer = extract_final_var(response, env)
    if answer is not None:
        return answer

    return None
