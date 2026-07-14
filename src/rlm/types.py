"""Type definitions for RLM."""

from typing import Any, Callable, Dict, List, Optional, TypedDict


class Message(TypedDict):
    """LLM message format."""

    role: str
    content: str


class RLMConfig(TypedDict, total=False):
    """Configuration for RLM instance."""

    model: str
    recursive_model: Optional[str]
    api_base: Optional[str]
    api_key: Optional[str]
    max_depth: int
    max_iterations: int
    repl_timeout: float
    max_output_chars: int
    max_concurrent_subcalls: int
    temperature: float
    timeout: int


class REPLEnvironment(TypedDict, total=False):
    """REPL execution environment."""

    context: str
    query: str
    answer: Dict[str, Any]
    llm_query: Callable[[str, str], str]
    rlm_query: Callable[[str, str], str]
    recursive_llm: Callable[[str, str], str]
    llm_query_batched: Callable[[List[str], Optional[List[str]]], List[str]]
    rlm_query_batched: Callable[[List[str], Optional[List[str]]], List[str]]
    re: Any


class CompletionResult(TypedDict):
    """Result from RLM completion."""

    answer: str
    iterations: int
    depth: int
    llm_calls: int
