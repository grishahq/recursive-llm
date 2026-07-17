"""Type definitions for RLM."""

from typing import Any, Callable, Dict, List, Optional, TypedDict

FinalAnswerValidator = Callable[[str], Optional[str]]


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
    repl_memory_limit_mb: Optional[int]
    repl_cpu_time_limit_seconds: Optional[int]
    repl_max_open_files: Optional[int]
    max_concurrent_subcalls: int
    max_total_calls: Optional[int]
    max_total_tokens: Optional[int]
    max_total_cost_usd: Optional[float]
    max_elapsed_seconds: Optional[float]
    max_retries: int
    retry_backoff_seconds: float
    final_answer_validator: Optional[FinalAnswerValidator]
    capture_trajectory_content: bool
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
