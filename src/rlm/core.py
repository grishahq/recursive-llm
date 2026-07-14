"""Core Recursive Language Model implementation."""

from __future__ import annotations

import asyncio
import concurrent.futures
import re
from typing import Any, Callable, Coroutine, Dict, List, Optional, Sequence, TypeVar, cast

import litellm

from .parser import extract_final, extract_final_var_name
from .prompts import build_system_prompt
from .repl import REPLError, REPLExecutor
from .stats import UsageTracker
from .types import Message


class RLMError(Exception):
    """Base error for RLM."""


class MaxIterationsError(RLMError):
    """Maximum root or child RLM iterations exceeded."""


class MaxDepthError(RLMError):
    """An invalid internal RLM depth was requested."""


T = TypeVar("T")


def _run_sync(awaitable: Coroutine[Any, Any, T]) -> T:
    """Run an awaitable from synchronous code, including inside a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(awaitable)).result()


class RLM:
    """Recursive Language Model with paper-aligned depth semantics."""

    def __init__(
        self,
        model: str,
        recursive_model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_depth: int = 1,
        max_iterations: int = 30,
        repl_timeout: float = 5,
        max_output_chars: int = 2000,
        max_concurrent_subcalls: int = 4,
        _current_depth: int = 0,
        _usage_tracker: Optional[UsageTracker] = None,
        **llm_kwargs: Any,
    ) -> None:
        """Initialize an RLM.

        ``max_depth`` describes available subcall capability, not the number of
        RLM objects. At depth 0 the root has a REPL but no subcalls. At depth 1
        it can call a plain LM. At depth 2 it can create one child RLM, whose
        boundary falls back to a plain LM call.
        """
        if max_depth < 0:
            raise ValueError("max_depth must be zero or greater")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be greater than zero")
        if repl_timeout <= 0:
            raise ValueError("repl_timeout must be greater than zero")
        if max_output_chars <= 0:
            raise ValueError("max_output_chars must be greater than zero")
        if max_concurrent_subcalls <= 0:
            raise ValueError("max_concurrent_subcalls must be greater than zero")
        if _current_depth < 0:
            raise ValueError("_current_depth must be zero or greater")

        self.model = model
        self.recursive_model = recursive_model or model
        self.api_base = api_base
        self.api_key = api_key
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.repl_timeout = repl_timeout
        self.max_output_chars = max_output_chars
        self.max_concurrent_subcalls = max_concurrent_subcalls
        self._current_depth = _current_depth
        self._usage_tracker = _usage_tracker or UsageTracker()
        self.llm_kwargs = llm_kwargs
        self._active_loop: Optional[asyncio.AbstractEventLoop] = None

        self._llm_calls = 0
        self._iterations = 0

    def complete(self, query: str = "", context: str = "", **kwargs: Any) -> str:
        """Synchronously complete a query over an external context."""
        return _run_sync(self.acomplete(query, context, **kwargs))

    async def acomplete(self, query: str = "", context: str = "", **kwargs: Any) -> str:
        """Complete a query through the root or a child RLM loop."""
        if query and not context:
            context = query
            query = ""

        if self._current_depth > 0 and self._current_depth >= self.max_depth:
            raise MaxDepthError(
                f"RLM depth {self._current_depth} is not available with max_depth={self.max_depth}"
            )

        if self._current_depth == 0:
            self._usage_tracker = UsageTracker()
            self._llm_calls = 0
            self._iterations = 0

        self._active_loop = asyncio.get_running_loop()
        repl_env = self._build_repl_env(query, context)
        repl = REPLExecutor(timeout=self.repl_timeout, max_output_chars=self.max_output_chars)
        system_prompt = build_system_prompt(
            len(context),
            depth=self._current_depth,
            max_depth=self.max_depth,
        )
        messages: List[Message] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        try:
            for iteration in range(self.max_iterations):
                self._iterations = iteration + 1
                self._usage_tracker.record_iteration()
                response = await self._call_llm(messages, **kwargs)

                if not response.strip():
                    messages.append({"role": "assistant", "content": response})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your response was empty. Return one short executable Python step "
                                "or a standalone final directive."
                            ),
                        }
                    )
                    continue

                direct_answer = extract_final(response)
                if direct_answer is not None:
                    return direct_answer

                final_var_name = extract_final_var_name(response)
                if final_var_name is not None:
                    try:
                        found, value = await asyncio.to_thread(
                            repl.get_variable, final_var_name
                        )
                    except REPLError:
                        if final_var_name in repl_env:
                            return str(repl_env[final_var_name])
                    else:
                        if found:
                            return str(value)
                    messages.append({"role": "assistant", "content": response})
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Error: variable {final_var_name!r} was not found",
                        }
                    )
                    continue

                try:
                    exec_result = await asyncio.to_thread(repl.execute, response, repl_env)
                    published_answer = repl.pop_final_answer()
                    if published_answer is not None:
                        return published_answer
                except REPLError as exc:
                    exec_result = f"Error: {exc}"
                except Exception as exc:
                    exec_result = f"Unexpected error: {exc}"

                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": exec_result})
        finally:
            await asyncio.to_thread(repl.close)
            self._active_loop = None

        raise MaxIterationsError(
            f"Max iterations ({self.max_iterations}) exceeded without a final answer"
        )

    async def _call_llm(self, messages: List[Message], **kwargs: Any) -> str:
        """Call the root or child RLM model and record its usage."""
        self._llm_calls += 1
        default_model = self.model if self._current_depth == 0 else self.recursive_model
        model = cast(str, kwargs.get("model", default_model))
        call_overrides = dict(kwargs)
        call_overrides.pop("model", None)
        self._usage_tracker.record_call(model, self._current_depth)
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            **self._completion_kwargs(call_overrides),
        )
        self._record_response(model, response)
        return self._response_text(response)

    async def _call_leaf(
        self,
        sub_query: str,
        sub_context: str = "",
        model: Optional[str] = None,
    ) -> str:
        """Call a plain LM without creating another REPL loop."""
        selected_model = model or self.recursive_model
        user_content = sub_query
        if sub_context:
            user_content = f"Task:\n{sub_query}\n\nContext:\n{sub_context}"
        messages: List[Message] = [
            {
                "role": "system",
                "content": (
                    "Answer the subproblem using only the supplied context. "
                    "Return the answer directly and do not emit REPL code or FINAL directives."
                ),
            },
            {"role": "user", "content": user_content},
        ]
        call_depth = self._current_depth + 1
        self._usage_tracker.record_call(selected_model, call_depth, is_leaf=True)
        response = await litellm.acompletion(
            model=selected_model,
            messages=messages,
            **self._completion_kwargs({}),
        )
        self._record_response(selected_model, response)
        return self._response_text(response)

    def _completion_kwargs(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Merge common provider arguments for one LiteLLM request."""
        call_kwargs = {**self.llm_kwargs, **overrides}
        if self.api_base:
            call_kwargs["api_base"] = self.api_base
        if self.api_key:
            call_kwargs["api_key"] = self.api_key
        return call_kwargs

    def _record_response(self, model: str, response: Any) -> None:
        self._usage_tracker.record_response(model, response, self._get_response_cost(response))

    @staticmethod
    def _response_text(response: Any) -> str:
        content = response.choices[0].message.content
        if content is None:
            raise RLMError("LLM response did not contain text content")
        return cast(str, content)

    @staticmethod
    def _get_response_cost(response: Any) -> Optional[float]:
        """Return LiteLLM's best-effort response cost without affecting completion."""
        hidden_params = getattr(response, "_hidden_params", None)
        if isinstance(hidden_params, dict):
            response_cost = hidden_params.get("response_cost")
            if isinstance(response_cost, (int, float)):
                return float(response_cost)
        try:
            response_cost = litellm.completion_cost(completion_response=response)
        except Exception:
            return None
        if isinstance(response_cost, (int, float)):
            return float(response_cost)
        return None

    def _build_repl_env(self, query: str, context: str) -> Dict[str, Any]:
        """Build the names exposed to restricted Python code."""
        env: Dict[str, Any] = {
            "context": context,
            "query": query,
            "answer": {"content": "", "ready": False},
            "re": re,
        }
        if self.max_depth == 0:
            return env

        llm_query = self._make_llm_query()
        rlm_query = self._make_rlm_query()
        env.update(
            {
                "llm_query": llm_query,
                "rlm_query": rlm_query,
                "recursive_llm": rlm_query,
                "llm_query_batched": self._make_batched_query(llm_query),
                "rlm_query_batched": self._make_batched_query(rlm_query),
            }
        )
        return env

    def _make_llm_query(self) -> Callable[..., str]:
        """Create the direct plain-LM function exposed in the REPL."""

        def llm_query(
            sub_query: str,
            sub_context: str = "",
            model: Optional[str] = None,
        ) -> str:
            return self._run_callback(self._call_leaf(sub_query, sub_context, model))

        return llm_query

    def _make_rlm_query(self) -> Callable[[str, str], str]:
        """Create the recursive function with a plain-LM boundary fallback."""

        async def call(sub_query: str, sub_context: str = "") -> str:
            if self._current_depth + 1 >= self.max_depth:
                return await self._call_leaf(sub_query, sub_context)

            child = RLM(
                model=self.recursive_model,
                recursive_model=self.recursive_model,
                api_base=self.api_base,
                api_key=self.api_key,
                max_depth=self.max_depth,
                max_iterations=self.max_iterations,
                repl_timeout=self.repl_timeout,
                max_output_chars=self.max_output_chars,
                max_concurrent_subcalls=self.max_concurrent_subcalls,
                _current_depth=self._current_depth + 1,
                _usage_tracker=self._usage_tracker,
                **self.llm_kwargs,
            )
            return await child.acomplete(sub_query, sub_context)

        def rlm_query(sub_query: str, sub_context: str = "") -> str:
            return self._run_callback(call(sub_query, sub_context))

        return rlm_query

    def _make_batched_query(
        self,
        query_fn: Callable[[str, str], str],
    ) -> Callable[[Sequence[str], Optional[Sequence[str]]], List[str]]:
        """Create an ordered, bounded-concurrency batch wrapper."""

        async def run_batch(
            queries: Sequence[str],
            contexts: Optional[Sequence[str]],
        ) -> List[str]:
            query_list = list(queries)
            context_list = list(contexts) if contexts is not None else [""] * len(query_list)
            if len(query_list) != len(context_list):
                raise ValueError("queries and contexts must have the same length")
            semaphore = asyncio.Semaphore(self.max_concurrent_subcalls)

            async def run_one(item_query: str, item_context: str) -> str:
                async with semaphore:
                    try:
                        return await asyncio.to_thread(query_fn, item_query, item_context)
                    except Exception as exc:
                        return f"Error: {exc}"

            return await asyncio.gather(
                *(run_one(item_query, item_context) for item_query, item_context in zip(
                    query_list, context_list
                ))
            )

        def batched(
            queries: Sequence[str],
            contexts: Optional[Sequence[str]] = None,
        ) -> List[str]:
            return self._run_callback(run_batch(queries, contexts))

        return batched

    def _run_callback(self, awaitable: Coroutine[Any, Any, T]) -> T:
        """Run a REPL callback on its owning completion loop when available."""
        loop = self._active_loop
        if loop is not None and loop.is_running():
            return asyncio.run_coroutine_threadsafe(awaitable, loop).result()
        return _run_sync(awaitable)

    @property
    def stats(self) -> Dict[str, Any]:
        """Return aggregate statistics for the latest full recursion tree."""
        stats = self._usage_tracker.snapshot()
        stats["iterations"] = self._iterations
        stats["depth"] = self._current_depth
        return stats
