"""Per-invocation state shared within one RLM recursion tree."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import Future, InvalidStateError
from copy import deepcopy
from threading import Lock
from typing import Any, Callable, Coroutine, Dict, Optional, Tuple, TypeVar

from .budget import RunBudget
from .results import TrajectoryEvent
from .stats import UsageTracker

T = TypeVar("T")


class RunState:
    """Own usage, budget, loop, and iteration data for one completion tree."""

    def __init__(
        self,
        usage: UsageTracker,
        budget: RunBudget,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        event_handler: Optional[Callable[[TrajectoryEvent], None]] = None,
    ) -> None:
        self.usage = usage
        self.budget = budget
        self._loop = loop
        self._iterations_by_depth: Dict[int, int] = {}
        self._started_at = time.monotonic()
        self._next_node_number = 0
        self._events: list[TrajectoryEvent] = []
        self._event_handler = event_handler
        self._lock = Lock()
        self._callbacks: Dict[Future[Any], Optional[asyncio.Task[Any]]] = {}
        self._cancelled = False

    def submit_callback(self, awaitable: Coroutine[Any, Any, T]) -> Future[T]:
        """Own both the thread bridge and its task until async cleanup finishes."""
        loop = self.loop
        if loop is None:
            awaitable.close()
            raise RuntimeError("REPL callback requires an owning event loop")
        future: Future[T] = Future()
        with self._lock:
            if self._cancelled:
                awaitable.close()
                future.cancel()
                return future
            # Track scheduled callbacks too: cancellation can precede task creation.
            self._callbacks[future] = None

        def finished(task: asyncio.Task[T]) -> None:
            try:
                value = task.result()
            except asyncio.CancelledError:
                future.cancel()
            except BaseException as exc:
                try:
                    future.set_exception(exc)
                except InvalidStateError:
                    pass
            else:
                try:
                    future.set_result(value)
                except InvalidStateError:
                    pass
            finally:
                with self._lock:
                    self._callbacks.pop(future, None)

        def start() -> None:
            with self._lock:
                cancelled = self._cancelled
                if cancelled:
                    self._callbacks.pop(future, None)
            if cancelled:
                awaitable.close()
                future.cancel()
                return
            # Task factories may start the coroutine synchronously. Never hold
            # the state lock while callback code can re-enter this run state.
            task = loop.create_task(awaitable)
            with self._lock:
                self._callbacks[future] = task
                cancelled = self._cancelled
            task.add_done_callback(finished)
            if cancelled:
                task.cancel()

        loop.call_soon_threadsafe(start)
        return future

    def cancel_callbacks(self) -> None:
        """Wake REPL threads awaiting child calls belonging to this run only."""
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            callbacks = tuple(self._callbacks.items())
        for future, task in callbacks:
            future.cancel()
            if task is not None:
                task.get_loop().call_soon_threadsafe(task.cancel)

    async def finish_callbacks(self) -> None:
        """Cancel and drain all run-owned work before freezing its result."""
        self.cancel_callbacks()
        while True:
            with self._lock:
                if not self._callbacks:
                    return
                tasks = [task for task in self._callbacks.values() if task is not None]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            # Let pending submissions and task done callbacks release ownership.
            await asyncio.sleep(0)

    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """Return the event loop that owns this invocation."""
        with self._lock:
            return self._loop

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Attach a loop when protected call helpers are used directly."""
        with self._lock:
            if self._loop is None:
                self._loop = loop

    def record_iteration(self, depth: int) -> None:
        """Record one iteration globally and for a specific RLM depth."""
        self.usage.record_iteration()
        with self._lock:
            self._iterations_by_depth[depth] = self._iterations_by_depth.get(depth, 0) + 1

    def iterations_at(self, depth: int) -> int:
        """Return loop iterations performed by one RLM depth."""
        with self._lock:
            return self._iterations_by_depth.get(depth, 0)

    def next_node_id(self, prefix: str) -> str:
        """Return a deterministic unique node identifier for this run."""
        with self._lock:
            self._next_node_number += 1
            return f"{prefix}-{self._next_node_number}"

    def record_event(
        self,
        kind: str,
        depth: int,
        node_id: str,
        parent_id: str = "",
        **data: Any,
    ) -> TrajectoryEvent:
        """Append an event and notify the optional best-effort handler."""
        with self._lock:
            event = TrajectoryEvent(
                sequence=len(self._events) + 1,
                kind=kind,
                elapsed_seconds=round(time.monotonic() - self._started_at, 6),
                depth=depth,
                node_id=node_id,
                parent_id=parent_id,
                data=deepcopy(data),
            )
            self._events.append(event)
        if self._event_handler is not None:
            try:
                self._event_handler(event)
            except Exception:
                pass
        return event

    def trajectory(self) -> Tuple[TrajectoryEvent, ...]:
        """Return a detached immutable snapshot of all events so far."""
        with self._lock:
            return tuple(
                TrajectoryEvent(
                    sequence=event.sequence,
                    kind=event.kind,
                    elapsed_seconds=event.elapsed_seconds,
                    depth=event.depth,
                    node_id=event.node_id,
                    parent_id=event.parent_id,
                    data=deepcopy(event.data),
                )
                for event in self._events
            )
