"""A failed or cancelled recursion tree must finish all work before returning."""

import asyncio
from types import SimpleNamespace

import pytest

from rlm import BudgetExceededError, RLM, RunBudget


def response(text, *, tokens=1, cost=0.0):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage={"total_tokens": tokens},
        _hidden_params={"response_cost": cost},
    )


@pytest.mark.parametrize(
    "kwargs,usage,metric",
    [
        ({"max_tokens": 10}, (11, 0.0), "total_tokens"),
        ({"max_cost_usd": 0.01}, (1, 0.02), "estimated_cost_usd"),
    ],
)
def test_exhausted_usage_budget_rejects_future_reservations(kwargs, usage, metric):
    budget = RunBudget(**kwargs)
    budget.reserve_call()
    with pytest.raises(BudgetExceededError):
        budget.record_usage(*usage)
    with pytest.raises(BudgetExceededError) as raised:
        budget.reserve_call()
    assert raised.value.metric == metric
    assert budget.snapshot()["reserved_calls"] == 1


@pytest.mark.asyncio
async def test_cancelled_run_closes_callbacks_scheduled_before_task_creation():
    state = RLM(model="test")._new_run_state(asyncio.get_running_loop())
    entered = []

    async def callback():
        entered.append(True)

    pending = state.submit_callback(callback())
    state.cancel_callbacks()
    rejected = state.submit_callback(callback())
    await state.finish_callbacks()
    assert pending.cancelled() and rejected.cancelled()
    assert not state._callbacks
    assert not entered


@pytest.mark.asyncio
@pytest.mark.parametrize("max_depth", [1, 2])
@pytest.mark.parametrize("metric", ["tokens", "cost"])
async def test_exhausted_batch_cancels_nested_work_and_freezes_diagnostics(max_depth, metric):
    slow_started = asyncio.Event()
    slow_cleaned_up = asyncio.Event()
    healthy_started = asyncio.Event()
    healthy_release = asyncio.Event()
    started = []

    async def complete(*, messages, **_kwargs):
        query = messages[1]["content"]
        is_leaf = messages[0]["content"].startswith("Answer the subproblem")
        if query == "healthy":
            healthy_started.set()
            await healthy_release.wait()
            return response('FINAL("healthy answer")')
        if query == "root":
            return response("result = rlm_query_batched(['over', 'slow', 'queued'], [context] * 3)")
        if not is_leaf:
            return response("answer['content'] = llm_query(query, context)\nanswer['ready'] = True")
        item = query.splitlines()[1]
        started.append(item)
        if item == "over":
            await slow_started.wait()
            return response("over budget", tokens=20, cost=0.02)
        assert item == "slow", "Queued provider work started after the budget was exhausted"
        slow_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            # Cancelling only the thread bridge does not await this provider cleanup.
            await asyncio.sleep(0.02)
            slow_cleaned_up.set()

    kwargs = {"max_total_tokens": 10} if metric == "tokens" else {"max_total_cost_usd": 0.01}
    rlm = RLM(
        model="test",
        completion_handler=complete,
        max_depth=max_depth,
        max_concurrent_subcalls=2,
        **kwargs,
    )
    healthy = asyncio.create_task(rlm.acomplete_result("healthy", "context"))
    try:
        await healthy_started.wait()
        result = await asyncio.wait_for(rlm.atry_complete_result("root", "context"), 40)
        assert result.error_type == "BudgetExceededError"
        assert sorted(started) == ["over", "slow"]
        assert slow_cleaned_up.is_set()
        assert not healthy.done()
        assert not rlm._last_run_state._callbacks
        await asyncio.sleep(0.05)
        assert result.stats["total_tokens"] == rlm.stats["total_tokens"]
        assert result.stats["llm_calls"] == rlm.stats["llm_calls"]
        assert result.trajectory == rlm.trajectory
        assert result.trajectory[-1].kind == "run_error"
        healthy_release.set()
        assert (await healthy).answer == "healthy answer"
    finally:
        healthy_release.set()
        healthy.cancel()
        await asyncio.gather(healthy, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelling_nested_batch_waits_for_provider_cleanup():
    leaf_started = asyncio.Event()
    leaf_cleaned_up = asyncio.Event()

    async def complete(*, messages, **_kwargs):
        if messages[0]["content"].startswith("Answer the subproblem"):
            leaf_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0.02)
                leaf_cleaned_up.set()
        query = messages[1]["content"]
        if query == "root":
            return response("rlm_query_batched(['child'], [context])")
        return response("llm_query_batched(['leaf'], [context])")

    rlm = RLM(model="test", completion_handler=complete, max_depth=2)
    task = asyncio.create_task(rlm.acomplete_result("root", "context"))
    try:
        await asyncio.wait_for(leaf_started.wait(), 40)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert leaf_cleaned_up.is_set()
        assert not rlm._last_run_state._callbacks
        trajectory = rlm.trajectory
        await asyncio.sleep(0.05)
        assert trajectory == rlm.trajectory
        assert trajectory[-1].kind == "run_error"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
