"""Tests for bounded retry / exponential backoff behaviour."""

from __future__ import annotations

import pytest

from app.utils.retry import (
    NonRetryableError,
    RetryPolicy,
    RetryStats,
    RetryableError,
    retry_async,
)


class _Recorder:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.delays.append(seconds)


def _policy(attempts: int = 3, base: float = 0.1, max_delay: float = 1.0) -> RetryPolicy:
    return RetryPolicy(attempts=attempts, base_delay=base, max_delay=max_delay, jitter=0)


@pytest.mark.asyncio
async def test_success_without_retry() -> None:
    stats = RetryStats()
    recorder = _Recorder()

    async def op() -> str:
        return "ok"

    result = await retry_async(op, policy=_policy(), description="unit", stats=stats,
                               sleep=recorder.sleep)
    assert result == "ok"
    assert stats.attempts == 0
    assert recorder.delays == []


@pytest.mark.asyncio
async def test_recovers_after_transient_failures() -> None:
    stats = RetryStats()
    recorder = _Recorder()
    calls = {"n": 0}

    async def op() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RetryableError("boom")
        return "recovered"

    result = await retry_async(op, policy=_policy(), description="unit", stats=stats,
                               sleep=recorder.sleep)
    assert result == "recovered"
    assert stats.attempts == 2
    assert stats.recoveries == 1
    assert stats.failures["unit"] == 2
    assert recorder.delays == [0.1, 0.2]  # exponential backoff without jitter


@pytest.mark.asyncio
async def test_gives_up_after_bounded_attempts() -> None:
    stats = RetryStats()
    recorder = _Recorder()

    async def op() -> str:
        raise RetryableError("always down")

    with pytest.raises(RetryableError, match="always down"):
        await retry_async(op, policy=_policy(attempts=3), description="unit",
                          stats=stats, sleep=recorder.sleep)

    assert stats.exhausted == 1
    assert stats.failures["unit"] == 3
    assert len(recorder.delays) == 2  # no sleep after the final attempt


@pytest.mark.asyncio
async def test_non_retryable_fails_fast() -> None:
    stats = RetryStats()
    recorder = _Recorder()
    calls = {"n": 0}

    async def op() -> str:
        calls["n"] += 1
        raise NonRetryableError("validation error")

    with pytest.raises(NonRetryableError):
        await retry_async(op, policy=_policy(attempts=5), description="unit",
                          stats=stats, sleep=recorder.sleep)

    assert calls["n"] == 1  # not retried
    assert stats.attempts == 0
    assert recorder.delays == []


@pytest.mark.asyncio
async def test_delay_is_capped() -> None:
    recorder = _Recorder()
    policy = RetryPolicy(attempts=6, base_delay=1.0, max_delay=4.0, jitter=0)

    async def op() -> str:
        raise RetryableError("down")

    with pytest.raises(RetryableError):
        await retry_async(op, policy=policy, description="unit", sleep=recorder.sleep)

    assert max(recorder.delays) <= 4.0
    assert recorder.delays[0] == 1.0


def test_retry_policy_delay_growth() -> None:
    policy = RetryPolicy(attempts=5, base_delay=1.0, max_delay=8.0, jitter=0)
    assert [policy.delay_for(i) for i in (1, 2, 3, 4)] == [1.0, 2.0, 4.0, 8.0]
