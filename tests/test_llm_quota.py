"""Tests for Gemini 429 classification, quota circuit breaking and hint-aware backoff.

These cover the free-tier behaviour added on top of the base retry design:

* a hard daily-quota 429 must fail fast (no retries) and open a circuit,
* a transient rate-limit 429 must be retried with backoff honouring the hint,
* the agent must mark quota-blocked records explicitly (never an offline fill).
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.agent.agent import IntakeAgent
from app.agent.llm import (
    BaseLLMProvider,
    GeminiProvider,
    LLMQuotaError,
    LLMRequest,
)
from app.models.unified_record import UnifiedRecord
from app.utils.retry import RetryPolicy, RetryStats, RetryableError, retry_async

_QUOTA_BODY = json.dumps(
    {
        "error": {
            "code": 429,
            "message": (
                "You exceeded your current quota, please check your plan and billing "
                "details. * Quota exceeded for metric: "
                "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
                "limit: 20, model: gemini-3.6-flash\nPlease retry in 4h55m2s."
            ),
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": "17702s",
                }
            ],
        }
    }
)

_RATE_BODY = json.dumps(
    {
        "error": {
            "code": 429,
            "message": "Rate limit exceeded. Please retry in 0h0m0.05s.",
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": "0.05s",
                }
            ],
        }
    }
)

_OK_BODY = json.dumps(
    {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]}
)


def _provider(handler: httpx.MockTransport | object, attempts: int = 3) -> GeminiProvider:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    client = httpx.AsyncClient(transport=transport)
    return GeminiProvider(
        api_key="test-key",
        model="gemini-3.6-flash",
        client=client,
        policy=RetryPolicy(attempts=attempts, base_delay=0.0, max_delay=0.0, jitter=0),
    )


_REQUEST = LLMRequest(system_prompt="system", user_prompt="user")


@pytest.mark.asyncio
async def test_daily_quota_429_fails_fast_without_retries() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, text=_QUOTA_BODY)

    provider = _provider(handler)
    with pytest.raises(LLMQuotaError) as exc_info:
        await provider.complete(_REQUEST)

    assert calls["n"] == 1  # no retries against a multi-hour daily quota
    assert exc_info.value.retry_after == pytest.approx(17702)
    assert exc_info.value.circuit_open is False
    await provider.aclose()


@pytest.mark.asyncio
async def test_quota_opens_circuit_and_fails_fast_without_http() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, text=_QUOTA_BODY)

    provider = _provider(handler)
    with pytest.raises(LLMQuotaError):
        await provider.complete(_REQUEST)
    assert calls["n"] == 1

    # Circuit is open: the second request must not touch the network at all.
    with pytest.raises(LLMQuotaError) as exc_info:
        await provider.complete(_REQUEST)
    assert exc_info.value.circuit_open is True
    assert calls["n"] == 1
    await provider.aclose()


@pytest.mark.asyncio
async def test_short_hint_429_is_retried_then_recovers() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text=_RATE_BODY)
        return httpx.Response(200, text=_OK_BODY)

    provider = _provider(handler)
    result = await provider.complete(_REQUEST)

    assert json.loads(result) == {"ok": True}
    assert calls["n"] == 2  # transient 429 was retried once and recovered
    await provider.aclose()


@pytest.mark.asyncio
async def test_agent_marks_quota_blocked_record(
    lead_record: UnifiedRecord,
) -> None:
    class _QuotaLLM(BaseLLMProvider):
        name = "gemini"

        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: LLMRequest) -> str:
            self.calls += 1
            raise LLMQuotaError("free-tier quota exhausted", retry_after=17702)

    stats = RetryStats()
    llm = _QuotaLLM()
    agent = IntakeAgent(llm, stats=stats)
    outcome = await agent.process(lead_record)

    assert not outcome.ok
    assert outcome.result is None
    assert outcome.quota_blocked is True
    assert "quota" in (outcome.error or "").lower()
    assert llm.calls == 1
    assert stats.failures.get("llm_quota") == 1


class _Hinted(RetryableError):
    def __init__(self, message: str, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class _Recorder:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.delays.append(seconds)


@pytest.mark.asyncio
async def test_retry_honours_server_retry_hint() -> None:
    recorder = _Recorder()
    policy = RetryPolicy(attempts=2, base_delay=0.1, max_delay=1.0, jitter=0)

    async def op() -> str:
        raise _Hinted("rate limited", retry_after=0.5)

    with pytest.raises(RetryableError):
        await retry_async(
            op, policy=policy, description="unit", sleep=recorder.sleep
        )

    # backoff would be 0.1s; the server hint (0.5s) wins.
    assert recorder.delays == [0.5]


@pytest.mark.asyncio
async def test_retry_hint_is_capped_by_max_server_delay() -> None:
    recorder = _Recorder()
    policy = RetryPolicy(attempts=2, base_delay=0.1, max_delay=1.0, jitter=0)

    async def op() -> str:
        raise _Hinted("rate limited", retry_after=10_000)

    with pytest.raises(RetryableError):
        await retry_async(
            op, policy=policy, description="unit", sleep=recorder.sleep
        )

    assert recorder.delays == [policy.max_server_delay]
