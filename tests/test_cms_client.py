"""Tests for the downstream CMS HTTP client (retry + failure handling)."""

from __future__ import annotations

import copy

import httpx
import pytest

from app.cms.client import CMSClient, CMSDeliveryError
from app.models.agent_result import AgentResult, validate_agent_result
from app.models.unified_record import RecordType, SourceType
from app.utils.retry import RetryPolicy, RetryStats
from mock_cms.main import app

VALID_PAYLOAD = {
    "record_id": "test-client-001",
    "source": "linkedin",
    "record_type": "lead",
    "classification": "WARM",
    "priority": "MEDIUM",
    "action": "FOLLOW_UP",
    "channel": "linkedin",
    "tone": "professional",
    "message": "Following up on your recent activity.",
    "rationale": "Warm because the lead recently engaged with relevant content.",
}


def _result(**overrides: object) -> AgentResult:
    payload = copy.deepcopy(VALID_PAYLOAD)
    payload.update(overrides)
    return validate_agent_result(
        payload["record_id"], SourceType.LINKEDIN, RecordType.LEAD, payload
    )


def _client(**kwargs) -> CMSClient:
    transport = httpx.ASGITransport(app=app)
    async_client = httpx.AsyncClient(transport=transport, base_url="http://cms.test")
    policy = RetryPolicy(attempts=3, base_delay=0.0, max_delay=0.0, jitter=0)
    return CMSClient("http://cms.test", policy=policy, client=async_client, **kwargs)


@pytest.mark.asyncio
async def test_deliver_success() -> None:
    stats = RetryStats()
    async with _client(stats=stats) as client:
        body = await client.deliver(_result(record_id="test-client-ok"))

    assert body["status"] == "stored"
    assert body["record_id"] == "test-client-ok"
    assert stats.exhausted == 0


@pytest.mark.asyncio
async def test_deliver_retries_transient_503_then_succeeds() -> None:
    stats = RetryStats()
    async with _client(stats=stats) as client:
        body = await client.deliver(
            _result(record_id="test-client-retry"), simulate="once"
        )

    assert body["status"] == "stored"
    assert stats.recoveries == 1  # recovered on retry


@pytest.mark.asyncio
async def test_deliver_persistent_failure_raises() -> None:
    stats = RetryStats()
    async with _client(stats=stats) as client:
        with pytest.raises(CMSDeliveryError, match="after 3 attempt"):
            await client.deliver(
                _result(record_id="test-client-down"), simulate="always"
            )

    assert stats.exhausted == 1


@pytest.mark.asyncio
async def test_deliver_rejects_bad_payload() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(422, json={"detail": "invalid"})
    )
    async_client = httpx.AsyncClient(transport=transport, base_url="http://cms.test")
    policy = RetryPolicy(attempts=3, base_delay=0.0, jitter=0)
    client = CMSClient("http://cms.test", policy=policy, client=async_client)

    with pytest.raises(CMSDeliveryError, match="422"):
        await client.deliver(_result(record_id="test-client-422"))

    await client.aclose()


@pytest.mark.asyncio
async def test_health_reports_unreachable_cms() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient(transport=transport, base_url="http://cms.test")
    client = CMSClient("http://cms.test", client=async_client)

    with pytest.raises(CMSDeliveryError, match="unreachable"):
        await client.health()

    await client.aclose()


@pytest.mark.asyncio
async def test_health_success() -> None:
    async with _client() as client:
        body = await client.health()
    assert body["status"] == "ok"
