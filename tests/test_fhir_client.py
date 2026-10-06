"""Tests for the FHIR client: retries, error mapping and resource validation."""

from __future__ import annotations

import json

import httpx
import pytest

from app.ingestion.fhir_client import FHIRClient, FHIRClientError
from app.utils.retry import NonRetryableError, RetryPolicy, RetryStats

BASE = "https://r4.smarthealthit.org"


def _patient(index: int) -> dict:
    return {
        "resourceType": "Patient",
        "id": f"p{index:03d}",
        "name": [{"given": ["Pat"], "family": f"Test{index}"}],
    }


def _bundle(resources: list[dict], next_url: str | None = None) -> dict:
    links = [{"relation": "next", "url": next_url}] if next_url else []
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "link": links,
        "entry": [{"resource": resource} for resource in resources],
    }


def _client(handler, **kwargs) -> FHIRClient:
    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient(transport=transport, base_url=BASE)
    policy = RetryPolicy(attempts=3, base_delay=0.0, max_delay=0.0, jitter=0)
    return FHIRClient(BASE, policy=policy, client=async_client, **kwargs)


@pytest.mark.asyncio
async def test_fetches_requested_patients() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/Patient"
        return httpx.Response(200, json=_bundle([_patient(i) for i in range(15)]))

    stats = RetryStats()
    async with _client(handler, stats=stats) as client:
        patients = await client.fetch_patients(15)

    assert len(patients) == 15
    assert patients[0]["id"] == "p000"
    assert stats.attempts == 0


@pytest.mark.asyncio
async def test_retries_on_500_then_recovers() -> None:
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] < 3:
            return httpx.Response(503, text="service unavailable")
        return httpx.Response(200, json=_bundle([_patient(1)]))

    stats = RetryStats()
    async with _client(handler, stats=stats) as client:
        patients = await client.fetch_patients(1)

    assert len(patients) == 1
    assert state["calls"] == 3
    assert stats.attempts == 2
    assert stats.recoveries == 1


@pytest.mark.asyncio
async def test_404_is_not_retried() -> None:
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        return httpx.Response(404, text="not found")

    stats = RetryStats()
    async with _client(handler, stats=stats) as client:
        with pytest.raises(NonRetryableError, match="404"):
            await client.fetch_patients(1)

    assert state["calls"] == 1
    assert stats.attempts == 0


@pytest.mark.asyncio
async def test_malformed_json_gives_up_after_retries() -> None:
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        return httpx.Response(200, text="{ this is not json")

    stats = RetryStats()
    async with _client(handler, stats=stats) as client:
        with pytest.raises(Exception):
            await client.fetch_patients(1)

    assert state["calls"] == 3  # bounded retries, then failure
    assert stats.exhausted == 1


@pytest.mark.asyncio
async def test_simulated_transient_failure_recovers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_bundle([_patient(1)]))

    stats = RetryStats()
    async with _client(handler, stats=stats, simulated_failures=1) as client:
        patients = await client.fetch_patients(1)

    assert len(patients) == 1
    assert stats.attempts == 1
    assert stats.recoveries == 1


@pytest.mark.asyncio
async def test_malformed_resources_are_skipped_not_fatal() -> None:
    resources = [
        {"resourceType": "Patient", "id": "good-1"},
        "not-a-dict",
        {"resourceType": "Observation", "id": "wrong"},
        {"resourceType": "Patient"},  # missing id
        {"resourceType": "Patient", "id": "good-2"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_bundle(resources))

    async with _client(handler) as client:
        patients = await client.fetch_patients(10)

    assert [p["id"] for p in patients] == ["good-1", "good-2"]


@pytest.mark.asyncio
async def test_pagination_follows_next_link() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("page") == "2":
            return httpx.Response(200, json=_bundle([_patient(2)]))
        return httpx.Response(
            200,
            json=_bundle([_patient(1)], next_url=f"{BASE}/Patient?_count=1&page=2"),
        )

    async with _client(handler) as client:
        patients = await client.fetch_patients(2)

    assert [p["id"] for p in patients] == ["p001", "p002"]


@pytest.mark.asyncio
async def test_empty_bundle_raises_fhir_client_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"resourceType": "Bundle", "entry": []})

    async with _client(handler) as client:
        with pytest.raises(FHIRClientError, match="No usable Patient"):
            await client.fetch_patients(5)


@pytest.mark.asyncio
async def test_connection_error_is_retried() -> None:
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] == 1:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, json=_bundle([_patient(1)]))

    stats = RetryStats()
    async with _client(handler, stats=stats) as client:
        patients = await client.fetch_patients(1)

    assert len(patients) == 1
    assert stats.recoveries == 1


@pytest.mark.asyncio
async def test_fetch_conditions_returns_display_labels() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/Condition":
            bundle = _bundle(
                [{"resourceType": "Condition", "code": {"text": "Asthma"}}]
            )
            return httpx.Response(200, json=bundle)
        return httpx.Response(200, json=_bundle([_patient(1)]))

    async with _client(handler) as client:
        conditions = await client.fetch_conditions("p000")

    assert conditions[0]["code"]["text"] == "Asthma"
    assert json.dumps(conditions)  # serializable
