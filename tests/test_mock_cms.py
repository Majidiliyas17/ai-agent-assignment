"""Tests for the mock CMS endpoints (FastAPI + SQLite)."""

from __future__ import annotations

import copy
from datetime import datetime, timezone

import httpx
import pytest
from fastapi import FastAPI

from mock_cms.main import app

VALID_PAYLOAD = {
    "record_id": "test-cms-001",
    "source": "linkedin",
    "record_type": "lead",
    "classification": "HOT",
    "priority": "HIGH",
    "action": "SEND_OUTREACH",
    "channel": "linkedin",
    "tone": "professional",
    "message": "Hi Rebecca, following up on your interest in clinic automation.",
    "rationale": "High priority because the lead requested a demo.",
    "processed_at": datetime.now(timezone.utc).isoformat(),
    "model_provider": "gemini",
}


@pytest.fixture
def client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.fixture
def payload() -> dict:
    return copy.deepcopy(VALID_PAYLOAD)


def test_app_is_fastapi() -> None:
    assert isinstance(app, FastAPI)


@pytest.mark.asyncio
async def test_health(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "mock-cms"
    assert isinstance(body["record_count"], int)


@pytest.mark.asyncio
async def test_root_lists_endpoints(client: httpx.AsyncClient) -> None:
    response = await client.get("/")
    assert response.status_code == 200
    assert "/records" in response.json()["endpoints"]


@pytest.mark.asyncio
async def test_post_record_and_fetch_it_back(
    client: httpx.AsyncClient, payload: dict
) -> None:
    created = await client.post("/records", json=payload)
    assert created.status_code == 201
    assert created.json() == {
        "status": "stored",
        "record_id": "test-cms-001",
        "created": created.json()["created"],
    }

    fetched = await client.get("/records/test-cms-001")
    assert fetched.status_code == 200
    body = fetched.json()
    assert body["record_id"] == "test-cms-001"
    assert body["classification"] == "HOT"
    assert body["action"] == "SEND_OUTREACH"
    assert body["rationale"]


@pytest.mark.asyncio
async def test_post_is_idempotent(
    client: httpx.AsyncClient, payload: dict
) -> None:
    payload["record_id"] = "test-cms-idempotent-001"
    first = await client.post("/records", json=payload)
    second = await client.post("/records", json=payload)
    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["created"] is True
    assert second.json()["created"] is False  # upsert, not duplicate


@pytest.mark.asyncio
async def test_list_records(client: httpx.AsyncClient, payload: dict) -> None:
    payload["record_id"] = "test-cms-list-001"
    payload["classification"] = "COLD"
    await client.post("/records", json=payload)

    response = await client.get("/records", params={"classification": "COLD"})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] >= 1
    assert all(r["classification"] == "COLD" for r in body["records"])


@pytest.mark.asyncio
async def test_get_missing_record_returns_404(client: httpx.AsyncClient) -> None:
    response = await client.get("/records/does-not-exist")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_invalid_payload_returns_422(
    client: httpx.AsyncClient, payload: dict
) -> None:
    payload["classification"] = "VERY_HOT"
    response = await client.post("/records", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_missing_field_returns_422(
    client: httpx.AsyncClient, payload: dict
) -> None:
    payload.pop("rationale")
    response = await client.post("/records", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_extra_field_returns_422(
    client: httpx.AsyncClient, payload: dict
) -> None:
    payload["unexpected"] = "nope"
    response = await client.post("/records", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_simulate_error_always_returns_503(
    client: httpx.AsyncClient, payload: dict
) -> None:
    payload["record_id"] = "test-cms-always-001"
    response = await client.post(
        "/records", json=payload, headers={"X-Simulate-Error": "always"}
    )
    assert response.status_code == 503
    assert "Simulated CMS outage" in response.json()["detail"]


@pytest.mark.asyncio
async def test_simulate_error_once_then_recovers(
    client: httpx.AsyncClient, payload: dict
) -> None:
    payload["record_id"] = "test-cms-once-001"
    headers = {"X-Simulate-Error": "once"}

    first = await client.post("/records", json=payload, headers=headers)
    second = await client.post("/records", json=payload, headers=headers)

    assert first.status_code == 503
    assert second.status_code == 201


@pytest.mark.asyncio
async def test_openapi_document_available(client: httpx.AsyncClient) -> None:
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "/records" in schema["paths"]
    assert "/health" in schema["paths"]
    assert schema["info"]["title"] == "Mock CMS API"
