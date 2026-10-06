"""Mock CMS service.

A deliberately small FastAPI service that behaves like the downstream CMS the
agent pipeline delivers to:

* ``GET  /health``            liveness + record count
* ``POST /records``           accept a structured AgentResult-like payload
* ``GET  /records``           list stored records (filter/limit)
* ``GET  /records/{id}`       fetch one record

Payloads are validated with Pydantic and persisted in SQLite.

Controlled failure simulation (used by the pipeline's ``--demo-failure`` mode
and by tests):

    X-Simulate-Error: once   -> 503 for the FIRST request carrying a given
                                record_id, then succeeds (exercises retry)
    X-Simulate-Error: always -> 503 for every request (exercises exhaustion)

Run with::

    uvicorn mock_cms.main:app --reload --port 8000
"""

from __future__ import annotations

import os
import threading

from fastapi import FastAPI, Header, HTTPException, Query

from mock_cms.db import RecordStore
from mock_cms.schemas import (
    CMSRecord,
    CMSRecordCreate,
    CMSRecordList,
    CMSRecordResponse,
    HealthResponse,
)

DB_PATH = os.getenv("CMS_DB_PATH", "./output/mock_cms.sqlite3")

store = RecordStore(DB_PATH)
_failed_once: set[str] = set()
_sim_lock = threading.Lock()

app = FastAPI(
    title="Mock CMS API",
    version="1.0.0",
    description=(
        "Downstream content-management endpoint for the AI intake pipeline. "
        "It accepts structured `AgentResult` payloads (JSON, never free text), "
        "validates them with Pydantic and stores them in SQLite.\n\n"
        "Use `X-Simulate-Error: once|always` on `POST /records` to simulate a "
        "transient or persistent downstream outage for retry/failure demos."
    ),
    contact={"name": "AI Intake Assignment", "url": "https://example.invalid"},
    license_info={"name": "Internal demo only"},
)


@app.get("/", tags=["meta"], summary="Service root")
def root() -> dict:
    """Basic service metadata and a pointer to the OpenAPI docs."""
    return {
        "service": "mock-cms",
        "status": "ok",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "endpoints": ["/health", "/records", "/records/{record_id}"],
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["meta"],
    summary="Health check",
    description="Liveness probe. Returns the number of records currently stored.",
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        record_count=store.count(),
        database=DB_PATH,
    )


@app.post(
    "/records",
    response_model=CMSRecordResponse,
    status_code=201,
    tags=["records"],
    summary="Store a processed record",
    description=(
        "Accepts a structured agent decision (classification, action, channel, "
        "tone, message, rationale, ...). The payload is validated, then "
        "upserted by `record_id`, making delivery idempotent.\n\n"
        "Optional header `X-Simulate-Error: once|always` forces a 503 response "
        "for failure/retry demonstrations."
    ),
    responses={
        422: {"description": "Payload failed Pydantic validation"},
        503: {"description": "Simulated downstream outage (X-Simulate-Error header)"},
    },
)
def create_record(
    payload: CMSRecordCreate,
    x_simulate_error: str | None = Header(default=None),
) -> CMSRecordResponse:
    if x_simulate_error:
        mode = x_simulate_error.strip().lower()
        if mode == "always":
            raise HTTPException(
                status_code=503,
                detail=f"Simulated CMS outage (always) for record_id={payload.record_id}",
            )
        if mode == "once":
            with _sim_lock:
                first_time = payload.record_id not in _failed_once
                _failed_once.add(payload.record_id)
            if first_time:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Simulated transient CMS outage (once) for "
                        f"record_id={payload.record_id}"
                    ),
                )

    record = CMSRecord.model_validate(payload.model_dump())
    created = store.upsert(record)
    return CMSRecordResponse(
        status="stored",
        record_id=record.record_id,
        created=created,
    )


@app.get(
    "/records",
    response_model=CMSRecordList,
    tags=["records"],
    summary="List stored records",
    description="Returns stored records, newest first, optionally filtered by classification.",
)
def list_records(
    limit: int = Query(default=100, ge=1, le=1000),
    classification: str | None = Query(default=None),
) -> CMSRecordList:
    records = store.list(limit=limit, classification=classification)
    return CMSRecordList(count=len(records), records=records)


@app.get(
    "/records/{record_id}",
    response_model=CMSRecord,
    tags=["records"],
    summary="Fetch one stored record",
    description="Returns a single stored record by its `record_id`.",
)
def get_record(record_id: str) -> CMSRecord:
    record = store.get(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"record_id={record_id} not found")
    return record
