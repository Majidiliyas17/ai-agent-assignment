"""Pydantic contracts for the mock CMS API.

The CMS is a *separate service boundary*: it declares its own payload schema
instead of importing the pipeline's models, so a contract drift between the
producer (agent) and the consumer (CMS) fails loudly at validation time.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Classification = Literal["HOT", "WARM", "COLD", "HIGH_PRIORITY", "STANDARD", "LOW_PRIORITY"]
Action = Literal[
    "SEND_OUTREACH", "FOLLOW_UP", "NURTURE", "NO_ACTION",
    "SCHEDULE_OUTREACH", "ADMIN_REVIEW",
]
Priority = Literal["HIGH", "MEDIUM", "LOW"]
Channel = Literal["linkedin", "email", "phone", "sms", "internal_note", "none"]
Tone = Literal["professional", "friendly", "direct", "empathetic", "neutral"]
Source = Literal["linkedin", "fhir"]
RecordType = Literal["lead", "patient"]


class CMSRecordCreate(BaseModel):
    """Payload accepted by ``POST /records`` (structured, never a text blob)."""

    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(min_length=1, max_length=200, examples=["lead-001"])
    source: Source
    record_type: RecordType
    classification: Classification
    priority: Priority
    action: Action
    channel: Channel
    tone: Tone
    message: str = Field(min_length=1, max_length=800)
    rationale: str = Field(min_length=1, max_length=500)
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_provider: str = Field(default="unknown", max_length=50)


class CMSRecord(CMSRecordCreate):
    """Stored record plus CMS-side metadata."""

    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CMSRecordResponse(BaseModel):
    status: str
    record_id: str
    created: bool


class CMSRecordList(BaseModel):
    count: int
    records: list[CMSRecord]


class HealthResponse(BaseModel):
    status: str
    service: str = "mock-cms"
    record_count: int
    database: str
