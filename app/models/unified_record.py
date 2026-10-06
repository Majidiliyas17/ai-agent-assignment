"""Unified internal record schema.

Every source (LinkedIn-style CSV leads, FHIR patients) is normalized into this
single shape so the AI agent and the downstream CMS only ever see one schema.

Design notes
------------
* ``person`` / ``contact_info`` hold the *person-centric* fields that exist for
  both leads and patients.
* ``context`` is a small, explicit, per-source bag (``facts``) so we never have
  to force LinkedIn fields and healthcare fields into one giant flat schema.
* ``raw_payload`` preserves the untouched source record for audit/debug.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class UnifiedRecordValidationError(ValueError):
    """Raised when a source record cannot be normalized into a UnifiedRecord."""


class SourceType(str, Enum):
    LINKEDIN = "linkedin"
    FHIR = "fhir"


class RecordType(str, Enum):
    LEAD = "lead"
    PATIENT = "patient"


class Person(BaseModel):
    """Person-level identity shared by leads and patients."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200, description="Display name.")
    given_name: str | None = Field(default=None, max_length=100)
    family_name: str | None = Field(default=None, max_length=100)
    role: str | None = Field(
        default=None, max_length=200,
        description="Job title for leads; empty for patients.",
    )
    organization: str | None = Field(
        default=None, max_length=200,
        description="Company/clinic for leads; managing organization for patients.",
    )


class ContactInfo(BaseModel):
    """Best-effort contact details. Missing values stay ``None`` - never invented."""

    model_config = ConfigDict(extra="forbid")

    email: str | None = Field(default=None, max_length=254)
    phone: str | None = Field(default=None, max_length=50)
    linkedin_url: str | None = Field(default=None, max_length=500)
    location: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _strip(self) -> "ContactInfo":
        for field_name in ("email", "phone", "linkedin_url", "location"):
            value = getattr(self, field_name)
            if value is not None:
                value = value.strip() or None
                setattr(self, field_name, value)
        return self


class RecordContext(BaseModel):
    """Flexible, source-aware context.

    ``summary`` is a one-line human description; ``facts`` carries structured
    source-specific attributes; ``tags`` holds short observable labels.
    """

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=1000)
    facts: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class UnifiedRecord(BaseModel):
    """The one schema produced by normalization and consumed by the agent."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=200, description="Stable internal id.")
    source: SourceType
    record_type: RecordType
    person: Person
    contact_info: ContactInfo
    context: RecordContext
    priority_signal: str = Field(
        min_length=1,
        max_length=300,
        description="Short observable signal observed in the source data.",
    )
    raw_payload: dict[str, Any] = Field(
        description="Untouched source record preserved for audit/debug."
    )
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _check_source_matches_record_type(self) -> "UnifiedRecord":
        expected = {
            SourceType.LINKEDIN: RecordType.LEAD,
            SourceType.FHIR: RecordType.PATIENT,
        }[self.source]
        if self.record_type is not expected:
            raise ValueError(
                f"source={self.source.value!r} requires record_type="
                f"{expected.value!r}, got {self.record_type.value!r}"
            )
        return self
