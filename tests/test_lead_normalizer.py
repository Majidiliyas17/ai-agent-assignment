"""Tests for LinkedIn lead -> UnifiedRecord normalization."""

import pytest
from pydantic import ValidationError

from app.ingestion.csv_loader import RawLead
from app.models.unified_record import (
    RecordType,
    SourceType,
    UnifiedRecord,
    UnifiedRecordValidationError,
)
from app.normalization.lead_normalizer import internal_lead_id, normalize_lead


def test_normalize_lead_basic(raw_lead: RawLead) -> None:
    record = normalize_lead(raw_lead)
    assert record.id == "lead-001"
    assert record.source is SourceType.LINKEDIN
    assert record.record_type is RecordType.LEAD
    assert record.person.name == "Rebecca Hart"
    assert record.person.role == "Practice Manager"
    assert record.person.organization == "Northgate Family Clinic"
    assert record.contact_info.location == "Austin TX"
    assert record.priority_signal
    assert record.raw_payload["lead_id"] == "L001"
    assert record.context.facts["industry"] == "Healthcare"
    assert "engaged" in record.context.tags or "intent" in record.context.tags


def test_normalize_lead_preserves_raw_payload(raw_lead: RawLead) -> None:
    record = normalize_lead(raw_lead)
    assert record.raw_payload == raw_lead.raw_payload
    assert record.raw_payload is not raw_lead.raw_payload  # defensive copy


def test_normalize_lead_intent_signal() -> None:
    raw = RawLead(
        row_number=3,
        data={
            "lead_id": "L009",
            "full_name": "Emily Zhang",
            "recent_activity": "Requested pricing info via LinkedIn message",
        },
        raw_payload={"lead_id": "L009"},
    )
    record = normalize_lead(raw)
    assert "intent" in record.context.tags
    assert "explicit intent" in record.priority_signal


def test_normalize_lead_missing_name_raises() -> None:
    raw = RawLead(row_number=4, data={"lead_id": "L050", "full_name": None}, raw_payload={})
    with pytest.raises(UnifiedRecordValidationError):
        normalize_lead(raw)


def test_normalize_lead_requires_valid_person() -> None:
    raw = RawLead(row_number=5, data={"lead_id": "", "full_name": ""}, raw_payload={})
    with pytest.raises(UnifiedRecordValidationError):
        normalize_lead(raw)


@pytest.mark.parametrize(
    ("lead_id", "expected"),
    [
        ("L001", "lead-001"),
        ("L12", "lead-012"),
        ("42", "lead-042"),
        ("weird-id", "lead-weird-id"),
        ("", "lead-unknown"),
    ],
)
def test_internal_lead_id(lead_id: str, expected: str) -> None:
    assert internal_lead_id(lead_id) == expected


def test_unified_record_rejects_bad_source_combination(raw_lead: RawLead) -> None:
    record = normalize_lead(raw_lead)
    with pytest.raises(ValidationError):
        UnifiedRecord(
            id=record.id,
            source=SourceType.FHIR,  # FHIR source requires record_type=patient
            record_type=RecordType.LEAD,
            person=record.person,
            contact_info=record.contact_info,
            context=record.context,
            priority_signal=record.priority_signal,
            raw_payload=record.raw_payload,
        )
