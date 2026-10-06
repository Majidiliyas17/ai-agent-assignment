"""Tests for the unified record schema itself."""

import pytest
from pydantic import ValidationError

from app.models.unified_record import (
    ContactInfo,
    Person,
    RecordContext,
    RecordType,
    SourceType,
    UnifiedRecord,
)


def _base(**overrides: object) -> dict:
    payload: dict = {
        "id": "lead-001",
        "source": SourceType.LINKEDIN,
        "record_type": RecordType.LEAD,
        "person": Person(name="Ann Lee"),
        "contact_info": ContactInfo(),
        "context": RecordContext(summary="A summary"),
        "priority_signal": "engaged",
        "raw_payload": {"lead_id": "L001"},
    }
    payload.update(overrides)
    return payload


def test_valid_lead_record() -> None:
    record = UnifiedRecord(**_base())
    assert record.id == "lead-001"
    assert record.ingested_at is not None
    assert record.context.facts == {}
    assert record.context.tags == []


def test_valid_patient_record() -> None:
    record = UnifiedRecord(
        **_base(
            id="patient-abc",
            source=SourceType.FHIR,
            record_type=RecordType.PATIENT,
        )
    )
    assert record.source is SourceType.FHIR
    assert record.record_type is RecordType.PATIENT


@pytest.mark.parametrize(
    "overrides",
    [
        {"source": SourceType.FHIR, "record_type": RecordType.LEAD},
        {"source": SourceType.LINKEDIN, "record_type": RecordType.PATIENT},
    ],
    ids=["fhir-with-lead", "linkedin-with-patient"],
)
def test_source_and_record_type_must_match(overrides: dict) -> None:
    with pytest.raises(ValidationError, match="requires record_type"):
        UnifiedRecord(**_base(**overrides))


def test_missing_id_rejected() -> None:
    with pytest.raises(ValidationError):
        UnifiedRecord(**_base(id=""))


def test_missing_person_name_rejected() -> None:
    with pytest.raises(ValidationError):
        UnifiedRecord(**_base(person=Person(name="")))


def test_missing_summary_rejected() -> None:
    with pytest.raises(ValidationError):
        UnifiedRecord(**_base(context=RecordContext(summary="")))


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        UnifiedRecord(**_base(unexpected_field=1))


def test_raw_payload_is_required() -> None:
    payload = _base()
    payload.pop("raw_payload")
    with pytest.raises(ValidationError):
        UnifiedRecord(**payload)


def test_contact_info_blank_values_become_none() -> None:
    contact = ContactInfo(email="  ", phone=" 555-1 ")
    assert contact.email is None
    assert contact.phone == "555-1"


def test_context_facts_are_flexible() -> None:
    context = RecordContext(
        summary="s",
        facts={"nested": {"a": 1}, "list": [1, 2, "three"], "flag": True},
        tags=["t1"],
    )
    assert context.facts["nested"]["a"] == 1
    assert context.tags == ["t1"]
