"""Tests for FHIR Patient -> UnifiedRecord normalization."""

import pytest
from pydantic import ValidationError

from app.models.unified_record import (
    RecordType,
    SourceType,
    UnifiedRecord,
    UnifiedRecordValidationError,
)
from app.normalization.patient_normalizer import normalize_patient


def test_normalize_patient_basic(sample_patient: dict) -> None:
    record = normalize_patient(sample_patient)
    assert record.id == "patient-test-patient-1"
    assert record.source is SourceType.FHIR
    assert record.record_type is RecordType.PATIENT
    assert record.person.name == "Ada Lovelace"  # official name preferred
    assert record.person.given_name == "Ada"
    assert record.person.family_name == "Lovelace"
    assert record.contact_info.phone == "555-0100"
    assert record.contact_info.email == "ada@example.org"
    assert "Boston" in (record.contact_info.location or "")
    assert record.context.facts["fhir_patient_id"] == "test-patient-1"
    assert record.context.facts["fhir_status"] == "active"
    assert record.context.facts["gender"] == "female"
    assert "fhir" in record.context.tags


def test_raw_fhir_payload_preserved(sample_patient: dict) -> None:
    record = normalize_patient(sample_patient)
    assert record.raw_payload == sample_patient
    assert record.raw_payload["resourceType"] == "Patient"


def test_age_computed(sample_patient: dict) -> None:
    record = normalize_patient(sample_patient)
    age = record.context.facts["age_years"]
    assert isinstance(age, int)
    assert 20 <= age <= 120


def test_name_falls_back_to_first_available(sample_patient: dict) -> None:
    sample_patient["name"] = [{"given": ["Grace"], "family": "Hopper"}]
    record = normalize_patient(sample_patient)
    assert record.person.name == "Grace Hopper"


def test_tolerates_family_as_list(sample_patient: dict) -> None:
    sample_patient["name"] = [{"given": ["Katherine"], "family": ["Johnson"]}]
    record = normalize_patient(sample_patient)
    assert record.person.name == "Katherine Johnson"


def test_missing_name_raises(sample_patient: dict) -> None:
    sample_patient.pop("name")
    with pytest.raises(UnifiedRecordValidationError):
        normalize_patient(sample_patient)


def test_missing_id_raises(sample_patient: dict) -> None:
    sample_patient.pop("id")
    with pytest.raises(UnifiedRecordValidationError):
        normalize_patient(sample_patient)


def test_wrong_resource_type_raises(sample_patient: dict) -> None:
    sample_patient["resourceType"] = "Observation"
    with pytest.raises(UnifiedRecordValidationError):
        normalize_patient(sample_patient)


def test_conditions_become_administrative_context(sample_patient: dict) -> None:
    conditions = [
        {"code": {"coding": [{"display": "Essential hypertension"}]}},
        {"code": {"text": "Type 2 diabetes mellitus"}},
    ]
    record = normalize_patient(sample_patient, conditions=conditions)
    assert record.context.facts["conditions"] == [
        "Essential hypertension",
        "Type 2 diabetes mellitus",
    ]
    assert record.context.facts["condition_count"] == 2
    assert "conditions-on-file" in record.context.tags
    # administrative signal only - no clinical advice
    assert "administrative follow-up" in record.priority_signal


def test_deceased_record_flagged(sample_patient: dict) -> None:
    sample_patient["deceasedBoolean"] = True
    record = normalize_patient(sample_patient)
    assert "deceased" in record.priority_signal
    assert record.context.facts["deceased"] is True


def test_missing_optional_fields_are_tolerated() -> None:
    record = normalize_patient({"resourceType": "Patient", "id": "bare", "name": [{"text": "Solo"}]})
    assert record.person.name == "Solo"
    assert record.contact_info.email is None
    assert record.contact_info.phone is None
    with pytest.raises(ValidationError):
        UnifiedRecord(
            id=record.id,
            source=record.source,
            record_type=RecordType.LEAD,  # fhir source requires patient type
            person=record.person,
            contact_info=record.contact_info,
            context=record.context,
            priority_signal=record.priority_signal,
            raw_payload=record.raw_payload,
        )
