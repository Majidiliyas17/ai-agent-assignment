"""Tests for the structured AgentResult schema and contextual validation."""

import pytest

from app.models.agent_result import (
    AgentResult,
    ResultValidationError,
    validate_agent_result,
)
from app.models.unified_record import RecordType, SourceType


def _lead_payload(**overrides: object) -> dict:
    payload: dict = {
        "record_id": "lead-001",
        "source": "linkedin",
        "record_type": "lead",
        "classification": "HOT",
        "priority": "HIGH",
        "action": "SEND_OUTREACH",
        "channel": "linkedin",
        "tone": "friendly",
        "message": "Hi Rebecca, saw your post about clinic automation.",
        "rationale": "High priority because the lead recently engaged with AI receptionist content.",
    }
    payload.update(overrides)
    return payload


def _patient_payload(**overrides: object) -> dict:
    payload = _lead_payload(
        record_id="patient-abc",
        source="fhir",
        record_type="patient",
        classification="HIGH_PRIORITY",
        action="SCHEDULE_OUTREACH",
        channel="phone",
        tone="professional",
        message="Administrative next step: call to confirm contact preferences.",
        rationale="High administrative priority: reachable contact details on file.",
    )
    payload.update(overrides)
    return payload


def test_valid_lead_result() -> None:
    result = validate_agent_result(
        "lead-001", SourceType.LINKEDIN, RecordType.LEAD, _lead_payload()
    )
    assert isinstance(result, AgentResult)
    assert result.classification == "HOT"
    assert result.action == "SEND_OUTREACH"
    assert result.processed_at is not None
    assert result.model_provider == "unknown"


def test_valid_patient_result() -> None:
    result = validate_agent_result(
        "patient-abc", SourceType.FHIR, RecordType.PATIENT, _patient_payload()
    )
    assert result.classification == "HIGH_PRIORITY"
    assert result.action == "SCHEDULE_OUTREACH"


def test_record_id_is_taken_from_the_record_not_the_model() -> None:
    result = validate_agent_result(
        "lead-001", SourceType.LINKEDIN, RecordType.LEAD,
        _lead_payload(record_id="hacked-id"),
    )
    assert result.record_id == "lead-001"


def test_source_and_record_type_cannot_be_spoofed() -> None:
    result = validate_agent_result(
        "lead-001", SourceType.LINKEDIN, RecordType.LEAD,
        _lead_payload(source="fhir", record_type="patient"),
    )
    assert result.source is SourceType.LINKEDIN
    assert result.record_type is RecordType.LEAD


def test_lead_cannot_use_patient_classification() -> None:
    with pytest.raises(ResultValidationError, match="classification"):
        validate_agent_result(
            "lead-001", SourceType.LINKEDIN, RecordType.LEAD,
            _lead_payload(classification="HIGH_PRIORITY"),
        )


def test_patient_cannot_use_lead_actions() -> None:
    with pytest.raises(ResultValidationError, match="action"):
        validate_agent_result(
            "patient-abc", SourceType.FHIR, RecordType.PATIENT,
            _patient_payload(action="NURTURE"),
        )


def test_patient_cannot_be_sent_sales_outreach() -> None:
    with pytest.raises(ResultValidationError):
        validate_agent_result(
            "patient-abc", SourceType.FHIR, RecordType.PATIENT,
            _patient_payload(action="SEND_OUTREACH"),
        )


def test_lead_cannot_use_internal_note_channel() -> None:
    with pytest.raises(ResultValidationError, match="channel"):
        validate_agent_result(
            "lead-001", SourceType.LINKEDIN, RecordType.LEAD,
            _lead_payload(channel="internal_note"),
        )


def test_missing_field_rejected() -> None:
    payload = _lead_payload()
    payload.pop("rationale")
    with pytest.raises(ResultValidationError, match="rationale"):
        validate_agent_result("lead-001", SourceType.LINKEDIN, RecordType.LEAD, payload)


def test_invalid_enum_rejected() -> None:
    with pytest.raises(ResultValidationError):
        validate_agent_result(
            "lead-001", SourceType.LINKEDIN, RecordType.LEAD,
            _lead_payload(classification="VERY_HOT"),
        )


def test_extra_keys_rejected() -> None:
    with pytest.raises(ResultValidationError):
        validate_agent_result(
            "lead-001", SourceType.LINKEDIN, RecordType.LEAD,
            _lead_payload(hallucinated_field="nope"),
        )


def test_non_object_payload_rejected() -> None:
    with pytest.raises(ResultValidationError, match="JSON object"):
        validate_agent_result("lead-001", SourceType.LINKEDIN, RecordType.LEAD, ["not", "a", "dict"])  # type: ignore[arg-type]


def test_message_and_rationale_length_limits() -> None:
    with pytest.raises(ResultValidationError):
        validate_agent_result(
            "lead-001", SourceType.LINKEDIN, RecordType.LEAD,
            _lead_payload(message="x" * 900),
        )
    with pytest.raises(ResultValidationError):
        validate_agent_result(
            "lead-001", SourceType.LINKEDIN, RecordType.LEAD,
            _lead_payload(rationale="y" * 600),
        )


def test_agent_result_is_json_serializable() -> None:
    result = validate_agent_result(
        "lead-001", SourceType.LINKEDIN, RecordType.LEAD, _lead_payload()
    )
    dumped = result.model_dump(mode="json")
    assert dumped["classification"] == "HOT"
    assert isinstance(dumped["processed_at"], str)
