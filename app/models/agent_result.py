"""Structured output of the AI agent (and the payload sent to the mock CMS).

The agent must always return this exact JSON shape. Validation happens in two
layers:

1. ``AgentResult`` - field types / enums / lengths (Pydantic).
2. ``validate_agent_result`` - *contextual* rules, e.g. a patient record may
   only receive administrative actions and never a diagnosis-like action.

Contextual violations raise :class:`ResultValidationError`, which the agent
treats as repairable (one more LLM call with an explicit repair prompt).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.models.unified_record import RecordType, SourceType

# --------------------------------------------------------------------------- #
# Controlled vocabularies
# --------------------------------------------------------------------------- #

#: Classification allowed for LinkedIn-style leads.
LEAD_CLASSIFICATIONS = ("HOT", "WARM", "COLD")
#: Classification allowed for FHIR patient records (administrative, not clinical).
PATIENT_CLASSIFICATIONS = ("HIGH_PRIORITY", "STANDARD", "LOW_PRIORITY")

Classification = Literal["HOT", "WARM", "COLD", "HIGH_PRIORITY", "STANDARD", "LOW_PRIORITY"]

#: Actions for leads (sales/marketing workflow).
LEAD_ACTIONS = ("SEND_OUTREACH", "FOLLOW_UP", "NURTURE", "NO_ACTION")
#: Actions for patients - administrative/workflow only, never clinical.
PATIENT_ACTIONS = ("FOLLOW_UP", "SCHEDULE_OUTREACH", "ADMIN_REVIEW", "NO_ACTION")

Action = Literal[
    "SEND_OUTREACH", "FOLLOW_UP", "NURTURE", "NO_ACTION",
    "SCHEDULE_OUTREACH", "ADMIN_REVIEW",
]

PRIORITIES = ("HIGH", "MEDIUM", "LOW")
Priority = Literal["HIGH", "MEDIUM", "LOW"]

CHANNELS = ("linkedin", "email", "phone", "sms", "internal_note", "none")
Channel = Literal["linkedin", "email", "phone", "sms", "internal_note", "none"]

TONES = ("professional", "friendly", "direct", "empathetic", "neutral")
Tone = Literal["professional", "friendly", "direct", "empathetic", "neutral"]


class ResultValidationError(ValueError):
    """Raised when the LLM output does not satisfy the contextual contract."""


class AgentResult(BaseModel):
    """Structured, CMS-ready outcome for a single processed record."""

    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(min_length=1, max_length=200)
    source: SourceType
    record_type: RecordType
    classification: Classification
    priority: Priority
    action: Action
    channel: Channel
    tone: Tone
    message: str = Field(
        min_length=1,
        max_length=800,
        description="Personalized outreach message (lead) or next-step note (patient).",
    )
    rationale: str = Field(
        min_length=1,
        max_length=500,
        description="Concise decision rationale based only on observable record factors.",
    )
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_provider: str = Field(
        default="unknown",
        max_length=50,
        description="LLM provider that produced this decision (audit aid).",
    )


# --------------------------------------------------------------------------- #
# Contextual validation
# --------------------------------------------------------------------------- #

_ALLOWED_BY_RECORD_TYPE: dict[RecordType, dict[str, tuple[str, ...]]] = {
    RecordType.LEAD: {
        "classification": LEAD_CLASSIFICATIONS,
        "action": LEAD_ACTIONS,
        "allowed_channels": ("linkedin", "email", "phone", "sms", "none"),
    },
    RecordType.PATIENT: {
        "classification": PATIENT_CLASSIFICATIONS,
        "action": PATIENT_ACTIONS,
        "allowed_channels": ("phone", "email", "sms", "internal_note", "none"),
    },
}


def validate_agent_result(record_id: str, source: SourceType, record_type: RecordType,
                          payload: dict[str, Any]) -> AgentResult:
    """Validate raw LLM JSON against the schema *and* the record-type contract.

    Parameters
    ----------
    record_id, source, record_type:
        Trusted values taken from the :class:`UnifiedRecord` (never from the LLM).
    payload:
        The decoded JSON returned by the model.
    """
    if not isinstance(payload, dict):
        raise ResultValidationError(f"LLM output must be a JSON object, got {type(payload).__name__}")

    # Never let the model spoof identity fields.
    payload = dict(payload)
    payload["record_id"] = record_id
    payload["source"] = source.value
    payload["record_type"] = record_type.value

    try:
        result = AgentResult.model_validate(payload)
    except ValidationError as exc:
        raise ResultValidationError(_short_validation_error(exc)) from exc

    rules = _ALLOWED_BY_RECORD_TYPE[record_type]
    if result.classification not in rules["classification"]:
        raise ResultValidationError(
            f"classification={result.classification!r} is not valid for a "
            f"{record_type.value} record; allowed: {rules['classification']}"
        )
    if result.action not in rules["action"]:
        raise ResultValidationError(
            f"action={result.action!r} is not valid for a {record_type.value} record; "
            f"allowed: {rules['action']}"
        )
    if result.channel not in rules["allowed_channels"]:
        raise ResultValidationError(
            f"channel={result.channel!r} is not valid for a {record_type.value} record; "
            f"allowed: {rules['allowed_channels']}"
        )
    if record_type is RecordType.PATIENT and result.action == "SEND_OUTREACH":
        raise ResultValidationError(
            "SEND_OUTREACH is reserved for lead records; patient records must use "
            "administrative actions only."
        )
    return result


def _short_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for err in exc.errors()[:5]:
        location = ".".join(str(item) for item in err["loc"]) or "<root>"
        parts.append(f"{location}: {err['msg']}")
    return "invalid AgentResult: " + "; ".join(parts)
