"""Pydantic models: the unified record schema shared by every data source."""

from app.models.agent_result import (
    AgentResult,
    CHANNELS,
    LEAD_ACTIONS,
    LEAD_CLASSIFICATIONS,
    PATIENT_ACTIONS,
    PATIENT_CLASSIFICATIONS,
    PRIORITIES,
    TONES,
    ResultValidationError,
    validate_agent_result,
)
from app.models.unified_record import (
    ContactInfo,
    Person,
    RecordContext,
    RecordType,
    SourceType,
    UnifiedRecord,
    UnifiedRecordValidationError,
)

__all__ = [
    "AgentResult",
    "CHANNELS",
    "ContactInfo",
    "LEAD_ACTIONS",
    "LEAD_CLASSIFICATIONS",
    "PATIENT_ACTIONS",
    "PATIENT_CLASSIFICATIONS",
    "PRIORITIES",
    "Person",
    "RecordContext",
    "RecordType",
    "ResultValidationError",
    "SourceType",
    "TONES",
    "UnifiedRecord",
    "UnifiedRecordValidationError",
    "validate_agent_result",
]
