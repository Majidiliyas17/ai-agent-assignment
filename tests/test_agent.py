"""Tests for the agent processing loop (LLM mocked - no API key needed)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.agent.agent import IntakeAgent, extract_json
from app.agent.llm import (
    BaseLLMProvider,
    FaultInjectionProvider,
    LLMFatalError,
    LLMRequest,
    OfflineProvider,
)
from app.agent.prompts import SYSTEM_PROMPT, build_repair_prompt, build_user_prompt
from app.models.unified_record import UnifiedRecord
from app.utils.retry import RetryStats


class FakeLLM(BaseLLMProvider):
    """Scripted provider: returns the next item (string or exception)."""

    name = "fake"

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        return str(item)


def _lead_result_json(**overrides: object) -> str:
    payload = {
        "record_id": "lead-001",
        "source": "linkedin",
        "record_type": "lead",
        "classification": "WARM",
        "priority": "MEDIUM",
        "action": "FOLLOW_UP",
        "channel": "linkedin",
        "tone": "professional",
        "message": "Following up on your recent post about clinic automation.",
        "rationale": "Warm because the lead recently engaged with relevant content.",
    }
    payload.update(overrides)
    return json.dumps(payload)


def _patient_result_json(record_id: str, **overrides: object) -> str:
    payload = {
        "record_id": record_id,
        "source": "fhir",
        "record_type": "patient",
        "classification": "STANDARD",
        "priority": "MEDIUM",
        "action": "FOLLOW_UP",
        "channel": "phone",
        "tone": "professional",
        "message": "Administrative follow-up: confirm contact preferences.",
        "rationale": "Standard priority: active record with a reachable phone number.",
    }
    payload.update(overrides)
    return json.dumps(payload)


@pytest.mark.asyncio
async def test_valid_output_produces_agent_result(lead_record: UnifiedRecord) -> None:
    llm = FakeLLM([_lead_result_json()])
    agent = IntakeAgent(llm, stats=RetryStats())
    outcome = await agent.process(lead_record)

    assert outcome.ok
    assert outcome.result is not None
    assert outcome.result.classification == "WARM"
    assert outcome.result.action == "FOLLOW_UP"
    assert outcome.result.record_id == "lead-001"
    assert outcome.result.model_provider == "fake"
    assert outcome.llm_calls == 1
    # the system prompt must carry the JSON contract + medical restriction
    assert "Return ONLY a single JSON object" in llm.calls[0].system_prompt
    assert "NEVER provide medical advice" in llm.calls[0].system_prompt


@pytest.mark.asyncio
async def test_invalid_json_triggers_repair_call(lead_record: UnifiedRecord) -> None:
    llm = FakeLLM(["this is not json at all", _lead_result_json()])
    agent = IntakeAgent(llm, stats=RetryStats())
    outcome = await agent.process(lead_record)

    assert outcome.ok
    assert outcome.llm_calls == 2
    assert llm.calls[0].is_repair is False
    assert llm.calls[1].is_repair is True
    assert "VALIDATION ERROR" in llm.calls[1].user_prompt


@pytest.mark.asyncio
async def test_contextual_violation_triggers_repair(lead_record: UnifiedRecord) -> None:
    bad = _lead_result_json(classification="HIGH_PRIORITY", action="ADMIN_REVIEW")
    llm = FakeLLM([bad, _lead_result_json()])
    agent = IntakeAgent(llm, stats=RetryStats())
    outcome = await agent.process(lead_record)

    assert outcome.ok
    assert outcome.llm_calls == 2
    assert outcome.result is not None
    assert outcome.result.classification == "WARM"


@pytest.mark.asyncio
async def test_invalid_output_exhausts_bounded_retries(lead_record: UnifiedRecord) -> None:
    llm = FakeLLM(["{{{ definitely broken"])
    stats = RetryStats()
    agent = IntakeAgent(llm, stats=stats)

    outcome = await agent.process(lead_record)

    assert not outcome.ok
    assert outcome.result is None
    assert "invalid LLM output after 3 attempts" in (outcome.error or "")
    assert outcome.llm_calls == 3  # 1 initial + 2 repairs
    assert stats.failures.get("llm_invalid_output") == 1


@pytest.mark.asyncio
async def test_patient_record_rejects_lead_only_action(
    patient_record: UnifiedRecord,
) -> None:
    wrong = _patient_result_json(patient_record.id, action="SEND_OUTREACH", channel="linkedin")
    llm = FakeLLM([wrong, _patient_result_json(patient_record.id)])
    agent = IntakeAgent(llm, stats=RetryStats())
    outcome = await agent.process(patient_record)

    assert outcome.ok
    assert outcome.result is not None
    assert outcome.result.action in {"FOLLOW_UP", "SCHEDULE_OUTREACH", "ADMIN_REVIEW", "NO_ACTION"}


@pytest.mark.asyncio
async def test_provider_failure_is_reported_not_raised(
    lead_record: UnifiedRecord,
) -> None:
    llm = FakeLLM([LLMFatalError("Gemini rejected the credentials (HTTP 401)")])
    agent = IntakeAgent(llm, stats=RetryStats())
    outcome = await agent.process(lead_record)

    assert not outcome.ok
    assert "LLM call failed" in (outcome.error or "")
    assert outcome.llm_calls == 1


@pytest.mark.asyncio
async def test_process_many_continues_after_a_failure(
    lead_record: UnifiedRecord, patient_record: UnifiedRecord
) -> None:
    llm = FakeLLM(["broken", _patient_result_json(patient_record.id)])
    agent = IntakeAgent(llm, stats=RetryStats())
    outcomes = await agent.process_many([lead_record, patient_record])

    assert len(outcomes) == 2
    assert not outcomes[0].ok  # lead consumed the broken responses
    assert outcomes[1].ok


@pytest.mark.asyncio
async def test_fault_injection_then_recovery(lead_record: UnifiedRecord) -> None:
    provider = FaultInjectionProvider(OfflineProvider(), failures=1)
    agent = IntakeAgent(provider, stats=RetryStats())
    outcome = await agent.process(lead_record)

    assert outcome.ok
    assert outcome.llm_calls == 2


@pytest.mark.asyncio
async def test_offline_provider_output_is_valid_structured_json(
    lead_record: UnifiedRecord, patient_record: UnifiedRecord
) -> None:
    agent = IntakeAgent(OfflineProvider(), stats=RetryStats())
    for record in (lead_record, patient_record):
        outcome = await agent.process(record)
        assert outcome.ok, outcome.error


def test_extract_json_plain() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_markdown_fenced() -> None:
    raw = '```json\n{"a": 1}\n```'
    assert extract_json(raw) == {"a": 1}


def test_extract_json_with_surrounding_prose() -> None:
    raw = 'Sure! Here is the JSON:\n{"a": 1}\nHope that helps.'
    assert extract_json(raw) == {"a": 1}


def test_extract_json_nested_object() -> None:
    raw = 'noise {"outer": {"inner": [1, 2, {"deep": true}]}} tail'
    assert extract_json(raw) == {"outer": {"inner": [1, 2, {"deep": True}]}}


def test_extract_json_invalid_raises() -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json("no json here")


def test_prompts_include_record_context(lead_record: UnifiedRecord) -> None:
    user = build_user_prompt(lead_record)
    assert lead_record.id in user
    assert lead_record.person.name in user
    assert "RECORD:" in user

    repair = build_repair_prompt(lead_record, "bad output", "invalid AgentResult")
    assert "bad output" in repair
    assert "invalid AgentResult" in repair
    assert lead_record.person.name in repair


def test_system_prompt_distinguishes_leads_and_patients() -> None:
    assert "Record type: lead" in SYSTEM_PROMPT
    assert "Record type: patient" in SYSTEM_PROMPT
    assert "diagnosis" in SYSTEM_PROMPT.lower()
