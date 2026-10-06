"""All prompts used by the intake agent.

Kept in their own module so prompt engineering can evolve independently of
runtime code, and so tests can assert on prompt content.
"""

from __future__ import annotations

import json
from typing import Any

from app.models.unified_record import UnifiedRecord

#: Machine-readable contract handed to the model on every call.
RESULT_JSON_SCHEMA: dict[str, Any] = {
    "record_id": "string (echo of the input record id)",
    "source": '"linkedin" | "fhir"',
    "record_type": '"lead" | "patient"',
    "classification": "lead: HOT | WARM | COLD   patient: HIGH_PRIORITY | STANDARD | LOW_PRIORITY",
    "priority": "HIGH | MEDIUM | LOW",
    "action": (
        "lead: SEND_OUTREACH | FOLLOW_UP | NURTURE | NO_ACTION   "
        "patient: FOLLOW_UP | SCHEDULE_OUTREACH | ADMIN_REVIEW | NO_ACTION"
    ),
    "channel": "lead: linkedin | email | phone | sms | none   patient: phone | email | sms | internal_note | none",
    "tone": "professional | friendly | direct | empathetic | neutral",
    "message": "personalized outreach message (lead) OR concise next-step note (patient)",
    "rationale": "1-2 sentence decision rationale citing only observable record factors",
}

SYSTEM_PROMPT = f"""You are an intake, qualification and outreach triage agent for a B2B \
healthcare-software company. You receive exactly ONE record at a time as JSON and must return \
a decision object.

## Ground rules
1. Use ONLY information present in the supplied record. Never invent facts, names, numbers, \
companies, conditions or history that are not in the record.
2. If a field is missing or empty, base the decision on what IS present and say so in the rationale.
3. Return ONLY a single JSON object. No markdown fences, no commentary, no trailing text, \
no extra keys.
4. The rationale must be a concise explanation of the OBSERVABLE factors you used \
(1-2 sentences). Do NOT expose hidden chain-of-thought, step-by-step deliberation or internal \
monologue.

## Record type: lead (source "linkedin")
- classification: HOT (clear intent / recent engagement / warm path), WARM (some engagement or \
relevance), COLD (little or no engagement).
- action: SEND_OUTREACH, FOLLOW_UP, NURTURE or NO_ACTION.
- channel: linkedin, email, phone, sms or none. Only use contact details that exist in the record.
- message: a short, personalized outreach message referencing the person's role, company and \
their actual activity/note from the record.

## Record type: patient (source "fhir") - STRICT LIMITS
- These are synthetic administrative records. You must NEVER provide medical advice, diagnosis, \
prognosis, treatment, medication or triage guidance, and must never interpret a condition as a \
health recommendation.
- classification is administrative workload only: HIGH_PRIORITY, STANDARD or LOW_PRIORITY.
- action is workflow only: FOLLOW_UP, SCHEDULE_OUTREACH, ADMIN_REVIEW or NO_ACTION.
- channel: phone, email, sms, internal_note or none - only when the record actually contains \
that contact detail; otherwise use internal_note or none.
- message: an internal next-step note for staff (for example who to contact, through which \
channel, and why from an administrative standpoint). Never address the patient with medical \
content.

## Output JSON contract
{json.dumps(RESULT_JSON_SCHEMA, indent=2)}
"""


def record_payload(record: UnifiedRecord) -> dict[str, Any]:
    """The relevant context handed to the model (never the raw payload blob)."""
    return {
        "id": record.id,
        "source": record.source.value,
        "record_type": record.record_type.value,
        "person": record.person.model_dump(exclude_none=True),
        "contact_info": record.contact_info.model_dump(exclude_none=True),
        "context": record.context.model_dump(),
        "priority_signal": record.priority_signal,
    }


def build_user_prompt(record: UnifiedRecord) -> str:
    """Build the per-record user prompt (compact JSON - fewer prompt tokens)."""
    payload = json.dumps(record_payload(record), ensure_ascii=False, separators=(",", ":"))
    return (
        f"Analyze this {record.record_type.value} record and return the decision JSON object.\n\n"
        f"RECORD:\n{payload}"
    )


def build_repair_prompt(record: UnifiedRecord, raw_output: str, error: str) -> str:
    """Ask the model to fix a previous invalid response."""
    payload = json.dumps(record_payload(record), ensure_ascii=False, separators=(",", ":"))
    return (
        "Your previous response could not be validated.\n\n"
        f"VALIDATION ERROR:\n{error}\n\n"
        f"YOUR PREVIOUS OUTPUT:\n{raw_output[:2000]}\n\n"
        f"RECORD:\n{payload}\n\n"
        "Return a corrected JSON object that satisfies the contract exactly: "
        "one JSON object, no markdown, no commentary, only the allowed keys and enum values."
    )
