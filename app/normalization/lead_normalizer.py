"""Normalize LinkedIn-style CSV leads into :class:`UnifiedRecord`.

Only facts that literally exist in the CSV are used - nothing is invented.
``priority_signal`` is a short *observable* signal (not an AI decision): the
agent still performs the actual classification.
"""

from __future__ import annotations

import re

from app.ingestion.csv_loader import RawLead
from app.models.unified_record import (
    ContactInfo,
    Person,
    RecordContext,
    RecordType,
    SourceType,
    UnifiedRecord,
    UnifiedRecordValidationError,
)

_INTENT_PHRASES = (
    "requested a demo",
    "requested pricing",
    "requested a call",
    "requested pricing info",
    "requested a demo via website form",
)
_WARM_PHRASES = ("warm intro", "mutual connection", "mutual ")
_ENGAGEMENT_PHRASES = (
    "liked", "commented", "shared", "attended", "posted", "connected", "clicked",
)
_COLD_PHRASES = (
    "no recent activity", "dormant", "cold", "no note", "imported", "badge scan",
    "group member", "saved search", "trade show",
)


def internal_lead_id(lead_id: str) -> str:
    """Map ``L001`` -> ``lead-001`` while staying stable for unusual ids."""
    cleaned = lead_id.strip()
    match = re.fullmatch(r"[A-Za-z]?0*(\d+)", cleaned)
    if match:
        return f"lead-{int(match.group(1)):03d}"
    slug = re.sub(r"[^a-z0-9]+", "-", cleaned.lower()).strip("-")
    return f"lead-{slug or 'unknown'}"


def _signal(text: str) -> tuple[str, list[str]]:
    lowered = text.lower()
    if any(phrase in lowered for phrase in _INTENT_PHRASES):
        return "explicit intent captured (demo/pricing/call request)", ["intent"]
    if any(phrase in lowered for phrase in _WARM_PHRASES):
        return "warm introduction or mutual connection", ["warm-intro"]
    if any(phrase in lowered for phrase in _ENGAGEMENT_PHRASES):
        return "recent engagement with relevant content", ["engaged"]
    if any(phrase in lowered for phrase in _COLD_PHRASES):
        return "cold or dormant signal, no recent interaction", ["cold"]
    return "no strong engagement signal in source fields", ["neutral"]


def normalize_lead(raw: RawLead) -> UnifiedRecord:
    """Convert one accepted CSV row into a :class:`UnifiedRecord`.

    Raises :class:`UnifiedRecordValidationError` for rows that cannot be
    normalized (the pipeline logs the failure and continues).
    """
    data = raw.data
    name = (data.get("full_name") or "").strip()
    lead_id = (data.get("lead_id") or "").strip()
    if not name or not lead_id:
        raise UnifiedRecordValidationError(
            f"row={raw.row_number}: lead requires both lead_id and full_name"
        )

    job_title = data.get("job_title")
    company = data.get("company")
    industry = data.get("industry")
    company_size = data.get("company_size")
    location = data.get("location")
    headline = data.get("linkedin_headline")
    activity = data.get("recent_activity")
    note = data.get("connection_note")

    signal_text = " ".join(part for part in (activity, note) if part)
    priority_signal, tags = _signal(signal_text or headline or "")
    if industry:
        tags = [*tags, industry.lower()]

    facts: dict[str, object] = {
        key: value
        for key, value in {
            "lead_id": lead_id,
            "job_title": job_title,
            "company": company,
            "industry": industry,
            "company_size": company_size,
            "location": location,
            "linkedin_headline": headline,
            "recent_activity": activity,
            "connection_note": note,
        }.items()
        if value
    }

    summary_bits: list[str] = [name]
    if job_title and company:
        summary_bits.append(f", {job_title} at {company}")
    elif job_title or company:
        summary_bits.append(f", {job_title or company}")
    if industry:
        summary_bits.append(
            f" ({industry}, {company_size})" if company_size else f" ({industry})"
        )
    if location:
        summary_bits.append(f" in {location}")
    summary = "".join(summary_bits).strip()
    if activity:
        summary += f". Recent activity: {activity}."
    if note:
        summary += f" Source note: {note}."

    return UnifiedRecord(
        id=internal_lead_id(lead_id),
        source=SourceType.LINKEDIN,
        record_type=RecordType.LEAD,
        person=Person(
            name=name,
            given_name=name.split(" ")[0] if name else None,
            family_name=" ".join(name.split()[1:]) or None,
            role=job_title,
            organization=company,
        ),
        contact_info=ContactInfo(location=location),
        context=RecordContext(
            summary=summary[:1000],
            facts=facts,
            tags=tags,
        ),
        priority_signal=priority_signal,
        raw_payload=raw.raw_payload,
    )
