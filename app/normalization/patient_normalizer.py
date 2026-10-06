"""Normalize FHIR R4 ``Patient`` resources into :class:`UnifiedRecord`.

Purely administrative/demographic mapping - no clinical inference, no
diagnosis, no treatment signal. Raw FHIR JSON is preserved verbatim.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from app.models.unified_record import (
    ContactInfo,
    Person,
    RecordContext,
    RecordType,
    SourceType,
    UnifiedRecord,
    UnifiedRecordValidationError,
)


def _pick_name(name_entries: list[dict[str, Any]]) -> tuple[str, str | None, str | None]:
    """Prefer an ``official``/``usual`` human name, else the first usable one."""
    if not name_entries:
        raise UnifiedRecordValidationError("FHIR Patient has no name")

    ordered = sorted(
        name_entries,
        key=lambda entry: 0 if entry.get("use") in {"official", "usual"} else 1,
    )
    for entry in ordered:
        given = entry.get("given") or []
        family = entry.get("family")
        if isinstance(family, list):  # tolerate non-conforming resources
            family = " ".join(str(part) for part in family) or None
        text = entry.get("text")
        if text:
            return str(text), None, None
        if given or family:
            parts = [str(part) for part in given] + ([str(family)] if family else [])
            full_name = " ".join(parts).strip()
            if full_name:
                return full_name, (str(given[0]) if given else None), str(family) if family else None
    raise UnifiedRecordValidationError("FHIR Patient name entries are empty")


def _age_years(birth_date: str | None) -> int | None:
    """Age in whole years from a FHIR date (YYYY, YYYY-MM, YYYY-MM-DD)."""
    if not birth_date:
        return None
    try:
        parts = [int(part) for part in birth_date.split("-")]
    except ValueError:
        return None
    if len(parts) == 1:
        return max(datetime.now(timezone.utc).year - parts[0], 0)
    try:
        born = date(parts[0], parts[1], parts[2] if len(parts) > 2 else 1)
    except (ValueError, IndexError):
        return None
    today = datetime.now(timezone.utc).date()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def _contact(telecom: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    email: str | None = None
    phone: str | None = None
    for item in telecom:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        if not value:
            continue
        system = item.get("system")
        if system == "email" and email is None:
            email = str(value)
        elif system == "phone" and phone is None:
            phone = str(value)
    return email, phone


def _location(addresses: list[dict[str, Any]]) -> str | None:
    for address in addresses:
        if not isinstance(address, dict):
            continue
        line = address.get("line") or []
        parts = [*line, address.get("city"), address.get("state"), address.get("postalCode")]
        rendered = ", ".join(str(part) for part in parts if part)
        if rendered:
            return rendered
    return None


def _condition_labels(conditions: list[dict[str, Any]]) -> list[str]:
    """Extract display labels only - administrative context, not advice."""
    labels: list[str] = []
    for condition in conditions:
        code = condition.get("code") or {}
        text = code.get("text")
        if text:
            labels.append(str(text))
            continue
        for coding in code.get("coding") or []:
            display = coding.get("display")
            if display:
                labels.append(str(display))
                break
    return labels


def normalize_patient(
    resource: dict[str, Any],
    conditions: list[dict[str, Any]] | None = None,
) -> UnifiedRecord:
    """Convert a raw FHIR Patient resource into a :class:`UnifiedRecord`.

    Raises :class:`UnifiedRecordValidationError` for resources that are missing
    critical fields; the pipeline logs and skips them.
    """
    if not isinstance(resource, dict) or resource.get("resourceType") != "Patient":
        raise UnifiedRecordValidationError("Expected a FHIR Patient resource")
    patient_id = resource.get("id")
    if not patient_id:
        raise UnifiedRecordValidationError("FHIR Patient is missing id")

    name, given, family = _pick_name(list(resource.get("name") or []))
    email, phone = _contact(list(resource.get("telecom") or []))
    location = _location(list(resource.get("address") or []))

    status = resource.get("active")
    gender = resource.get("gender")
    birth_date = resource.get("birthDate")
    age = _age_years(birth_date)
    organization = None
    managing = resource.get("managingOrganization")
    if isinstance(managing, dict):
        organization = managing.get("display") or (
            (managing.get("reference") or "").split("/")[-1] or None
        )

    condition_labels = _condition_labels(conditions or [])
    deceased = bool(resource.get("deceasedBoolean") or resource.get("deceasedDateTime"))

    facts: dict[str, Any] = {
        "fhir_patient_id": patient_id,
        "fhir_status": "active" if status is True else ("inactive" if status is False else "unknown"),
        "gender": gender,
        "birth_date": birth_date,
        "age_years": age,
        "managing_organization": organization,
        "condition_count": len(condition_labels),
        "conditions": condition_labels,
        "deceased": deceased,
    }
    facts = {key: value for key, value in facts.items() if value not in (None, [], "")}

    tags = ["fhir", "patient", str(facts.get("fhir_status", "unknown"))]
    if condition_labels:
        tags.append("conditions-on-file")
    if phone or email:
        tags.append("contactable")

    summary = f"FHIR Patient {patient_id}: {name}"
    demographics = [value for value in (gender, f"age {age}" if age is not None else None) if value]
    if demographics:
        summary += f" ({', '.join(demographics)})"
    summary += f", status={facts.get('fhir_status', 'unknown')}"
    if condition_labels:
        summary += f", {len(condition_labels)} condition(s) on file"
    summary += "."

    if deceased:
        priority_signal = "record flagged as deceased - administrative review only"
    elif condition_labels:
        priority_signal = f"{len(condition_labels)} recorded condition(s) require administrative follow-up"
    elif phone or email:
        priority_signal = "active record with reachable contact details"
    else:
        priority_signal = "active record without contact details"

    return UnifiedRecord(
        id=f"patient-{patient_id}",
        source=SourceType.FHIR,
        record_type=RecordType.PATIENT,
        person=Person(
            name=name,
            given_name=given,
            family_name=family,
            role=None,
            organization=organization,
        ),
        contact_info=ContactInfo(email=email, phone=phone, location=location),
        context=RecordContext(summary=summary[:1000], facts=facts, tags=tags),
        priority_signal=priority_signal,
        raw_payload=resource,
    )
