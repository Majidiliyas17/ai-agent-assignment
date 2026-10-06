"""Shared pytest fixtures."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Point the mock CMS at a throwaway SQLite file before the app module is
# imported (mock_cms.main creates its store at import time).
_TMP_DIR = tempfile.mkdtemp(prefix="cms-test-")
os.environ.setdefault("CMS_DB_PATH", str(Path(_TMP_DIR) / "cms_test.sqlite3"))

from app.ingestion.csv_loader import RawLead  # noqa: E402
from app.models.unified_record import (  # noqa: E402
    ContactInfo,
    Person,
    RecordContext,
    RecordType,
    SourceType,
    UnifiedRecord,
)

SAMPLE_LEAD_ROW: dict[str, str | None] = {
    "lead_id": "L001",
    "full_name": "Rebecca Hart",
    "job_title": "Practice Manager",
    "company": "Northgate Family Clinic",
    "industry": "Healthcare",
    "company_size": "11-50",
    "location": "Austin TX",
    "linkedin_headline": "Practice Manager helping small clinics run smoother ops",
    "recent_activity": "Liked a post about AI receptionists for medical clinics",
    "connection_note": "Sent a connection request after webinar on clinic automation",
}

SAMPLE_PATIENT: dict = {
    "resourceType": "Patient",
    "id": "test-patient-1",
    "active": True,
    "name": [
        {"use": "official", "given": ["Ada"], "family": "Lovelace"},
        {"use": "nickname", "given": ["Ada"]},
    ],
    "telecom": [
        {"system": "phone", "value": "555-0100"},
        {"system": "email", "value": "ada@example.org"},
    ],
    "gender": "female",
    "birthDate": "1990-12-10",
    "address": [
        {"line": ["123 Main St"], "city": "Boston", "state": "MA", "postalCode": "02108"}
    ],
    "managingOrganization": {"display": "Test Hospital"},
}


@pytest.fixture
def sample_lead_row() -> dict[str, str | None]:
    return dict(SAMPLE_LEAD_ROW)


@pytest.fixture
def raw_lead(sample_lead_row: dict[str, str | None]) -> RawLead:
    return RawLead(row_number=2, data=dict(sample_lead_row), raw_payload=dict(sample_lead_row))


@pytest.fixture
def sample_patient() -> dict:
    import copy

    return copy.deepcopy(SAMPLE_PATIENT)


@pytest.fixture
def lead_record() -> UnifiedRecord:
    return UnifiedRecord(
        id="lead-001",
        source=SourceType.LINKEDIN,
        record_type=RecordType.LEAD,
        person=Person(
            name="Rebecca Hart",
            role="Practice Manager",
            organization="Northgate Family Clinic",
        ),
        contact_info=ContactInfo(location="Austin TX"),
        context=RecordContext(
            summary="Rebecca Hart, Practice Manager at Northgate Family Clinic",
            facts={"recent_activity": "Liked a post about AI receptionists"},
            tags=["engaged", "healthcare"],
        ),
        priority_signal="recent engagement with relevant content",
        raw_payload={"lead_id": "L001"},
    )


@pytest.fixture
def patient_record(sample_patient: dict) -> UnifiedRecord:
    from app.normalization.patient_normalizer import normalize_patient

    return normalize_patient(sample_patient)
