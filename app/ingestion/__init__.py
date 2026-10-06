"""Ingestion layer: raw data acquisition (CSV leads + FHIR patients)."""

from app.ingestion.csv_loader import CSVLoadResult, RawLead, discover_csv, load_leads
from app.ingestion.fhir_client import FHIRClient, FHIRClientError

__all__ = [
    "CSVLoadResult",
    "FHIRClient",
    "FHIRClientError",
    "RawLead",
    "discover_csv",
    "load_leads",
]
