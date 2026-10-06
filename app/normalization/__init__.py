"""Normalization layer: source-specific mapping into :class:`UnifiedRecord`."""

from app.normalization.lead_normalizer import normalize_lead
from app.normalization.patient_normalizer import normalize_patient

__all__ = ["normalize_lead", "normalize_patient"]
