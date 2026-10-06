"""Environment-driven configuration for the intake pipeline.

All runtime knobs are read from environment variables (optionally loaded from a
``.env`` file) so that the same code runs unchanged in CI, locally, or in a
container. Secrets are never hardcoded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

#: The assignment requires at least 15 synthetic FHIR patients.
MIN_FHIR_PATIENTS = 15
SUPPORTED_LLM_PROVIDERS = ("gemini", "offline")


class ConfigError(RuntimeError):
    """Raised when required environment configuration is missing or invalid."""


def _read_str(key: str, default: str = "") -> str:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip()


def _read_int(key: str, default: int) -> int:
    raw = _read_str(key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {key}={raw!r} is not an integer") from exc


def _read_float(key: str, default: float) -> float:
    raw = _read_str(key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {key}={raw!r} is not a number") from exc


def _read_bool(key: str, default: bool) -> bool:
    raw = _read_str(key).lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"Environment variable {key}={raw!r} is not a boolean")


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of every configuration value used by the pipeline."""

    # LLM
    gemini_api_key: str
    gemini_model: str
    llm_provider: str
    llm_temperature: float
    llm_max_tokens: int

    # Downstream CMS
    cms_base_url: str
    cms_db_path: str

    # Sources
    leads_csv_path: str
    fhir_base_url: str
    fhir_patient_count: int
    fhir_fetch_conditions: bool

    # Resilience
    http_timeout_seconds: float
    max_retries: int
    retry_backoff_seconds: float

    # Observability
    log_level: str
    output_dir: Path

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Settings":
        """Load settings from the process environment, optionally via ``.env``."""
        load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)

        provider = _read_str("LLM_PROVIDER", "gemini").lower()
        if provider not in SUPPORTED_LLM_PROVIDERS:
            raise ConfigError(
                f"Unsupported LLM_PROVIDER={provider!r}. "
                f"Expected one of {SUPPORTED_LLM_PROVIDERS}."
            )

        patient_count = _read_int("FHIR_PATIENT_COUNT", 20)
        if patient_count < MIN_FHIR_PATIENTS:
            patient_count = MIN_FHIR_PATIENTS

        output_dir = Path(_read_str("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir

        return cls(
            gemini_api_key=_read_str("GEMINI_API_KEY"),
            gemini_model=_read_str("GEMINI_MODEL", "gemini-flash-latest"),
            llm_provider=provider,
            llm_temperature=_read_float("LLM_TEMPERATURE", 0.2),
            llm_max_tokens=_read_int("LLM_MAX_TOKENS", 1024),
            cms_base_url=_read_str("CMS_BASE_URL", "http://localhost:8000").rstrip("/"),
            cms_db_path=_read_str("CMS_DB_PATH", "./output/mock_cms.sqlite3"),
            leads_csv_path=_read_str("LEADS_CSV_PATH"),
            fhir_base_url=_read_str("FHIR_BASE_URL", "https://r4.smarthealthit.org").rstrip("/"),
            fhir_patient_count=patient_count,
            fhir_fetch_conditions=_read_bool("FHIR_FETCH_CONDITIONS", False),
            http_timeout_seconds=_read_float("HTTP_TIMEOUT_SECONDS", 20.0),
            max_retries=_read_int("MAX_RETRIES", 3),
            retry_backoff_seconds=_read_float("RETRY_BACKOFF_SECONDS", 1.0),
            log_level=_read_str("LOG_LEVEL", "INFO").upper(),
            output_dir=output_dir,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def with_overrides(self, **changes: object) -> "Settings":
        """Return a copy with CLI/runtime overrides applied."""
        return replace(self, **changes)  # type: ignore[arg-type]

    @property
    def uses_gemini(self) -> bool:
        return self.llm_provider == "gemini"

    def require_llm_credentials(self) -> None:
        """Fail fast with an actionable message when the LLM cannot run."""
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            raise ConfigError(
                "GEMINI_API_KEY is missing. The pipeline cannot call the LLM.\n"
                "  1. Get a free key at https://aistudio.google.com/apikey\n"
                "  2. Copy .env.example to .env and set GEMINI_API_KEY=<your key>\n"
                "     (or export GEMINI_API_KEY in your shell)\n"
                "  3. Re-run the pipeline.\n"
                "For a fully offline demo without an LLM, set LLM_PROVIDER=offline."
            )
