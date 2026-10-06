"""Pipeline entry point.

Run with::

    python -m app.main

Flow: CSV + FHIR ingestion -> normalization -> AI agent -> mock CMS -> artifacts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app import __version__
from app.agent.agent import IntakeAgent
from app.agent.llm import FaultInjectionProvider, build_llm_provider
from app.cms.client import CMSClient, CMSDeliveryError
from app.config import ConfigError, Settings
from app.ingestion.csv_loader import CSVLoadError, RawLead, discover_csv, load_leads
from app.ingestion.fhir_client import FHIRClient, FHIRClientError
from app.models.agent_result import AgentResult
from app.models.unified_record import UnifiedRecord, UnifiedRecordValidationError
from app.normalization.lead_normalizer import normalize_lead
from app.normalization.patient_normalizer import normalize_patient
from app.utils.logger import get_logger, setup_logging
from app.utils.retry import RetryPolicy, RetryStats

logger = get_logger("main")

#: Deliberately malformed rows injected in --demo-failure mode.
DEMO_MALFORMED_LEAD = RawLead(
    row_number=9999,
    data={"lead_id": "", "full_name": "", "company": None},
    raw_payload={"lead_id": "", "full_name": ""},
)
DEMO_MALFORMED_PATIENTS: list[dict[str, Any]] = [
    {"resourceType": "Patient"},  # missing id and name
    {"resourceType": "Patient", "id": "demo-no-name", "gender": "unknown"},  # missing name
]


@dataclass
class PipelineSummary:
    """Counters reported at the end of a run."""

    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0
    csv_records_loaded: int = 0
    fhir_records_fetched: int = 0
    normalized_records: int = 0
    malformed_skipped: int = 0
    total_records: int = 0
    processed_ok: int = 0
    failed: int = 0
    cms_delivered: int = 0
    cms_failed: int = 0
    retries: int = 0
    recoveries: int = 0
    llm_failures: int = 0
    quota_blocked: int = 0
    provider: str = ""
    demo_failure_mode: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Stage 1 + 2: ingestion and normalization
# --------------------------------------------------------------------------- #
def ingest_and_normalize_leads(settings: Settings, summary: PipelineSummary) -> list[UnifiedRecord]:
    """Stage 1A/2A: CSV -> UnifiedRecord."""
    try:
        csv_path = Path(settings.leads_csv_path) if settings.leads_csv_path else discover_csv()
        load_result = load_leads(csv_path)
    except CSVLoadError as exc:
        logger.error("CSV ingestion failed error=%s", exc)
        summary.notes.append(f"CSV ingestion failed: {exc}")
        return []

    summary.csv_records_loaded = load_result.loaded_count
    records: list[UnifiedRecord] = []

    rows = list(load_result.leads)
    if summary.demo_failure_mode:
        logger.warning("Demo failure mode: injecting 1 malformed lead row")
        rows.append(DEMO_MALFORMED_LEAD)

    for raw in rows:
        try:
            records.append(normalize_lead(raw))
        except UnifiedRecordValidationError as exc:
            summary.malformed_skipped += 1
            logger.error(
                "Record validation failed record_id=%s row=%s error=%s",
                raw.lead_id or "<missing>", raw.row_number, exc,
            )
            logger.warning("Skipping malformed record and continuing")

    summary.normalized_records = len(records)
    logger.info("Normalized %s LinkedIn lead records", len(records))
    return records


async def ingest_and_normalize_patients(
    settings: Settings, summary: PipelineSummary, stats: RetryStats
) -> list[UnifiedRecord]:
    """Stage 1B/2B: FHIR Patient -> UnifiedRecord."""
    policy = RetryPolicy(
        attempts=settings.max_retries,
        base_delay=settings.retry_backoff_seconds,
    )
    simulated = 1 if summary.demo_failure_mode else 0
    records: list[UnifiedRecord] = []

    try:
        async with FHIRClient(
            settings.fhir_base_url,
            timeout=settings.http_timeout_seconds,
            policy=policy,
            stats=stats,
            simulated_failures=simulated,
        ) as client:
            resources = await client.fetch_patients(settings.fhir_patient_count)

            conditions_by_patient: dict[str, list[dict[str, Any]]] = {}
            if settings.fhir_fetch_conditions:
                for resource in resources:
                    conditions_by_patient[str(resource["id"])] = await client.fetch_conditions(
                        str(resource["id"])
                    )
    except FHIRClientError as exc:
        logger.error("FHIR ingestion failed error=%s", exc)
        summary.notes.append(f"FHIR ingestion failed: {exc}")
        return []
    except Exception as exc:  # noqa: BLE001 - one bad source must not crash the run
        logger.exception("FHIR ingestion unexpected failure error=%s", exc)
        summary.notes.append(f"FHIR ingestion failed unexpectedly: {exc}")
        return []

    summary.fhir_records_fetched = len(resources)

    if summary.demo_failure_mode:
        logger.warning("Demo failure mode: injecting 2 malformed FHIR Patient resources")
        resources = [*resources, *DEMO_MALFORMED_PATIENTS]

    for resource in resources:
        patient_id = resource.get("id", "<missing>")
        try:
            records.append(
                normalize_patient(
                    resource,
                    conditions=conditions_by_patient.get(str(patient_id)),
                )
            )
        except UnifiedRecordValidationError as exc:
            summary.malformed_skipped += 1
            logger.error(
                "Record validation failed record_id=patient-%s error=%s", patient_id, exc
            )
            logger.warning("Skipping malformed FHIR record and continuing")

    logger.info("Normalized %s FHIR patient records", len(records))
    return records


# --------------------------------------------------------------------------- #
# Stage 3 + 4: agent, validation and CMS delivery
# --------------------------------------------------------------------------- #
async def deliver_to_cms(
    results: list[tuple[UnifiedRecord, AgentResult]],
    settings: Settings,
    summary: PipelineSummary,
    stats: RetryStats,
    *,
    demo_failure: bool,
    skip_cms: bool,
) -> dict[str, str]:
    """Push structured results to the mock CMS over real HTTP POSTs."""
    statuses: dict[str, str] = {}
    if skip_cms:
        logger.warning("CMS delivery skipped by --skip-cms")
        for record, _ in results:
            statuses[record.id] = "skipped"
        return statuses

    policy = RetryPolicy(
        attempts=settings.max_retries,
        base_delay=settings.retry_backoff_seconds,
    )
    async with CMSClient(
        settings.cms_base_url,
        timeout=settings.http_timeout_seconds,
        policy=policy,
        stats=stats,
    ) as client:
        try:
            health = await client.health()
            logger.info(
                "CMS health check ok status=%s records=%s url=%s",
                health.get("status"), health.get("record_count"), settings.cms_base_url,
            )
        except CMSDeliveryError as exc:
            logger.error("CMS health check failed error=%s", exc)
            logger.warning("Continuing anyway; each delivery will be retried individually")

        attempted = 0
        for record, result in results:
            simulate: str | None = None
            if demo_failure:
                if attempted == 0:
                    simulate = "once"
                    logger.warning(
                        "Demo failure mode: first delivery will receive a transient 503"
                    )
                elif attempted == 2:
                    simulate = "always"
                    logger.warning(
                        "Demo failure mode: delivery for %s will receive persistent 503s",
                        record.id,
                    )
            attempted += 1

            try:
                await client.deliver(result, simulate=simulate)
            except CMSDeliveryError as exc:
                summary.cms_failed += 1
                statuses[record.id] = f"failed: {exc}"
                logger.error(
                    "CMS delivery failed record_id=%s error=%s", record.id, exc
                )
                logger.warning("Continuing with remaining records")
                continue

            summary.cms_delivered += 1
            statuses[record.id] = "delivered"

    return statuses


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
def write_artifacts(
    output_dir: Path,
    *,
    records: list[UnifiedRecord],
    outcomes: list[Any],
    statuses: dict[str, str],
    summary: PipelineSummary,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    processed: list[dict[str, Any]] = []
    for outcome in outcomes:
        if outcome.result is None:
            continue
        record = outcome.record
        processed.append(
            {
                "record": {
                    "id": record.id,
                    "source": record.source.value,
                    "record_type": record.record_type.value,
                    "name": record.person.name,
                    "priority_signal": record.priority_signal,
                },
                "agent_result": outcome.result.model_dump(mode="json"),
                "delivery": {"status": statuses.get(record.id, "not_attempted")},
            }
        )

    _write_json(output_dir / "processed_records.json", processed)
    _write_json(
        output_dir / "normalized_records.json",
        [record.model_dump(mode="json") for record in records],
    )
    _write_json(output_dir / "pipeline_summary.json", summary.to_dict())
    logger.info(
        "Artifacts written to %s (processed_records.json, normalized_records.json, pipeline_summary.json)",
        output_dir,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
async def run_pipeline(
    settings: Settings,
    *,
    skip_csv: bool = False,
    skip_fhir: bool = False,
    skip_cms: bool = False,
    demo_failure: bool = False,
    limit: int | None = None,
) -> PipelineSummary:
    started = datetime.now(timezone.utc)
    summary = PipelineSummary(
        started_at=started.isoformat(),
        demo_failure_mode=demo_failure,
        provider=settings.llm_provider,
    )
    stats = RetryStats()
    logger.info(
        "Pipeline started version=%s provider=%s model=%s cms=%s",
        __version__,
        settings.llm_provider,
        settings.gemini_model,
        settings.cms_base_url,
    )

    # Fail fast before any network work if the LLM cannot run.
    settings.require_llm_credentials()

    # ---- Stage 1 + 2 -------------------------------------------------- #
    records: list[UnifiedRecord] = []
    if not skip_csv:
        records.extend(ingest_and_normalize_leads(settings, summary))
    else:
        logger.warning("CSV ingestion skipped by --skip-csv")

    if not skip_fhir:
        records.extend(await ingest_and_normalize_patients(settings, summary, stats))
    else:
        logger.warning("FHIR ingestion skipped by --skip-fhir")

    summary.normalized_records = len(records)
    logger.info("Normalized %s records in total", len(records))

    if limit is not None and limit < len(records):
        logger.info("Limiting run to %s of %s records (--limit)", limit, len(records))
        records = records[:limit]

    if not records:
        logger.error("No records to process - aborting pipeline")
        summary.notes.append("no records available")
        return summary

    # ---- Stage 3: AI agent -------------------------------------------- #
    provider = build_llm_provider(settings, stats)
    if demo_failure:
        provider = FaultInjectionProvider(provider, failures=1)
        logger.warning("Demo failure mode: first LLM response will be invalid JSON")

    agent = IntakeAgent(provider, stats=stats)
    try:
        outcomes = await agent.process_many(records)
    finally:
        await provider.aclose()

    summary.total_records = len(outcomes)
    summary.processed_ok = sum(1 for outcome in outcomes if outcome.ok)
    summary.failed = summary.total_records - summary.processed_ok
    # LLM failures = provider call failures (timeouts/5xx/bad key) plus
    # invalid-output events, including the ones later repaired.
    summary.llm_failures = sum(
        count
        for label, count in stats.failures.items()
        if "llm" in label.lower() or "gemini" in label.lower()
    )
    summary.quota_blocked = sum(1 for outcome in outcomes if outcome.quota_blocked)
    if summary.quota_blocked:
        summary.notes.append(
            f"Gemini free-tier quota exhausted during the run: "
            f"{summary.quota_blocked} record(s) were NOT processed. "
            f"No offline/hardcoded fallback was used - they failed loudly."
        )
        logger.error(
            "Gemini quota blocked %s record(s); processed_by=gemini for the rest only",
            summary.quota_blocked,
        )

    results = [
        (outcome.record, outcome.result)
        for outcome in outcomes
        if outcome.result is not None
    ]

    # ---- Stage 4: mock CMS -------------------------------------------- #
    statuses = await deliver_to_cms(
        results, settings, summary, stats,
        demo_failure=demo_failure, skip_cms=skip_cms,
    )

    # ---- Artifacts + summary ------------------------------------------ #
    summary.retries = stats.attempts
    summary.recoveries = stats.recoveries
    finished = datetime.now(timezone.utc)
    summary.finished_at = finished.isoformat()
    summary.duration_seconds = round((finished - started).total_seconds(), 2)

    write_artifacts(
        settings.output_dir,
        records=records,
        outcomes=outcomes,
        statuses=statuses,
        summary=summary,
    )

    logger.info(
        "Pipeline completed total=%s processed_ok=%s failed=%s cms_delivered=%s "
        "cms_failed=%s retries=%s llm_failures=%s quota_blocked=%s "
        "malformed_skipped=%s duration=%ss",
        summary.total_records,
        summary.processed_ok,
        summary.failed,
        summary.cms_delivered,
        summary.cms_failed,
        summary.retries,
        summary.llm_failures,
        summary.quota_blocked,
        summary.malformed_skipped,
        summary.duration_seconds,
    )
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.main",
        description="AI agent pipeline: ingest leads + FHIR patients, reason, deliver to CMS.",
    )
    parser.add_argument("--csv", help="Path to the leads CSV (default: auto-discover data/mock_leads*.csv)")
    parser.add_argument("--cms-url", help="Mock CMS base URL (default: env CMS_BASE_URL)")
    parser.add_argument("--fhir-count", type=int, help="Number of FHIR patients to fetch (min 15)")
    parser.add_argument("--limit", type=int, help="Process at most N records")
    parser.add_argument("--provider", choices=["gemini", "offline"], help="Override LLM provider")
    parser.add_argument("--skip-csv", action="store_true", help="Skip the CSV source")
    parser.add_argument("--skip-fhir", action="store_true", help="Skip the FHIR source")
    parser.add_argument("--skip-cms", action="store_true", help="Skip CMS delivery (offline analysis)")
    parser.add_argument(
        "--demo-failure",
        action="store_true",
        help="Inject controlled failures (malformed record, invalid LLM output, CMS 503s)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        print(f"[config error] {exc}", file=sys.stderr)
        return 2

    overrides: dict[str, Any] = {}
    if args.csv:
        overrides["leads_csv_path"] = args.csv
    if args.cms_url:
        overrides["cms_base_url"] = args.cms_url.rstrip("/")
    if args.fhir_count:
        overrides["fhir_patient_count"] = max(args.fhir_count, 15)
    if args.provider:
        overrides["llm_provider"] = args.provider
    if overrides:
        settings = settings.with_overrides(**overrides)

    setup_logging(settings.log_level, settings.output_dir / "pipeline.log")

    try:
        summary = asyncio.run(
            run_pipeline(
                settings,
                skip_csv=args.skip_csv,
                skip_fhir=args.skip_fhir,
                skip_cms=args.skip_cms,
                demo_failure=args.demo_failure,
                limit=args.limit,
            )
        )
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        print(f"\n[config error] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        logger.warning("Pipeline interrupted by user")
        return 130

    print(_render_summary(summary))
    return 0 if summary.processed_ok > 0 else 1


def _render_summary(summary: PipelineSummary) -> str:
    lines = [
        "",
        "=" * 62,
        " PIPELINE SUMMARY",
        "=" * 62,
        f" Total records ........... {summary.total_records}",
        f" CSV leads loaded ........ {summary.csv_records_loaded}",
        f" FHIR patients fetched ... {summary.fhir_records_fetched}",
        f" Normalized records ...... {summary.normalized_records}",
        f" Successfully processed .. {summary.processed_ok}",
        f" Failed .................. {summary.failed}",
        f" Malformed skipped ....... {summary.malformed_skipped}",
        f" CMS delivered ........... {summary.cms_delivered}",
        f" CMS failed .............. {summary.cms_failed}",
        f" Retries ................. {summary.retries}",
        f" LLM failures ............ {summary.llm_failures}",
        f" Gemini quota blocked .... {summary.quota_blocked}",
        f" Provider ................ {summary.provider}",
        f" Duration ................ {summary.duration_seconds}s",
        "=" * 62,
    ]
    if summary.notes:
        lines.append(" Notes:")
        lines.extend(f"   - {note}" for note in summary.notes)
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
