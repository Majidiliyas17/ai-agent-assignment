"""Robust loader for the LinkedIn-style ``mock_leads.csv`` export.

The loader is intentionally defensive:

* inspects the header instead of assuming a fixed column order,
* requires only the fields we genuinely need (``lead_id``, ``full_name``),
* normalizes empty cells to ``None``,
* skips (and logs) malformed rows instead of aborting the whole run,
* preserves the original raw record for every accepted row,
* never modifies the source file.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.config import DEFAULT_DATA_DIR

logger = logging.getLogger("app.ingestion.csv")

#: Minimum columns that must exist for a CSV to be treated as a lead export.
REQUIRED_COLUMNS: tuple[str, ...] = ("lead_id", "full_name")


class CSVLoadError(ValueError):
    """Raised when the CSV file itself cannot be used (missing/empty/invalid)."""


@dataclass
class RawLead:
    """A single accepted CSV row."""

    row_number: int
    data: dict[str, str | None]
    raw_payload: dict[str, str | None] = field(default_factory=dict)

    @property
    def lead_id(self) -> str:
        return self.data.get("lead_id") or ""


@dataclass
class SkippedRow:
    row_number: int
    reason: str


@dataclass
class CSVLoadResult:
    path: Path
    columns: list[str]
    leads: list[RawLead]
    skipped: list[SkippedRow]
    total_rows: int

    @property
    def loaded_count(self) -> int:
        return len(self.leads)


def discover_csv(data_dir: Path | None = None) -> Path:
    """Locate ``mock_leads*.csv`` inside the data directory."""
    directory = data_dir or DEFAULT_DATA_DIR
    if not directory.exists():
        raise CSVLoadError(f"Data directory not found: {directory}")

    candidates = sorted(directory.glob("mock_leads*.csv"))
    if not candidates:
        raise CSVLoadError(
            f"No mock_leads*.csv file found in {directory}. "
            "Provide one with --csv or set LEADS_CSV_PATH."
        )
    if len(candidates) > 1:
        logger.warning("Multiple lead CSVs found; using %s", candidates[0].name)
    return candidates[0]


def _clean_row(row: dict[str, str | None]) -> dict[str, str | None]:
    cleaned: dict[str, str | None] = {}
    for key, value in row.items():
        if key is None:  # extra cells beyond the header - ignore them
            continue
        if isinstance(value, str):
            value = value.strip()
            cleaned[key] = value or None
        else:
            cleaned[key] = value
    return cleaned


def load_leads(path: Path | str) -> CSVLoadResult:
    """Load and validate lead rows from ``path``.

    Raises :class:`CSVLoadError` for unusable files; individual bad rows are
    skipped and reported instead.
    """
    path = Path(path)
    if not path.exists():
        raise CSVLoadError(f"Lead CSV not found: {path}")
    if path.stat().st_size == 0:
        raise CSVLoadError(f"Lead CSV is empty: {path}")

    # utf-8-sig tolerates a BOM produced by Excel.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise CSVLoadError(f"Lead CSV has no header row: {path}")

        columns = [str(name).strip() for name in reader.fieldnames]
        missing = [col for col in REQUIRED_COLUMNS if col not in columns]
        if missing:
            raise CSVLoadError(
                f"Lead CSV {path.name} is missing required column(s): {missing}. "
                f"Found columns: {columns}"
            )

        leads: list[RawLead] = []
        skipped: list[SkippedRow] = []
        total_rows = 0

        for index, row in enumerate(reader, start=2):  # start=2: row 1 is the header
            total_rows += 1
            raw = {k: (v if isinstance(v, str) else v) for k, v in row.items() if k is not None}
            cleaned = _clean_row(row)

            missing_values = [col for col in REQUIRED_COLUMNS if not cleaned.get(col)]
            if missing_values:
                reason = f"missing required field(s): {', '.join(missing_values)}"
                skipped.append(SkippedRow(row_number=index, reason=reason))
                logger.error(
                    "Record validation failed row=%s reason=%s", index, reason
                )
                continue

            leads.append(RawLead(row_number=index, data=cleaned, raw_payload=raw))

    if not leads:
        raise CSVLoadError(
            f"Lead CSV {path.name} contained {total_rows} row(s) but 0 usable records."
        )

    logger.info(
        "Loaded %s LinkedIn leads from %s (rows=%s, skipped=%s)",
        len(leads), path.name, total_rows, len(skipped),
    )
    if skipped:
        logger.warning(
            "Skipped %s malformed CSV row(s); continuing with %s valid records",
            len(skipped), len(leads),
        )
    return CSVLoadResult(
        path=path,
        columns=columns,
        leads=leads,
        skipped=skipped,
        total_rows=total_rows,
    )
