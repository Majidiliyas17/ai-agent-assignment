"""Tests for the CSV ingestion layer."""

from pathlib import Path

import pytest

from app.ingestion.csv_loader import CSVLoadError, discover_csv, load_leads

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_discover_csv_finds_provided_file() -> None:
    found = discover_csv(PROJECT_ROOT / "data")
    assert found.exists()
    assert found.name.startswith("mock_leads")


def test_discover_csv_missing_dir() -> None:
    with pytest.raises(CSVLoadError):
        discover_csv(PROJECT_ROOT / "does-not-exist")


def test_load_real_leads_file(tmp_path: Path) -> None:
    result = load_leads(discover_csv(PROJECT_ROOT / "data"))
    assert result.loaded_count == 18
    assert "lead_id" in result.columns
    assert "full_name" in result.columns
    assert result.skipped == []
    assert result.leads[0].lead_id == "L001"
    # raw payload preserved untouched
    assert result.leads[0].raw_payload["company"] == "Northgate Family Clinic"
    # no file modification
    assert result.path.exists()


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CSVLoadError, match="not found"):
        load_leads(tmp_path / "nope.csv")


def test_load_empty_file(tmp_path: Path) -> None:
    path = _write(tmp_path / "empty.csv", "")
    with pytest.raises(CSVLoadError, match="empty"):
        load_leads(path)


def test_load_missing_required_columns(tmp_path: Path) -> None:
    path = _write(tmp_path / "bad.csv", "foo,bar\n1,2\n")
    with pytest.raises(CSVLoadError, match="missing required column"):
        load_leads(path)


def test_missing_required_values_are_skipped(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "mock_leads_partial.csv",
        "lead_id,full_name,company\n"
        "L001,Valid Person,Acme\n"
        ",No Id Person,Acme\n"
        "L003,,Acme\n",
    )
    result = load_leads(path)
    assert result.loaded_count == 1
    assert len(result.skipped) == 2
    assert result.total_rows == 3
    assert "missing required field(s)" in result.skipped[0].reason


def test_all_rows_invalid_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "mock_leads_allbad.csv", "lead_id,full_name\n,\n")
    with pytest.raises(CSVLoadError, match="0 usable records"):
        load_leads(path)


def test_null_cells_normalized(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "mock_leads_nulls.csv",
        "lead_id,full_name,company,industry\nL001,Ann Lee,,  \n",
    )
    result = load_leads(path)
    assert result.leads[0].data["company"] is None
    assert result.leads[0].data["industry"] is None
