"""Tiny SQLite persistence layer for the mock CMS.

SQLite is used instead of an in-memory dict so records survive a restart and
the demo genuinely persists downstream data.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from mock_cms.schemas import CMSRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    record_id     TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    record_type   TEXT NOT NULL,
    classification TEXT NOT NULL,
    action        TEXT NOT NULL,
    payload       TEXT NOT NULL,
    processed_at  TEXT,
    received_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_classification ON records(classification);
"""


class RecordStore:
    """Thread-safe SQLite store (one connection, guarded by a lock)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.path), check_same_thread=False, autocommit=True
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ #
    def upsert(self, record: CMSRecord) -> bool:
        """Insert or replace a record. Returns True when it was newly created."""
        payload = record.model_dump(mode="json", exclude={"received_at"})
        with self._lock:
            existing = self._conn.execute(
                "SELECT 1 FROM records WHERE record_id = ?", (record.record_id,)
            ).fetchone()
            self._conn.execute(
                """
                INSERT INTO records (
                    record_id, source, record_type, classification, action,
                    payload, processed_at, received_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(record_id) DO UPDATE SET
                    source = excluded.source,
                    record_type = excluded.record_type,
                    classification = excluded.classification,
                    action = excluded.action,
                    payload = excluded.payload,
                    processed_at = excluded.processed_at,
                    received_at = excluded.received_at
                """,
                (
                    record.record_id,
                    record.source,
                    record.record_type,
                    record.classification,
                    record.action,
                    json.dumps(payload, ensure_ascii=False),
                    record.processed_at.isoformat(),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return existing is None

    def get(self, record_id: str) -> CMSRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload, received_at FROM records WHERE record_id = ?",
                (record_id,),
            ).fetchone()
        if row is None:
            return None
        data = json.loads(row["payload"])
        data["received_at"] = row["received_at"]
        return CMSRecord.model_validate(data)

    def list(self, *, limit: int = 100, classification: str | None = None) -> list[CMSRecord]:
        query = "SELECT payload, received_at FROM records"
        params: list[object] = []
        if classification:
            query += " WHERE classification = ?"
            params.append(classification)
        query += " ORDER BY received_at DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            rows = self._conn.execute(query, params).fetchall()

        records: list[CMSRecord] = []
        for row in rows:
            data = json.loads(row["payload"])
            data["received_at"] = row["received_at"]
            records.append(CMSRecord.model_validate(data))
        return records

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM records").fetchone()
        return int(row["n"]) if row else 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()
