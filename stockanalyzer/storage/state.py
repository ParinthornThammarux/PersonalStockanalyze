"""SQLite application state: update log, watchlists, alerts, transactions, caches.

SQLite (not DuckDB) on purpose: the nightly job and the Streamlit app touch this at the
same time, and SQLite in WAL mode handles that where a DuckDB file would lock.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS update_log (
    symbol TEXT NOT NULL, interval TEXT NOT NULL, last_attempt TEXT NOT NULL,
    status TEXT NOT NULL, message TEXT NOT NULL DEFAULT '', rows_added INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (symbol, interval)
);
CREATE TABLE IF NOT EXISTS watchlists (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS watchlist_items (
    watchlist_id INTEGER NOT NULL REFERENCES watchlists(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL, note TEXT NOT NULL DEFAULT '', added_at TEXT NOT NULL,
    PRIMARY KEY (watchlist_id, symbol)
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, kind TEXT NOT NULL,
    params TEXT NOT NULL DEFAULT '{}', active INTEGER NOT NULL DEFAULT 1,
    cooldown_hours REAL NOT NULL DEFAULT 24, created_at TEXT NOT NULL, last_triggered_at TEXT
);
CREATE TABLE IF NOT EXISTS alert_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, alert_id INTEGER, symbol TEXT NOT NULL,
    triggered_at TEXT NOT NULL, message TEXT NOT NULL, value REAL
);
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, portfolio TEXT NOT NULL DEFAULT 'default',
    ts TEXT NOT NULL, symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    quantity REAL NOT NULL CHECK (quantity > 0), price REAL NOT NULL CHECK (price >= 0),
    fees REAL NOT NULL DEFAULT 0, note TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS fundamentals_cache (
    symbol TEXT PRIMARY KEY, fetched_at TEXT NOT NULL, data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT NOT NULL, finished_at TEXT,
    status TEXT NOT NULL, summary TEXT
);
"""


def to_iso(moment: datetime) -> str:
    """UTC, second precision, naive text - the format every timestamp column uses."""
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment.replace(microsecond=0).isoformat(sep=" ")


def utcnow_iso() -> str:
    return to_iso(datetime.now(timezone.utc))


class StateDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Run a write statement; returns ``lastrowid``."""
        with self.connect() as conn:
            return conn.execute(sql, params).lastrowid

    def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        with self.connect() as conn:
            conn.executemany(sql, rows)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict]:
        with self.connect() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> dict | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None
