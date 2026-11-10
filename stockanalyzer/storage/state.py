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

TRANSACTION_COLUMNS = """
    id INTEGER PRIMARY KEY AUTOINCREMENT, portfolio TEXT NOT NULL DEFAULT 'default',
    ts TEXT NOT NULL, symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL', 'DIVIDEND', 'DEPOSIT', 'WITHDRAW')),
    quantity REAL NOT NULL CHECK (quantity > 0), price REAL NOT NULL CHECK (price >= 0),
    fees REAL NOT NULL DEFAULT 0, note TEXT NOT NULL DEFAULT '', currency TEXT NOT NULL DEFAULT ''
"""

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
    cooldown_hours REAL NOT NULL DEFAULT 24, created_at TEXT NOT NULL, last_triggered_at TEXT,
    last_bar TEXT, in_zone INTEGER
);
CREATE TABLE IF NOT EXISTS alert_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, alert_id INTEGER, symbol TEXT NOT NULL,
    triggered_at TEXT NOT NULL, message TEXT NOT NULL, value REAL
);
CREATE TABLE IF NOT EXISTS transactions ({TRANSACTION_COLUMNS});
CREATE TABLE IF NOT EXISTS fundamentals_cache (
    symbol TEXT PRIMARY KEY, fetched_at TEXT NOT NULL, data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS statements_archive (
    symbol TEXT NOT NULL, freq TEXT NOT NULL CHECK (freq IN ('a', 'q')),
    statement TEXT NOT NULL, period TEXT NOT NULL,
    first_seen TEXT NOT NULL, last_seen TEXT NOT NULL, data TEXT NOT NULL,
    PRIMARY KEY (symbol, freq, statement, period)
);
CREATE TABLE IF NOT EXISTS estimates_snapshots (
    symbol TEXT NOT NULL, day TEXT NOT NULL, fetched_at TEXT NOT NULL, data TEXT NOT NULL,
    PRIMARY KEY (symbol, day)
);
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT NOT NULL, finished_at TEXT,
    status TEXT NOT NULL, summary TEXT
);
CREATE TABLE IF NOT EXISTS saved_screens (
    name TEXT PRIMARY KEY, spec TEXT NOT NULL, saved_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    symbol TEXT NOT NULL, kind TEXT NOT NULL, day TEXT NOT NULL, title TEXT NOT NULL,
    region TEXT NOT NULL DEFAULT '', at TEXT, data TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (symbol, kind, day, title, region)
);
CREATE INDEX IF NOT EXISTS events_by_day ON events(day);
CREATE TABLE IF NOT EXISTS events_log (
    scope TEXT PRIMARY KEY, fetched_at TEXT NOT NULL, status TEXT NOT NULL, message TEXT NOT NULL DEFAULT ''
);
"""

SCHEMA = SCHEMA.replace("{TRANSACTION_COLUMNS}", TRANSACTION_COLUMNS)


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
            self._migrate_transactions(conn)
            self._migrate_alerts(conn)

    @staticmethod
    def _migrate_alerts(conn: sqlite3.Connection) -> None:
        """Databases from before alerts tracked what they had already seen: add the two state columns."""
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(alerts)")}
        for name, kind in (("last_bar", "TEXT"), ("in_zone", "INTEGER")):
            if name not in columns:
                conn.execute(f"ALTER TABLE alerts ADD COLUMN {name} {kind}")

    @staticmethod
    def _migrate_transactions(conn: sqlite3.Connection) -> None:
        """Databases from before dividends, cash movements and currencies: rebuild ``transactions``.

        SQLite cannot widen a CHECK constraint in place, so the table is copied into the new definition (ids and all).
        """
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(transactions)")}
        if "currency" in columns:
            return
        conn.execute("DROP TABLE IF EXISTS transactions_new")
        conn.execute(f"CREATE TABLE transactions_new ({TRANSACTION_COLUMNS})")
        conn.execute("INSERT INTO transactions_new(id, portfolio, ts, symbol, side, quantity, price, fees, note) "
                     "SELECT id, portfolio, ts, symbol, side, quantity, price, fees, note FROM transactions")
        conn.execute("DROP TABLE transactions")
        conn.execute("ALTER TABLE transactions_new RENAME TO transactions")

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
