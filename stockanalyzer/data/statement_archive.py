"""Every statement period Yahoo ever returned, kept after Yahoo stops returning it.

Yahoo serves about four annual and five quarterly periods and drops the oldest when a new one is
filed, so a valuation history built from live calls never grows. Each fetch is merged into SQLite
instead; the history then lengthens by one period per report, and ``first_seen`` records when we
first saw a period (a rough point-in-time stamp - Yahoo gives no filing dates).

Yahoo restates per-share figures for splits, but a period it no longer returns was last restated
when we last saw it. ``load`` therefore applies every split dated after ``last_seen`` itself.
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from stockanalyzer.domain.financials import FinancialStatements
from stockanalyzer.storage.state import StateDB, to_iso

STATEMENTS = ("income", "balance", "cashflow")
PER_SHARE = frozenset({"Diluted EPS", "Basic EPS"})  # divide by the split ratio
SHARE_COUNTS = frozenset({"Diluted Average Shares", "Basic Average Shares", "Ordinary Shares Number",
                          "Share Issued", "Treasury Shares Number"})  # multiply by it


def _freq(quarterly: bool) -> str:
    return "q" if quarterly else "a"


def _split_adjust(items: dict[str, float], ratio: float) -> dict[str, float]:
    return {k: v / ratio if k in PER_SHARE else v * ratio if k in SHARE_COUNTS else v for k, v in items.items()}


class StatementArchive:
    def __init__(self, state: StateDB):
        self.state = state

    def save(self, stmts: FinancialStatements, now: datetime) -> None:
        """Upsert every period of ``stmts`` (a restated period replaces the stored one, ``first_seen`` stays)."""
        seen = to_iso(now)
        rows = []
        for name in STATEMENTS:
            frame = getattr(stmts, name)
            for period in frame.columns:
                items = {str(k): float(v) for k, v in frame[period].dropna().items()}
                if items:
                    rows.append((stmts.symbol, _freq(stmts.quarterly), name, pd.Timestamp(period).strftime("%Y-%m-%d"),
                                 seen, seen, json.dumps(items)))
        if rows:
            self.state.executemany(
                "INSERT INTO statements_archive(symbol, freq, statement, period, first_seen, last_seen, data) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(symbol, freq, statement, period) "
                "DO UPDATE SET last_seen=excluded.last_seen, data=excluded.data", rows)

    def load(self, symbol: str, quarterly: bool, splits: pd.Series | None = None) -> FinancialStatements:
        """The archive as statements (oldest period first); ``splits`` is ratio by split date (10.0 = 10-for-1)."""
        rows = self.state.query(
            "SELECT statement, period, last_seen, data FROM statements_archive WHERE symbol=? AND freq=?",
            (symbol, _freq(quarterly)))
        if splits is None:
            splits = pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
        splits = splits[splits > 0]
        columns: dict[str, dict[pd.Timestamp, dict[str, float]]] = {name: {} for name in STATEMENTS}
        for r in rows:
            items = json.loads(r["data"])
            last_seen = pd.Timestamp(r["last_seen"]).normalize()
            for ratio in splits[splits.index > last_seen].to_numpy():
                items = _split_adjust(items, float(ratio))
            columns[r["statement"]][pd.Timestamp(r["period"])] = items
        frames = {name: pd.DataFrame(cols).sort_index(axis=1) if cols else pd.DataFrame()
                  for name, cols in columns.items()}
        return FinancialStatements(symbol, quarterly=quarterly, **frames)

    def merge(self, fresh: FinancialStatements, now: datetime, splits: pd.Series | None = None) -> FinancialStatements:
        """Store what Yahoo returned and hand back everything we have, old periods included."""
        self.save(fresh, now)
        return self.load(fresh.symbol, fresh.quarterly, splits)
