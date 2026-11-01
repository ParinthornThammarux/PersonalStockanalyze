"""Fundamentals access: cached per-symbol snapshots for screening, live statements for deep dives."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

import pandas as pd

from stockanalyzer.analysis.fundamental.valuation import snapshot_from_info
from stockanalyzer.config import Settings, get_settings
from stockanalyzer.data.providers.base import FundamentalsProvider
from stockanalyzer.data.providers.yfinance_provider import YFinanceProvider
from stockanalyzer.domain.financials import FinancialStatements
from stockanalyzer.storage.state import StateDB, to_iso

log = logging.getLogger(__name__)

EMPTY_RETRY_DAYS = 1  # symbols Yahoo returned nothing for are retried sooner than real data


@dataclass
class RefreshReport:
    requested: int = 0
    fresh: int = 0
    refreshed: int = 0
    empty: list[str] = field(default_factory=list)
    elapsed: float = 0.0

    def summary(self) -> str:
        return (f"{self.requested} requested, {self.fresh} fresh, {self.refreshed} refreshed, "
                f"{len(self.empty)} empty in {self.elapsed:.1f}s")


class FundamentalsService:
    def __init__(
        self,
        provider: FundamentalsProvider | None = None,
        state: StateDB | None = None,
        *,
        settings: Settings | None = None,
        now: Callable[[], datetime] | None = None,
    ):
        self.settings = settings or get_settings()
        self.provider = provider or YFinanceProvider()
        self.state = state or StateDB(self.settings.state_db)
        self._now = now or (lambda: datetime.now(timezone.utc))

    # -- live ---------------------------------------------------------------------------------
    def statements(self, symbol: str, *, quarterly: bool = False) -> FinancialStatements:
        return self.provider.statements(symbol.upper(), quarterly=quarterly)

    def info(self, symbol: str) -> dict:
        return self.provider.info(symbol.upper())

    # -- cached snapshots ---------------------------------------------------------------------
    def _cached(self, symbols: Iterable[str] | None = None) -> dict[str, dict]:
        rows = self.state.query("SELECT symbol, fetched_at, data FROM fundamentals_cache")
        wanted = None if symbols is None else {s.upper() for s in symbols}
        return {r["symbol"]: {"fetched_at": r["fetched_at"], "data": json.loads(r["data"])}
                for r in rows if wanted is None or r["symbol"] in wanted}

    def _is_stale(self, entry: dict | None, max_age_days: float) -> bool:
        if entry is None:
            return True
        age_limit = EMPTY_RETRY_DAYS if entry["data"].get("_empty") else max_age_days
        fetched = datetime.fromisoformat(entry["fetched_at"]).replace(tzinfo=timezone.utc)
        return self._now() - fetched > timedelta(days=age_limit)

    def _store(self, symbol: str, data: dict) -> None:
        self.state.execute(
            "INSERT INTO fundamentals_cache(symbol, fetched_at, data) VALUES (?,?,?) "
            "ON CONFLICT(symbol) DO UPDATE SET fetched_at=excluded.fetched_at, data=excluded.data",
            (symbol, to_iso(self._now()), json.dumps(data)),
        )

    def snapshot(self, symbol: str, *, max_age_days: float = 7, refresh: bool = False) -> dict | None:
        symbol = symbol.upper()
        entry = self._cached([symbol]).get(symbol)
        if refresh or self._is_stale(entry, max_age_days):
            self.refresh([symbol], max_age_days=0 if refresh else max_age_days, workers=1)
            entry = self._cached([symbol]).get(symbol)
        data = entry["data"] if entry else None
        return None if not data or data.get("_empty") else data

    def _fetch_one(self, symbol: str) -> tuple[str, dict | None]:
        info = self.provider.info(symbol)
        if not info or not (info.get("marketCap") or info.get("quoteType")):
            return symbol, None
        return symbol, snapshot_from_info(info)

    def refresh(
        self,
        symbols: Iterable[str],
        *,
        max_age_days: float = 7,
        workers: int = 4,
        limit: int | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> RefreshReport:
        """Fetch snapshots for missing/stale symbols in parallel (oldest first, at most ``limit``)."""
        t0 = time.monotonic()
        syms = list(dict.fromkeys(s.upper() for s in symbols))
        report = RefreshReport(requested=len(syms))
        cached = self._cached(syms)
        stale = [s for s in syms if self._is_stale(cached.get(s), max_age_days)]
        report.fresh = len(syms) - len(stale)
        stale.sort(key=lambda s: cached[s]["fetched_at"] if s in cached else "")
        if limit is not None:
            stale = stale[:limit]

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(self._fetch_one, s) for s in stale]
            for done, fut in enumerate(as_completed(futures), start=1):
                try:
                    symbol, data = fut.result()
                except Exception as exc:  # one bad symbol must not stop the batch
                    log.warning("fundamentals fetch failed: %s", exc)
                    continue
                if data is None:
                    report.empty.append(symbol)
                    self._store(symbol, {"_empty": True})
                else:
                    report.refreshed += 1
                    self._store(symbol, data)
                if progress:
                    progress(done, len(stale), symbol)
        report.elapsed = time.monotonic() - t0
        return report

    def snapshots_frame(self, symbols: Iterable[str] | None = None) -> pd.DataFrame:
        """Cached snapshots as a frame indexed by symbol (for the screener)."""
        rows = {s: {**e["data"], "fundamentals_at": e["fetched_at"]}
                for s, e in self._cached(symbols).items() if not e["data"].get("_empty")}
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "symbol"
        return df.sort_index()
