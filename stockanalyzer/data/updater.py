"""Incremental price updates for any number of symbols.

* Only symbols that are behind the last completed session are fetched.
* Existing symbols are re-fetched from ``last bar - overlap`` so late revisions are caught.
* If the overlap no longer matches what we stored (split / dividend re-adjusted the
  history) that symbol is downloaded in full and replaced.
* Symbols are grouped by fetch-start so a whole chunk is one batched provider call, with
  retries and exponential backoff for rate limits.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from stockanalyzer.data.market_clock import last_completed_session
from stockanalyzer.data.providers.base import PriceProvider, RateLimitError, is_intraday
from stockanalyzer.data.providers.fred import FRED_PREFIX
from stockanalyzer.data.quality import clean_ohlcv
from stockanalyzer.data.store import PriceStore
from stockanalyzer.storage.state import StateDB, to_iso

log = logging.getLogger(__name__)


THROTTLE_MIN_BATCH = 10  # smaller batches are too small to tell throttling from a few dead tickers
THROTTLE_EMPTY_SHARE = 0.5


@dataclass
class UpdateReport:
    interval: str = "1d"
    requested: int = 0
    skipped_fresh: int = 0
    skipped_known_dead: int = 0
    updated: int = 0
    new_symbols: int = 0
    full_refreshes: list[str] = field(default_factory=list)
    rows_added: int = 0
    no_data: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    aborted: bool = False
    elapsed: float = 0.0

    def summary(self) -> str:
        parts = [
            f"{self.requested} requested", f"{self.updated} updated ({self.new_symbols} new)",
            f"{self.skipped_fresh} fresh", f"{self.rows_added} rows added",
        ]
        if self.full_refreshes:
            parts.append(f"{len(self.full_refreshes)} full refreshes (split/dividend)")
        if self.no_data:
            parts.append(f"{len(self.no_data)} no data")
        if self.skipped_known_dead:
            parts.append(f"{self.skipped_known_dead} skipped (no data recently)")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        if self.aborted:
            parts.append("ABORTED after repeated provider failures")
        return ", ".join(parts) + f" in {self.elapsed:.1f}s"


def _is_price_series(symbol: str) -> bool:
    """Macro series (``FRED:*``) may be zero or negative; real prices cannot."""
    return not symbol.startswith(FRED_PREFIX)


def adjustment_changed(old: pd.DataFrame, new: pd.DataFrame, rtol: float = 1e-4) -> bool:
    """True if bars present in both frames disagree on close / adj_close (split or dividend)."""
    common = old.index.intersection(new.index)
    if common.empty:
        return False
    for col in ("close", "adj_close"):
        a, b = old.loc[common, col].to_numpy(), new.loc[common, col].to_numpy()
        if not np.allclose(a, b, rtol=rtol, atol=0.0):
            return True
    return False


class Updater:
    def __init__(
        self,
        provider: PriceProvider,
        store: PriceStore,
        state: StateDB | None = None,
        *,
        chunk_size: int = 100,
        max_retries: int = 3,
        backoff: float = 2.0,
        max_consecutive_failures: int = 3,
        overlap_days: int = 7,
        default_history_years: int = 10,
        no_data_retry_days: int = 7,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] | None = None,
    ):
        self.provider = provider
        self.store = store
        self.state = state
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.backoff = backoff
        self.max_consecutive_failures = max_consecutive_failures
        self.overlap_days = overlap_days
        self.default_history_years = default_history_years
        self.no_data_retry_days = no_data_retry_days
        self._sleep = sleep
        self._now = now or (lambda: datetime.now(timezone.utc))

    # -- helpers ----------------------------------------------------------------------------
    def _default_start(self) -> pd.Timestamp:
        today = pd.Timestamp(self._now().date())
        return today - pd.DateOffset(years=self.default_history_years)

    def _log(self, symbol: str, interval: str, status: str, message: str = "", rows: int = 0) -> None:
        if self.state is None:
            return
        self.state.execute(
            "INSERT INTO update_log(symbol, interval, last_attempt, status, message, rows_added) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(symbol, interval) DO UPDATE SET "
            "last_attempt=excluded.last_attempt, status=excluded.status, "
            "message=excluded.message, rows_added=excluded.rows_added",
            (symbol, interval, to_iso(self._now()), status, message[:500], rows),
        )

    def _recent_no_data(self, symbols: list[str], interval: str) -> set[str]:
        if self.state is None or not symbols:
            return set()
        cutoff = to_iso(self._now() - timedelta(days=self.no_data_retry_days))
        rows = self.state.query(
            "SELECT symbol FROM update_log WHERE interval=? AND status='no_data' AND last_attempt>=?",
            (interval, cutoff),
        )
        wanted = set(symbols)
        return {r["symbol"] for r in rows if r["symbol"] in wanted}

    def _fetch(self, symbols: list[str], start: pd.Timestamp, interval: str) -> dict[str, pd.DataFrame]:
        """One batched provider call with retry/backoff; raises after the last attempt."""
        for attempt in range(self.max_retries):
            try:
                return self.provider.history_many(symbols, start=start, interval=interval)
            except Exception as exc:  # providers raise many types; all are retried the same way
                if attempt == self.max_retries - 1:
                    raise
                delay = self.backoff * (2**attempt) * (2 if isinstance(exc, RateLimitError) else 1)
                log.warning("fetch of %d symbols failed (%s); retry in %.0fs", len(symbols), exc, delay)
                self._sleep(delay)
        return {}  # unreachable

    # -- main entry -------------------------------------------------------------------------
    def update(
        self,
        symbols: Iterable[str],
        interval: str = "1d",
        *,
        start: str | pd.Timestamp | None = None,
        force: bool = False,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> UpdateReport:
        if is_intraday(interval):
            raise ValueError("the price store keeps completed daily-or-longer bars only")
        t0 = time.monotonic()
        syms = list(dict.fromkeys(s.strip().upper() for s in symbols if s and s.strip()))
        report = UpdateReport(interval=interval, requested=len(syms))
        last = self.store.last_timestamps(syms, interval)
        first_start = pd.Timestamp(start) if start is not None else self._default_start()

        # `force` is an explicit request to retry, so it ignores the "no data recently" memory
        known_dead = set() if force else self._recent_no_data([s for s in syms if s not in last], interval)
        report.skipped_known_dead = len(known_dead)

        todo: dict[pd.Timestamp, list[str]] = {}
        for s in syms:
            if s in known_dead:
                continue
            target = last_completed_session(s, self._now())
            if s in last:
                if not force and last[s] >= target:
                    report.skipped_fresh += 1
                    continue
                fetch_from = (last[s] - pd.Timedelta(days=self.overlap_days)).normalize()
            else:
                fetch_from = first_start
            todo.setdefault(fetch_from, []).append(s)

        batches = [(st, grp[i : i + self.chunk_size]) for st, grp in todo.items()
                   for i in range(0, len(grp), self.chunk_size)]
        done, total, consecutive_failures = 0, sum(len(b) for _, b in batches), 0

        for fetch_from, batch in batches:
            try:
                frames = self._fetch(batch, fetch_from, interval)
            except Exception as exc:
                consecutive_failures += 1
                for s in batch:
                    report.failed[s] = str(exc)
                    self._log(s, interval, "error", str(exc))
                done += len(batch)
                if progress:
                    progress(done, total, f"failed batch of {len(batch)}")
                if consecutive_failures >= self.max_consecutive_failures:
                    report.aborted = True
                    break
                continue

            missing = {s for s in batch if frames.get(s) is None or frames[s].empty}
            # Yahoo answers a throttled batch with empty frames instead of an error. Treating those as
            # "delisted" would hide valid tickers for a week, so a mostly-empty batch is a transient failure.
            throttled = len(batch) >= THROTTLE_MIN_BATCH and len(missing) > THROTTLE_EMPTY_SHARE * len(batch)
            for s in batch:
                if throttled and s in missing and s not in last:
                    msg = f"empty response for {len(missing)}/{len(batch)} symbols (rate limited?)"
                    report.failed[s] = msg
                    self._log(s, interval, "error", msg)
                else:
                    self._ingest(s, frames.get(s), s in last, fetch_from, interval, report)
            consecutive_failures = consecutive_failures + 1 if throttled else 0
            if consecutive_failures >= self.max_consecutive_failures:
                report.aborted = True
                done += len(batch)
                break
            done += len(batch)
            if progress:
                progress(done, total, batch[-1])

        report.elapsed = time.monotonic() - t0
        return report

    def _ingest(self, symbol, frame, existed: bool, fetch_from, interval, report: UpdateReport) -> None:
        if frame is None or frame.empty:
            if not existed:
                report.no_data.append(symbol)
                self._log(symbol, interval, "no_data", "provider returned no rows")
            else:
                self._log(symbol, interval, "ok", "no new rows")
            return

        frame = frame[frame.index <= last_completed_session(symbol, self._now())]  # drop partial bar
        frame, quality = clean_ohlcv(frame, interval=interval, require_positive=_is_price_series(symbol))
        if frame.empty:
            if not existed:
                report.no_data.append(symbol)
                self._log(symbol, interval, "no_data", "no completed bars")
            return

        try:
            replace = False
            if existed:
                old = self.store.read(symbol, interval, start=frame.index.min())
                if adjustment_changed(old, frame):
                    log.info("%s: history re-adjusted, downloading in full", symbol)
                    full = self.provider.history(symbol, start=self._default_start(), interval=interval)
                    full = full[full.index <= last_completed_session(symbol, self._now())]
                    frame, _ = clean_ohlcv(full, interval=interval, require_positive=_is_price_series(symbol))
                    replace = True
                    report.full_refreshes.append(symbol)
            added = self.store.write(symbol, frame, interval, replace=replace)
        except Exception as exc:
            report.failed[symbol] = str(exc)
            self._log(symbol, interval, "error", str(exc))
            return

        report.rows_added += added
        report.updated += 1
        if not existed:
            report.new_symbols += 1
        self._log(symbol, interval, "ok", quality.summary() if not quality.clean else "", added)
