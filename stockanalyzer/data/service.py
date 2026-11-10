"""``MarketData``: the one entry point the UI, screener, backtester and ML code use for prices.

Daily bars come from the local store (filled on demand, kept fresh incrementally);
weekly/monthly bars are resampled from daily; intraday bars are fetched live and not stored.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

import pandas as pd

from stockanalyzer.config import Settings, get_settings
from stockanalyzer.data.market_clock import last_completed_session
from stockanalyzer.data.providers import default_provider
from stockanalyzer.data.providers.base import (
    PriceProvider,
    empty_ohlcv,
    is_intraday,
    normalize_ohlcv,
    period_to_start,
)
from stockanalyzer.data.store import PriceStore
from stockanalyzer.data.updater import Updater, UpdateReport
from stockanalyzer.storage.state import StateDB

log = logging.getLogger(__name__)

_RESAMPLE = {"1wk": "W-FRI", "1mo": "ME"}
_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "adj_close": "last",
        "volume": "sum", "dividends": "sum", "splits": "sum"}


def adjust_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Scale open/high/low/close by ``adj_close / close`` (splits *and* dividends)."""
    if df.empty:
        return df
    factor = (df["adj_close"] / df["close"]).where(df["close"] > 0, 1.0)
    out = df.copy()
    for col in ("open", "high", "low"):
        out[col] = df[col] * factor
    out["close"] = df["adj_close"]
    return out


def resample_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    rule = _RESAMPLE[interval]
    out = df.resample(rule).agg(_AGG).dropna(subset=["close"])
    out.index.name = "ts"
    return out


class MarketData:
    def __init__(
        self,
        provider: PriceProvider | None = None,
        store: PriceStore | None = None,
        state: StateDB | None = None,
        *,
        settings: Settings | None = None,
        now: Callable[[], datetime] | None = None,
        retry_minutes: int = 30,
    ):
        self.settings = settings or get_settings()
        self.provider = provider or default_provider()
        self.store = store or PriceStore(self.settings.prices_dir)
        self.state = state or StateDB(self.settings.state_db)
        self._now = now or (lambda: datetime.now(timezone.utc))
        self.retry_minutes = retry_minutes
        self.updater = Updater(self.provider, self.store, self.state, now=self._now)

    # -- store maintenance -------------------------------------------------------------------
    def update(self, symbols: Iterable[str], **kwargs) -> UpdateReport:
        return self.updater.update(symbols, **kwargs)

    def backfill(self, symbols: Iterable[str], **kwargs) -> UpdateReport:
        """Replace each symbol with its whole history (back to 1993 for SPY, 2008 for TDEX), see ``Updater.backfill``."""
        return self.updater.backfill(symbols, **kwargs)

    def first_timestamps(self, symbols: Iterable[str]) -> dict[str, pd.Timestamp]:
        """Oldest stored daily bar per symbol (symbols with nothing stored are left out)."""
        out = {}
        for s in dict.fromkeys(x.strip().upper() for x in symbols):
            df = self.store.read(s)
            if not df.empty:
                out[s] = df.index.min()
        return out

    def last_error(self, symbol: str) -> str | None:
        row = self.state.query_one(
            "SELECT status, message FROM update_log WHERE symbol=? AND interval='1d'", (symbol.upper(),)
        )
        return row["message"] if row and row["status"] in ("error", "no_data") else None

    def _recently_attempted(self, symbol: str) -> bool:
        row = self.state.query_one(
            "SELECT last_attempt FROM update_log WHERE symbol=? AND interval='1d'", (symbol,)
        )
        if not row:
            return False
        at = datetime.fromisoformat(row["last_attempt"]).replace(tzinfo=timezone.utc)
        return self._now() - at < timedelta(minutes=self.retry_minutes)

    def ensure(self, symbol: str, *, force: bool = False) -> None:
        """Bring one symbol up to date, at most once per ``retry_minutes`` unless forced."""
        symbol = symbol.strip().upper()
        last = self.store.last_timestamps([symbol]).get(symbol)
        if not force and last is not None and last >= last_completed_session(symbol, self._now()):
            return
        if not force and self._recently_attempted(symbol):
            return
        report = self.updater.update([symbol], force=force)
        if report.failed:
            log.warning("update of %s failed: %s", symbol, report.failed[symbol])

    # -- prices ------------------------------------------------------------------------------
    def prices(
        self,
        symbol: str,
        *,
        period: str | None = "1y",
        start=None,
        end=None,
        interval: str = "1d",
        adjusted: bool = True,
        live: bool = True,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Canonical OHLCV for one symbol (empty frame if nothing is available)."""
        symbol = symbol.strip().upper()

        if is_intraday(interval):
            try:
                kw = {"start": start, "end": end} if start is not None else {"period": period or "5d"}
                return self.provider.history(symbol, interval=interval, **kw)
            except Exception as exc:
                log.warning("intraday fetch for %s failed: %s", symbol, exc)
                return empty_ohlcv()

        self.ensure(symbol, force=refresh)
        df = self.store.read(symbol)
        if live:
            df = self._with_live_bar(symbol, df)
        if df.empty:
            return df

        if interval in _RESAMPLE:
            df = resample_ohlcv(df, interval)
        elif interval != "1d":
            raise ValueError(f"unsupported interval {interval!r}")

        lo = pd.Timestamp(start) if start is not None else (
            period_to_start(period, df.index[-1]) if period else None
        )
        if lo is not None or end is not None:
            df = df.loc[lo : pd.Timestamp(end) if end is not None else None]
        return adjust_ohlc(df) if adjusted else df

    def _with_live_bar(self, symbol: str, stored: pd.DataFrame) -> pd.DataFrame:
        """Append today's in-progress bar (not persisted) so charts show the latest price."""
        completed = last_completed_session(symbol, self._now())
        if not stored.empty and stored.index[-1] > completed:
            return stored
        try:
            recent = normalize_ohlcv(self.provider.history(symbol, period="5d"))
        except Exception as exc:
            log.debug("live bar for %s unavailable: %s", symbol, exc)
            return stored
        newer = recent[recent.index > (stored.index[-1] if not stored.empty else pd.Timestamp.min)]
        if stored.empty or newer.empty:
            return stored
        return pd.concat([stored, newer])

    def prices_many(
        self, symbols: Iterable[str], *, period: str | None = "1y", start=None, adjusted: bool = True, live: bool = True
    ) -> dict[str, pd.DataFrame]:
        return {s: df for s in dict.fromkeys(x.upper() for x in symbols)
                if not (df := self.prices(s, period=period, start=start, adjusted=adjusted, live=live)).empty}

    def close_matrix(
        self, symbols: Iterable[str], *, period: str | None = "2y", start=None, live: bool = True,
        ffill: bool = True,
    ) -> pd.DataFrame:
        """Adjusted closes side by side (dates x symbols), forward-filled over non-shared holidays.

        ``live=False`` leaves out today's unfinished bar and saves one network call per symbol.
        ``ffill=False`` leaves NaN where a symbol did not trade (a Thai holiday next to a US stock), which
        correlation and cointegration need: a filled holiday is a fake 0 % return.
        """
        frames = self.prices_many(symbols, period=period, start=start, live=live)
        if not frames:
            return pd.DataFrame()
        close = pd.DataFrame({s: df["close"] for s, df in frames.items()}).sort_index()
        return close.ffill() if ffill else close
