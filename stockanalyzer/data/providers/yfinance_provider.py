"""Yahoo Finance (via yfinance): prices, statements and company info."""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

from stockanalyzer.data.providers.base import (
    ProviderError,
    RateLimitError,
    empty_ohlcv,
    normalize_ohlcv,
)
from stockanalyzer.domain.financials import FinancialStatements

log = logging.getLogger(__name__)


def _wrap(exc: Exception) -> ProviderError:
    text = f"{type(exc).__name__}: {exc}"
    if "RateLimit" in type(exc).__name__ or "Too Many Requests" in text:
        return RateLimitError(text)
    return ProviderError(text)


def _day(value) -> str | None:
    return None if value is None else pd.Timestamp(value).strftime("%Y-%m-%d")


def _orient_statement(df: pd.DataFrame | None) -> pd.DataFrame:
    """Yahoo returns newest-first columns; we want oldest-first, numeric, datetime columns."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = pd.DatetimeIndex(pd.to_datetime(out.columns)).tz_localize(None)
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return out.sort_index(axis=1)


class YFinanceProvider:
    name = "yfinance"

    def __init__(self, *, threads: bool = True):
        self.threads = threads

    # -- prices ---------------------------------------------------------------------------
    def history(self, symbol, *, start=None, end=None, period=None, interval="1d") -> pd.DataFrame:
        import yfinance as yf

        kwargs: dict = {"interval": interval, "auto_adjust": False, "actions": True}
        if start is not None:
            kwargs.update(start=_day(start), end=_day(end))
        else:
            kwargs["period"] = period or "1y"
        try:
            raw = yf.Ticker(symbol).history(**kwargs)
        except Exception as exc:  # yfinance raises many unrelated types
            raise _wrap(exc) from exc
        return normalize_ohlcv(raw, interval=interval)

    def history_many(
        self, symbols: Sequence[str], *, start=None, end=None, period=None, interval="1d"
    ) -> dict[str, pd.DataFrame]:
        import yfinance as yf

        symbols = list(dict.fromkeys(symbols))
        if not symbols:
            return {}
        if len(symbols) == 1:
            df = self.history(symbols[0], start=start, end=end, period=period, interval=interval)
            return {symbols[0]: df} if not df.empty else {}

        kwargs: dict = {
            "interval": interval,
            "auto_adjust": False,
            "actions": True,
            "group_by": "ticker",
            "threads": self.threads,
            "progress": False,
        }
        if start is not None:
            kwargs.update(start=_day(start), end=_day(end))
        else:
            kwargs["period"] = period or "1y"
        try:
            raw = yf.download(symbols, **kwargs)
        except Exception as exc:
            raise _wrap(exc) from exc
        if raw is None or raw.empty:
            return {}

        out: dict[str, pd.DataFrame] = {}
        if isinstance(raw.columns, pd.MultiIndex):
            for ticker in raw.columns.get_level_values(0).unique():
                frame = normalize_ohlcv(raw[ticker], interval=interval)
                if not frame.empty:
                    out[str(ticker)] = frame
        return out

    # -- fundamentals ----------------------------------------------------------------------
    def statements(self, symbol: str, *, quarterly: bool = False) -> FinancialStatements:
        import yfinance as yf

        try:
            t = yf.Ticker(symbol)
            if quarterly:
                inc, bal, cf = t.quarterly_income_stmt, t.quarterly_balance_sheet, t.quarterly_cashflow
            else:
                inc, bal, cf = t.income_stmt, t.balance_sheet, t.cashflow
        except Exception as exc:
            raise _wrap(exc) from exc
        return FinancialStatements(
            symbol=symbol,
            income=_orient_statement(inc),
            balance=_orient_statement(bal),
            cashflow=_orient_statement(cf),
            quarterly=quarterly,
        )

    def info(self, symbol: str) -> dict:
        import yfinance as yf

        try:
            return dict(yf.Ticker(symbol).info or {})
        except Exception as exc:
            log.warning("info(%s) failed: %s", symbol, exc)
            return {}


__all__ = ["YFinanceProvider", "empty_ohlcv"]
