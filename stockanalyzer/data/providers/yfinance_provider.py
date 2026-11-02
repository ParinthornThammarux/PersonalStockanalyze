"""Yahoo Finance (via yfinance): prices, statements, company info and event calendars."""

from __future__ import annotations

import logging
import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from stockanalyzer.data.providers.base import (
    EARNINGS_COLUMNS,
    ECONOMIC_COLUMNS,
    ProviderError,
    RateLimitError,
    _canon,
    empty_ohlcv,
    normalize_ohlcv,
)
from stockanalyzer.domain.financials import FinancialStatements

log = logging.getLogger(__name__)

ECONOMIC_PAGE = 100  # Yahoo serves the economic calendar 100 rows at a time
ECONOMIC_MAX_PAGES = 60


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


def _numbers(raw: pd.DataFrame, *names: str) -> np.ndarray:
    """First of ``names`` (canonical spelling) present in ``raw`` as floats, else all-NaN."""
    cols = {_canon(c): c for c in raw.columns}
    for name in names:
        if name in cols:
            return pd.to_numeric(raw[cols[name]], errors="coerce").to_numpy(dtype="float64")
    return np.full(len(raw), np.nan)


def earnings_frame(raw: pd.DataFrame | None) -> pd.DataFrame:
    """``Ticker.get_earnings_dates()`` -> ``EARNINGS_COLUMNS``, oldest first."""
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=EARNINGS_COLUMNS)
    at = pd.DatetimeIndex(raw.index)
    at = (at.tz_localize("America/New_York") if at.tz is None else at).tz_convert("UTC")  # Yahoo quotes New York time
    out = pd.DataFrame({
        "at": at,
        "eps_estimate": _numbers(raw, "eps_estimate"),
        "eps_actual": _numbers(raw, "reported_eps", "eps_actual"),
        "surprise_pct": _numbers(raw, "surprise(%)", "surprise_(%)"),
    })
    return out.dropna(subset=["at"]).drop_duplicates("at").sort_values("at", ignore_index=True)


def _iso_day(value) -> str | None:
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(ts) else ts.strftime("%Y-%m-%d")


def _number(value) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else x


def calendar_dict(raw: dict | None) -> dict:
    """``Ticker.calendar`` -> the plain dict described by ``EventsProvider.calendar``."""
    raw = dict(raw or {})
    earnings = raw.get("Earnings Date") or []
    if not isinstance(earnings, (list, tuple)):
        earnings = [earnings]
    return {
        "earnings": [d for d in map(_iso_day, earnings) if d],
        "ex_dividend": _iso_day(raw.get("Ex-Dividend Date")),
        "dividend_payment": _iso_day(raw.get("Dividend Date")),
        "eps_estimate": _number(raw.get("Earnings Average")),
        "eps_low": _number(raw.get("Earnings Low")),
        "eps_high": _number(raw.get("Earnings High")),
        "revenue_estimate": _number(raw.get("Revenue Average")),
    }


def economic_frame(raw: pd.DataFrame | None) -> pd.DataFrame:
    """Pages of ``Calendars.get_economic_events_calendar()`` (index reset) -> ``ECONOMIC_COLUMNS``."""
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=ECONOMIC_COLUMNS)
    df = raw.copy()
    df.columns = [_canon(c) for c in df.columns]

    def text(col: str) -> pd.Series:
        s = df[col].astype("string").str.strip() if col in df.columns else pd.Series("", index=df.index, dtype="string")
        return s.replace({"N/A": "", "nan": ""}).fillna("")

    out = pd.DataFrame({
        "at": pd.to_datetime(df["event_time"], utc=True, errors="coerce"),
        "event": text("event").str.rstrip("* ").str.strip(),  # Yahoo marks some names with a trailing '*'
        "region": text("region").str.upper(),
        "period": text("for"),
        "actual": _numbers(df, "actual"),
        "expected": _numbers(df, "expected"),
        "last": _numbers(df, "last"),
        "revised": _numbers(df, "revised"),
    })
    out = out.dropna(subset=["at"])
    out = out[out["event"] != ""]
    return out.drop_duplicates(["event", "region", "at"]).sort_values(["at", "region", "event"], ignore_index=True)


def news_items(raw: list[dict], symbol: str) -> list[dict]:
    """``Search(...).news`` -> headline dicts, keeping only stories Yahoo tags with ``symbol``."""
    items = []
    for n in raw or []:
        if symbol.upper() not in {t.upper() for t in n.get("relatedTickers") or []} or not n.get("title"):
            continue
        items.append({"title": n["title"].strip(), "source": n.get("publisher") or "", "url": n.get("link") or "",
                      "published": pd.to_datetime(n.get("providerPublishTime"), unit="s", utc=True, errors="coerce")})
    return items


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

    # -- calendars -------------------------------------------------------------------------
    def earnings_dates(self, symbol: str, *, limit: int = 12) -> pd.DataFrame:
        import yfinance as yf

        try:
            with warnings.catch_warnings():  # yfinance vs pandas 3 deprecation noise
                warnings.simplefilter("ignore")
                raw = yf.Ticker(symbol).get_earnings_dates(limit=limit)
        except Exception as exc:
            raise _wrap(exc) from exc
        return earnings_frame(raw)

    def calendar(self, symbol: str) -> dict:
        import yfinance as yf

        try:
            raw = yf.Ticker(symbol).calendar
        except Exception as exc:
            raise _wrap(exc) from exc
        return calendar_dict(raw)

    def economic_events(self, start, end) -> pd.DataFrame:
        import yfinance as yf

        if not hasattr(yf, "Calendars"):
            raise ProviderError("this yfinance version has no economic calendar - run: pip install -U yfinance")
        cal = yf.Calendars(start=_day(start), end=_day(end))
        pages = []
        for page in range(ECONOMIC_MAX_PAGES):
            try:
                df = cal.get_economic_events_calendar(limit=ECONOMIC_PAGE, offset=page * ECONOMIC_PAGE, force=True)
            except Exception as exc:
                raise _wrap(exc) from exc
            if df is None or df.empty:
                break
            pages.append(df.reset_index())
            if len(df) < ECONOMIC_PAGE:
                break
        return economic_frame(pd.concat(pages, ignore_index=True) if pages else None)

    def news(self, symbol: str, *, limit: int = 20) -> list[dict]:
        import yfinance as yf

        if not hasattr(yf, "Search"):
            raise ProviderError("this yfinance version has no news search - run: pip install -U yfinance")
        try:
            raw = yf.Search(symbol, news_count=limit, max_results=1, enable_fuzzy_query=False).news
        except Exception as exc:
            raise _wrap(exc) from exc
        return news_items(raw, symbol)


__all__ = ["YFinanceProvider", "calendar_dict", "earnings_frame", "economic_frame", "empty_ohlcv", "news_items"]
