"""Provider contracts and the canonical OHLCV frame every provider must return.

Canonical frame: ``DatetimeIndex`` named ``ts`` (tz-naive, ``datetime64[ns]``) and
float64 columns ``open high low close adj_close volume dividends splits``.
Daily-or-longer bars are indexed at local midnight of the exchange date; intraday
bars are naive UTC.
"""

from __future__ import annotations

import math
import re
from typing import Protocol, Sequence, runtime_checkable

import pandas as pd

CANONICAL_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume", "dividends", "splits"]
PRICE_COLUMNS = ["open", "high", "low", "close"]
INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}


class ProviderError(RuntimeError):
    """The provider failed (network, parsing, unknown response)."""


class RateLimitError(ProviderError):
    """The provider asked us to slow down; callers should back off and retry."""


def is_intraday(interval: str) -> bool:
    return interval in INTRADAY_INTERVALS


def empty_ohlcv() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], name="ts", dtype="datetime64[ns]")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in CANONICAL_COLUMNS}, index=idx)


def _canon(name: object) -> str:
    return str(name).strip().lower().replace(" ", "_")


def normalize_ohlcv(df: pd.DataFrame | None, *, interval: str = "1d") -> pd.DataFrame:
    """Coerce whatever a provider returned into the canonical frame (see module doc)."""
    if df is None or len(df) == 0:
        return empty_ohlcv()

    out = df.copy()
    out.columns = [_canon(c) for c in out.columns]
    out = out.loc[:, ~out.columns.duplicated()]
    if "close" not in out.columns:
        return empty_ohlcv()

    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None) if is_intraday(interval) else idx.tz_localize(None)
    if not is_intraday(interval):
        idx = idx.normalize()
    out.index = pd.DatetimeIndex(idx).astype("datetime64[ns]")
    out.index.name = "ts"

    for col in ("open", "high", "low", "close", "adj_close", "volume", "dividends", "splits"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["close"])

    for col in ("open", "high", "low"):
        out[col] = out[col].fillna(out["close"]) if col in out.columns else out["close"]
    out["adj_close"] = out["adj_close"].fillna(out["close"]) if "adj_close" in out.columns else out["close"]
    for col, default in (("volume", 0.0), ("dividends", 0.0), ("splits", 0.0)):
        out[col] = out[col].fillna(default) if col in out.columns else default

    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out[CANONICAL_COLUMNS].astype("float64")


_PERIOD = re.compile(r"^(\d+)(d|wk|mo|y)$")


def period_to_start(period: str, end: pd.Timestamp) -> pd.Timestamp | None:
    """Translate a yfinance-style period ('6mo', '5y', 'ytd', 'max') to a start timestamp."""
    if period == "max":
        return None
    if period == "ytd":
        return pd.Timestamp(year=end.year, month=1, day=1)
    m = _PERIOD.match(period)
    if not m:
        raise ValueError(f"unsupported period {period!r}")
    n, unit = int(m.group(1)), m.group(2)
    if unit == "d":  # trading days -> calendar days
        return end - pd.Timedelta(days=math.ceil(n * 7 / 5) + 1)
    if unit == "wk":
        return end - pd.DateOffset(weeks=n)
    if unit == "mo":
        return end - pd.DateOffset(months=n)
    return end - pd.DateOffset(years=n)


@runtime_checkable
class PriceProvider(Protocol):
    name: str

    def history(
        self,
        symbol: str,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        period: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """One symbol -> canonical frame (empty if the symbol has no data)."""

    def history_many(
        self,
        symbols: Sequence[str],
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        period: str | None = None,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """Many symbols -> {symbol: canonical frame}; symbols without data are omitted."""


@runtime_checkable
class FundamentalsProvider(Protocol):
    def statements(self, symbol: str, *, quarterly: bool = False): ...

    def info(self, symbol: str) -> dict: ...
