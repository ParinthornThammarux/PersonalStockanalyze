"""When is a daily bar final? Only completed bars go into the store.

Yahoo returns the in-progress bar for today. Persisting it would freeze a half-day
candle until the next overlap fetch, so the updater drops everything after
:func:`last_completed_session`.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from stockanalyzer.domain.assets import AssetClass, infer_asset_class

# symbol suffix -> (timezone, hour, minute after which that day's bar is final)
_MARKETS: dict[str, tuple[str, int, int]] = {
    "": ("America/New_York", 16, 15),
    ".BK": ("Asia/Bangkok", 17, 0),
    ".HK": ("Asia/Hong_Kong", 16, 45),
    ".T": ("Asia/Tokyo", 15, 45),
    ".L": ("Europe/London", 16, 45),
    ".SS": ("Asia/Shanghai", 15, 15),
    ".SZ": ("Asia/Shanghai", 15, 15),
    ".AX": ("Australia/Sydney", 16, 15),
}


def _market_for(symbol: str) -> tuple[str, int, int]:
    s = symbol.upper()
    for suffix, spec in _MARKETS.items():
        if suffix and s.endswith(suffix):
            return spec
    return _MARKETS[""]


def _previous_weekday(d):
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def last_completed_session(symbol: str, now: datetime | None = None) -> pd.Timestamp:
    """Date (midnight) of the most recent daily bar that will no longer change."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    cls = infer_asset_class(symbol)

    if cls is AssetClass.CRYPTO:  # 24/7: yesterday (UTC) is the last full day
        return pd.Timestamp((now.astimezone(timezone.utc) - timedelta(days=1)).date())
    if cls is AssetClass.FX:  # trades weekdays around the clock
        return pd.Timestamp(_previous_weekday((now.astimezone(timezone.utc) - timedelta(days=1)).date()))

    tz, hour, minute = _market_for(symbol)
    local = now.astimezone(ZoneInfo(tz))
    day = local.date()
    if local.time() < time(hour, minute):
        day -= timedelta(days=1)
    return pd.Timestamp(_previous_weekday(day))
