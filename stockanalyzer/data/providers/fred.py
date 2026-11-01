"""FRED macro series via the public CSV endpoint (no API key)."""

from __future__ import annotations

import io
from typing import Sequence

import pandas as pd
import requests

from stockanalyzer.data.providers.base import ProviderError, empty_ohlcv, normalize_ohlcv

FRED_PREFIX = "FRED:"


class FredProvider:
    name = "fred"
    URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    def history(self, symbol, *, start=None, end=None, period=None, interval="1d") -> pd.DataFrame:
        series_id = symbol.removeprefix(FRED_PREFIX)
        params = {"id": series_id}
        if start is not None:
            params["cosd"] = pd.Timestamp(start).strftime("%Y-%m-%d")
        if end is not None:
            params["coed"] = pd.Timestamp(end).strftime("%Y-%m-%d")
        try:
            resp = requests.get(self.URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise ProviderError(f"FRED {series_id}: {exc}") from exc

        raw = pd.read_csv(io.StringIO(resp.text))
        if raw.shape[1] < 2:
            return empty_ohlcv()
        # .to_numpy(): passing Series next to an explicit index would align on labels (all-NaN)
        values = pd.to_numeric(raw.iloc[:, 1], errors="coerce").to_numpy()  # '.' = missing
        frame = pd.DataFrame(
            {"open": values, "high": values, "low": values, "close": values},
            index=pd.DatetimeIndex(pd.to_datetime(raw.iloc[:, 0])),
        )
        return normalize_ohlcv(frame, interval="1d")

    def history_many(self, symbols: Sequence[str], **kwargs) -> dict[str, pd.DataFrame]:
        out = {}
        for s in symbols:
            df = self.history(s, **kwargs)
            if not df.empty:
                out[s] = df
        return out
