"""Market-data providers. ``default_provider()`` routes ``FRED:*`` to FRED and the rest to Yahoo."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from stockanalyzer.data.providers.base import (
    CANONICAL_COLUMNS,
    FundamentalsProvider,
    PriceProvider,
    ProviderError,
    RateLimitError,
    normalize_ohlcv,
)
from stockanalyzer.data.providers.fred import FRED_PREFIX, FredProvider
from stockanalyzer.data.providers.yfinance_provider import YFinanceProvider


class RoutingProvider:
    """Send symbols to a provider by prefix (``FRED:DGS10``); everything else to the default."""

    def __init__(self, default: PriceProvider, routes: dict[str, PriceProvider] | None = None):
        self.default = default
        self.routes = routes or {}
        self.name = "routing"

    def _pick(self, symbol: str) -> PriceProvider:
        for prefix, provider in self.routes.items():
            if symbol.startswith(prefix):
                return provider
        return self.default

    def history(self, symbol, **kwargs) -> pd.DataFrame:
        return self._pick(symbol).history(symbol, **kwargs)

    def history_many(self, symbols: Sequence[str], **kwargs) -> dict[str, pd.DataFrame]:
        groups: dict[int, tuple[PriceProvider, list[str]]] = {}
        for s in symbols:
            p = self._pick(s)
            groups.setdefault(id(p), (p, []))[1].append(s)
        out: dict[str, pd.DataFrame] = {}
        for provider, group in groups.values():
            out.update(provider.history_many(group, **kwargs))
        return out


def default_provider() -> RoutingProvider:
    return RoutingProvider(YFinanceProvider(), {FRED_PREFIX: FredProvider()})


__all__ = [
    "CANONICAL_COLUMNS",
    "FRED_PREFIX",
    "FredProvider",
    "FundamentalsProvider",
    "PriceProvider",
    "ProviderError",
    "RateLimitError",
    "RoutingProvider",
    "YFinanceProvider",
    "default_provider",
    "normalize_ohlcv",
]
