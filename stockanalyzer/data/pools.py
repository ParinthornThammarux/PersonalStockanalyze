"""Stock pools: index membership lists (packaged snapshot) and pools cut from the packaged US stock list.

The index lists come from ``scripts/build_pools.py`` and are a snapshot of the day it ran - indexes change every few
months. Ranking by size uses the market caps of the packaged US list, which are a snapshot too; symbols it does not
know (the Thai ones) keep the order they are listed in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

from stockanalyzer.data import universe as uni


@dataclass(frozen=True)
class StockPool:
    key: str
    label: str
    description: str
    source: str
    as_of: str
    symbols: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.symbols)


@lru_cache(maxsize=1)
def load_pools() -> dict[str, StockPool]:
    raw = json.loads(resources.files("stockanalyzer.resources").joinpath("pools.json").read_text("utf-8"))
    return {p["key"]: StockPool(p["key"], p["label"], p["description"], p["source"], p["as_of"], tuple(p["symbols"]))
            for p in raw["pools"]}


@lru_cache(maxsize=1)
def _caps() -> dict[str, float]:
    us = uni.load_us_stocks()
    return dict(zip(us["symbol"], us["market_cap"].fillna(0.0)))


def largest(symbols: list[str] | tuple[str, ...], n: int | None = None) -> list[str]:
    """The ``n`` biggest by market cap; symbols without a known cap follow in their original order."""
    caps = _caps()
    known = sorted((s for s in symbols if caps.get(s, 0.0) > 0), key=lambda s: -caps[s])
    ranked = known + [s for s in symbols if caps.get(s, 0.0) <= 0]
    return ranked if n is None else ranked[: max(n, 0)]


@lru_cache(maxsize=1)
def industries() -> dict[str, int]:
    """US industries (SPAC shells excluded) with their number of stocks, biggest group first."""
    us = uni.load_us_stocks()
    counts = us.loc[us["industry"] != uni.SPAC_INDUSTRY, "industry"].value_counts()
    return {k: int(v) for k, v in counts.items()}


def us_by_market_cap(n: int, industry: str | None = None) -> list[str]:
    """The ``n`` largest US-listed stocks, optionally within one industry (SPAC shells excluded)."""
    us = uni.filter_universe(uni.load_us_stocks(), industries=[industry] if industry else None, exclude_spacs=True, limit=n)
    return us["symbol"].tolist()
