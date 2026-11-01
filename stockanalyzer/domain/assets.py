"""Asset model: what a symbol *is*, independent of any data provider."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class AssetClass(str, Enum):
    EQUITY = "equity"
    ETF = "etf"
    INDEX = "index"
    CRYPTO = "crypto"
    COMMODITY = "commodity"  # futures / commodities, e.g. GC=F
    FX = "fx"
    RATE = "rate"
    OTHER = "other"

    @property
    def trades_weekends(self) -> bool:
        return self in (AssetClass.CRYPTO, AssetClass.FX)


_RATE_INDEXES = {"^TNX", "^TYX", "^FVX", "^IRX"}
_CRYPTO = re.compile(r"[A-Z0-9]{2,10}-(USD|USDT|EUR|GBP|JPY|THB|BTC)")


def infer_asset_class(symbol: str) -> AssetClass:
    """Best-effort class from a Yahoo-style symbol (ETFs cannot be told apart by name)."""
    s = symbol.strip().upper()
    if s in _RATE_INDEXES:
        return AssetClass.RATE
    if s.startswith("^"):
        return AssetClass.INDEX
    if s.endswith("=F"):
        return AssetClass.COMMODITY
    if s.endswith("=X"):
        return AssetClass.FX
    if _CRYPTO.fullmatch(s):
        return AssetClass.CRYPTO
    return AssetClass.EQUITY


@dataclass(frozen=True)
class Asset:
    symbol: str
    name: str = ""
    asset_class: AssetClass = AssetClass.EQUITY
    currency: str = "USD"
    exchange: str = ""
    sector: str = ""
    industry: str = ""
    market_cap: float | None = None

    @classmethod
    def from_symbol(cls, symbol: str, **kwargs) -> "Asset":
        sym = symbol.strip().upper()
        kwargs.setdefault("asset_class", infer_asset_class(sym))
        return cls(symbol=sym, **kwargs)


_CAP_UNITS = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def parse_market_cap(text: str | float | None) -> float | None:
    """'34.44B' -> 34_440_000_000.0; '-', '', None -> None."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text) if text == text else None
    t = text.strip().replace(",", "").upper()
    if not t or t in {"-", "NAN", "N/A"}:
        return None
    unit = _CAP_UNITS.get(t[-1])
    try:
        return float(round(float(t[:-1]) * unit)) if unit else float(t)
    except ValueError:
        return None
