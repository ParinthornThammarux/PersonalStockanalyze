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


# Yahoo exchange suffix -> currency the listing is quoted in (the stock itself may report in another one)
SUFFIX_CURRENCY = {
    ".BK": "THB", ".T": "JPY", ".HK": "HKD", ".SI": "SGD", ".L": "GBp", ".TO": "CAD", ".AX": "AUD",
    ".DE": "EUR", ".PA": "EUR", ".AS": "EUR", ".MI": "EUR", ".MC": "EUR", ".SS": "CNY", ".SZ": "CNY",
    ".KS": "KRW", ".TW": "TWD", ".NS": "INR", ".BO": "INR", ".JK": "IDR", ".KL": "MYR", ".SW": "CHF",
}
# Local market benchmark by suffix; anything else is measured against the S&P 500
MARKET_BENCHMARKS = {".BK": "TDEX.BK"}
DEFAULT_BENCHMARK = "SPY"
THAI_BOARD_LOT = 100
MINOR_UNIT_CODES = ("GBp", "GBX", "ZAc", "ILA")  # quoted in a hundredth of the major unit; the only mixed-case codes


def clean_currency(code: str) -> str:
    """Currency codes are upper case, except the minor units Yahoo quotes (GBp = pence)."""
    code = code.strip()
    return code if code in MINOR_UNIT_CODES else code.upper()


def currency_of(symbol: str) -> str:
    """Currency a Yahoo symbol is quoted in: its exchange suffix, the quote side of a crypto pair, else USD."""
    s = symbol.strip().upper()
    for suffix, ccy in SUFFIX_CURRENCY.items():
        if s.endswith(suffix):
            return ccy
    pair = _CRYPTO.fullmatch(s)
    if pair:
        quote = pair.group(1)
        return "USD" if quote in ("USDT", "BTC") else quote
    return "USD"


def benchmark_for(symbol: str) -> str:
    """The index/ETF a symbol's beta and relative strength are measured against (SET50 ETF for Thai listings)."""
    s = symbol.strip().upper()
    for suffix, bench in MARKET_BENCHMARKS.items():
        if s.endswith(suffix) and s != bench:
            return bench
    return DEFAULT_BENCHMARK


def lot_size(symbol: str) -> int:
    """Shares per board lot: 100 on the SET, 1 elsewhere."""
    return THAI_BOARD_LOT if symbol.strip().upper().endswith(".BK") else 1


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
