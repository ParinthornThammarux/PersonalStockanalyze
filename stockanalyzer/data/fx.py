"""Currency conversion from Yahoo FX pairs (latest close)."""

from __future__ import annotations

import pandas as pd

from stockanalyzer.data.service import MarketData

# Yahoo quotes some listings in a minor unit: code -> (major currency, minor units per major unit)
MINOR_UNITS = {"GBp": ("GBP", 100.0), "GBX": ("GBP", 100.0), "ZAc": ("ZAR", 100.0), "ILA": ("ILS", 100.0)}


def _latest(md: MarketData, symbol: str) -> float | None:
    df = md.prices(symbol, period="1mo", live=False, adjusted=False)
    if df.empty:
        return None
    px = float(df["close"].iloc[-1])
    return px if px > 0 else None


def _major_pair(md: MarketData, src: str, dst: str) -> float | None:
    """``dst`` per 1 ``src`` for two major currencies, trying the pair, its USD shorthand, then the inverse."""
    candidates: list[tuple[str, bool]] = [(f"{src}{dst}=X", False)]
    if src == "USD":
        candidates.append((f"{dst}=X", False))  # Yahoo's USD-based pairs have no "USD" prefix: JPY=X
    if dst == "USD":
        candidates.append((f"{src}=X", True))
    candidates.append((f"{dst}{src}=X", True))
    for symbol, inverse in candidates:
        px = _latest(md, symbol)
        if px is not None:
            return 1.0 / px if inverse else px
    return None


def _series(md: MarketData, symbol: str) -> pd.Series | None:
    df = md.prices(symbol, period="max", live=False, adjusted=False)
    if df.empty:
        return None
    close = df["close"]
    return close[close > 0]


def _major_history(md: MarketData, src: str, dst: str) -> pd.Series | None:
    """``dst`` per 1 ``src`` over time for two major currencies (same pair lookup as ``_major_pair``)."""
    candidates: list[tuple[str, bool]] = []
    if src == "USD":  # Yahoo's own USD-based symbols (THB=X) first: the long-form ones are often missing
        candidates.append((f"{dst}=X", False))
    if dst == "USD":
        candidates.append((f"{src}=X", True))
    candidates += [(f"{src}{dst}=X", False), (f"{dst}{src}=X", True)]
    for symbol, inverse in candidates:
        px = _series(md, symbol)
        if px is not None and not px.empty:
            return 1.0 / px if inverse else px
    return None


def pair_symbols(from_ccy: str, to_ccy: str) -> list[str]:
    """Every Yahoo symbol ``fx_history`` may read to convert ``from_ccy`` into ``to_ccy`` (direct, inverse and via USD),
    so a caller can download their full history too."""
    src, dst = MINOR_UNITS.get(from_ccy, (from_ccy, 1.0))[0], MINOR_UNITS.get(to_ccy, (to_ccy, 1.0))[0]
    if not src or not dst or src == dst:
        return []
    legs = [(src, dst)] + ([(src, "USD"), ("USD", dst)] if "USD" not in (src, dst) else [])
    out: list[str] = []
    for a, b in legs:
        if a == "USD":
            out.append(f"{b}=X")
        if b == "USD":
            out.append(f"{a}=X")
        out += [f"{a}{b}=X", f"{b}{a}=X"]
    return list(dict.fromkeys(out))


def fx_history(md: MarketData, from_ccy: str, to_ccy: str) -> pd.Series | None:
    """Daily units of ``to_ccy`` per one ``from_ccy``; ``None`` if no Yahoo pair exists, even through the US dollar.

    Pairs without a direct Yahoo symbol are crossed through USD, e.g. EUR -> THB = EURUSD x USDTHB. The same
    currency gives an empty series, which ``rate_on`` reads as 1.0.
    """
    if not from_ccy or not to_ccy:
        return None
    if from_ccy == to_ccy:
        return pd.Series(dtype="float64")
    src, src_minor = MINOR_UNITS.get(from_ccy, (from_ccy, 1.0))
    dst, dst_minor = MINOR_UNITS.get(to_ccy, (to_ccy, 1.0))
    scale = dst_minor / src_minor
    if src == dst:  # pence <-> pounds: a constant, valid for all dates
        return pd.Series([scale], index=pd.DatetimeIndex([pd.Timestamp("1970-01-01")]), dtype="float64")
    major = _major_history(md, src, dst)
    if major is None and "USD" not in (src, dst):
        first, second = _major_history(md, src, "USD"), _major_history(md, "USD", dst)
        if first is not None and second is not None:
            both = pd.concat([first, second], axis=1, keys=["a", "b"]).sort_index().ffill().dropna()
            major = both["a"] * both["b"]
    return None if major is None else major * scale


def rate_on(series: pd.Series | None, when) -> float:
    """The rate in force on ``when`` (the last known one; the first one if ``when`` precedes the history).

    An empty series means "same currency" (1.0); ``None`` means no rate is known (NaN).
    """
    if series is None:
        return float("nan")
    if series.empty:
        return 1.0
    value = series.asof(pd.Timestamp(when))
    return float(series.iloc[0] if pd.isna(value) else value)


def fx_rate(md: MarketData, from_ccy: str, to_ccy: str) -> float | None:
    """Units of ``to_ccy`` per one ``from_ccy``; ``None`` if no Yahoo pair is available.

    Handles minor-unit quotes (``GBp`` pence), e.g. USD -> GBp = USDGBP x 100.
    """
    if not from_ccy or not to_ccy:
        return None
    if from_ccy == to_ccy:
        return 1.0
    src, src_minor = MINOR_UNITS.get(from_ccy, (from_ccy, 1.0))
    dst, dst_minor = MINOR_UNITS.get(to_ccy, (to_ccy, 1.0))
    major = 1.0 if src == dst else _major_pair(md, src, dst)
    return None if major is None else major * dst_minor / src_minor
