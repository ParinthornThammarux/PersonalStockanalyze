"""Currency conversion from Yahoo FX pairs (latest close)."""

from __future__ import annotations

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
