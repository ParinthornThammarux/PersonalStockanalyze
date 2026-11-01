"""ATR-adaptive support/resistance zones from confirmed pivots (+ Fibonacci retracement).

Ported from the original Custom Analysis page and moved out of the UI so the screener can
use it ("price within 1 ATR of support").

Method
1. A bar is a *pivot high/low* when it is the extreme of the ``2*strength+1`` bars around it.
   Pivots are only known ``strength`` bars after they happen, so running this on data
   truncated at bar t never peeks at the future (safe for backtests).
2. Pivots are processed in time order and merged into the tightest existing zone of the same
   side whose total height stays within ``atr_mult * ATR``; otherwise they start a new zone.
3. Zones with fewer than ``min_touches`` pivots are discarded; the rest are ranked by
   touches, recency and time span, keeping the best ``max_per_side`` supports/resistances.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from stockanalyzer.analysis.technical.indicators import atr as atr_indicator


@dataclass
class Zone:
    side: str  # "support" | "resistance" (by the pivots that formed it)
    low: float
    high: float
    touches: int
    first_idx: int
    last_idx: int
    pivots: list[tuple[int, float]] = field(default_factory=list)

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2

    @property
    def height(self) -> float:
        return self.high - self.low

    def role(self, price: float) -> str:
        """What the zone is *now*: support if price is above it, resistance if below."""
        if price > self.high:
            return "support"
        if price < self.low:
            return "resistance"
        return "in-zone"


def find_pivots(
    df: pd.DataFrame, strength: int = 4, use_hl: bool = True
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Return (pivot_highs, pivot_lows) as lists of (bar index, price)."""
    highs = df["high" if use_hl else "close"].to_numpy(dtype=float)
    lows = df["low" if use_hl else "close"].to_numpy(dtype=float)
    n, width = len(highs), 2 * strength + 1
    if n < width:
        return [], []
    centers = np.arange(strength, n - strength)
    is_high = highs[centers] == sliding_window_view(highs, width).max(axis=1)
    is_low = lows[centers] == sliding_window_view(lows, width).min(axis=1)
    return (
        [(int(i), float(highs[i])) for i in centers[is_high]],
        [(int(i), float(lows[i])) for i in centers[is_low]],
    )


def _zone_width(sub: pd.DataFrame, atr_period: int, atr_mult: float) -> float:
    atr_now = atr_indicator(sub["high"], sub["low"], sub["close"], atr_period).iloc[-1]
    if pd.isna(atr_now) or atr_now <= 0:
        atr_now = float(np.std(sub["close"].to_numpy(dtype=float)) * 0.5)
    return max(float(atr_now) * atr_mult, 1e-6)


def build_zones(
    df: pd.DataFrame,
    *,
    lookback: int = 300,
    strength: int = 4,
    atr_period: int = 14,
    atr_mult: float = 1.0,
    min_touches: int = 3,
    use_hl: bool = True,
    max_per_side: int = 6,
) -> list[Zone]:
    """Zones with bar indices relative to the *full* ``df`` (not the lookback slice)."""
    if df is None or len(df) < max(50, strength * 2 + 10):
        return []

    sub = df.iloc[max(0, len(df) - lookback):]
    offset = len(df) - len(sub)
    width = _zone_width(sub, atr_period, atr_mult)

    highs, lows = find_pivots(sub, strength, use_hl)
    pivots = sorted([(i, p, "resistance") for i, p in highs] + [(i, p, "support") for i, p in lows])

    zones: list[Zone] = []
    for idx, price, side in pivots:
        best, best_height = None, None
        for z in zones:
            if z.side != side:
                continue
            height = max(z.high, price) - min(z.low, price)
            if height <= width and (best_height is None or height < best_height):
                best, best_height = z, height
        if best is None:
            zones.append(Zone(side, price, price, 1, idx, idx, [(idx, price)]))
        else:
            best.low, best.high = min(best.low, price), max(best.high, price)
            best.touches += 1
            best.last_idx = idx
            best.pivots.append((idx, price))

    zones = [z for z in zones if z.touches >= min_touches]
    n_sub = len(sub)

    def score(z: Zone) -> float:
        recency = z.last_idx / max(1, n_sub)
        span = min(z.last_idx - z.first_idx, 200) / 200.0
        return z.touches * 2.0 + recency + span

    zones.sort(key=score, reverse=True)
    keep = [z for z in zones if z.side == "support"][:max_per_side]
    keep += [z for z in zones if z.side == "resistance"][:max_per_side]

    for z in keep:
        z.first_idx += offset
        z.last_idx += offset
        z.pivots = [(i + offset, p) for i, p in z.pivots]
    return keep


def nearest_levels(zones: list[Zone], price: float) -> dict:
    """Closest zone below (support) and above (resistance) the price, with % distance."""
    below = [z for z in zones if z.high < price]
    above = [z for z in zones if z.low > price]
    sup = max(below, key=lambda z: z.high, default=None)
    res = min(above, key=lambda z: z.low, default=None)
    return {
        "support": sup,
        "resistance": res,
        "support_dist_pct": None if sup is None else (price - sup.high) / price * 100,
        "resistance_dist_pct": None if res is None else (res.low - price) / price * 100,
        "inside": next((z for z in zones if z.low <= price <= z.high), None),
    }


FIB_RATIOS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)


def fibonacci_levels(df: pd.DataFrame, lookback: int = 200) -> dict[float, float]:
    """Retracement prices between the lookback high (ratio 0) and low (ratio 1)."""
    sub = df.iloc[max(0, len(df) - lookback):]
    if len(sub) < 10:
        return {}
    hi, lo = float(sub["high"].max()), float(sub["low"].min())
    if hi <= lo:
        return {}
    return {r: hi - (hi - lo) * r for r in FIB_RATIOS}
