"""Candlestick patterns (simplified, documented definitions - not identical to TA-Lib's).

Each detector returns +100 (bullish), -100 (bearish) or 0, aligned with the input index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _parts(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    lower = df[["open", "close"]].min(axis=1) - df["low"]
    return body, rng, upper, lower


def doji(df: pd.DataFrame, max_body: float = 0.1) -> pd.Series:
    """Body at most ``max_body`` of the bar's range. Neutral, reported as +100 when present."""
    body, rng, _, _ = _parts(df)
    return ((body <= max_body * rng) * 100).fillna(0).astype(int)


def hammer(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Small body near the high, lower shadow >= 2x body, after a decline over ``lookback`` bars."""
    body, rng, upper, lower = _parts(df)
    shape = (lower >= 2 * body) & (upper <= 0.1 * rng) & (body > 0) & (body <= 0.35 * rng)
    downtrend = df["close"] < df["close"].shift(lookback)
    return ((shape & downtrend) * 100).fillna(0).astype(int)


def shooting_star(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Small body near the low, upper shadow >= 2x body, after an advance over ``lookback`` bars."""
    body, rng, upper, lower = _parts(df)
    shape = (upper >= 2 * body) & (lower <= 0.1 * rng) & (body > 0) & (body <= 0.35 * rng)
    uptrend = df["close"] > df["close"].shift(lookback)
    return ((shape & uptrend) * -100).fillna(0).astype(int)


def engulfing(df: pd.DataFrame) -> pd.Series:
    """+100 bullish engulfing (up bar swallows prior down bar), -100 bearish engulfing."""
    o, c = df["open"], df["close"]
    po, pc = o.shift(1), c.shift(1)
    bull = (pc < po) & (c > o) & (o <= pc) & (c >= po)
    bear = (pc > po) & (c < o) & (o >= pc) & (c <= po)
    return (bull.astype(int) * 100 - bear.astype(int) * 100).fillna(0).astype(int)


DETECTORS = {"doji": doji, "hammer": hammer, "shooting_star": shooting_star, "engulfing": engulfing}


def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({name: fn(df) for name, fn in DETECTORS.items()})
