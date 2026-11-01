"""Event detection on top of indicators: crossovers, RSI reversals, MACD flips, trend state.

All event series mark the bar on which the event is *visible at the close*, so they can be
traded from the next bar without look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockanalyzer.analysis.technical import indicators as ind


def crossover(a: pd.Series, b: pd.Series | float) -> pd.Series:
    """+1 on the bar ``a`` crosses above ``b``, -1 when it crosses below, else 0."""
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    above = a > b
    prev_above = above.shift(1, fill_value=False)
    prev_valid = a.shift(1).notna() & b.shift(1).notna()
    up = above & ~prev_above & prev_valid & (a.shift(1) <= b.shift(1))
    down = (a < b) & prev_valid & (a.shift(1) >= b.shift(1))
    return up.astype(int) - down.astype(int)


def ma_cross(close: pd.Series, fast: int = 12, slow: int = 26, kind: str = "ema") -> pd.Series:
    fn = ind.ema if kind == "ema" else ind.sma
    return crossover(fn(close, fast), fn(close, slow))


def rsi_events(rsi: pd.Series, lower: float = 30, upper: float = 70) -> pd.DataFrame:
    """Entries into and exits from the extreme zones.

    ``bullish_reversal`` = RSI climbs back above ``lower`` (the useful buy trigger);
    ``bearish_reversal`` = RSI falls back below ``upper``.
    """
    return pd.DataFrame(
        {
            "enter_oversold": crossover(rsi, lower).eq(-1),
            "bullish_reversal": crossover(rsi, lower).eq(1),
            "enter_overbought": crossover(rsi, upper).eq(1),
            "bearish_reversal": crossover(rsi, upper).eq(-1),
        }
    )


def macd_cross(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    m = ind.macd(close, fast, slow, signal)
    return crossover(m["macd"], m["macd_signal"])


_LABELS = {
    ("ema_cross", 1): ("EMA 12/26 bullish cross", "bullish"),
    ("ema_cross", -1): ("EMA 12/26 bearish cross", "bearish"),
    ("sma_cross", 1): ("Golden cross (SMA 50/200)", "bullish"),
    ("sma_cross", -1): ("Death cross (SMA 50/200)", "bearish"),
    ("macd_cross", 1): ("MACD bullish cross", "bullish"),
    ("macd_cross", -1): ("MACD bearish cross", "bearish"),
}


def event_log(df: pd.DataFrame, last_n: int = 250) -> pd.DataFrame:
    """Dated list of signals across indicators over the last ``last_n`` bars (newest first)."""
    close = df["close"]
    series = {
        "ema_cross": ma_cross(close, 12, 26, "ema"),
        "sma_cross": ma_cross(close, 50, 200, "sma"),
        "macd_cross": macd_cross(close),
    }
    rows = []
    for key, s in series.items():
        for ts, v in s.tail(last_n)[lambda x: x != 0].items():
            label, direction = _LABELS[(key, int(v))]
            rows.append((ts, label, direction))
    events = ind.rsi(close, 14).pipe(rsi_events).tail(last_n)
    for col, label, direction in [
        ("enter_oversold", "RSI entered oversold (<30)", "bullish-watch"),
        ("bullish_reversal", "RSI back above 30", "bullish"),
        ("enter_overbought", "RSI entered overbought (>70)", "bearish-watch"),
        ("bearish_reversal", "RSI back below 70", "bearish"),
    ]:
        rows += [(ts, label, direction) for ts in events.index[events[col]]]
    out = pd.DataFrame(rows, columns=["date", "signal", "direction"])
    return out.sort_values("date", ascending=False, ignore_index=True)


def trend_regression(close: pd.Series, n: int = 60) -> dict:
    """OLS fit of log price on time over the last ``n`` bars: annualised slope and R²."""
    y = np.log(close.dropna().tail(n).to_numpy(dtype=float))
    if len(y) < max(10, n // 2):
        return {"slope_annual_pct": np.nan, "r2": np.nan}
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid**2).sum() / ss_tot if ss_tot > 0 else 0.0
    return {"slope_annual_pct": float((np.exp(slope * ind.TRADING_DAYS) - 1) * 100), "r2": float(r2)}


def trend_state(df: pd.DataFrame) -> dict:
    """Classify the current trend from moving averages, ADX and regression quality.

    Score from -5 to +5 (price vs SMA50/200, SMA50 vs SMA200, SMA50 slope, DI direction).
    """
    close = df["close"]
    sma50, sma200 = ind.sma(close, 50), ind.sma(close, 200)
    adx = ind.adx(df["high"], df["low"], close)
    last = close.iloc[-1]

    points, notes = 0, []
    for cond, up_note, down_note in [
        (last > sma50.iloc[-1], "above SMA50", "below SMA50"),
        (last > sma200.iloc[-1], "above SMA200", "below SMA200"),
        (sma50.iloc[-1] > sma200.iloc[-1], "SMA50 > SMA200", "SMA50 < SMA200"),
        (sma50.iloc[-1] > sma50.iloc[-11] if len(sma50) > 11 else False, "SMA50 rising", "SMA50 falling"),
        (adx["plus_di"].iloc[-1] > adx["minus_di"].iloc[-1], "+DI > -DI", "-DI > +DI"),
    ]:
        if pd.isna(cond):
            continue
        points += 1 if cond else -1
        notes.append(up_note if cond else down_note)

    strength = adx["adx"].iloc[-1]
    if points >= 3:
        label = "Uptrend"
    elif points <= -3:
        label = "Downtrend"
    else:
        label = "Sideways / mixed"
    return {
        "label": label,
        "score": points,
        "adx": None if pd.isna(strength) else float(strength),
        "trending": bool(strength >= 25) if not pd.isna(strength) else False,
        "notes": notes,
        **trend_regression(close),
    }
