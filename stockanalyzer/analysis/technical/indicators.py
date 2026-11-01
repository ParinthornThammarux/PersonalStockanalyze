"""Technical indicators as pure pandas/numpy functions (TA-Lib is *not* required).

Conventions
* Inputs are Series sharing one DatetimeIndex; outputs keep that index, NaN during warm-up.
* Wilder smoothing (RSI, ATR, ADX) is ``ewm(alpha=1/n, adjust=False)``. TA-Lib seeds the
  recursion with an SMA instead of the first value, so the two converge after the warm-up
  window (verified in ``tests/test_indicators.py``).
* Everything is vectorised so a whole universe can be processed in a loop of a few ms/symbol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

TRADING_DAYS = 252


def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


# -- trend / overlays -------------------------------------------------------------------------
def sma(close: pd.Series, n: int = 20) -> pd.Series:
    return close.rolling(n).mean()


def ema(close: pd.Series, n: int = 20) -> pd.Series:
    return close.ewm(span=n, adjust=False, min_periods=n).mean()


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    upper, lower = mid + k * std, mid - k * std
    width = (upper - lower).replace(0, np.nan)
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
         "bb_pctb": (close - lower) / width, "bb_bandwidth": width / mid}
    )


def donchian(high: pd.Series, low: pd.Series, n: int = 20) -> pd.DataFrame:
    upper, lower = high.rolling(n).max(), low.rolling(n).min()
    return pd.DataFrame({"dc_upper": upper, "dc_mid": (upper + lower) / 2, "dc_lower": lower})


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 20) -> pd.Series:
    """Rolling volume-weighted average price over the last ``n`` bars."""
    typical = (high + low + close) / 3
    return (typical * volume).rolling(n).sum() / volume.rolling(n).sum().replace(0, np.nan)


# -- momentum ---------------------------------------------------------------------------------
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    avg_gain = _wilder(delta.clip(lower=0), n)
    avg_loss = _wilder((-delta).clip(lower=0), n)
    total = avg_gain + avg_loss
    return (100 * avg_gain / total.replace(0, np.nan)).fillna(50.0).where(total.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k: int = 14, d: int = 3, smooth: int = 3) -> pd.DataFrame:
    """Slow stochastic by default (``smooth=1`` gives the fast version)."""
    lowest, highest = low.rolling(k).min(), high.rolling(k).max()
    raw = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    stoch_k = raw.rolling(smooth).mean()
    return pd.DataFrame({"stoch_k": stoch_k, "stoch_d": stoch_k.rolling(d).mean()})


def roc(close: pd.Series, n: int = 12) -> pd.Series:
    return 100 * (close / close.shift(n) - 1)


def mom(close: pd.Series, n: int = 10) -> pd.Series:
    return close - close.shift(n)


def _rolling_mean_abs_dev(s: pd.Series, n: int) -> pd.Series:
    values = s.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) >= n:
        windows = sliding_window_view(values, n)
        out[n - 1:] = np.abs(windows - windows.mean(axis=1, keepdims=True)).mean(axis=1)
    return pd.Series(out, index=s.index)


def cci(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20) -> pd.Series:
    typical = (high + low + close) / 3
    dev = _rolling_mean_abs_dev(typical, n).replace(0, np.nan)
    return (typical - typical.rolling(n).mean()) / (0.015 * dev)


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    highest, lowest = high.rolling(n).max(), low.rolling(n).min()
    return -100 * (highest - close) / (highest - lowest).replace(0, np.nan)


# -- trend strength / volatility ---------------------------------------------------------------
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return _wilder(true_range(high, low, close), n)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.DataFrame:
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    smoothed_tr = _wilder(true_range(high, low, close), n)
    plus_di = 100 * _wilder(plus_dm, n) / smoothed_tr
    minus_di = 100 * _wilder(minus_dm, n) / smoothed_tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({"adx": _wilder(dx, n), "plus_di": plus_di, "minus_di": minus_di})


def historical_volatility(close: pd.Series, n: int = 20, periods_per_year: int = TRADING_DAYS) -> pd.Series:
    """Annualised standard deviation of log returns, in percent."""
    return 100 * np.log(close / close.shift(1)).rolling(n).std() * np.sqrt(periods_per_year)


# -- volume -------------------------------------------------------------------------------------
def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    direction.iloc[0] = 1.0  # TA-Lib convention: the series starts at the first bar's volume
    return (direction * volume).cumsum()


def accumulation_distribution(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    span = (high - low).replace(0, np.nan)
    multiplier = (((close - low) - (high - close)) / span).fillna(0.0)
    return (multiplier * volume).cumsum()


def volume_ma(volume: pd.Series, n: int = 20) -> pd.Series:
    return volume.rolling(n).mean()


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 14) -> pd.Series:
    typical = (high + low + close) / 3
    flow = typical * volume
    change = typical.diff()
    positive = flow.where(change > 0, 0.0).rolling(n).sum()
    negative = flow.where(change < 0, 0.0).rolling(n).sum()
    total = (positive + negative).replace(0, np.nan)
    return (100 * positive / total).where(change.rolling(n).count() >= n)


# -- cross-asset --------------------------------------------------------------------------------
def relative_strength(close: pd.Series, benchmark: pd.Series) -> pd.Series:
    """Price ratio to a benchmark, rebased to 100 at the first common bar (rising = outperforming)."""
    bench = benchmark.reindex(close.index).ffill()
    ratio = (close / bench).dropna()
    return 100 * ratio / ratio.iloc[0] if not ratio.empty else ratio
