"""Indicator registry: one declarative table that drives the UI, screener and backtester.

Each :class:`Indicator` says which OHLCV columns it needs, which parameters it takes (with
ranges for UI widgets), and which of its outputs belong on the price pane. Output columns
are suffixed with the parameter values, so ``sma(20)`` and ``sma(50)`` coexist:
``sma_20``, ``sma_50``, ``macd_12_26_9``, ``macd_signal_12_26_9`` ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import pandas as pd

from stockanalyzer.analysis.technical import indicators as ind


@dataclass(frozen=True)
class Param:
    name: str
    default: int | float
    min: int | float
    max: int | float
    step: int | float = 1

    @property
    def is_int(self) -> bool:
        return isinstance(self.default, int)


@dataclass(frozen=True)
class Indicator:
    name: str
    label: str
    category: str  # trend | momentum | volatility | volume
    fn: Callable
    inputs: tuple[str, ...]
    params: tuple[Param, ...] = ()
    pane: str = "separate"  # "price" overlays the candles, "separate" gets its own panel
    outputs: tuple[str, ...] = ()  # column names the function returns (default: the indicator name)
    plot: tuple[str, ...] | None = None  # subset of outputs to draw (default: all)
    levels: tuple[float, ...] = ()  # horizontal reference lines (RSI 30/70 ...)

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}

    def suffix(self, params: dict) -> str:
        values = [params[p.name] for p in self.params]
        return "_" + "_".join(str(v) for v in values) if values else ""

    def compute(self, df: pd.DataFrame, **overrides) -> pd.DataFrame:
        params = {**self.defaults(), **overrides}
        unknown = set(params) - {p.name for p in self.params}
        if unknown:
            raise TypeError(f"{self.name}: unknown parameter(s) {sorted(unknown)}")
        out = self.fn(*(df[c] for c in self.inputs), **params)
        if isinstance(out, pd.Series):
            out = out.to_frame(self.name)
        sfx = self.suffix(params)
        return out.rename(columns={c: f"{c}{sfx}" for c in out.columns})

    def plot_columns(self, params: dict | None = None) -> list[str]:
        merged = {**self.defaults(), **(params or {})}
        sfx = self.suffix(merged)
        names = self.plot or self.outputs or (self.name,)
        return [f"{c}{sfx}" for c in names]


def _n(default: int, lo: int = 2, hi: int = 400) -> Param:
    return Param("n", default, lo, hi)


_ALL = [
    Indicator("sma", "Simple Moving Average", "trend", ind.sma, ("close",), (_n(20),), "price"),
    Indicator("ema", "Exponential Moving Average", "trend", ind.ema, ("close",), (_n(20),), "price"),
    Indicator("bbands", "Bollinger Bands", "volatility", ind.bollinger, ("close",),
              (_n(20), Param("k", 2.0, 0.5, 4.0, 0.1)), "price",
              outputs=("bb_mid", "bb_upper", "bb_lower", "bb_pctb", "bb_bandwidth"),
              plot=("bb_upper", "bb_mid", "bb_lower")),
    Indicator("donchian", "Donchian Channel", "trend", ind.donchian, ("high", "low"), (_n(20),), "price",
              outputs=("dc_upper", "dc_mid", "dc_lower"), plot=("dc_upper", "dc_lower")),
    Indicator("vwap", "Rolling VWAP", "volume", ind.vwap, ("high", "low", "close", "volume"),
              (_n(20),), "price"),
    Indicator("rsi", "RSI", "momentum", ind.rsi, ("close",), (_n(14),), levels=(30, 70)),
    Indicator("macd", "MACD", "momentum", ind.macd, ("close",),
              (Param("fast", 12, 2, 100), Param("slow", 26, 3, 200), Param("signal", 9, 2, 100)),
              outputs=("macd", "macd_signal", "macd_hist"), levels=(0,)),
    Indicator("stoch", "Stochastic", "momentum", ind.stochastic, ("high", "low", "close"),
              (Param("k", 14, 2, 100), Param("d", 3, 1, 50), Param("smooth", 3, 1, 50)),
              outputs=("stoch_k", "stoch_d"), levels=(20, 80)),
    Indicator("roc", "Rate of Change %", "momentum", ind.roc, ("close",), (_n(12),), levels=(0,)),
    Indicator("mom", "Momentum", "momentum", ind.mom, ("close",), (_n(10),), levels=(0,)),
    Indicator("cci", "Commodity Channel Index", "momentum", ind.cci, ("high", "low", "close"),
              (_n(20),), levels=(-100, 100)),
    Indicator("willr", "Williams %R", "momentum", ind.williams_r, ("high", "low", "close"),
              (_n(14),), levels=(-80, -20)),
    Indicator("adx", "ADX / Directional Index", "trend", ind.adx, ("high", "low", "close"),
              (_n(14),), outputs=("adx", "plus_di", "minus_di"), levels=(20, 25)),
    Indicator("atr", "Average True Range", "volatility", ind.atr, ("high", "low", "close"), (_n(14),)),
    Indicator("hvol", "Historical Volatility %", "volatility", ind.historical_volatility, ("close",),
              (Param("n", 20, 5, 250),)),
    Indicator("obv", "On-Balance Volume", "volume", ind.obv, ("close", "volume")),
    Indicator("ad", "Accumulation / Distribution", "volume", ind.accumulation_distribution,
              ("high", "low", "close", "volume")),
    Indicator("vma", "Volume Moving Average", "volume", ind.volume_ma, ("volume",), (_n(20),)),
    Indicator("mfi", "Money Flow Index", "volume", ind.mfi, ("high", "low", "close", "volume"),
              (_n(14),), levels=(20, 80)),
]

REGISTRY: dict[str, Indicator] = {i.name: i for i in _ALL}


def get(name: str) -> Indicator:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown indicator {name!r}; available: {sorted(REGISTRY)}") from None


def compute(name: str, df: pd.DataFrame, **params) -> pd.DataFrame:
    return get(name).compute(df, **params)


def compute_many(df: pd.DataFrame, requests: Sequence[str | tuple[str, dict]]) -> pd.DataFrame:
    """Compute several indicators at once: ``compute_many(df, ["rsi", ("sma", {"n": 50})])``."""
    frames = []
    for req in requests:
        name, params = (req, {}) if isinstance(req, str) else req
        frames.append(compute(name, df, **params))
    return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=df.index)


def by_category() -> dict[str, list[Indicator]]:
    out: dict[str, list[Indicator]] = {}
    for i in REGISTRY.values():
        out.setdefault(i.category, []).append(i)
    return out
