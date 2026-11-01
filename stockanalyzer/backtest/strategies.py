"""Rule-based strategies. Each returns the *desired position after each bar's close*.

They only use information up to bar t (the engine applies the position from t+1), and are
declared in a registry with parameter ranges so the UI can build forms and parameter sweeps.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from stockanalyzer.analysis.risk import metrics as risk
from stockanalyzer.analysis.technical import indicators as ind
from stockanalyzer.analysis.technical.registry import Param
from stockanalyzer.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from stockanalyzer.data.service import adjust_ohlc


def _latch(enter: pd.Series, exit_: pd.Series) -> pd.Series:
    """Stateful 0/1 position: 1 from an ``enter`` bar until the next ``exit_`` bar."""
    state = pd.Series(np.nan, index=enter.index)
    state[exit_.fillna(False)] = 0.0
    state[enter.fillna(False)] = 1.0  # enter wins if both fire on one bar
    return state.ffill().fillna(0.0)


def buy_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


def sma_cross(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
    return (ind.sma(df["close"], fast) > ind.sma(df["close"], slow)).astype(float)


def ema_cross(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    return (ind.ema(df["close"], fast) > ind.ema(df["close"], slow)).astype(float)


def macd_trend(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    m = ind.macd(df["close"], fast, slow, signal)
    return (m["macd"] > m["macd_signal"]).astype(float)


def rsi_reversion(df: pd.DataFrame, n: int = 14, entry: int = 30, exit: int = 55) -> pd.Series:
    r = ind.rsi(df["close"], n)
    return _latch(r < entry, r > exit)


def bollinger_reversion(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.Series:
    bb = ind.bollinger(df["close"], n, k)
    return _latch(df["close"] < bb["bb_lower"], df["close"] > bb["bb_mid"])


def donchian_breakout(df: pd.DataFrame, entry: int = 55, exit: int = 20) -> pd.Series:
    upper = df["high"].rolling(entry).max().shift(1)
    lower = df["low"].rolling(exit).min().shift(1)
    return _latch(df["close"] > upper, df["close"] < lower)


def time_series_momentum(df: pd.DataFrame, lookback: int = 126, skip: int = 0) -> pd.Series:
    close = df["close"]
    return (close.shift(skip) / close.shift(lookback + skip) - 1 > 0).astype(float)


@dataclass(frozen=True)
class Strategy:
    name: str
    label: str
    fn: Callable[..., pd.Series]
    params: tuple[Param, ...] = ()
    description: str = ""

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}

    def positions(self, df: pd.DataFrame, **params) -> pd.Series:
        unknown = set(params) - {p.name for p in self.params}
        if unknown:
            raise TypeError(f"{self.name}: unknown parameter(s) {sorted(unknown)}")
        return self.fn(adjust_ohlc(df), **{**self.defaults(), **params})


STRATEGIES: dict[str, Strategy] = {s.name: s for s in [
    Strategy("buy_hold", "Buy & hold", buy_hold, (), "Always long - the benchmark to beat."),
    Strategy("sma_cross", "SMA crossover", sma_cross,
             (Param("fast", 50, 2, 200), Param("slow", 200, 5, 400)), "Long while the fast SMA is above the slow SMA."),
    Strategy("ema_cross", "EMA crossover", ema_cross,
             (Param("fast", 12, 2, 100), Param("slow", 26, 5, 300)), "Long while the fast EMA is above the slow EMA."),
    Strategy("macd_trend", "MACD trend", macd_trend,
             (Param("fast", 12, 2, 60), Param("slow", 26, 5, 120), Param("signal", 9, 2, 50)),
             "Long while MACD is above its signal line."),
    Strategy("rsi_reversion", "RSI mean reversion", rsi_reversion,
             (Param("n", 14, 2, 50), Param("entry", 30, 5, 50), Param("exit", 55, 40, 90)),
             "Buy when RSI drops below `entry`, sell when it recovers above `exit`."),
    Strategy("bollinger_reversion", "Bollinger mean reversion", bollinger_reversion,
             (Param("n", 20, 5, 100), Param("k", 2.0, 0.5, 4.0, 0.1)),
             "Buy a close below the lower band, sell at the middle band."),
    Strategy("donchian_breakout", "Donchian breakout", donchian_breakout,
             (Param("entry", 55, 5, 200), Param("exit", 20, 3, 100)),
             "Buy a close above the prior N-bar high, exit below the prior M-bar low (turtle style)."),
    Strategy("momentum", "Time-series momentum", time_series_momentum,
             (Param("lookback", 126, 10, 400), Param("skip", 0, 0, 60)),
             "Long when the trailing return over `lookback` bars is positive."),
]}


def run_strategy(df: pd.DataFrame, name: str, params: dict | None = None,
                 config: BacktestConfig | None = None) -> BacktestResult:
    return run_backtest(df, STRATEGIES[name].positions(df, **(params or {})), config)


def sweep(
    df: pd.DataFrame,
    name: str,
    grid: dict[str, list],
    config: BacktestConfig | None = None,
    *,
    metric: str = "sharpe",
    split: float | None = None,
) -> pd.DataFrame:
    """Run every parameter combination; with ``split`` also report in-sample vs out-of-sample.

    The best in-sample row is *expected* to look worse out of sample - that gap is the
    overfitting you pay for choosing parameters on the same data.
    """
    cfg = config or BacktestConfig()
    keys = list(grid)
    cut = int(len(df) * split) if split else None
    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        if "fast" in params and "slow" in params and params["fast"] >= params["slow"]:
            continue
        res = run_strategy(df, name, params, cfg)
        row = {**params, "sharpe": res.metrics["sharpe"], "cagr": res.metrics["cagr"],
               "max_drawdown": res.metrics["max_drawdown"], "n_trades": res.metrics["n_trades"],
               "total_return": res.metrics["total_return"]}
        if cut:
            r_in, r_out = res.returns.iloc[:cut], res.returns.iloc[cut:]
            row.update(sharpe_in=risk.sharpe(r_in, cfg.risk_free), sharpe_out=risk.sharpe(r_out, cfg.risk_free),
                       cagr_in=risk.annualized_return(r_in), cagr_out=risk.annualized_return(r_out))
        rows.append(row)
    out = pd.DataFrame(rows)
    sort_col = f"{metric}_in" if cut and f"{metric}_in" in out else metric
    return out.sort_values(sort_col, ascending=False, na_position="last", ignore_index=True) if len(out) else out


# -- cross-sectional --------------------------------------------------------------------------------
def rebalance_dates(index: pd.DatetimeIndex, rule: str = "ME") -> pd.DatetimeIndex:
    """Last trading day of each period (``ME`` month, ``W-FRI`` week, ``QE`` quarter)."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(s.index.to_period(_PERIOD[rule])).max().to_numpy())


_PERIOD = {"ME": "M", "W-FRI": "W-FRI", "QE": "Q"}


def momentum_rotation_weights(
    close: pd.DataFrame, top_n: int = 5, lookback: int = 126, skip: int = 21,
    rebalance: str = "ME", require_positive: bool = True,
) -> pd.DataFrame:
    """Equal-weight the ``top_n`` strongest assets by ``lookback`` return (skipping the last ``skip``
    bars). If fewer than ``top_n`` have positive momentum the rest stays in cash."""
    mom = close.shift(skip) / close.shift(lookback) - 1
    weights = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for date in rebalance_dates(close.index, rebalance):
        scores = mom.loc[date].dropna()
        if scores.empty:
            continue
        winners = scores.nlargest(top_n)
        if require_positive:
            winners = winners[winners > 0]
        weights.loc[date] = 0.0
        weights.loc[date, winners.index] = 1.0 / top_n
    return weights
