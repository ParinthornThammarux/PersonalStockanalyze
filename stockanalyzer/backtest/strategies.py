"""Rule-based strategies. Each returns the *desired position after each bar's close*.

They only use information up to bar t (the engine applies the position from t+1), and are
declared in a registry with parameter ranges so the UI can build forms and parameter sweeps.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from functools import reduce
from typing import Callable

import numpy as np
import pandas as pd

from stockanalyzer.analysis.risk import metrics as risk
from stockanalyzer.analysis.technical import indicators as ind
from stockanalyzer.analysis.technical.registry import Param
from stockanalyzer.backtest.engine import (
    BacktestConfig, BacktestResult, PortfolioBacktestResult, run_backtest, run_portfolio_backtest,
)
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


# -- one strategy on many assets, held at the same time ---------------------------------------------
ALLOCATIONS: dict[str, tuple[str, str]] = {
    "slots": ("Fixed slots", "Every asset owns its share of the equity; when its signal is off that share stays in cash."),
    "active": ("Fully invested in active signals",
               "All the equity is split between the assets whose signal is on, in proportion to their weights; cash only when none is."),
}


def allocation_weights(
    signals: pd.DataFrame, allocation: str = "slots", max_leverage: float = 1.0, weights: pd.Series | None = None,
) -> pd.DataFrame:
    """Per-asset signals (desired position after each close) -> the portfolio weights to hold.

    ``weights`` are each asset's relative share (default equal); they are scaled to add up to 1. Gross exposure is
    capped at ``max_leverage``; the remainder of the equity is cash.
    """
    if allocation not in ALLOCATIONS:
        raise ValueError(f"unknown allocation {allocation!r}; choose from {sorted(ALLOCATIONS)}")
    base = pd.Series(1.0, index=signals.columns) if weights is None else weights.reindex(signals.columns)
    if base.isna().any() or (base < 0).any() or not base.sum() > 0:
        raise ValueError("weights must be given for every asset, not be negative and not all be zero")
    w = signals.mul(base / base.sum(), axis=1)
    if allocation == "active":
        gross = w.abs().sum(axis=1)
        w = w.div(gross.where(gross > 0), axis=0).fillna(0.0)
    gross = w.abs().sum(axis=1)
    return w.mul((max_leverage / gross.where(gross > 0, 1.0)).clip(upper=1.0), axis=0)


@dataclass
class MultiStrategyResult:
    portfolio: PortfolioBacktestResult
    benchmark: PortfolioBacktestResult  # buy & hold of the same assets at the same weights, same costs
    signals: pd.DataFrame  # each asset's desired position after each close
    weights: pd.DataFrame  # the portfolio weights those signals ask for after each close
    assets: dict[str, BacktestResult]  # each asset's signal run alone with the whole capital, same bars
    alone: pd.DataFrame  # per asset: total return of that run and of simply holding it, dividends included


def _yield_on_grid(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.Series:
    """Dividends paid between consecutive shared bars as a fraction of the earlier bar's close.

    An ex-date on a day that is not a shared bar lands on the next shared bar, so no dividend is lost.
    """
    if "dividends" not in df:
        return pd.Series(0.0, index=idx)
    paid = df["dividends"].fillna(0.0).cumsum().loc[idx].diff().fillna(0.0)
    return (paid / df["close"].loc[idx].shift(1)).fillna(0.0)


def drop_short_histories(
    frames: dict[str, pd.DataFrame], min_share: float = 0.8
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Timestamp]]:
    """Leave out assets whose history is shorter than ``min_share`` of the longest one, and say where each began.

    The shared window starts when the youngest asset does, so one recent listing would shorten the test for the whole
    pool. Everything is kept if fewer than two assets would be left.
    """
    span = {s: (df.index[-1] - df.index[0]).days for s, df in frames.items()}
    kept = {s: df for s, df in frames.items() if span[s] >= min_share * max(span.values())}
    if len(kept) < 2:
        return frames, {}
    return kept, {s: df.index[0] for s, df in frames.items() if s not in kept}


def run_multi_signals(
    frames: dict[str, pd.DataFrame], signals: dict[str, pd.Series], config: BacktestConfig | None = None,
    allocation: str = "slots", rebalance: str | None = None, weights: pd.Series | None = None,
) -> MultiStrategyResult:
    """Hold every asset whose signal is on, at the same time, out of one pool of equity.

    * ``frames`` are stored prices (split-adjusted ``close`` plus ``adj_close`` and ``dividends``), not already
      dividend-adjusted. ``config.reinvest_dividends`` chooses between total return (dividends bought back on the
      ex-date) and price return with the dividends paid out as cash.
    * The assets are aligned on the dates they **all** trade (a US ETF, a Thai stock and a coin share Mon-Fri
      minus either holiday). Moves on days only some of them trade are not lost: they land in the next shared
      bar's overnight gap. The window starts when the youngest asset does.
    * ``weights`` (relative, default equal) set each asset's share; ``allocation`` says whether a switched-off
      asset's share stays in cash or goes to the active ones. Weights are re-set only when some signal changes (and
      on ``rebalance`` dates, ``ME`` / ``QE`` / ``W-FRI``); in between they drift with prices, and turnover is
      costed from the drifted weights.
    * Timing and costs follow ``config`` exactly as in ``run_backtest``.
    """
    cfg = config or BacktestConfig()
    if not frames:
        raise ValueError("need at least one asset")
    if rebalance is not None and rebalance not in _PERIOD:
        raise ValueError(f"rebalance must be one of {sorted(_PERIOD)} or None")
    basis = {s: df if cfg.reinvest_dividends else df.assign(adj_close=df["close"]) for s, df in frames.items()}
    adj = {s: adjust_ohlc(df)[["open", "close"]].astype("float64") for s, df in basis.items()}
    idx = reduce(lambda a, b: a.intersection(b), (d.index for d in adj.values())).sort_values()
    if len(idx) < 3:
        raise ValueError("the assets share fewer than 3 trading days")
    close = pd.DataFrame({s: d["close"] for s, d in adj.items()}).loc[idx]
    opens = pd.DataFrame({s: d["open"] for s, d in adj.items()}).loc[idx]
    yields = pd.DataFrame({s: _yield_on_grid(df, idx) for s, df in frames.items()})

    # bars per year of the shared calendar; keep the configured 252 unless it is clearly not a weekday market
    derived = (len(idx) - 1) / max((idx[-1] - idx[0]).days / 365.25, 1e-9)
    if abs(derived / cfg.periods_per_year - 1) > 0.05:
        cfg = replace(cfg, periods_per_year=max(int(round(derived)), 1))

    lo = -cfg.max_leverage if cfg.allow_short else 0.0
    sig = (pd.DataFrame({s: signals[s].reindex(idx) for s in frames}).astype("float64")
           .fillna(0.0).clip(lo, cfg.max_leverage))
    target_w = allocation_weights(sig, allocation, cfg.max_leverage, weights)
    act = sig.ne(sig.shift(1).fillna(0.0)).any(axis=1)
    if rebalance:
        act = act | idx.isin(rebalance_dates(idx, rebalance))
    target = target_w.copy()
    target.loc[~act] = np.nan

    hold = pd.DataFrame(np.nan, index=idx, columns=close.columns)
    hold.iloc[0] = (1.0 / close.shape[1]) if weights is None else (weights.reindex(close.columns) / weights.sum()).to_numpy()

    assets = {s: run_backtest(basis[s].loc[idx], sig[s], cfg) for s in frames}
    rows = {}
    for s, a in assets.items():
        alone, held_it = a.metrics["total_return"], a.metrics["buy_hold_return"]
        if not cfg.reinvest_dividends:  # run_backtest saw price-only bars: add the cash it collected
            before = a.positions.shift(1).fillna(0.0) if cfg.execution == "next_open" else a.positions
            alone += float((before * yields[s] * a.equity.shift(1).fillna(cfg.initial_capital)).sum()) / cfg.initial_capital
            held_it += float((yields[s] * (close[s] / close[s].iloc[0]).shift(1)).sum())
        rows[s] = {"alone": alone, "buy_hold": held_it}
    return MultiStrategyResult(
        run_portfolio_backtest(close, target, cfg, open_=opens, dividend_yield=yields),
        run_portfolio_backtest(close, hold, cfg, open_=opens, dividend_yield=yields),
        sig, target_w, assets, pd.DataFrame(rows).T,
    )


def run_multi_strategy(
    frames: dict[str, pd.DataFrame], name: str, params: dict | None = None, config: BacktestConfig | None = None,
    allocation: str = "slots", rebalance: str | None = None, weights: pd.Series | None = None,
) -> MultiStrategyResult:
    """Run strategy ``name`` on every frame and hold the resulting positions together."""
    strat = STRATEGIES[name]
    signals = {s: strat.positions(df, **(params or {})) for s, df in frames.items()}
    return run_multi_signals(frames, signals, config, allocation, rebalance, weights)
