"""Vectorised single-asset backtester and a drift-aware multi-asset portfolio backtester.

Timing contract (this is what prevents look-ahead bias):

* A strategy returns the position it wants *after seeing bar t's close*.
* ``execution="next_open"`` (default): the trade happens at the open of bar t+1. The old
  position still earns the overnight gap, the new one earns open->close of t+1.
* ``execution="close"``: the trade happens at the close of bar t (market-on-close);
  the new position earns bar t+1's close-to-close return. Slightly optimistic.

Costs are ``(commission_bps + slippage_bps)`` per unit of traded notional (one-way).
Prices are split/dividend adjusted, i.e. dividends are reinvested on the ex-date. Only the portfolio backtester can
instead pay them out as cash (``reinvest_dividends=False``). Trade statistics treat every trade at full size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stockanalyzer.analysis.risk import metrics as risk
from stockanalyzer.data.service import adjust_ohlc


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    execution: str = "next_open"  # "next_open" | "close"
    allow_short: bool = False
    max_leverage: float = 1.0
    risk_free: float = 0.0
    periods_per_year: int = 252
    reinvest_dividends: bool = True  # portfolio backtester only: False pays dividends out as cash that is never invested

    def __post_init__(self):
        if self.execution not in ("next_open", "close"):
            raise ValueError("execution must be 'next_open' or 'close'")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be positive")

    @property
    def cost_rate(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 1e4


@dataclass
class BacktestResult:
    config: BacktestConfig
    positions: pd.Series  # position actually held during each bar
    returns: pd.Series  # net strategy return per bar
    gross_returns: pd.Series
    costs: pd.Series
    equity: pd.Series
    benchmark_returns: pd.Series  # buy & hold, close to close
    trades: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def benchmark_equity(self) -> pd.Series:
        return self.config.initial_capital * (1 + self.benchmark_returns.fillna(0)).cumprod()


TRADE_COLUMNS = ["entry_ts", "exit_ts", "side", "entry_price", "exit_price", "return", "bars", "open"]


def _extract_trades(d: pd.DataFrame, held: pd.Series, cfg: BacktestConfig) -> pd.DataFrame:
    side = np.sign(held.to_numpy())
    open_, close, idx = d["open"].to_numpy(), d["close"].to_numpy(), d.index
    next_open = cfg.execution == "next_open"
    rows, cur, start = [], 0.0, 0

    def entry_of(i):  # (timestamp, price) when a position held from bar i was established
        return (idx[i], open_[i]) if next_open else (idx[i - 1], close[i - 1])

    def close_trade(i, still_open=False):
        if still_open:
            ts, px = idx[-1], close[-1]
        else:
            ts, px = (idx[i], open_[i]) if next_open else (idx[i - 1], close[i - 1])
        e_ts, e_px = entry_of(start)
        gross = cur * (px / e_px - 1)
        rows.append((e_ts, ts, "long" if cur > 0 else "short", e_px, px,
                     gross - 2 * cfg.cost_rate, int(i - start), still_open))

    for i in range(len(side)):
        if side[i] != cur:
            if cur != 0:
                close_trade(i)
            cur, start = side[i], i
    if cur != 0:
        close_trade(len(side), still_open=True)
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def _trade_stats(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {"n_trades": 0, "win_rate": float("nan"), "profit_factor": float("nan"),
                "avg_trade": float("nan"), "avg_win": float("nan"), "avg_loss": float("nan"),
                "best_trade": float("nan"), "worst_trade": float("nan"), "avg_bars_held": float("nan")}
    r = trades["return"]
    wins, losses = r[r > 0], r[r < 0]
    return {
        "n_trades": int(len(r)),
        "win_rate": float((r > 0).mean()),
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else float("inf") if len(wins) else float("nan"),
        "avg_trade": float(r.mean()),
        "avg_win": float(wins.mean()) if len(wins) else float("nan"),
        "avg_loss": float(losses.mean()) if len(losses) else float("nan"),
        "best_trade": float(r.max()),
        "worst_trade": float(r.min()),
        "avg_bars_held": float(trades["bars"].mean()),
    }


def run_backtest(df: pd.DataFrame, position: pd.Series, config: BacktestConfig | None = None) -> BacktestResult:
    """Backtest ``position`` (target after each bar's close; -1..1) on an OHLC frame."""
    cfg = config or BacktestConfig()
    d = adjust_ohlc(df)[["open", "high", "low", "close"]].astype("float64")
    if len(d) < 3:
        raise ValueError("need at least 3 bars to backtest")
    lo = -cfg.max_leverage if cfg.allow_short else 0.0
    sig = position.reindex(d.index).astype("float64").fillna(0.0).clip(lo, cfg.max_leverage)

    held = sig.shift(1).fillna(0.0)  # position during bar t = decision after bar t-1
    if cfg.execution == "close":
        ret = d["close"].pct_change().fillna(0.0)
        gross = held * ret
        turnover = sig.diff().abs()
        turnover.iloc[0] = abs(sig.iloc[0])
    else:
        prev_held = held.shift(1).fillna(0.0)
        gap = (d["open"] / d["close"].shift(1) - 1).fillna(0.0)
        intraday = d["close"] / d["open"] - 1
        gross = (1 + prev_held * gap) * (1 + held * intraday) - 1
        turnover = (held - prev_held).abs()
    costs = turnover * cfg.cost_rate
    net = gross - costs
    bench = d["close"].pct_change().fillna(0.0)
    equity = cfg.initial_capital * (1 + net).cumprod()

    trades = _extract_trades(d, held, cfg)
    years = max(len(d) / cfg.periods_per_year, 1e-9)
    metrics = risk.summary(net, bench, cfg.risk_free, cfg.periods_per_year)
    metrics.update(_trade_stats(trades))
    metrics.update(
        exposure=float((held != 0).mean()),
        turnover_per_year=float(turnover.sum() / years),
        total_costs=float(costs.sum()),
        buy_hold_return=risk.total_return(bench),
        buy_hold_cagr=risk.annualized_return(bench, cfg.periods_per_year),
        buy_hold_sharpe=risk.sharpe(bench, cfg.risk_free, cfg.periods_per_year),
        buy_hold_max_drawdown=risk.max_drawdown(bench).max_drawdown,
        final_equity=float(equity.iloc[-1]),
    )
    return BacktestResult(cfg, held, net, gross, costs, equity, bench, trades, metrics)


# -- multi-asset ------------------------------------------------------------------------------------
@dataclass
class PortfolioBacktestResult:
    config: BacktestConfig
    returns: pd.Series
    equity: pd.Series  # invested equity plus any dividends paid out so far
    weights: pd.DataFrame  # weights held at each close, after any trade made by then (share of invested equity)
    turnover: pd.Series
    costs: pd.Series
    metrics: dict[str, float]
    contribution: pd.DataFrame = field(default_factory=pd.DataFrame)  # each asset's part of every bar's gross return
    dividends: pd.DataFrame = field(default_factory=pd.DataFrame)  # money each asset's holding was paid, per ex-date bar
    invested: pd.Series | None = None  # equity in positions; below ``equity`` once dividends are being paid out

    def pnl_by_asset(self) -> pd.Series:
        """Money each asset made, dividends included: its contribution to a bar's return times the invested equity at
        the start of that bar (plus its dividends when they are paid out rather than reinvested).

        Together with the costs (``costs`` times the same starting equity) this adds up exactly to the change in equity.
        """
        base = self.equity if self.invested is None else self.invested
        pnl = self.contribution.mul(base.shift(1).fillna(self.config.initial_capital), axis=0).sum()
        return pnl if self.config.reinvest_dividends else pnl + self.dividends.sum()


def _drift(w: np.ndarray, r: np.ndarray) -> tuple[float, np.ndarray]:
    """Portfolio return of weights ``w`` over asset returns ``r`` and the weights after it (cash earns nothing)."""
    port = float(w @ r)
    return port, (w * (1 + r) / (1 + port) if 1 + port > 0 else w)


def run_portfolio_backtest(
    close: pd.DataFrame, target_weights: pd.DataFrame, config: BacktestConfig | None = None,
    open_: pd.DataFrame | None = None, dividend_yield: pd.DataFrame | None = None,
) -> PortfolioBacktestResult:
    """Rebalance to ``target_weights`` on the dates it has values (NaN rows = hold and drift).

    Weights chosen at close t earn returns from t+1; the remainder (1 - sum) sits in cash at 0%.
    Costs are charged on the turnover needed to get from the drifted weights to the target.

    By default the trade happens at close t. Pass ``open_`` (opens on the same dates and columns) together with
    ``config.execution == "next_open"`` to trade at the open of t+1 instead: the old weights earn the overnight gap
    and the new ones the open-to-close move, exactly as ``run_backtest`` does for one asset.

    ``dividend_yield`` is what one unit held at the previous close was paid on each ex-date bar (dividend / that
    close). It always fills ``dividends`` (money). With ``config.reinvest_dividends`` the prices must already include
    the dividends (dividend-adjusted), so it only reports them; otherwise the prices must be price-only and the
    dividends go to a cash balance that is never invested - it is added to ``equity`` but not to ``invested``.
    """
    cfg = config or BacktestConfig()
    close = close.sort_index()
    rets = close.pct_change(fill_method=None).fillna(0.0).to_numpy()
    targets = target_weights.reindex(index=close.index, columns=close.columns)
    rebal = targets.notna().any(axis=1).to_numpy()
    tw = targets.fillna(0.0).to_numpy()
    dy = (np.zeros(rets.shape) if dividend_yield is None
          else dividend_yield.reindex(index=close.index, columns=close.columns).fillna(0.0).to_numpy())

    next_open = open_ is not None and cfg.execution == "next_open"
    if next_open:
        opens = open_.reindex(index=close.index, columns=close.columns)
        opens = opens.where(opens.notna(), close)  # no open printed: treat the whole move as the gap
        gap = (opens / close.shift(1) - 1).fillna(0.0).to_numpy()
        intraday = (close / opens - 1).fillna(0.0).to_numpy()

    n, m = rets.shape
    w = np.zeros(m)
    net, turnover, costs = np.zeros(n), np.zeros(n), np.zeros(n)
    held, contrib, paid = np.zeros((n, m)), np.zeros((n, m)), np.zeros((n, m))
    invested = np.zeros(n)
    value = cfg.initial_capital
    trades = 0

    def trade(target: np.ndarray, w: np.ndarray, t: int) -> np.ndarray:
        nonlocal trades
        turnover[t] = np.abs(target - w).sum()
        costs[t] = turnover[t] * cfg.cost_rate
        trades += 1
        return target.copy()

    for t in range(n):
        paid[t] = value * w * dy[t]  # whoever held at the previous close gets the dividend
        if next_open:
            gap_part = w * gap[t]
            port_gap, w = _drift(w, gap[t])
            if t and rebal[t - 1]:
                w = trade(tw[t - 1], w, t)
            day_part = w * intraday[t]
            port_day, w = _drift(w, intraday[t])
            gross = (1 + port_gap) * (1 + port_day) - 1
            contrib[t] = gap_part + (1 + port_gap) * day_part
        else:
            contrib[t] = w * rets[t]
            gross, w = _drift(w, rets[t])
            if rebal[t]:
                w = trade(tw[t], w, t)
        net[t] = gross - costs[t]
        value *= 1 + net[t]
        invested[t], held[t] = value, w

    idx, cols = close.index, close.columns
    invested_s = pd.Series(invested, index=idx)
    payout = np.zeros(n) if cfg.reinvest_dividends else paid.sum(axis=1)
    equity = invested_s + pd.Series(payout, index=idx).cumsum()
    returns = pd.Series(net, index=idx) if cfg.reinvest_dividends else equity.div(equity.shift(1).fillna(cfg.initial_capital)) - 1
    years = max(n / cfg.periods_per_year, 1e-9)
    received = float(paid.sum())
    metrics = risk.summary(returns, None, cfg.risk_free, cfg.periods_per_year)
    metrics.update(turnover_per_year=float(turnover.sum() / years), total_costs=float(costs.sum()),
                   final_equity=float(equity.iloc[-1]), rebalances=trades,
                   exposure=float(np.abs(held).sum(axis=1).mean()),
                   avg_positions=float((np.abs(held) > 1e-9).sum(axis=1).mean()),
                   dividends_received=received, dividend_yield=received / years / max(float(equity.mean()), 1e-9))
    return PortfolioBacktestResult(cfg, returns, equity, pd.DataFrame(held, index=idx, columns=cols),
                                   pd.Series(turnover, index=idx), pd.Series(costs, index=idx), metrics,
                                   pd.DataFrame(contrib, index=idx, columns=cols), pd.DataFrame(paid, index=idx, columns=cols),
                                   invested_s)
