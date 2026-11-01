"""Risk and performance metrics on return series (daily by default).

Conventions: returns are simple periodic returns; VaR/CVaR are reported as *positive loss*
fractions; ``rf`` is an annual risk-free rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
_EPS = 1e-12  # std below this is floating-point noise around a constant series


def to_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return prices.pct_change(fill_method=None).dropna(how="all")


# -- performance ---------------------------------------------------------------------------------
def total_return(returns: pd.Series) -> float:
    return float((1 + returns.dropna()).prod() - 1)


def annualized_return(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = returns.dropna()
    if r.empty:
        return float("nan")
    growth = (1 + r).prod()
    return float(growth ** (periods / len(r)) - 1) if growth > 0 else -1.0


def annualized_volatility(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(returns.dropna().std(ddof=1) * np.sqrt(periods))


def sharpe(returns: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    excess = returns.dropna() - rf / periods
    sd = excess.std(ddof=1)
    return float(excess.mean() / sd * np.sqrt(periods)) if sd > _EPS else float("nan")


def sortino(returns: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    excess = returns.dropna() - rf / periods
    downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
    return float(excess.mean() / downside * np.sqrt(periods)) if downside > _EPS else float("nan")


# -- drawdowns -------------------------------------------------------------------------------------
def drawdown_series(returns: pd.Series) -> pd.DataFrame:
    equity = (1 + returns.fillna(0)).cumprod()
    peak = equity.cummax()
    return pd.DataFrame({"equity": equity, "peak": peak, "drawdown": equity / peak - 1})


@dataclass
class DrawdownInfo:
    max_drawdown: float  # negative fraction
    peak_date: pd.Timestamp | None
    trough_date: pd.Timestamp | None
    recovery_date: pd.Timestamp | None  # None = not recovered yet
    duration_bars: int  # peak -> trough


def max_drawdown(returns: pd.Series) -> DrawdownInfo:
    dd = drawdown_series(returns)
    if dd.empty:
        return DrawdownInfo(float("nan"), None, None, None, 0)
    trough = dd["drawdown"].idxmin()
    mdd = float(dd.loc[trough, "drawdown"])
    if mdd == 0:
        return DrawdownInfo(0.0, None, None, None, 0)
    peak = dd.loc[:trough, "equity"].idxmax()
    after = dd.loc[trough:]
    recovered = after[after["equity"] >= dd.loc[peak, "equity"]]
    return DrawdownInfo(mdd, peak, trough, recovered.index[0] if len(recovered) else None,
                        int(dd.index.get_loc(trough) - dd.index.get_loc(peak)))


def calmar(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(returns).max_drawdown
    return float(annualized_return(returns, periods) / abs(mdd)) if mdd and mdd < 0 else float("nan")


def monthly_table(returns: pd.Series) -> pd.DataFrame:
    """Calendar-month returns, years as rows and months as columns, plus a full-year column."""
    r = returns.dropna()
    monthly = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    table = monthly.unstack()
    table.columns = [pd.Timestamp(2000, int(m), 1).strftime("%b") for m in table.columns]
    table["Year"] = (1 + r).groupby(r.index.year).prod() - 1
    table.index.name = "year"
    return table


# -- tail risk -------------------------------------------------------------------------------------
def var_historical(returns: pd.Series, level: float = 0.95) -> float:
    r = returns.dropna()
    return float(-np.quantile(r, 1 - level)) if len(r) else float("nan")


def cvar_historical(returns: pd.Series, level: float = 0.95) -> float:
    """Expected loss given the loss is at least the VaR (average of the worst tail)."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    cutoff = np.quantile(r, 1 - level)
    return float(-r[r <= cutoff].mean())


def var_parametric(returns: pd.Series, level: float = 0.95) -> float:
    """Gaussian VaR; understates tails for fat-tailed assets, compare with the historical one."""
    r = returns.dropna()
    return float(-(r.mean() + stats.norm.ppf(1 - level) * r.std(ddof=1))) if len(r) > 1 else float("nan")


# -- relative to a benchmark ---------------------------------------------------------------------------
def beta_alpha(returns: pd.Series, benchmark: pd.Series, rf: float = 0.0,
               periods: int = TRADING_DAYS) -> dict[str, float]:
    both = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(both) < 3:
        return {k: float("nan") for k in ("beta", "alpha", "r2", "correlation", "tracking_error", "information_ratio")}
    r, b = both.iloc[:, 0] - rf / periods, both.iloc[:, 1] - rf / periods
    var_b = b.var(ddof=1)
    beta = float(r.cov(b) / var_b) if var_b > _EPS**2 else float("nan")
    alpha = float((r.mean() - beta * b.mean()) * periods)
    corr = float(r.corr(b))
    active = both.iloc[:, 0] - both.iloc[:, 1]
    te = float(active.std(ddof=1) * np.sqrt(periods))
    return {"beta": beta, "alpha": alpha, "r2": corr**2, "correlation": corr, "tracking_error": te,
            "information_ratio": float(active.mean() * periods / te) if te > 0 else float("nan")}


def correlation_matrix(returns: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    return returns.corr(method=method)


def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 60) -> pd.Series:
    return a.rolling(window).corr(b)


def risk_contributions(weights: pd.Series, returns: pd.DataFrame) -> pd.DataFrame:
    """Each holding's share of portfolio variance: ``w_i * (Σw)_i / w'Σw``."""
    cols = list(weights.index)
    cov = returns[cols].cov().to_numpy()
    w = weights.to_numpy(dtype=float)
    marginal = cov @ w
    total = float(w @ marginal)
    contrib = w * marginal / total if total > 0 else np.full(len(w), np.nan)
    return pd.DataFrame({"weight": w, "risk_contribution": contrib}, index=cols)


# -- scenarios -----------------------------------------------------------------------------------------
HISTORICAL_SCENARIOS: dict[str, tuple[str, str]] = {
    "2008 financial crisis (Sep 12 2008 - Mar 9 2009)": ("2008-09-12", "2009-03-09"),
    "COVID crash (Feb 19 - Mar 23 2020)": ("2020-02-19", "2020-03-23"),
    "2022 rate shock (Jan 3 - Oct 12 2022)": ("2022-01-03", "2022-10-12"),
    "2025 tariff shock (Apr 2 - Apr 8 2025)": ("2025-04-02", "2025-04-08"),
}


def replay_scenarios(
    returns: pd.Series, scenarios: Mapping[str, tuple[str, str]] | None = None, min_coverage: float = 0.8
) -> pd.DataFrame:
    """Cumulative return of ``returns`` inside each historical window the data covers."""
    rows = []
    for name, (start, end) in (scenarios or HISTORICAL_SCENARIOS).items():
        window = returns.loc[start:end].dropna()
        expected = max(1, len(pd.bdate_range(start, end)) - 1)
        if len(window) >= min_coverage * expected and returns.index.min() <= pd.Timestamp(start):
            rows.append({"scenario": name, "return": total_return(window), "bars": len(window),
                         "worst_day": float(window.min())})
    return pd.DataFrame(rows, columns=["scenario", "return", "bars", "worst_day"])


def beta_shock(weights: pd.Series, betas: pd.Series, market_move: float) -> float:
    """First-order portfolio P&L for an instantaneous market move: Σ w_i β_i * move."""
    return float((weights * betas.reindex(weights.index).fillna(1.0)).sum() * market_move)


def days_to_liquidate(position_value: pd.Series, adv_dollars: pd.Series, participation: float = 0.2) -> pd.Series:
    """Trading days to exit each position when selling ``participation`` of average daily volume."""
    capacity = (adv_dollars.reindex(position_value.index) * participation).replace(0, np.nan)
    return (position_value.abs() / capacity).rename("days_to_exit")


# -- one-call summary -----------------------------------------------------------------------------------
def summary(returns: pd.Series, benchmark: pd.Series | None = None, rf: float = 0.0,
            periods: int = TRADING_DAYS) -> dict[str, float]:
    r = returns.dropna()
    dd = max_drawdown(r)
    out = {
        "total_return": total_return(r),
        "cagr": annualized_return(r, periods),
        "volatility": annualized_volatility(r, periods),
        "sharpe": sharpe(r, rf, periods),
        "sortino": sortino(r, rf, periods),
        "max_drawdown": dd.max_drawdown,
        "calmar": calmar(r, periods),
        "var_95": var_historical(r, 0.95),
        "cvar_95": cvar_historical(r, 0.95),
        "var_99": var_historical(r, 0.99),
        "skew": float(r.skew()) if len(r) > 2 else float("nan"),
        "kurtosis": float(r.kurt()) if len(r) > 3 else float("nan"),
        "best_day": float(r.max()) if len(r) else float("nan"),
        "worst_day": float(r.min()) if len(r) else float("nan"),
        "hit_rate": float((r > 0).mean()) if len(r) else float("nan"),
    }
    if benchmark is not None:
        out.update({f"vs_benchmark_{k}": v for k, v in beta_alpha(r, benchmark, rf, periods).items()})
    return out
