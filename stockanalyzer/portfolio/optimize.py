"""Long-only portfolio construction: min variance, max Sharpe, risk parity, inverse vol.

Inputs are noisy: sample means in particular are close to useless for forecasting, so
asset means are shrunk toward their cross-sectional average (``return_shrinkage``) and the
covariance uses Ledoit-Wolf shrinkage. Treat the output as a starting point, not a forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252
METHODS = ("equal_weight", "inverse_vol", "min_variance", "max_sharpe", "risk_parity")


@dataclass
class OptResult:
    method: str
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float
    converged: bool = True


def covariance(returns: pd.DataFrame, periods: int = TRADING_DAYS, shrink: bool = True) -> pd.DataFrame:
    r = returns.dropna()
    if len(r) < 2:
        raise ValueError("need at least 2 overlapping return observations")
    cov = LedoitWolf().fit(r.to_numpy()).covariance_ if shrink else r.cov().to_numpy()
    return pd.DataFrame(cov * periods, index=returns.columns, columns=returns.columns)


def expected_returns(returns: pd.DataFrame, periods: int = TRADING_DAYS, shrinkage: float = 0.5) -> pd.Series:
    mu = returns.mean() * periods
    return (1 - shrinkage) * mu + shrinkage * mu.mean()


def _stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> tuple[float, float, float]:
    ret, vol = float(mu @ w), float(np.sqrt(w @ cov @ w))
    return ret, vol, (ret - rf) / vol if vol > 0 else float("nan")


def optimize(
    returns: pd.DataFrame,
    method: str = "min_variance",
    *,
    rf: float = 0.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    periods: int = TRADING_DAYS,
    shrink_cov: bool = True,
    return_shrinkage: float = 0.5,
    mu: pd.Series | None = None,
) -> OptResult:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    n = returns.shape[1]
    if n < 2:
        raise ValueError("need at least two assets")
    if min_weight * n > 1 + 1e-9 or max_weight * n < 1 - 1e-9:
        raise ValueError(f"bounds [{min_weight}, {max_weight}] cannot sum to 1 with {n} assets")

    cov = covariance(returns, periods, shrink_cov)
    exp = (mu if mu is not None else expected_returns(returns, periods, return_shrinkage)).reindex(cov.index)
    S, m = cov.to_numpy(), exp.to_numpy()
    w0 = np.full(n, 1 / n)
    bounds = [(min_weight, max_weight)] * n
    budget = {"type": "eq", "fun": lambda w: w.sum() - 1}

    if method in ("equal_weight", "inverse_vol"):
        w = w0 if method == "equal_weight" else (1 / np.sqrt(np.diag(S))) / (1 / np.sqrt(np.diag(S))).sum()
        if w.min() < min_weight - 1e-9 or w.max() > max_weight + 1e-9:
            raise ValueError(f"{method} weights violate the bounds; use an optimising method instead")
        converged = True
    else:
        if method == "min_variance":
            objective = lambda w: w @ S @ w  # noqa: E731
        elif method == "max_sharpe":
            objective = lambda w: -(m @ w - rf) / np.sqrt(max(w @ S @ w, 1e-16))  # noqa: E731
        else:  # risk parity: every asset contributes the same share of variance
            def objective(w):
                rc = w * (S @ w)
                return float(((rc[:, None] - rc[None, :]) ** 2).sum()) / max(w @ S @ w, 1e-16) ** 2
        res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=[budget],
                       options={"maxiter": 500, "ftol": 1e-12})
        w, converged = np.clip(res.x, min_weight, max_weight), bool(res.success)
        w = w / w.sum()

    ret, vol, sharpe = _stats(w, m, S, rf)
    return OptResult(method, pd.Series(w, index=cov.index, name=method), ret, vol, sharpe, converged)


def _max_return(mu: np.ndarray, max_weight: float) -> float:
    """Highest return reachable long-only with a per-asset cap: fill the best assets first."""
    remaining, total = 1.0, 0.0
    for value in sorted(mu, reverse=True):
        take = min(max_weight, remaining)
        total += take * value
        remaining -= take
        if remaining <= 1e-12:
            break
    return total


def efficient_frontier(returns: pd.DataFrame, n_points: int = 25, *, rf: float = 0.0, max_weight: float = 1.0,
                       periods: int = TRADING_DAYS, return_shrinkage: float = 0.5) -> pd.DataFrame:
    """Minimum-variance portfolios for target returns from the min-variance point to the best asset."""
    cov = covariance(returns, periods)
    mu = expected_returns(returns, periods, return_shrinkage).reindex(cov.index)
    S, m, n = cov.to_numpy(), mu.to_numpy(), len(cov)
    base = optimize(returns, "min_variance", rf=rf, max_weight=max_weight, periods=periods,
                    return_shrinkage=return_shrinkage)
    rows = []
    for target in np.linspace(base.expected_return, max(_max_return(m, max_weight), base.expected_return), n_points):
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}, {"type": "eq", "fun": lambda w, t=target: m @ w - t}]
        res = minimize(lambda w: w @ S @ w, base.weights.to_numpy(), method="SLSQP",
                       bounds=[(0, max_weight)] * n, constraints=cons, options={"maxiter": 300, "ftol": 1e-12})
        if res.success:
            w = res.x / res.x.sum()
            ret, vol, sharpe = _stats(w, m, S, rf)
            rows.append({"return": ret, "volatility": vol, "sharpe": sharpe, **dict(zip(cov.index, w))})
    return pd.DataFrame(rows)
