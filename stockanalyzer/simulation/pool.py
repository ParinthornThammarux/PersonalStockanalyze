"""The pool of historical weeks a simulation draws from, one row per week of the driver's history.

The regime model is fitted on a long index history (SPY since 1993, TDEX since 2008), but most holdings have shorter
histories - a stock listed in 2015 never went through 2008. So that every holding has a return in every pooled week,
the weeks before a holding traded are **estimated**: the index's return that week times the holding's beta in that
state, plus a residual taken from a real week of the holding in the same state. The residual week is picked with one
shared draw per week, so holdings that traded over the same years get their residuals from the same week and keep
their co-movement beyond the index (two Thai banks stay close). Weeks the holding did trade are used as they are.

Returns are in the portfolio's base currency; the driver's are in its own currency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stockanalyzer.analysis.regimes import WEEKS_PER_YEAR, RegimeModel

MIN_ASSET_WEEKS = 52  # less real history than this and a holding cannot be described
MIN_STATE_WEEKS = 26  # a beta for one state needs this many real weeks in it, else the all-weeks beta is used
MIN_DONORS = 8  # residual weeks needed in a state before falling back to any of the holding's weeks
FLOOR = -0.95  # a weekly loss larger than this is a data error, not a market move

RETURN_MODES = {
    "market": "Market-implied (cash rate + beta x index)",
    "half": "Halfway",
    "own": "Each holding's own history",
}


def to_annual(weekly: np.ndarray | float) -> np.ndarray | float:
    return (1 + np.asarray(weekly)) ** WEEKS_PER_YEAR - 1


def to_weekly(annual: np.ndarray | float) -> np.ndarray | float:
    return (1 + np.asarray(annual)) ** (1 / WEEKS_PER_YEAR) - 1


def weekly_frame(close: pd.DataFrame) -> pd.DataFrame:
    """Weekly simple returns per column from daily closes (NaN before a column's first price)."""
    if close.empty:
        return close
    wk = close.resample("W-FRI").last()
    return wk.pct_change(fill_method=None).iloc[1:]


@dataclass
class ReturnPool:
    dates: pd.DatetimeIndex  # one per pooled week
    state: np.ndarray  # state of each week (index into ``names``)
    driver: np.ndarray  # the driver's weekly return
    returns: np.ndarray  # weeks x assets, real where ``real`` else estimated
    real: np.ndarray  # weeks x assets
    assets: list[str]
    names: tuple[str, ...]
    beta: pd.DataFrame  # assets x states
    dropped: dict[str, str] = field(default_factory=dict)  # holdings left out and why

    @property
    def n_weeks(self) -> int:
        return len(self.dates)

    def coverage(self) -> pd.Series:
        """Share of pooled weeks with real data, per holding."""
        return pd.Series(self.real.mean(axis=0), index=self.assets)

    def first_real(self) -> pd.Series:
        return pd.Series([self.dates[np.argmax(self.real[:, j])] for j in range(len(self.assets))], index=self.assets)

    def state_means(self, returns: np.ndarray | None = None) -> np.ndarray:
        """states x assets: average weekly return of the pooled weeks in each state."""
        r = self.returns if returns is None else returns
        return np.vstack([r[self.state == k].mean(axis=0) if (self.state == k).any() else np.zeros(r.shape[1])
                          for k in range(len(self.names))])

    def driver_state_means(self) -> np.ndarray:
        return np.array([self.driver[self.state == k].mean() if (self.state == k).any() else 0.0
                         for k in range(len(self.names))])

    def own_mean(self, mix: np.ndarray) -> np.ndarray:
        """Weekly mean per holding when the states occur in proportions ``mix`` (the model's long run)."""
        return mix @ self.state_means()

    def market_mean(self, mix: np.ndarray, *, rf: float = 0.0, index_return: float | None = None) -> np.ndarray:
        """Weekly mean per holding if it earned only the cash rate plus its beta times the index's excess return (CAPM):
        no skill and no luck of its own. ``rf`` and ``index_return`` are weekly; the index defaults to its own history.

        Without the cash rate a bond fund (beta ~0 to stocks) would be assumed to earn nothing.
        """
        beta = self.beta.to_numpy()  # assets x states
        m = self.driver_state_means()
        if index_return is not None:
            m = m + (index_return - float(mix @ m))  # same shift in every state: stress still hurts
        return rf + (mix * (m - rf)) @ beta.T

    def targets(self, mode: str, mix: np.ndarray, *, rf: float = 0.0, index_return: float | None = None) -> np.ndarray:
        own, market = self.own_mean(mix), self.market_mean(mix, rf=rf, index_return=index_return)
        if mode == "own":
            return own
        if mode == "market":
            return market
        if mode == "half":
            return 0.5 * (own + market)
        raise ValueError(f"unknown return mode {mode!r}")

    def shifted(self, targets: np.ndarray, mix: np.ndarray) -> np.ndarray:
        """Returns moved by a constant per holding so the long-run weekly mean equals ``targets``.

        The shift is the same in every state, so the differences between states (stress hurts) stay; it is worked
        out for the model's long-run mix, so a scenario that makes stress longer still lowers the result.
        """
        delta = np.asarray(targets, dtype=float) - self.own_mean(mix)
        return np.maximum(self.returns + delta, FLOOR)


def _beta(r: np.ndarray, m: np.ndarray) -> float:
    if len(r) < 3:
        return 1.0
    vm = np.var(m)
    return float(np.cov(r, m, bias=True)[0, 1] / vm) if vm > 0 else 1.0


def build_pool(model: RegimeModel, asset_weekly: pd.DataFrame, *, seed: int = 0) -> ReturnPool:
    """Line holdings' weekly returns up with the driver's weeks and estimate the weeks before each one traded."""
    dates = model.weekly.index
    state = model.labels.to_numpy().astype(int)
    m = model.weekly.to_numpy()
    k = model.k
    a = asset_weekly.reindex(dates)
    dropped: dict[str, str] = {}
    keep = []
    for c in a.columns:
        n = int(a[c].notna().sum())
        if n < MIN_ASSET_WEEKS:
            dropped[c] = f"only {n} weeks of prices overlap the {model.driver} history (needs {MIN_ASSET_WEEKS})"
        else:
            keep.append(c)
    a = a[keep]
    rng = np.random.default_rng(seed)
    u = rng.random(len(dates))  # one draw per week, shared by every holding (keeps their residual co-movement)
    out = np.empty((len(dates), len(keep)))
    real = a.notna().to_numpy()
    betas = np.ones((len(keep), k))
    for j, c in enumerate(keep):
        r = a[c].to_numpy()
        ok = real[:, j]
        b_all = _beta(r[ok], m[ok])
        for s in range(k):
            sel = ok & (state == s)
            betas[j, s] = _beta(r[sel], m[sel]) if sel.sum() >= MIN_STATE_WEEKS else b_all
        resid = np.where(ok, r - betas[j, state] * m, np.nan)
        col = np.where(ok, r, np.nan)
        missing = np.flatnonzero(~ok)
        if len(missing):
            donors_all = np.flatnonzero(ok)
            donors = {s: np.flatnonzero(ok & (state == s)) for s in range(k)}
            for t in missing:
                d = donors[state[t]] if len(donors[state[t]]) >= MIN_DONORS else donors_all
                col[t] = betas[j, state[t]] * m[t] + resid[d[int(u[t] * len(d))]]
        out[:, j] = np.maximum(col, FLOOR)
    return ReturnPool(dates=dates, state=state, driver=m, returns=out, real=real, assets=keep, names=model.names,
                      beta=pd.DataFrame(betas, index=keep, columns=list(model.names)), dropped=dropped)
