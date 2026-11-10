"""Market regimes: a hidden Markov (Markov-switching) model on a market index's weekly returns.

Every week is in one of 2 (or 3) states that differ mostly in **volatility**: "Calm" and "Stressed" (and "Crisis").
The model also gives how long each state tends to last and the chance of switching each week, which is what a
Monte Carlo needs to simulate realistic runs of calm and stress.

Why weekly and why 2 states by default (measured on the local store, Nov 2026):
- Daily fits only pick up short volatility bursts (SPY states lasted 53 / 21 days, Bitcoin 11 / 6 days).
- Weekly on SPY since 1993: Calm lasts ~37 weeks on average (11 %/yr volatility), Stressed ~14 weeks (27 %), with 28-29
  separate episodes of each; Stressed covers 87 % of the 2008 weeks, 92 % of early 2020 and all of 2022.
- A third state becomes a rare "crash" state (SPY: 5 episodes, 2 % of weeks; on 10 years of data it was a single
  2-week COVID rebound), so it is offered but flagged when it rests on few episodes.
- The average return inside a state is noisy (in Thai stocks and Bitcoin the high-volatility state had a *positive*
  mean); volatility is what separates the states reliably. Treat the state means with suspicion.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WEEK = "W-FRI"
WEEKS_PER_YEAR = 52
NAMES = {2: ("Calm", "Stressed"), 3: ("Calm", "Stressed", "Crisis")}
MIN_WEEKS = 156  # three years of weeks before a fit is attempted
MIN_EPISODES = 5  # a state seen fewer separate times than this is described from too few events
SEARCH_REPS = 20  # random start points tried inside one fit (the likelihood has local optima)
TRIES = {2: 2, 3: 5}  # independent fits per model, the best likelihood wins


def weekly_returns(close: pd.Series) -> pd.Series:
    """Simple returns between Friday closes (the last close of each week; a week still in progress counts so far)."""
    c = close.dropna()
    c = c[c > 0]
    if c.empty:
        return pd.Series(dtype="float64")
    wk = c.resample(WEEK).last().dropna()
    return wk.pct_change().dropna().rename(close.name)


def stationary(P: np.ndarray) -> np.ndarray:
    """Long-run share of time in each state for a row-stochastic transition matrix."""
    vals, vecs = np.linalg.eig(np.asarray(P, dtype=float).T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
    v = np.abs(v)
    return v / v.sum()


def episodes_of(labels: pd.Series) -> pd.DataFrame:
    """Runs of consecutive weeks in the same state: start, end, state, weeks."""
    if labels.empty:
        return pd.DataFrame(columns=["start", "end", "state", "weeks"])
    run = (labels != labels.shift()).cumsum()
    g = labels.groupby(run)
    return pd.DataFrame({"start": g.apply(lambda s: s.index[0]), "end": g.apply(lambda s: s.index[-1]),
                         "state": g.first(), "weeks": g.size()}).reset_index(drop=True)


@dataclass
class RegimeModel:
    """A fitted regime model. States are ordered from the lowest to the highest volatility."""

    driver: str
    names: tuple[str, ...]
    weekly: pd.Series  # the driver's weekly returns the model was fitted on
    labels: pd.Series  # most likely state of each week (index into ``names``), using all weeks (smoothed)
    smoothed: pd.DataFrame  # weeks x names: P(state) using all weeks
    filtered: pd.DataFrame  # weeks x names: P(state) using only the weeks up to then (what was knowable at the time)
    transition: np.ndarray  # P[i, j] = chance of moving from state i to state j next week
    mean: np.ndarray  # model mean weekly return per state
    vol: np.ndarray  # model weekly volatility per state
    params: np.ndarray = field(repr=False, default=None)  # raw statsmodels parameters (its own state order)
    order: np.ndarray = field(repr=False, default=None)  # statsmodels state of each ordered state

    @property
    def k(self) -> int:
        return len(self.names)

    @property
    def current(self) -> np.ndarray:
        """State probabilities for the latest week, from data up to that week only."""
        return self.filtered.iloc[-1].to_numpy()

    @property
    def current_state(self) -> str:
        return self.names[int(np.argmax(self.current))]

    @property
    def long_run(self) -> np.ndarray:
        return stationary(self.transition)

    @property
    def avg_weeks(self) -> np.ndarray:
        """Expected length of a stay in each state (weeks): 1 / (1 - chance of staying)."""
        stay = np.clip(np.diag(self.transition), 0.0, 1 - 1e-9)
        return 1.0 / (1.0 - stay)

    def episodes(self) -> pd.DataFrame:
        ep = episodes_of(self.labels)
        if ep.empty:
            return ep
        growth = (1 + self.weekly).cumprod()
        before = growth.shift(1).fillna(1.0)
        ep["return"] = [growth.loc[e] / before.loc[s] - 1 for s, e in zip(ep["start"], ep["end"])]
        ep["state"] = [self.names[int(i)] for i in ep["state"]]
        return ep

    def stats(self) -> pd.DataFrame:
        """Per state: model volatility and mean (per year), average stay, share of weeks, episodes, worst week."""
        ep = episodes_of(self.labels)
        rows = []
        for i, name in enumerate(self.names):
            inside = self.weekly[self.labels == i]
            rows.append({
                "state": name,
                "volatility": self.vol[i] * np.sqrt(WEEKS_PER_YEAR),
                "mean": (1 + self.mean[i]) ** WEEKS_PER_YEAR - 1,
                "avg_weeks": self.avg_weeks[i],
                "share": float((self.labels == i).mean()),
                "long_run_share": float(self.long_run[i]),
                "episodes": int((ep["state"] == i).sum()) if not ep.empty else 0,
                "weeks": int(len(inside)),
                "worst_week": float(inside.min()) if len(inside) else np.nan,
            })
        return pd.DataFrame(rows).set_index("state")

    def thin_states(self) -> list[str]:
        """States seen in fewer than ``MIN_EPISODES`` separate episodes."""
        s = self.stats()
        return list(s.index[s["episodes"] < MIN_EPISODES])


def _fit_raw(y_pct: np.ndarray, k: int, *, search_reps: int, seed: int, start_params=None):
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    model = MarkovRegression(pd.Series(y_pct), k_regimes=k, trend="c", switching_variance=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence chatter from the random restarts
        res = model.fit(search_reps=search_reps, disp=False, rng=seed, start_params=start_params)
    return model, res


def _ordered(res, k: int):
    """Parameters of each state, re-ordered from calm to wild."""
    p = np.asarray(res.params, dtype=float)
    names = list(res.model.param_names)
    const = np.array([p[names.index(f"const[{i}]")] for i in range(k)])
    sigma2 = np.array([p[names.index(f"sigma2[{i}]")] for i in range(k)])
    order = np.argsort(sigma2)
    T = np.asarray(res.model.regime_transition_matrix(p))[:, :, 0]  # statsmodels: T[to, from]
    P = T.T[np.ix_(order, order)]  # rows = from, columns = to
    P = P / P.sum(axis=1, keepdims=True)
    return order, const[order] / 100.0, np.sqrt(sigma2[order]) / 100.0, P


def fit(weekly: pd.Series, k: int = 2, *, driver: str = "", search_reps: int = SEARCH_REPS, seed: int = 0) -> RegimeModel:
    """Fit a ``k``-state Markov-switching model (switching mean and variance) to weekly returns."""
    if k not in NAMES:
        raise ValueError("k must be 2 or 3")
    y = weekly.dropna().astype("float64")
    if len(y) < MIN_WEEKS:
        raise ValueError(f"need at least {MIN_WEEKS} weeks of {driver or 'driver'} prices, have {len(y)}")
    # Several restarts, best likelihood kept: with 3 states single runs on SPY landed in junk optima (a "state" lasting
    # one week) in 2 of 5 seeds; 2 states gave the same answer every time.
    res = None
    for s in range(seed, seed + TRIES[k]):
        _, r = _fit_raw(y.to_numpy() * 100.0, k, search_reps=search_reps, seed=s)  # percent: better scaled for the optimiser
        if res is None or (np.isfinite(r.llf) and r.llf > res.llf):
            res = r
    order, mean, vol, P = _ordered(res, k)
    sm = np.asarray(res.smoothed_marginal_probabilities)[:, order]
    fl = np.asarray(res.filtered_marginal_probabilities)[:, order]
    names = NAMES[k]
    return RegimeModel(
        driver=driver or str(weekly.name or ""), names=names, weekly=y,
        labels=pd.Series(sm.argmax(axis=1), index=y.index, name="state"),
        smoothed=pd.DataFrame(sm, index=y.index, columns=list(names)),
        filtered=pd.DataFrame(fl, index=y.index, columns=list(names)),
        transition=P, mean=mean, vol=vol, params=np.asarray(res.params, dtype=float), order=order,
    )


def refilter(model: RegimeModel, weekly: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Run an already-fitted model over another stretch of weeks (its parameters kept).

    Returns (labels from smoothed probabilities, filtered probabilities of the last week), both in the model's order.
    Used by the walk-forward check, where the parameters are re-estimated only once a year.
    """
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    y = weekly.dropna().to_numpy() * 100.0
    m = MarkovRegression(pd.Series(y), k_regimes=model.k, trend="c", switching_variance=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = m.smooth(model.params)
    sm = np.asarray(res.smoothed_marginal_probabilities)[:, model.order]
    fl = np.asarray(res.filtered_marginal_probabilities)[:, model.order]
    return sm.argmax(axis=1), fl[-1]
