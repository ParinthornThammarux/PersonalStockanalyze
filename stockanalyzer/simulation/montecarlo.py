"""Monte Carlo of a portfolio, with monthly saving (DCA) or withdrawals, through simulated market regimes.

Each path is built in weekly steps:

1. A sequence of states (Calm / Stressed / Crisis) is drawn from the regime model's weekly transition matrix,
   starting from the market's state today (or a state you force, or a past crisis replayed week by week).
2. For every week a real week of history **in the same state** is drawn from the pool, as a whole row - every holding's
   return of that same week - so crisis correlations, fat tails and currency moves come along. Consecutive weeks are
   kept together for a few weeks on average (a stationary block bootstrap), which keeps short-term momentum and
   volatility clustering.
3. The portfolio is run through those weeks: money is added (or taken out) on the first week of each month, weights
   drift with prices and are reset on the chosen schedule, and every trade pays the fee.

This shows a range of outcomes under stated assumptions; it is not a forecast. Past price forecasting failed every
honest test in this project (see docs/ROADMAP.md), and nothing here pretends otherwise: the regime model describes how
calm and stress came and went in the past, and the return assumption is a choice, shown on screen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stockanalyzer.analysis.regimes import WEEK, WEEKS_PER_YEAR, RegimeModel
from stockanalyzer.simulation.pool import ReturnPool

REBALANCE = {"none": "Never (weights drift)", "monthly": "Monthly", "quarterly": "Quarterly", "yearly": "Yearly"}
BLOCK_WEEKS = 4.0  # average run of consecutive historical weeks kept together
PERCENTILES = (5, 25, 50, 75, 95)

# Peak-to-trough windows of the driver (the replay uses the weeks ending inside them)
CRISES: dict[str, tuple[str, str, str]] = {
    "dotcom": ("Dot-com bust (Mar 2000 - Oct 2002)", "2000-03-24", "2002-10-09"),
    "gfc": ("Global financial crisis (Oct 2007 - Mar 2009)", "2007-10-09", "2009-03-09"),
    "euro": ("Euro debt crisis (Apr - Oct 2011)", "2011-04-29", "2011-10-03"),
    "taper": ("Taper tantrum (May - Aug 2013)", "2013-05-21", "2013-08-30"),
    "oil": ("China slowdown and oil slump (May 2015 - Feb 2016)", "2015-05-21", "2016-02-11"),
    "q4_2018": ("Q4 2018 sell-off (Sep - Dec 2018)", "2018-09-20", "2018-12-24"),
    "covid": ("COVID crash (Feb - Mar 2020)", "2020-02-19", "2020-03-23"),
    "rates": ("2022 rate shock (Jan - Oct 2022)", "2022-01-03", "2022-10-12"),
    "tariff": ("2025 tariff shock (Feb - Apr 2025)", "2025-02-19", "2025-04-08"),
}


@dataclass
class Plan:
    initial: float = 100_000.0  # invested at the start (already in the market: no fee)
    monthly: float = 0.0  # added on the first week of each month; negative = taken out
    step_up: float = 0.0  # yearly growth of the monthly amount (0.03 = 3 % more each year)
    years: int = 10
    rebalance: str = "yearly"
    cost: float = 0.001  # fee as a share of every amount traded
    inflation: float = 0.0  # only for the "in today's money" figures

    @property
    def steps(self) -> int:
        return int(round(self.years * WEEKS_PER_YEAR))


@dataclass
class Scenario:
    name: str = "Base case"
    start: str | int = "now"  # "now" (today's state probabilities), "long_run", or a state index to start in
    stretch: dict[int, float] = field(default_factory=dict)  # state -> multiply its average stay
    frequency: dict[int, float] = field(default_factory=dict)  # state -> multiply the weekly chance of entering it
    replay: str | None = None  # a key of CRISES played first, the same weeks on every path
    replay_after: int = 0  # extra weeks of history after the crisis window to replay too (the recovery)
    transition: np.ndarray | None = None  # replaces the model's matrix outright


def adjusted_transition(P: np.ndarray, stretch: dict[int, float] | None = None,
                        frequency: dict[int, float] | None = None) -> np.ndarray:
    """The transition matrix with some states made longer (``stretch``) or more frequent (``frequency``).

    Stretching state i by f turns an average stay of d weeks into f x d (the chance of leaving divides by f, spread over
    the other states in their old proportions). Frequency g multiplies every other state's chance of moving into i.
    """
    Q = np.array(P, dtype=float)
    k = len(Q)
    for i, g in (frequency or {}).items():
        if g == 1:
            continue
        for r in range(k):
            if r == i:
                continue
            Q[r, i] = min(Q[r, i] * g, 0.999)
            others = [c for c in range(k) if c != i]
            rest = 1.0 - Q[r, i]
            tot = Q[r, others].sum()
            Q[r, others] = Q[r, others] / tot * rest if tot > 0 else rest / len(others)
    for i, f in (stretch or {}).items():
        if f == 1:
            continue
        leave = (1.0 - Q[i, i]) / max(f, 1e-6)
        leave = min(max(leave, 1e-4), 1.0)
        others = [c for c in range(k) if c != i]
        tot = Q[i, others].sum()
        Q[i, others] = Q[i, others] / tot * leave if tot > 0 else leave / len(others)
        Q[i, i] = 1.0 - leave
    return Q / Q.sum(axis=1, keepdims=True)


def crisis_weeks(pool: ReturnPool, key: str, after: int = 0) -> np.ndarray:
    """Pool rows of a crisis window (plus ``after`` weeks of what followed); empty if the history does not reach it."""
    _, start, end = CRISES[key]
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if pool.dates[0] > start or pool.dates[-1] < end:
        return np.array([], dtype=int)
    rows = np.flatnonzero((pool.dates >= start) & (pool.dates <= end))
    if len(rows) == 0:  # a window inside one week: the week that holds its end
        rows = np.flatnonzero(pool.dates >= end)[:1]
    if after and len(rows):
        rows = np.arange(rows[0], min(rows[-1] + 1 + after, pool.n_weeks))
    return rows


def available_crises(pool: ReturnPool) -> dict[str, str]:
    return {k: v[0] for k, v in CRISES.items() if len(crisis_weeks(pool, k))}


def crises_within(dates: pd.DatetimeIndex) -> dict[str, str]:
    """Crises whose whole window lies inside ``dates`` (before a pool is built)."""
    if not len(dates):
        return {}
    return {k: v[0] for k, v in CRISES.items() if dates[0] <= pd.Timestamp(v[1]) and dates[-1] >= pd.Timestamp(v[2])}


def start_distribution(model: RegimeModel, scenario: Scenario, P: np.ndarray) -> np.ndarray:
    k = model.k
    if isinstance(scenario.start, (int, np.integer)):
        p = np.zeros(k)
        p[int(scenario.start)] = 1.0
        return p
    if scenario.start == "long_run":
        return model.long_run
    return np.asarray(model.current) @ P  # next week's state, given what is known about this week


def state_paths(P: np.ndarray, start: np.ndarray, n_paths: int, n_steps: int, rng: np.random.Generator,
                fixed: np.ndarray | None = None) -> np.ndarray:
    """(paths x steps) states of a Markov chain; ``fixed`` sets the first weeks on every path (a replayed crisis)."""
    k = len(P)
    cum = np.cumsum(P, axis=1)
    cum[:, -1] = 1.0
    out = np.empty((n_paths, n_steps), dtype=np.int8)
    n_fixed = 0 if fixed is None else min(len(fixed), n_steps)
    if n_fixed:
        out[:, :n_fixed] = fixed[:n_fixed]
        cur = np.full(n_paths, fixed[n_fixed - 1], dtype=np.int64)
        first = n_fixed
    else:
        c0 = np.cumsum(start)
        c0[-1] = 1.0
        cur = np.minimum(np.searchsorted(c0, rng.random(n_paths), side="right"), k - 1)
        out[:, 0] = cur
        first = 1
    for t in range(first, n_steps):
        u = rng.random(n_paths)
        cur = np.minimum((u[:, None] >= cum[cur]).sum(axis=1), k - 1)
        out[:, t] = cur
    return out


def pool_rows(states: np.ndarray, pool_state: np.ndarray, rng: np.random.Generator, *, block: float = BLOCK_WEEKS,
              fixed: np.ndarray | None = None) -> np.ndarray:
    """(paths x steps) pool rows: a week of history in the same state, often the week after the previous draw.

    A path keeps walking forward through history while the next historical week is in the state it needs, ending each
    run with chance 1/``block`` per week (Politis-Romano stationary bootstrap inside each state).
    """
    n_paths, n_steps = states.shape
    W = len(pool_state)
    by_state = [np.flatnonzero(pool_state == s) for s in range(int(max(states.max(), pool_state.max())) + 1)]
    everything = np.arange(W)
    rows = np.empty((n_paths, n_steps), dtype=np.int32)
    n_fixed = 0 if fixed is None else min(len(fixed), n_steps)
    if n_fixed:
        rows[:, :n_fixed] = fixed[:n_fixed]
    keep = 1.0 - 1.0 / max(block, 1.0)
    for t in range(n_fixed, n_steps):
        s = states[:, t]
        if t > 0:
            nxt = rows[:, t - 1] + 1
            ok = (nxt < W) & (rng.random(n_paths) < keep)
            ok &= pool_state[np.minimum(nxt, W - 1)] == s
        else:
            nxt = np.zeros(n_paths, dtype=np.int64)
            ok = np.zeros(n_paths, dtype=bool)
        rows[ok, t] = nxt[ok]
        for st in np.unique(s[~ok]):
            m = (~ok) & (s == st)
            choices = by_state[st] if st < len(by_state) and len(by_state[st]) else everything
            rows[m, t] = choices[rng.integers(0, len(choices), int(m.sum()))]
    return rows


def schedule(start: pd.Timestamp, plan: Plan) -> pd.DataFrame:
    """Weekly dates of the simulation with the money added / taken out and the rebalancing weeks."""
    dates = pd.date_range(pd.Timestamp(start) + pd.offsets.Week(weekday=4), periods=plan.steps, freq=WEEK)
    prev = pd.DatetimeIndex([pd.Timestamp(start)]).append(dates[:-1])
    new_month = dates.month != prev.month
    years_in = np.arange(plan.steps) // WEEKS_PER_YEAR
    flow = np.where(new_month, plan.monthly * (1 + plan.step_up) ** years_in, 0.0)
    if plan.rebalance == "monthly":
        reb = new_month
    elif plan.rebalance == "quarterly":
        reb = new_month & np.isin(dates.month, [1, 4, 7, 10])
    elif plan.rebalance == "yearly":
        reb = new_month & (dates.month == 1)
    elif plan.rebalance == "none":
        reb = np.zeros(plan.steps, dtype=bool)
    else:
        raise ValueError(f"unknown rebalance {plan.rebalance!r}")
    return pd.DataFrame({"flow": flow, "rebalance": reb}, index=dates)


@dataclass
class SimResult:
    dates: pd.DatetimeIndex  # start + one per step
    value: np.ndarray  # paths x (steps + 1), base currency
    invested: np.ndarray  # (steps + 1) money put in so far, net of withdrawals
    states: np.ndarray  # paths x steps
    growth: np.ndarray  # per path: what 1 invested at the start became, ignoring the cash flows (time-weighted)
    max_drawdown: np.ndarray  # per path, of that time-weighted line
    ran_out: np.ndarray  # per path: step at which withdrawals emptied the portfolio, -1 if never
    names: tuple[str, ...]
    plan: Plan
    scenario: Scenario

    @property
    def years(self) -> float:
        return self.plan.steps / WEEKS_PER_YEAR

    def bands(self, qs=PERCENTILES) -> pd.DataFrame:
        return pd.DataFrame(np.percentile(self.value, qs, axis=0).T, index=self.dates, columns=[f"p{q}" for q in qs])

    def cagr(self) -> np.ndarray:
        return np.where(self.growth > 0, self.growth ** (1 / self.years) - 1, -1.0)

    def state_share(self) -> np.ndarray:
        """paths x states: share of the weeks each path spent in each state."""
        return np.stack([(self.states == s).mean(axis=1) for s in range(len(self.names))], axis=1)

    def summary(self) -> dict[str, float]:
        final, put_in = self.value[:, -1], self.invested[-1]
        cagr = self.cagr()
        deflator = (1 + self.plan.inflation) ** self.years
        return {
            "median": float(np.median(final)),
            "p5": float(np.percentile(final, 5)),
            "p95": float(np.percentile(final, 95)),
            "invested": float(put_in),
            "below_invested": float((final < put_in).mean()),
            "median_real": float(np.median(final) / deflator),
            "cagr_median": float(np.median(cagr)),
            "cagr_p5": float(np.percentile(cagr, 5)),
            "cagr_p95": float(np.percentile(cagr, 95)),
            "drawdown_median": float(np.median(self.max_drawdown)),
            "drawdown_p95": float(np.percentile(self.max_drawdown, 95)),
            "ran_out": float((self.ran_out >= 0).mean()),
        }

    def milestones(self, years=(1, 3, 5, 10, 15, 20, 25, 30, 40)) -> pd.DataFrame:
        rows = []
        for y in [y for y in years if y <= self.years] + ([self.years] if self.years not in years else []):
            i = min(int(round(y * WEEKS_PER_YEAR)), len(self.dates) - 1)
            v = self.value[:, i]
            p = np.percentile(v, PERCENTILES)
            rows.append({"year": y, "date": self.dates[i], "invested": self.invested[i],
                         **{f"p{q}": x for q, x in zip(PERCENTILES, p)},
                         "below_invested": float((v < self.invested[i]).mean()),
                         "median_real": float(np.median(v) / (1 + self.plan.inflation) ** y)})
        return pd.DataFrame(rows).drop_duplicates("year").set_index("year")

    def by_stress(self, state: int | None = None) -> pd.DataFrame:
        """Outcomes split by how much of the time a path spent in the most stressed state (thirds of the paths)."""
        s = len(self.names) - 1 if state is None else state
        share = self.state_share()[:, s]
        cagr = self.cagr()
        edges = np.unique(np.percentile(share, [0, 100 / 3, 200 / 3, 100]))
        if len(edges) < 2:
            return pd.DataFrame()
        groups = np.clip(np.searchsorted(edges, share, side="right") - 1, 0, len(edges) - 2)
        rows = []
        for g in range(len(edges) - 1):
            m = groups == g
            if not m.any():
                continue
            rows.append({"time in " + self.names[s]: f"{share[m].min():.0%} - {share[m].max():.0%}", "paths": int(m.sum()),
                         "median CAGR": float(np.median(cagr[m])), "bad case CAGR (5%)": float(np.percentile(cagr[m], 5)),
                         "median max drawdown": float(np.median(self.max_drawdown[m]))})
        return pd.DataFrame(rows)


def simulate(pool: ReturnPool, weights: pd.Series, model: RegimeModel, plan: Plan, scenario: Scenario | None = None, *,
             returns_mode: str = "market", rf: float = 0.0, index_return: float | None = None,
             targets: np.ndarray | None = None, n_paths: int = 5000, regimes: bool = True,
             block: float = BLOCK_WEEKS, seed: int = 0, start: pd.Timestamp | None = None) -> SimResult:
    """Run ``n_paths`` futures of the portfolio. ``weights`` are target weights by symbol (re-normalised to the pool).

    ``rf`` (cash rate) and ``index_return`` (the driver's long-run mean) are weekly and only matter for the market-implied
    part of ``returns_mode``; ``targets`` (weekly mean per pool asset) overrides all three. The same ``seed`` gives the same random draws,
    so two scenarios run with one seed differ because of the scenario, not because of luck.
    ``regimes=False`` draws from all past weeks alike (a plain block bootstrap): the start state, stretch and frequency
    of the scenario are then ignored, a replayed crisis still plays first.
    """
    scenario = scenario or Scenario()
    w = weights.reindex(pool.assets).fillna(0.0).to_numpy(dtype=float)
    if w.sum() <= 0:
        raise ValueError("no weight on any holding with enough history")
    w = w / w.sum()
    mix = model.long_run
    if targets is None:
        targets = pool.targets(returns_mode, mix, rf=rf, index_return=index_return)
    R = pool.shifted(targets, mix)

    P = scenario.transition if scenario.transition is not None else adjusted_transition(
        model.transition, scenario.stretch, scenario.frequency)
    fixed_rows = crisis_weeks(pool, scenario.replay, scenario.replay_after) if scenario.replay else None
    if fixed_rows is not None and len(fixed_rows) == 0:
        raise ValueError(f"the {model.driver} history does not reach the {CRISES[scenario.replay][0]}")
    fixed_states = pool.state[fixed_rows] if fixed_rows is not None else None

    rng = np.random.default_rng(seed)
    n_steps = plan.steps
    if regimes:
        names = model.names
        states = state_paths(P, start_distribution(model, scenario, P), n_paths, n_steps, rng, fixed_states)
        rows = pool_rows(states, pool.state, rng, block=block, fixed=fixed_rows)
    else:
        names = ("All weeks",)
        states = np.zeros((n_paths, n_steps), dtype=np.int8)
        rows = pool_rows(states, np.zeros(pool.n_weeks, dtype=int), rng, block=block, fixed=fixed_rows)

    t0 = pd.Timestamp(start).normalize() if start is not None else pd.Timestamp.today().normalize()
    sched = schedule(t0, plan)
    flows, reb = sched["flow"].to_numpy(), sched["rebalance"].to_numpy()
    cost = plan.cost

    h = np.outer(np.full(n_paths, plan.initial), w)
    value = np.empty((n_paths, n_steps + 1), dtype=np.float32)
    value[:, 0] = plan.initial
    growth = np.ones(n_paths)
    peak = np.ones(n_paths)
    mdd = np.zeros(n_paths)
    ran_out = np.full(n_paths, -1, dtype=np.int32)
    for t in range(n_steps):
        before = h.sum(axis=1)
        h *= 1.0 + R[rows[:, t]]
        after = h.sum(axis=1)
        step = np.divide(after, before, out=np.ones(n_paths), where=before > 0)
        f = flows[t]
        if f > 0:
            h += (f * (1 - cost)) * w
        elif f < 0:
            need = -f / (1 - cost)
            frac = np.divide(need, after, out=np.ones(n_paths), where=after > 0)
            empty = frac >= 1.0
            ran_out[empty & (ran_out < 0) & (after > 0)] = t
            h *= (1.0 - np.minimum(frac, 1.0))[:, None]
        if reb[t]:
            v = h.sum(axis=1)
            target = v[:, None] * w
            fee = cost * np.abs(target - h).sum(axis=1)
            v2 = np.maximum(v - fee, 0.0)
            h = v2[:, None] * w
            step *= np.divide(v2, v, out=np.ones(n_paths), where=v > 0)
        growth *= step
        peak = np.maximum(peak, growth)
        mdd = np.maximum(mdd, 1.0 - growth / peak)
        value[:, t + 1] = h.sum(axis=1)

    invested = plan.initial + np.concatenate([[0.0], np.cumsum(flows)])
    return SimResult(dates=pd.DatetimeIndex([t0]).append(sched.index), value=value, invested=invested,
                     states=states, growth=growth, max_drawdown=mdd, ran_out=ran_out, names=names, plan=plan,
                     scenario=scenario)


def crisis_table(pool: ReturnPool, weights: pd.Series) -> pd.DataFrame:
    """How today's portfolio (bought and held, weights drifting) went through each crisis the history reaches.

    Uses the pooled returns as they happened (real, or estimated where a holding had not listed yet), without the
    return assumption: this is history, not a simulation. ``recovery_weeks`` counts from the end of the window until the
    portfolio is back to where it started, NaN if it had not by the end of the data.
    """
    w = weights.reindex(pool.assets).fillna(0.0).to_numpy(dtype=float)
    if w.sum() <= 0:
        return pd.DataFrame()
    w = w / w.sum()
    rows = []
    for key, (label, _, _) in CRISES.items():
        idx = crisis_weeks(pool, key)
        if not len(idx):
            continue
        h = w.copy()
        line = []
        for t in range(idx[0], pool.n_weeks):
            h = h * (1 + pool.returns[t])
            line.append(h.sum())
            if t >= idx[-1] and h.sum() >= 1.0:
                break
        line = np.array(line)
        inside = line[: len(idx)]
        path = np.concatenate([[1.0], inside])
        dd = 1 - path / np.maximum.accumulate(path)
        rec = np.flatnonzero(line[len(idx):] >= 1.0)
        rows.append({
            "crisis": label, "key": key, "weeks": len(idx),
            "index": float(np.prod(1 + pool.driver[idx]) - 1),
            "portfolio": float(inside[-1] - 1),
            "worst week": float(np.min(np.diff(path) / path[:-1])),
            "max drawdown": float(dd.max()),
            "recovery_weeks": float(rec[0] + 1) if inside[-1] < 1.0 and len(rec) else (0.0 if inside[-1] >= 1.0 else np.nan),
            "estimated": float(w @ (~pool.real[idx]).mean(axis=0)),
        })
    return pd.DataFrame(rows)
