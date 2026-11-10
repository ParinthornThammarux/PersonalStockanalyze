"""Walk-forward check of the regime simulation on the index itself.

Every 4 weeks from the 10th year of history on, using only the weeks known at that point, simulate the index's next
3 months, 6 months and 1 year twice - with the regime model (state today + transition matrix + draws from same-state
weeks) and with a plain block bootstrap of all past weeks (no regimes) - then see where the real outcome landed:

- **coverage**: how often the real return fell inside the middle 50 % / 90 % of the simulated ones (should be ~50 / 90);
- **CRPS**: one score for the whole simulated distribution against what happened (lower is better);
- a Diebold-Mariano test on the CRPS differences with a Newey-West variance, because the 1-year windows overlap and
  consecutive checks are far from independent.

The regime parameters are re-estimated once a year (re-filtering the states every check), which is close to what a
user re-running the page would have seen. The same drift (each window's own past average) is used by both methods,
so only the shape of the distribution is being compared.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import Callable

import numpy as np
import pandas as pd

from stockanalyzer.analysis import regimes as rg
from stockanalyzer.simulation.montecarlo import BLOCK_WEEKS, pool_rows, state_paths

HORIZONS = {"3 months": 13, "6 months": 26, "1 year": 52}
MIN_TRAIN = 520  # ten years of weeks before the first check
STEP = 4  # weeks between checks
REFIT = 52  # weeks between re-estimates of the regime parameters
METHODS = ("Regimes", "Plain bootstrap")


def crps(samples: np.ndarray, y: float) -> float:
    """Continuous ranked probability score of a sample forecast: E|X - y| - E|X - X'| / 2."""
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    spread = 2.0 * np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / (n * n)
    return float(np.mean(np.abs(x - y)) - 0.5 * spread)


def diebold_mariano(d: np.ndarray, lag: int) -> tuple[float, float]:
    """Mean of the loss differences ``d`` and its two-sided p-value, Newey-West variance with ``lag`` lags."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return float("nan"), float("nan")
    e = d - d.mean()
    var = e @ e / n
    for j in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - j / (lag + 1)) * (e[j:] @ e[:-j]) / n
    if var <= 0:
        return float(d.mean()), float("nan")
    z = d.mean() / sqrt(var / n)
    return float(d.mean()), float(2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2)))))


@dataclass
class Calibration:
    driver: str
    checks: pd.DataFrame  # one row per (origin, horizon, method): pit, crps, start state
    first: pd.Timestamp
    last: pd.Timestamp

    def table(self) -> pd.DataFrame:
        g = self.checks.groupby(["horizon", "method"], sort=False)
        out = pd.DataFrame({
            "checks": g.size(),
            "inside 50%": g["pit"].apply(lambda p: ((p >= 0.25) & (p <= 0.75)).mean()),
            "inside 90%": g["pit"].apply(lambda p: ((p >= 0.05) & (p <= 0.95)).mean()),
            "below 5%": g["pit"].apply(lambda p: (p < 0.05).mean()),
            "above 95%": g["pit"].apply(lambda p: (p > 0.95).mean()),
            "CRPS": g["crps"].mean(),
        })
        return out

    def by_state(self) -> pd.DataFrame:
        """90 % coverage split by the state the market was in when the simulation started."""
        g = self.checks.groupby(["horizon", "start", "method"], sort=False)
        return pd.DataFrame({"checks": g.size(), "inside 90%": g["pit"].apply(lambda p: ((p >= 0.05) & (p <= 0.95)).mean()),
                             "CRPS": g["crps"].mean()})

    def verdict(self) -> pd.DataFrame:
        """Per horizon: CRPS of regimes minus plain bootstrap (negative = regimes better) and the overlap-aware p-value."""
        rows = []
        for name, h in HORIZONS.items():
            c = self.checks[self.checks["horizon"] == name]
            if c.empty:
                continue
            wide = c.pivot(index="origin", columns="method", values="crps").dropna()
            if wide.empty or not set(METHODS) <= set(wide.columns):
                continue
            diff, p = diebold_mariano((wide[METHODS[0]] - wide[METHODS[1]]).to_numpy(), lag=int(np.ceil(h / STEP)))
            rel = diff / wide[METHODS[1]].mean() if wide[METHODS[1]].mean() else np.nan
            rows.append({"horizon": name, "checks": len(wide), "independent windows": int(len(wide) * STEP / h),
                         "CRPS change": rel, "p-value": p})
        return pd.DataFrame(rows).set_index("horizon") if rows else pd.DataFrame()


def calibrate(weekly: pd.Series, k: int = 2, *, driver: str = "", horizons: dict[str, int] | None = None,
              min_train: int = MIN_TRAIN, step: int = STEP, refit: int = REFIT, n_paths: int = 2000,
              block: float = BLOCK_WEEKS, seed: int = 0, search_reps: int = 10,
              progress: Callable[[int, int], None] | None = None) -> Calibration:
    horizons = horizons or HORIZONS
    y = weekly.dropna().astype("float64")
    r = y.to_numpy()
    H = max(horizons.values())
    origins = list(range(min_train, len(r) - min(horizons.values()) + 1, step))
    if not origins:
        raise ValueError(f"need more than {min_train} weeks of {driver or 'driver'} history for the check, have {len(r)}")
    rng = np.random.default_rng(seed)
    rows, model, fitted_at = [], None, -10**9
    for n, t in enumerate(origins):
        train = y.iloc[:t]
        if t - fitted_at >= refit:
            model = rg.fit(train, k, driver=driver, search_reps=search_reps, seed=seed)
            labels, cur = model.labels.to_numpy(), model.current
            fitted_at = t
        else:
            labels, cur = rg.refilter(model, train)
        past = r[:t]
        start_state = model.names[int(np.argmax(cur))]
        steps = min(H, len(r) - t)
        states = state_paths(model.transition, np.asarray(cur) @ model.transition, n_paths, steps, rng)
        sims = {
            METHODS[0]: past[pool_rows(states, labels.astype(int), rng, block=block)],
            METHODS[1]: past[pool_rows(np.zeros((n_paths, steps), dtype=np.int8), np.zeros(t, dtype=int), rng, block=block)],
        }
        growth = {m: np.cumprod(1 + s, axis=1) for m, s in sims.items()}
        for name, h in horizons.items():
            if t + h > len(r):
                continue
            real = float(np.prod(1 + r[t:t + h]) - 1)
            for m in METHODS:
                outcome = growth[m][:, h - 1] - 1
                rows.append({"origin": y.index[t - 1], "horizon": name, "method": m, "start": start_state,
                             "pit": float((outcome <= real).mean()), "crps": crps(outcome, real), "real": real})
        if progress:
            progress(n + 1, len(origins))
    checks = pd.DataFrame(rows)
    return Calibration(driver=driver, checks=checks, first=checks["origin"].min(), last=checks["origin"].max())
