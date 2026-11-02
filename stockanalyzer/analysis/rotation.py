"""Sector rotation: who leads, who lags and how that changes over time.

Everything works on a ``close`` frame (dates x sectors, any level that moves like a price) and a
benchmark series on the same dates. Windows are calendar dates, so holidays and weekends never shift them.

"Where money is moving" is measured with two proxies, because Yahoo has no historical fund flows:

- **relative strength**: what the market paid for a sector compared with the benchmark, and how that
  ranking changed against the window before;
- **Chaikin money flow**: volume-weighted buying vs selling pressure inside each bar (+1 = every bar closed
  at its high on volume, -1 = at its low).

Neither is a count of dollars entering or leaving a sector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUADRANTS = ("Leading", "Weakening", "Lagging", "Improving")  # clockwise: the order a sector normally cycles through
# measured on 10 years of US sectors, Thai baskets and asset classes: ~4.5 quadrant changes per sector a year,
# a median of 8-10 weeks in one quadrant (3 / 26 / 5 flipped every 5 weeks and filled the log with noise)
RRG_SMOOTH, RRG_TREND, RRG_MOMENTUM = 6, 39, 13
REGIME_WINDOW, REGIME_BAND = 100, 0.01  # ~4.5 regime flips a year, a median of 40-50 days apart (no band: every 5 days)
MIN_REGIME_BARS = 20


# -- building blocks ----------------------------------------------------------------------------
def equal_weight_level(returns: pd.DataFrame, base: float = 100.0) -> pd.Series:
    """Level of an equal-weight basket rebalanced daily; a member counts from its first available return."""
    mean = returns.mean(axis=1, skipna=True)  # NaN where no member has a return
    level = base * (1 + mean.fillna(0.0)).cumprod()
    return level.where(mean.notna().cummax())


def _at(frame: pd.DataFrame | pd.Series, date) -> pd.Series | float:
    """Last row on or before ``date``; NaN (per column) when the data starts later."""
    pos = frame.index.searchsorted(pd.Timestamp(date), side="right") - 1
    if pos < 0:
        return pd.Series(np.nan, index=frame.columns) if isinstance(frame, pd.DataFrame) else np.nan
    return frame.iloc[pos]


def window_returns(close: pd.DataFrame | pd.Series, start, end) -> pd.Series | float:
    """Return from the close on or before ``start`` to the close on or before ``end`` (a Series per column for a frame)."""
    return _at(close, end) / _at(close, start) - 1


def chaikin_flow(mfv: pd.DataFrame, volume: pd.DataFrame, start, end) -> pd.Series:
    """Chaikin money flow per column over ``(start, end]``: sum of money-flow volume / sum of volume."""
    mask = (mfv.index > pd.Timestamp(start)) & (mfv.index <= pd.Timestamp(end))
    total = volume[mask].sum()
    return (mfv[mask].sum() / total.where(total > 0)).astype("float64")


def _rank(values: pd.Series) -> pd.Series:
    return values.rank(ascending=False, method="min")


# -- who leads, and what changed --------------------------------------------------------------------
def scoreboard(close: pd.DataFrame, bench: pd.Series, start, end,
               flow: pd.Series | None = None, prev_flow: pd.Series | None = None) -> pd.DataFrame:
    """Performance of every sector over ``(start, end]`` against the equal-length window before it.

    ``rank`` 1 is the best return; ``rank_change`` is positive when a sector climbed versus the previous window.
    ``rel`` is the return relative to the benchmark: (1 + r) / (1 + benchmark) - 1.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    prev = start - (end - start)
    ret, prev_ret = window_returns(close, start, end), window_returns(close, prev, start)
    b, prev_b = window_returns(bench, start, end), window_returns(bench, prev, start)
    out = pd.DataFrame({"ret": ret, "rel": (1 + ret) / (1 + b) - 1, "prev_ret": prev_ret,
                        "prev_rel": (1 + prev_ret) / (1 + prev_b) - 1})
    out["rank"] = _rank(ret)
    out["prev_rank"] = _rank(prev_ret)
    out["rank_change"] = out["prev_rank"] - out["rank"]
    out["flow"] = flow.reindex(out.index) if flow is not None else np.nan
    out["prev_flow"] = prev_flow.reindex(out.index) if prev_flow is not None else np.nan
    return out.dropna(subset=["ret"]).sort_values("rank")


def periodic_returns(close: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
    """Return of each sector in each calendar period (``W-FRI`` week, ``ME`` month, ``QE`` quarter), oldest first.

    The last period is only complete if the data ends on its last trading day.
    """
    level = close.resample(freq).last()
    return level.pct_change(fill_method=None).iloc[1:].dropna(how="all")


def rank_history(returns: pd.DataFrame) -> pd.DataFrame:
    """Rank of each sector in each period (1 = best)."""
    return returns.rank(axis=1, ascending=False, method="min")


def leaders(returns: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Per period: the ``n`` best sectors, the worst one, and where the lead moved to from the period before."""
    rows, previous = [], None
    for period, r in returns.iterrows():
        r = r.dropna().sort_values(ascending=False)
        if r.empty:
            continue
        top = list(r.index[:n])
        moved = f"{previous} → {top[0]}" if previous and previous != top[0] else ("held the lead" if previous else "")
        rows.append({"period": period, **{f"#{i + 1}": f"{s} ({r[s]:+.1%})" for i, s in enumerate(top)},
                     "worst": f"{r.index[-1]} ({r.iloc[-1]:+.1%})", "lead moved": moved})
        previous = top[0]
    return pd.DataFrame(rows).set_index("period") if rows else pd.DataFrame()


# -- relative rotation graph --------------------------------------------------------------------------
def rrg(close: pd.DataFrame, bench: pd.Series, *, bar: str | None = "W-FRI", trend: int = RRG_TREND,
        momentum: int = RRG_MOMENTUM, smooth: int = RRG_SMOOTH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """RS-Ratio and RS-Momentum in the style of a Relative Rotation Graph (not JdK's proprietary formula).

    ``rs`` = sector / benchmark. **RS-Ratio** = ``smooth``-bar average of ``rs`` over its ``trend``-bar average x 100:
    above 100 the sector is stronger than its own recent norm. **RS-Momentum** = RS-Ratio over its value
    ``momentum`` bars ago x 100: above 100 the relative strength is still improving.
    """
    px, b = (close, bench) if bar is None else (close.resample(bar).last(), bench.resample(bar).last())
    rs = px.div(b, axis=0)
    ratio = (100 * rs.rolling(smooth, min_periods=smooth).mean() / rs.rolling(trend, min_periods=trend).mean()).dropna(how="all")
    return ratio, 100 * ratio / ratio.shift(momentum)


def quadrants(ratio: pd.DataFrame, mom: pd.DataFrame) -> pd.DataFrame:
    """Quadrant name per date and sector; ``""`` where either input is missing."""
    r, m = ratio.to_numpy(), mom.reindex_like(ratio).to_numpy()
    labels = np.select([(r >= 100) & (m >= 100), (r >= 100) & (m < 100), (r < 100) & (m < 100), (r < 100) & (m >= 100)],
                       QUADRANTS, default="")
    return pd.DataFrame(labels, index=ratio.index, columns=ratio.columns)


def quadrant_age(quads: pd.DataFrame) -> pd.Series:
    """Bars each sector has been in its current quadrant (the last row of ``quads``)."""
    ages = {}
    for name in quads.columns:
        q = quads[name].replace("", np.nan).dropna()
        if not q.empty:
            run = (q != q.shift()).cumsum()
            ages[name] = int((run == run.iloc[-1]).sum())
    return pd.Series(ages, dtype="int64")


def quadrant_changes(quads: pd.DataFrame, since=None) -> pd.DataFrame:
    """Every move of a sector from one quadrant to another, newest first."""
    rows = []
    for name in quads.columns:
        q = quads[name].replace("", np.nan).dropna()
        for date, now, before in zip(q.index[1:], q.iloc[1:], q.iloc[:-1]):
            if now != before and (since is None or date >= pd.Timestamp(since)):
                rows.append({"date": date, "sector": name, "from": before, "to": now})
    out = pd.DataFrame(rows, columns=["date", "sector", "from", "to"])
    return out.sort_values(["date", "sector"], ascending=[False, True], ignore_index=True)


# -- risk appetite --------------------------------------------------------------------------------------
def offensive_defensive(close: pd.DataFrame, offensive, defensive, window: int = REGIME_WINDOW,
                        band: float = REGIME_BAND) -> pd.DataFrame:
    """Equal-weight offensive vs defensive groups, their ratio and the ratio's ``window``-bar average.

    ``regime`` flips to ``Risk-off`` once the ratio falls more than ``band`` (a fraction) below that average and back
    to ``Risk-on`` once it rises more than ``band`` above it; in between it keeps its last value, so a ratio
    hovering around its average does not flip every few days.
    """
    off, dfn = [c for c in offensive if c in close.columns], [c for c in defensive if c in close.columns]
    if not off or not dfn:
        return pd.DataFrame(columns=["offensive", "defensive", "ratio", "trend", "regime"])
    returns = close.pct_change(fill_method=None)
    out = pd.DataFrame({"offensive": equal_weight_level(returns[off]), "defensive": equal_weight_level(returns[dfn])})
    out = out.dropna()
    out["ratio"] = out["offensive"] / out["defensive"] * 100 / (out["offensive"].iloc[0] / out["defensive"].iloc[0])
    out["trend"] = out["ratio"].rolling(window, min_periods=min(window, MIN_REGIME_BARS)).mean()
    state, regime = "", []
    for ratio, trend in zip(out["ratio"].to_numpy(), out["trend"].to_numpy()):
        if np.isnan(trend):
            regime.append("")
            continue
        if not state:
            state = "Risk-on" if ratio >= trend else "Risk-off"
        elif state == "Risk-on" and ratio < trend * (1 - band):
            state = "Risk-off"
        elif state == "Risk-off" and ratio > trend * (1 + band):
            state = "Risk-on"
        regime.append(state)
    out["regime"] = regime
    return out


def regime_changes(regime: pd.Series) -> pd.DataFrame:
    """Dates on which the regime flipped, with how long the previous one lasted (calendar days), newest first."""
    r = regime[regime != ""]
    flips = r[r != r.shift()].iloc[1:]
    rows = [{"date": d, "to": to, "after_days": (d - start).days}
            for d, to, start in zip(flips.index, flips, r[r != r.shift()].index[:-1])]
    out = pd.DataFrame(rows, columns=["date", "to", "after_days"])
    return out.sort_values("date", ascending=False, ignore_index=True)
