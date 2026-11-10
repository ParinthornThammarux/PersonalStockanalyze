"""What analyst consensus says about growth, earnings, price and direction (pure functions over ``Estimates``)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stockanalyzer.domain.estimates import PERIOD_NAMES, Estimates

REVISION_BAND = 0.01  # an EPS estimate that moved less than 1% in a month counts as unchanged
GAP_BAND = 0.05  # growth within 5 points of the consensus counts as in line


def convert(est: Estimates, fx: float) -> Estimates:
    """The same consensus with EPS and revenue amounts multiplied by ``fx`` (reporting -> price currency).

    Price targets are already in the price currency, and growth rates, analyst counts and revision counts do not change.
    """
    if fx == 1.0:
        return est
    money = {"avg", "low", "high", "year_ago"}
    trend = {"current", "d7", "d30", "d60", "d90"}

    def scale(table: dict, keys: set[str]) -> dict:
        return {p: {k: v * fx if k in keys else v for k, v in row.items()} for p, row in table.items()}

    return Estimates(symbol=est.symbol, currency=est.currency, eps=scale(est.eps, money), revenue=scale(est.revenue, money),
                     eps_trend=scale(est.eps_trend, trend), eps_revisions=est.eps_revisions, target=est.target,
                     recommendation=est.recommendation, as_of=est.as_of)


def _cagr(base: float | None, end: float | None, years: int) -> float | None:
    if base is None or end is None or base <= 0 or end <= 0 or years <= 0:
        return None
    return float((end / base) ** (1 / years) - 1)


@dataclass(frozen=True)
class ConsensusGrowth:
    revenue: float | None  # yearly growth from the last reported fiscal year to the furthest estimate
    revenue_years: int  # 2 when next fiscal year is covered, 1 when only the current one is, 0 when neither
    revenue_analysts: int
    eps: float | None
    eps_years: int
    eps_analysts: int


def consensus_growth(est: Estimates) -> ConsensusGrowth:
    """Compound yearly growth the analysts expect, measured from the last *completed* fiscal year (``year_ago`` of ``0y``).

    Two years when the next fiscal year has an estimate, else one. Revenue and EPS are separate: EPS growth from a loss
    (or a zero) is undefined and left empty.
    """
    def run(table: dict) -> tuple[float | None, int, int]:
        this, nxt = table.get("0y"), table.get("+1y")
        if this and nxt and this.get("year_ago") and nxt.get("avg"):
            return _cagr(this["year_ago"], nxt["avg"], 2), 2, int(min(this.get("analysts", 0), nxt.get("analysts", 0)))
        if this and this.get("year_ago") and this.get("avg"):
            return _cagr(this["year_ago"], this["avg"], 1), 1, int(this.get("analysts", 0))
        return None, 0, 0

    rev, rev_years, rev_n = run(est.revenue)
    eps, eps_years, eps_n = run(est.eps)
    return ConsensusGrowth(rev, rev_years, rev_n, eps, eps_years, eps_n)


def forward_pe(price: float, est: Estimates) -> dict[str, float]:
    """Price over the EPS estimate of each fiscal year (``0y`` / ``+1y``); a year with no positive estimate is left out."""
    out = {}
    for period in ("0y", "+1y"):
        eps = est.eps.get(period, {}).get("avg")
        if eps and eps > 0 and price > 0:
            out[period] = price / eps
    return out


def forward_peg(price: float, est: Estimates) -> float | None:
    """P/E on the furthest EPS estimate divided by the yearly EPS growth (in points) the analysts expect; ``None`` if either is missing or growth is not positive."""
    pes = forward_pe(price, est)
    growth = consensus_growth(est).eps
    if not pes or growth is None or growth <= 0:
        return None
    return float(pes.get("+1y", pes.get("0y")) / (growth * 100))


@dataclass(frozen=True)
class TargetView:
    mean: float | None
    median: float | None
    high: float | None
    low: float | None
    upside_mean: float | None
    upside_high: float | None
    upside_low: float | None
    analysts: int | None
    recommendation_mean: float | None
    recommendation: str | None


def target_view(price: float, est: Estimates) -> TargetView | None:
    """Analyst price target against ``price`` (the upside is to the mean target); ``None`` when there is no target."""
    t = est.target
    if not t.get("mean") and not est.recommendation.get("mean"):
        return None

    def up(level: float | None) -> float | None:
        return None if not level or price <= 0 else float(level / price - 1)

    n = t.get("analysts") or est.recommendation.get("analysts")
    return TargetView(mean=t.get("mean"), median=t.get("median"), high=t.get("high"), low=t.get("low"),
                      upside_mean=up(t.get("mean")), upside_high=up(t.get("high")), upside_low=up(t.get("low")),
                      analysts=int(n) if n else None, recommendation_mean=est.recommendation.get("mean"),
                      recommendation=est.recommendation.get("key"))


RECOMMENDATION_WORDS = ((1.5, "strong buy"), (2.5, "buy"), (3.5, "hold"), (4.5, "underperform"), (5.01, "sell"))


def recommendation_word(mean: float) -> str:
    """Yahoo's 1-5 recommendation mean in words (1 = strong buy, 5 = sell)."""
    return next(word for limit, word in RECOMMENDATION_WORDS if mean < limit)


@dataclass(frozen=True)
class Revision:
    period: str
    change_30d: float | None  # change of the average EPS estimate over the last 30 days
    change_90d: float | None
    up_30d: int | None  # analysts who raised / cut it in the last 30 days
    down_30d: int | None
    direction: str  # "rising" / "falling" / "flat" / "unknown"


def revision(est: Estimates, period: str = "0y") -> Revision:
    """How the EPS estimate for ``period`` has moved: percentage change over 30 / 90 days and how many analysts moved which way.

    Rising or falling needs the 30-day change beyond ``REVISION_BAND``; with no 30-day figure the analysts' counts decide.
    """
    trend = est.eps_trend.get(period, {})
    counts = est.eps_revisions.get(period, {})

    def change(days: str) -> float | None:
        now, then = trend.get("current"), trend.get(days)
        return None if now is None or then is None or then == 0 else float((now - then) / abs(then))

    c30, c90 = change("d30"), change("d90")
    up, down = counts.get("up30"), counts.get("down30")
    if c30 is not None:
        direction = "rising" if c30 > REVISION_BAND else "falling" if c30 < -REVISION_BAND else "flat"
    elif up is not None and down is not None:
        direction = "rising" if up > down else "falling" if down > up else "flat"
    else:
        direction = "unknown"
    return Revision(period, c30, c90, None if up is None else int(up), None if down is None else int(down), direction)


def expectation_gap(implied: float | None, consensus: float | None) -> tuple[float, str] | None:
    """The growth a price implies minus the growth analysts expect, with a verdict in words; ``None`` when either is missing."""
    if implied is None or consensus is None or implied != implied or consensus != consensus:
        return None
    gap = implied - consensus
    if gap > GAP_BAND:
        return gap, "asks for more growth than analysts forecast"
    if gap < -GAP_BAND:
        return gap, "needs less growth than analysts forecast"
    return gap, "is about what analysts forecast"


def history_frame(snapshots: list[tuple[str, Estimates]]) -> pd.DataFrame:
    """Daily snapshots as a frame: mean price target, EPS estimates of both fiscal years, revenue of the current one, analysts."""
    rows = []
    for day, est in snapshots:
        rows.append({
            "day": pd.Timestamp(day), "target_mean": est.target.get("mean"),
            "eps_0y": est.eps.get("0y", {}).get("avg"), "eps_1y": est.eps.get("+1y", {}).get("avg"),
            "revenue_0y": est.revenue.get("0y", {}).get("avg"),
            "recommendation_mean": est.recommendation.get("mean"),
        })
    if not rows:
        return pd.DataFrame(columns=["target_mean", "eps_0y", "eps_1y", "revenue_0y", "recommendation_mean"])
    return pd.DataFrame(rows).set_index("day").sort_index().astype("float64")


def period_label(period: str) -> str:
    return PERIOD_NAMES.get(period, period)
