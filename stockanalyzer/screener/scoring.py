"""Composite 0-100 score from cross-sectional percentile ranks.

Each component is the average percentile rank (within the table passed in) of a few
columns; the composite is the weight-averaged score over the components that exist, so a
table without fundamentals still gets a technical score instead of NaN.

Three choices keep the composite from being one signal counted several times:

* every column in a component measures something the others do not (the 3-month return, the distance to the 200-day
  average and ATR / price were dropped: they correlate 0.7-0.94 with the 6-month return and volatility that stay);
* the trend-quality column is R² *signed by the slope* - a plain R² is as high for a steady slide as for a steady climb;
* value and quality are ranked **against peers** (same industry in the same market, else the whole market), because a bank
  at 1.2x book is not cheaper than a software company at 3x book - it is just a bank.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# component -> [(column, higher_is_better)]
COMPONENTS: dict[str, list[tuple[str, bool]]] = {
    "momentum": [("ret_126d", True), ("mom_12_1", True)],
    "trend": [("trend_score", True), ("trend_quality", True)],
    "risk": [("hvol_60", False), ("drawdown_1y_max", True)],
    "liquidity": [("dollar_volume_20", True)],
    "value": [("pe", False), ("pb", False), ("fcf_yield", True), ("ev_ebitda", False)],
    "quality": [("roe", True), ("operating_margin", True), ("debt_to_equity", False)],
}
DEFAULT_WEIGHTS = {"momentum": 0.25, "trend": 0.15, "risk": 0.15, "liquidity": 0.05, "value": 0.25, "quality": 0.15}
PEER_RELATIVE = {"value", "quality"}  # components ranked inside an industry / market rather than the whole pool
MIN_PEERS = 10  # a peer group with fewer stocks that have the figure falls back to the next broader group
POSITIVE_ONLY = {"pe", "pb", "ev_ebitda"}  # a negative P/E is not "cheap"


def with_trend_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``trend_quality`` = R² of the 60-day fit, negative when the fit slopes down (a smooth decline is not a good trend)."""
    if "trend_quality" in df.columns or not {"trend_r2", "trend_slope_annual"} <= set(df.columns):
        return df
    return df.assign(trend_quality=df["trend_r2"] * np.sign(df["trend_slope_annual"]))


def peer_groups(df: pd.DataFrame) -> list[pd.Series]:
    """Grouping keys from the broadest to the narrowest: the market (Thai ``.BK`` or other), then industry within the market.

    Rows with no industry get a missing key in the second group and so fall back to their market.
    """
    market = pd.Series(np.where(df.index.astype(str).str.endswith(".BK"), "TH", "other"), index=df.index, dtype="object")
    groups = [market]
    if "industry" in df.columns:
        industry = df["industry"].astype("object")
        known = industry.notna() & industry.astype(str).str.strip().ne("")
        groups.append((market + " | " + industry.astype(str)).where(known))
    return groups


def _rank(series: pd.Series, higher_is_better: bool, positive_only: bool = False,
          groups: list[pd.Series] | None = None) -> pd.Series:
    """Percentile rank 0-100 over the whole table, replaced by the rank inside each peer group that is big enough."""
    s = series.astype("float64")
    if positive_only:
        s = s.where(s > 0)
    out = s.rank(pct=True, ascending=higher_is_better) * 100
    for key in groups or []:  # broad -> narrow: the narrowest group with enough peers wins
        grouped = s.groupby(key)
        ranked = grouped.rank(pct=True, ascending=higher_is_better) * 100
        out = ranked.where(grouped.transform("count") >= MIN_PEERS, out)
    return out


def score_table(df: pd.DataFrame, weights: dict[str, float] | None = None, *, peer_relative: bool = True) -> pd.DataFrame:
    """Return ``df`` plus ``score_<component>`` columns and a composite ``score``.

    ``peer_relative=False`` ranks value and quality over the whole table like the other components.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    out = df.copy()
    ranked_from = with_trend_quality(df)
    groups = peer_groups(df) if peer_relative else None
    present: dict[str, pd.Series] = {}
    for comp, cols in COMPONENTS.items():
        ranks = [
            _rank(ranked_from[c], good, positive_only=c in POSITIVE_ONLY, groups=groups if comp in PEER_RELATIVE else None)
            for c, good in cols if c in ranked_from.columns
        ]
        if ranks:
            present[comp] = pd.concat(ranks, axis=1).mean(axis=1, skipna=True)
            out[f"score_{comp}"] = present[comp]
    if not present:
        out["score"] = np.nan
        return out

    frame = pd.DataFrame(present)
    w = pd.Series({c: weights.get(c, 0.0) for c in frame.columns})
    valid = frame.notna() * w
    out["score"] = (frame.fillna(0) * w).sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)
    return out
