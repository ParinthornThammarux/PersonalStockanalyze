"""Composite 0-100 score from cross-sectional percentile ranks.

Each component is the average percentile rank (within the table passed in) of a few
columns; the composite is the weight-averaged score over the components that exist, so a
table without fundamentals still gets a technical score instead of NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# component -> [(column, higher_is_better)]
COMPONENTS: dict[str, list[tuple[str, bool]]] = {
    "momentum": [("ret_126d", True), ("mom_12_1", True), ("ret_63d", True)],
    "trend": [("trend_score", True), ("sma200_dist", True), ("trend_r2", True)],
    "risk": [("hvol_60", False), ("drawdown_1y_max", True), ("atr_pct", False)],
    "liquidity": [("dollar_volume_20", True)],
    "value": [("pe", False), ("fcf_yield", True), ("ev_ebitda", False)],
    "quality": [("roe", True), ("operating_margin", True), ("debt_to_equity", False)],
}
DEFAULT_WEIGHTS = {"momentum": 0.30, "trend": 0.25, "risk": 0.15, "liquidity": 0.05, "value": 0.15, "quality": 0.10}


def _rank(series: pd.Series, higher_is_better: bool, positive_only: bool = False) -> pd.Series:
    s = series.astype("float64")
    if positive_only:  # a negative P/E is not "cheap"
        s = s.where(s > 0)
    return s.rank(pct=True, ascending=higher_is_better) * 100


def score_table(df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Return ``df`` plus ``score_<component>`` columns and a composite ``score``."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    out = df.copy()
    present: dict[str, pd.Series] = {}
    for comp, cols in COMPONENTS.items():
        ranks = [
            _rank(df[c], good, positive_only=c in {"pe", "ev_ebitda"})
            for c, good in cols if c in df.columns
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
