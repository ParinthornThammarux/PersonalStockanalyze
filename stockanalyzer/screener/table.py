"""Join features with universe metadata and cached fundamentals into one screening table."""

from __future__ import annotations

import pandas as pd


def build_screen_table(
    features: pd.DataFrame,
    universe: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per symbol. ``market_cap`` prefers the fundamentals snapshot over the static list."""
    table = features.copy()
    if universe is not None and not universe.empty:
        meta = universe.drop_duplicates("symbol").set_index("symbol")
        cols = [c for c in ("name", "industry", "asset_class", "market_cap") if c in meta.columns]
        table = table.join(meta[cols].rename(columns={"market_cap": "universe_market_cap"}), how="left")
    if fundamentals is not None and not fundamentals.empty:
        overlap = [c for c in fundamentals.columns if c in table.columns and c not in ("name", "industry")]
        table = table.join(fundamentals.drop(columns=overlap), how="left", rsuffix="_fund")
        if "name_fund" in table.columns:
            table["name"] = table["name"].fillna(table.pop("name_fund")) if "name" in table.columns else table.pop("name_fund")
        if "industry_fund" in table.columns:
            table["industry"] = table["industry"].fillna(table.pop("industry_fund")) if "industry" in table.columns else table.pop("industry_fund")
    if "universe_market_cap" in table.columns:
        base = table["market_cap"] if "market_cap" in table.columns else pd.Series(index=table.index, dtype="float64")
        table["market_cap"] = base.fillna(table["universe_market_cap"])
        table = table.drop(columns="universe_market_cap")
    return table
