"""Join features with universe metadata and cached fundamentals into one screening table."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from stockanalyzer.domain.assets import currency_of

USD_COLUMNS = ("market_cap", "dollar_volume_20")  # money columns that are compared across markets
# Average traded value per day (US dollars) a stock needs to count as tradable. Thailand's own floor is lower because the
# market is ~1/130 of the US: at the US figure only 60 of 883 Thai stocks pass, at this one about 170 (10 million baht a day).
MIN_TRADED_US = 5e6
MIN_TRADED_TH = 3e5


def _currencies(table: pd.DataFrame) -> pd.Series:
    """Currency each row is quoted in: the fundamentals snapshot's if there is one, else what the symbol implies."""
    implied = pd.Series([currency_of(s) for s in table.index], index=table.index, dtype="object")
    if "currency" not in table.columns:
        return implied
    return table["currency"].where(table["currency"].astype(str).str.len() > 0).fillna(implied)


def build_screen_table(
    features: pd.DataFrame,
    universe: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    usd_rate: Callable[[str], float | None] | None = None,
) -> pd.DataFrame:
    """One row per symbol. ``market_cap`` prefers the fundamentals snapshot over the static list.

    With ``usd_rate`` (currency -> US dollars per 1 unit, ``None`` when unknown) ``market_cap`` and ``dollar_volume_20``
    are expressed in US dollars, so a Thai stock and a US stock face the same liquidity and size filter. A currency
    with no rate leaves those two figures empty instead of passing baht off as dollars. ``close`` stays in the
    stock's own currency, named in the ``currency`` column.
    """
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

    currency = _currencies(table)
    if usd_rate is not None:
        rates = {c: (1.0 if c == "USD" else usd_rate(c)) for c in currency.dropna().unique()}
        rate = currency.map(lambda c: np.nan if rates.get(c) is None else rates[c]).astype("float64")
        for column in USD_COLUMNS:
            if column in table.columns:
                table[column] = table[column] * rate  # the packaged US list below is already in dollars

    if "universe_market_cap" in table.columns:
        base = table["market_cap"] if "market_cap" in table.columns else pd.Series(index=table.index, dtype="float64")
        table["market_cap"] = base.fillna(table["universe_market_cap"])
        table = table.drop(columns="universe_market_cap")
    if usd_rate is not None or "currency" in table.columns:  # a bare feature table keeps its columns
        table["currency"] = currency
    return table


def liquidity_mask(table: pd.DataFrame, min_us: float = MIN_TRADED_US, min_th: float = MIN_TRADED_TH) -> pd.Series:
    """Rows that trade at least ``min_th`` dollars a day if Thai (``.BK``), else ``min_us``; a floor of 0 keeps everything.

    Needs ``dollar_volume_20`` in US dollars (see :func:`build_screen_table`); without the column nothing is filtered.
    A missing figure fails any floor above zero (indices and crypto have no share volume).
    """
    if "dollar_volume_20" not in table.columns:
        return pd.Series(True, index=table.index)
    floor = pd.Series(np.where(table.index.astype(str).str.endswith(".BK"), min_th, min_us), index=table.index)
    return (table["dollar_volume_20"] >= floor).fillna(False) | (floor <= 0)
