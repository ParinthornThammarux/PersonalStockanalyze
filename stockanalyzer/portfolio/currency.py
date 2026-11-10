"""Express prices, holdings and trades in one base currency.

An exchange-rate table maps each currency to a daily series of "base currency per 1 unit of it" (an empty series for
the base itself, ``None`` when Yahoo has no pair), so a portfolio of Thai and US shares can be added up, compared with
its cost and ranked by risk in a single currency.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd

from stockanalyzer.data import fx as fx_data
from stockanalyzer.data.service import MarketData
from stockanalyzer.domain.assets import currency_of
from stockanalyzer.portfolio.ledger import FxAt

FxTable = Mapping[str, "pd.Series | None"]


def fx_table(md: MarketData, currencies: Iterable[str], base: str) -> dict[str, pd.Series | None]:
    """Daily rates into ``base`` for every currency asked for (a pair Yahoo does not have maps to ``None``)."""
    return {c: fx_data.fx_history(md, c, base) for c in dict.fromkeys(currencies)}


def lookup(table: FxTable) -> FxAt:
    """``(currency, date) -> rate`` for the ledger. Unknown currencies give NaN, which the ledger refuses."""
    def at(currency: str, when: pd.Timestamp) -> float:
        return fx_data.rate_on(table.get(currency), when) if currency in table else float("nan")
    return at


def latest(table: FxTable) -> dict[str, float]:
    """The newest rate of each currency (NaN where there is none)."""
    out = {}
    for currency, series in table.items():
        out[currency] = float("nan") if series is None else (1.0 if series.empty else float(series.iloc[-1]))
    return out


def missing(table: FxTable) -> list[str]:
    return [c for c, series in table.items() if series is None]


def currencies_of(symbols: Iterable[str]) -> list[str]:
    return sorted({currency_of(s) for s in symbols})


def to_base(close: pd.DataFrame, table: FxTable) -> pd.DataFrame:
    """Prices (columns = symbols) in the base currency, converted at each day's rate.

    A column whose currency has no rate is dropped rather than left in the wrong currency.
    """
    out = {}
    for symbol in close.columns:
        series = table.get(currency_of(symbol))
        if series is None:
            continue
        if series.empty:  # already in the base currency
            out[symbol] = close[symbol]
            continue
        rate = series.reindex(close.index.union(series.index)).ffill().reindex(close.index)
        out[symbol] = close[symbol] * rate.fillna(series.iloc[0])
    return pd.DataFrame(out, index=close.index)
