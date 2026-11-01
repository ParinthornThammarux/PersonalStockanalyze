"""Mark a book to market, build the daily value/return history and slice allocations."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from stockanalyzer.domain.assets import infer_asset_class
from stockanalyzer.portfolio.ledger import Book, Transaction


def mark_to_market(
    positions: pd.DataFrame, prices: Mapping[str, float] | pd.Series, meta: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Add price, market value, unrealised P&L, weights and classification to open positions."""
    out = positions.copy()
    px = pd.Series(prices, dtype="float64").reindex(out.index)
    out["price"] = px
    out["market_value"] = out["quantity"] * px
    out["unrealized_pnl"] = out["market_value"] - out["cost_basis"]
    out["unrealized_pct"] = out["unrealized_pnl"] / out["cost_basis"].replace(0, np.nan)
    total = out["market_value"].sum()
    out["weight"] = out["market_value"] / total if total else np.nan
    out["asset_class"] = [infer_asset_class(s).value for s in out.index]
    if meta is not None and not meta.empty:
        extra = meta.reindex(out.index)
        for col in ("name", "industry", "sector"):
            if col in extra.columns:
                out[col] = extra[col]
    return out


def allocation(marked: pd.DataFrame, by: str = "symbol") -> pd.Series:
    """Weights (summing to 1) grouped by a column such as ``asset_class`` or ``industry``."""
    key = marked.index.to_series() if by == "symbol" else marked[by].fillna("Unknown")
    grouped = marked.groupby(key)["market_value"].sum()
    total = grouped.sum()
    return (grouped / total).sort_values(ascending=False) if total else grouped


def summary(marked: pd.DataFrame, book: Book) -> dict[str, float]:
    value = float(marked["market_value"].sum())
    cost = float(marked["cost_basis"].sum())
    unrealized = value - cost
    realized = float(book.realized["realized_pnl"].sum()) if not book.realized.empty else 0.0
    return {
        "market_value": value, "cost_basis": cost, "unrealized_pnl": unrealized,
        "unrealized_pct": unrealized / cost if cost else float("nan"),
        "realized_pnl": realized, "total_pnl": unrealized + realized,
        "fees_paid": float(book.realized["fees_paid"].sum()) if not book.realized.empty else 0.0,
        "positions": int(len(marked)),
    }


def holdings_history(transactions: Iterable[Transaction], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Shares held at each date's close (rows = ``index``). Off-calendar trades roll to the next bar."""
    deltas: dict[str, np.ndarray] = {}
    for t in transactions:
        pos = index.searchsorted(t.ts.normalize())
        if pos >= len(index):
            continue
        arr = deltas.setdefault(t.symbol, np.zeros(len(index)))
        arr[pos] += t.quantity if t.side == "BUY" else -t.quantity
    if not deltas:
        return pd.DataFrame(index=index)
    return pd.DataFrame({s: np.cumsum(a) for s, a in deltas.items()}, index=index)


def portfolio_series(transactions: Iterable[Transaction], close: pd.DataFrame) -> pd.DataFrame:
    """Daily ``value`` and time-weighted ``ret`` of the invested portfolio.

    Trades are assumed at the day's close, so a day's return is the previous close's holdings
    times that day's price change - contributions and withdrawals do not distort it.
    """
    txs = list(transactions)
    held = holdings_history(txs, close.index)
    cols = [c for c in held.columns if c in close.columns]
    if not cols:
        return pd.DataFrame({"value": 0.0, "ret": np.nan}, index=close.index)
    q, px = held[cols], close[cols].ffill()
    value = (q * px).sum(axis=1)
    pnl = (q.shift(1) * px.diff()).sum(axis=1)
    ret = (pnl / value.shift(1).replace(0, np.nan)).where(value.shift(1) > 0)
    first = (q != 0).any(axis=1).idxmax() if (q != 0).any().any() else None
    out = pd.DataFrame({"value": value, "ret": ret})
    return out.loc[first:] if first is not None else out
