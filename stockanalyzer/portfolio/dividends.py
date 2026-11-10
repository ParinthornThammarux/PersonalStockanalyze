"""Dividends you should have received, worked out from the ex-dates in the price store and what you held.

Brokers pay weeks after the ex-date and withhold tax, so the amounts here are gross estimates: they let you spot
dividends missing from the ledger and record them in one click, then correct the tax. A share bought on the ex-date
itself is not entitled to that dividend, so only holdings at the close before the ex-date count.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from stockanalyzer.portfolio.ledger import TRADE_SIDES, Transaction

COLUMNS = ["symbol", "ex_date", "shares", "per_share", "gross", "currency", "recorded"]
PAY_TOLERANCE_DAYS = 3  # a payment booked a few days before the ex-date still belongs to it


def estimated_dividends(
    transactions: Iterable[Transaction], ex_dividends: Mapping[str, pd.Series], *, window_days: int = 120
) -> pd.DataFrame:
    """Every ex-dividend since the first purchase on a symbol you held, with whether a DIVIDEND row already covers it.

    ``ex_dividends`` maps a symbol to its dividends per share indexed by ex-date. A recorded dividend covers the latest
    ex-date up to ``window_days`` before its date that no other record covers.
    """
    txs = sorted(transactions, key=lambda t: (t.ts, t.id if t.id is not None else 0))
    by_symbol: dict[str, list[Transaction]] = {}
    for t in txs:
        by_symbol.setdefault(t.symbol, []).append(t)
    rows = []
    for symbol, history in by_symbol.items():
        divs = ex_dividends.get(symbol)
        trades = [t for t in history if t.side in TRADE_SIDES]
        if divs is None or divs.empty or not trades:
            continue
        days = np.array([t.ts.normalize().value for t in trades])
        shares = np.cumsum([t.quantity if t.side == "BUY" else -t.quantity for t in trades])
        paid = [t for t in history if t.side == "DIVIDEND"]
        covered: set[pd.Timestamp] = set()
        for t in paid:
            near = [e for e in divs.index if e - pd.Timedelta(days=PAY_TOLERANCE_DAYS) <= t.ts <= e + pd.Timedelta(days=window_days)
                    and e not in covered]
            if near:
                covered.add(max(near))
        for ex, per_share in divs.items():
            held_to = np.searchsorted(days, pd.Timestamp(ex).normalize().value, side="left")  # trades strictly before the ex-date
            held = float(shares[held_to - 1]) if held_to else 0.0
            if held > 1e-9 and per_share > 0:
                rows.append((symbol, pd.Timestamp(ex), held, float(per_share), held * float(per_share), trades[0].currency,
                             ex in covered))
    out = pd.DataFrame(rows, columns=COLUMNS)
    return out.sort_values(["ex_date", "symbol"], ignore_index=True)


def as_transactions(estimates: pd.DataFrame, portfolio: str = "default", tax_rate: float = 0.0) -> list[Transaction]:
    """DIVIDEND rows dated on the ex-date, with ``tax_rate`` of the gross withheld (edit them if the broker differs)."""
    if not 0 <= tax_rate < 1:
        raise ValueError("tax rate must be between 0 and 1")
    return [Transaction(r.ex_date, r.symbol, "DIVIDEND", r.shares, r.per_share, r.gross * tax_rate,
                        "estimated from the ex-dividend date", portfolio, None, r.currency)
            for r in estimates.itertuples()]
