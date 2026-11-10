"""What the investor actually earned: money-weighted return (XIRR), cash balance and where the profit came from.

Time-weighted return (``valuation.portfolio_series``) says how good the holdings were; the money-weighted return says
what *you* earned given when you put money in and took it out. Both are needed: they differ whenever contributions are
large or badly timed. All amounts are in the base currency, converted at the rate of each transaction's date.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from stockanalyzer.portfolio.ledger import CASH_SIDES, FxAt, Transaction

DAYS_PER_YEAR = 365.0


def xirr(amounts: Sequence[float], dates: Sequence) -> float:
    """Annualised rate that makes the dated cash flows sum to zero (Excel's XIRR); NaN if there is none.

    Needs at least one outflow and one inflow. Flows are discounted by actual days / 365.
    """
    a = np.asarray(amounts, dtype="float64")
    if len(a) < 2 or not (a < 0).any() or not (a > 0).any():
        return float("nan")
    d = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    t = np.asarray((d - d.min()).days, dtype="float64") / DAYS_PER_YEAR
    if t.max() == 0:
        return float("nan")

    def npv(rate: float) -> float:
        return float(np.sum(a / (1.0 + rate) ** t))

    lo, hi = -0.999999, 1.0
    if npv(lo) * npv(hi) > 0:  # a big gain over a short time needs a large annual rate: widen the bracket
        for _ in range(12):
            hi *= 10.0
            if npv(lo) * npv(hi) <= 0:
                break
        else:
            return float("nan")
    return float(brentq(npv, lo, hi, xtol=1e-12, maxiter=500))


def has_cash_flows(transactions: Iterable[Transaction]) -> bool:
    return any(t.side in CASH_SIDES for t in transactions)


def _rate(fx_at: FxAt | None, base: str | None, t: Transaction) -> float:
    if fx_at is None or t.currency == base:
        return 1.0
    value = fx_at(t.currency, t.ts)
    if value is None or value != value or value <= 0:
        raise ValueError(f"no {t.currency}/{base} exchange rate for {t.ts:%Y-%m-%d}")
    return float(value)


def investor_flows(
    transactions: Iterable[Transaction], fx_at: FxAt | None = None, base: str | None = None, *, track_cash: bool | None = None
) -> pd.DataFrame:
    """The investor's own cash flows in the base currency: negative = paid in, positive = taken out.

    With cash tracking (the default when the ledger has any DEPOSIT / WITHDRAW) only those two count - buying and selling
    stay inside the account, so idle cash dilutes the return like it does in real life. Without them every purchase
    is money in, and every sale or dividend is money out.
    """
    txs = sorted(transactions, key=lambda t: (t.ts, t.id if t.id is not None else 0))
    cash_mode = has_cash_flows(txs) if track_cash is None else track_cash
    rows = []
    for t in txs:
        fx = _rate(fx_at, base, t)
        if cash_mode:
            if t.side == "DEPOSIT":
                rows.append((t.ts, -t.gross * fx, "deposit"))
            elif t.side == "WITHDRAW":
                rows.append((t.ts, t.gross * fx, "withdrawal"))
        elif t.side == "BUY":
            rows.append((t.ts, -(t.gross + t.fees) * fx, "buy"))
        elif t.side == "SELL":
            rows.append((t.ts, (t.gross - t.fees) * fx, "sell"))
        elif t.side == "DIVIDEND":
            rows.append((t.ts, (t.gross - t.fees) * fx, "dividend"))
    return pd.DataFrame(rows, columns=["date", "amount", "kind"])


def cash_balance(transactions: Iterable[Transaction], fx_at: FxAt | None = None, base: str | None = None) -> float:
    """Cash left in the account in the base currency: deposits and sale/dividend proceeds minus withdrawals and purchases."""
    total = 0.0
    for t in transactions:
        fx = _rate(fx_at, base, t)
        total += fx * {"DEPOSIT": t.gross - t.fees, "WITHDRAW": -t.gross - t.fees, "BUY": -t.gross - t.fees,
                       "SELL": t.gross - t.fees, "DIVIDEND": t.gross - t.fees}[t.side]
    return total


@dataclass(frozen=True)
class MoneySummary:
    put_in: float  # all money paid in
    taken_out: float  # all money paid back out (before today's value)
    value: float  # what it is worth now
    gain: float
    multiple: float  # (taken out + value) / put in
    xirr: float  # NaN when it cannot be computed
    years: float
    first: pd.Timestamp | None


def money_summary(flows: pd.DataFrame, value: float, asof) -> MoneySummary:
    """Money-in / money-out totals and XIRR, treating today's ``value`` as a final withdrawal."""
    if flows.empty:
        return MoneySummary(0.0, 0.0, value, 0.0, float("nan"), float("nan"), 0.0, None)
    asof = pd.Timestamp(asof)
    put_in = float(-flows.loc[flows["amount"] < 0, "amount"].sum())
    taken = float(flows.loc[flows["amount"] > 0, "amount"].sum())
    first = pd.Timestamp(flows["date"].min())
    rate = xirr([*flows["amount"], value], [*flows["date"], asof])
    return MoneySummary(put_in, taken, float(value), taken + value - put_in,
                        (taken + value) / put_in if put_in else float("nan"), rate,
                        max((asof - first).days, 0) / 365.25, first)


def attribution(marked: pd.DataFrame, realized: pd.DataFrame) -> pd.DataFrame:
    """Total profit in the base currency split into where it came from.

    Needs the base-currency columns from ``mark_to_market(..., fx=...)`` and ``build_book(..., base=...)``. Fees are
    already inside the gains (buys capitalise them, sales net them off), so they are a memo line, not another term.
    """
    def total(frame: pd.DataFrame, column: str) -> float:
        return float(frame[column].sum()) if column in frame and not frame.empty else 0.0

    rows = [
        ("Unrealised · stock price", total(marked, "unrealized_price"), "open positions, in their own currency, at today's rate"),
        ("Unrealised · currency", total(marked, "unrealized_fx"), "what the exchange rate added to open positions since you bought"),
        ("Realised · stock price", total(realized, "realized_price_base"), "closed sales, in their own currency"),
        ("Realised · currency", total(realized, "realized_fx_base"), "what the exchange rate added between buying and selling"),
        ("Dividends", total(realized, "dividends_base"), "received, after tax withheld"),
    ]
    out = pd.DataFrame(rows, columns=["component", "amount", "meaning"]).set_index("component")
    return out

