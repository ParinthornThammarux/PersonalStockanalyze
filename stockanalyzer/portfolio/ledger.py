"""Transaction ledger -> open positions, cost basis and realised P&L (average-cost or FIFO)."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from stockanalyzer.storage.state import StateDB

EPS = 1e-9
POSITION_COLUMNS = ["quantity", "avg_cost", "cost_basis", "realized_pnl", "fees_paid", "first_buy"]
LOT_COLUMNS = ["symbol", "sell_date", "buy_date", "quantity", "cost", "proceeds", "gain", "holding_days"]


@dataclass(frozen=True)
class Transaction:
    ts: pd.Timestamp
    symbol: str
    side: str  # BUY | SELL
    quantity: float
    price: float
    fees: float = 0.0
    note: str = ""
    portfolio: str = "default"
    id: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "ts", pd.Timestamp(self.ts).tz_localize(None) if pd.Timestamp(self.ts).tzinfo else pd.Timestamp(self.ts))
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "side", self.side.strip().upper())
        if self.side not in ("BUY", "SELL"):
            raise ValueError(f"side must be BUY or SELL, got {self.side!r}")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.price < 0 or self.fees < 0:
            raise ValueError("price and fees cannot be negative")


@dataclass
class Book:
    positions: pd.DataFrame  # open positions, indexed by symbol
    realized: pd.DataFrame  # realised P&L per symbol (including fully closed ones)
    lots: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=LOT_COLUMNS))


def build_book(transactions: Iterable[Transaction], method: str = "average") -> Book:
    """Replay transactions in time order.

    Buy fees are added to cost; sell fees reduce proceeds. Selling more than is held is an
    error (no shorting in the ledger) rather than silently producing a negative position.
    """
    if method not in ("average", "fifo"):
        raise ValueError("method must be 'average' or 'fifo'")
    txs = sorted(transactions, key=lambda t: (t.ts, t.id if t.id is not None else 0))

    lots: dict[str, deque] = defaultdict(deque)  # fifo: [qty, unit_cost, buy_date]
    qty: dict[str, float] = defaultdict(float)
    cost: dict[str, float] = defaultdict(float)  # average: total cost of open shares
    realized: dict[str, float] = defaultdict(float)
    fees: dict[str, float] = defaultdict(float)
    first_buy: dict[str, pd.Timestamp] = {}
    sold_rows: list[tuple] = []

    for t in txs:
        s = t.symbol
        fees[s] += t.fees
        if t.side == "BUY":
            first_buy.setdefault(s, t.ts)
            unit = (t.quantity * t.price + t.fees) / t.quantity
            lots[s].append([t.quantity, unit, t.ts])
            qty[s] += t.quantity
            cost[s] += t.quantity * unit
            continue

        if t.quantity > qty[s] + EPS:
            raise ValueError(f"cannot sell {t.quantity:g} {s} on {t.ts:%Y-%m-%d}: only {qty[s]:g} held")
        proceeds_unit = (t.quantity * t.price - t.fees) / t.quantity
        if method == "average":
            avg = cost[s] / qty[s]
            gain = (proceeds_unit - avg) * t.quantity
            sold_rows.append((s, t.ts, first_buy[s], t.quantity, avg * t.quantity, proceeds_unit * t.quantity,
                              gain, (t.ts - first_buy[s]).days))
            realized[s] += gain
            cost[s] -= avg * t.quantity
        else:
            remaining = t.quantity
            while remaining > EPS:
                lot = lots[s][0]
                take = min(remaining, lot[0])
                gain = (proceeds_unit - lot[1]) * take
                sold_rows.append((s, t.ts, lot[2], take, lot[1] * take, proceeds_unit * take, gain,
                                  (t.ts - lot[2]).days))
                realized[s] += gain
                cost[s] -= lot[1] * take
                lot[0] -= take
                remaining -= take
                if lot[0] <= EPS:
                    lots[s].popleft()
        qty[s] -= t.quantity
        if qty[s] <= EPS:
            qty[s], cost[s] = 0.0, 0.0
            lots[s].clear()

    rows = {
        s: {"quantity": q, "avg_cost": cost[s] / q, "cost_basis": cost[s], "realized_pnl": realized[s],
            "fees_paid": fees[s], "first_buy": first_buy[s]}
        for s, q in qty.items() if q > EPS
    }
    positions = pd.DataFrame.from_dict(rows, orient="index", columns=POSITION_COLUMNS)
    positions.index.name = "symbol"
    real = pd.DataFrame({"realized_pnl": pd.Series(dict(realized)), "fees_paid": pd.Series(dict(fees))}).rename_axis("symbol")
    return Book(positions.sort_index(), real.sort_index(), pd.DataFrame(sold_rows, columns=LOT_COLUMNS))


class TransactionRepo:
    def __init__(self, state: StateDB):
        self.state = state

    def add(self, tx: Transaction) -> int:
        return self.state.execute(
            "INSERT INTO transactions(portfolio, ts, symbol, side, quantity, price, fees, note) VALUES (?,?,?,?,?,?,?,?)",
            (tx.portfolio, tx.ts.strftime("%Y-%m-%d %H:%M:%S"), tx.symbol, tx.side, tx.quantity, tx.price, tx.fees, tx.note),
        )

    def list(self, portfolio: str | None = None) -> list[Transaction]:
        sql, args = "SELECT * FROM transactions", []
        if portfolio:
            sql, args = sql + " WHERE portfolio=?", [portfolio]
        return [Transaction(pd.Timestamp(r["ts"]), r["symbol"], r["side"], r["quantity"], r["price"], r["fees"],
                            r["note"], r["portfolio"], r["id"])
                for r in self.state.query(sql + " ORDER BY ts, id", args)]

    def delete(self, tx_id: int) -> None:
        self.state.execute("DELETE FROM transactions WHERE id=?", (tx_id,))

    def portfolios(self) -> list[str]:
        return [r["portfolio"] for r in self.state.query("SELECT DISTINCT portfolio FROM transactions ORDER BY portfolio")]

    def to_frame(self, portfolio: str | None = None) -> pd.DataFrame:
        txs = self.list(portfolio)
        return pd.DataFrame(
            [(t.id, t.ts, t.symbol, t.side, t.quantity, t.price, t.fees, t.note, t.portfolio) for t in txs],
            columns=["id", "date", "symbol", "side", "quantity", "price", "fees", "note", "portfolio"])

    def import_frame(self, df: pd.DataFrame, portfolio: str = "default") -> int:
        """Import a CSV/Excel-like frame; all rows are validated before any is written."""
        names = {c.strip().lower(): c for c in df.columns}

        def pick(*options):
            for o in options:
                if o in names:
                    return names[o]
            raise ValueError(f"missing a '{options[0]}' column (accepted: {', '.join(options)})")

        c_date = pick("date", "ts", "time", "datetime")
        c_sym = pick("symbol", "ticker")
        c_side = pick("side", "type", "action")
        c_qty = pick("quantity", "qty", "shares")
        c_px = pick("price", "avg_price", "fill_price")
        c_fee = next((names[o] for o in ("fees", "fee", "commission") if o in names), None)

        txs, errors = [], []
        for i, row in df.reset_index(drop=True).iterrows():
            try:
                txs.append(Transaction(
                    pd.Timestamp(row[c_date]), str(row[c_sym]), str(row[c_side]), float(row[c_qty]),
                    float(row[c_px]), float(row[c_fee]) if c_fee and pd.notna(row[c_fee]) else 0.0,
                    portfolio=portfolio))
            except Exception as exc:
                errors.append(f"row {i + 2}: {exc}")
        if errors:
            raise ValueError("; ".join(errors[:5]) + (f" (+{len(errors) - 5} more)" if len(errors) > 5 else ""))
        build_book([t for t in self.list(portfolio)] + txs)  # surface oversells before writing anything
        for t in txs:
            self.add(t)
        return len(txs)
