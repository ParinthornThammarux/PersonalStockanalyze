"""Transaction ledger -> open positions, cost basis and realised P&L (average-cost or FIFO).

Besides BUY and SELL the ledger knows DIVIDEND (``quantity`` shares x ``price`` per share, ``fees`` = tax withheld) and the
cash movements DEPOSIT / WITHDRAW (``quantity`` = amount, symbol ``$CASH``). Every row has a currency (that of its symbol
unless stated). Given a base currency and an exchange-rate lookup, the book is also kept in that currency, at the rate of
each trade date, and every gain is split into the part the price produced and the part the exchange rate produced.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Iterable

import pandas as pd

from stockanalyzer.domain.assets import clean_currency, currency_of
from stockanalyzer.storage.state import StateDB

EPS = 1e-9
SIDES = ("BUY", "SELL", "DIVIDEND", "DEPOSIT", "WITHDRAW")
TRADE_SIDES = ("BUY", "SELL")
CASH_SIDES = ("DEPOSIT", "WITHDRAW")
CASH_SYMBOL = "$CASH"  # cannot clash with a ticker
POSITION_COLUMNS = ["quantity", "avg_cost", "cost_basis", "realized_pnl", "fees_paid", "first_buy", "currency", "dividends"]
LOT_COLUMNS = ["symbol", "sell_date", "buy_date", "quantity", "cost", "proceeds", "gain", "holding_days"]
LOT_COLUMNS_BASE = LOT_COLUMNS + ["gain_base", "fx_effect"]

FxAt = Callable[[str, pd.Timestamp], float]  # (currency, date) -> units of the base currency per 1 unit of it


@dataclass(frozen=True)
class Transaction:
    ts: pd.Timestamp
    symbol: str
    side: str  # BUY | SELL | DIVIDEND | DEPOSIT | WITHDRAW
    quantity: float
    price: float
    fees: float = 0.0
    note: str = ""
    portfolio: str = "default"
    id: int | None = None
    currency: str = ""  # the currency of price and fees; blank = the symbol's own

    def __post_init__(self):
        object.__setattr__(self, "ts", pd.Timestamp(self.ts).tz_localize(None) if pd.Timestamp(self.ts).tzinfo else pd.Timestamp(self.ts))
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "side", self.side.strip().upper())
        if self.side not in SIDES:
            raise ValueError(f"side must be one of {', '.join(SIDES)}, got {self.side!r}")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.price < 0 or self.fees < 0:
            raise ValueError("price and fees cannot be negative")
        if self.side in CASH_SIDES:
            if not self.currency.strip():
                raise ValueError(f"a {self.side.lower()} needs a currency")
            object.__setattr__(self, "symbol", CASH_SYMBOL)
        elif self.symbol == CASH_SYMBOL or not self.symbol:
            raise ValueError(f"a {self.side.lower()} needs a symbol")
        object.__setattr__(self, "currency", clean_currency(self.currency) if self.currency.strip() else currency_of(self.symbol))

    @classmethod
    def cash(cls, ts, amount: float, currency: str, *, deposit: bool = True, note: str = "", portfolio: str = "default",
             id: int | None = None) -> "Transaction":
        return cls(ts, CASH_SYMBOL, "DEPOSIT" if deposit else "WITHDRAW", amount, 1.0, 0.0, note, portfolio, id, currency)

    @property
    def gross(self) -> float:
        """Quantity x price: the amount traded, paid out as a dividend, or moved in or out of cash."""
        return self.quantity * self.price


@dataclass
class Book:
    positions: pd.DataFrame  # open positions, indexed by symbol
    realized: pd.DataFrame  # per symbol (including fully closed ones): realised P&L, fees, dividends
    lots: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=LOT_COLUMNS))
    base: str | None = None  # set when the book was also kept in a base currency


def build_book(
    transactions: Iterable[Transaction], method: str = "average", *, base: str | None = None, fx_at: FxAt | None = None
) -> Book:
    """Replay transactions in time order.

    Buy fees are added to cost; sell fees reduce proceeds. Selling more than is held is an
    error (no shorting in the ledger) rather than silently producing a negative position.
    Dividends are collected per symbol; cash movements are not part of the book (see ``portfolio.returns``).

    With ``base`` and ``fx_at`` the cost of every lot is also recorded in the base currency at its buy-date rate, and
    each sale is split into ``realized_price_base`` (the gain in the stock's own currency, at the sale-date rate) and
    ``realized_fx_base`` (what the exchange rate added between buying and selling).
    """
    if method not in ("average", "fifo"):
        raise ValueError("method must be 'average' or 'fifo'")
    if (base is None) != (fx_at is None):
        raise ValueError("base and fx_at go together")
    txs = sorted(transactions, key=lambda t: (t.ts, t.id if t.id is not None else 0))

    def rate(ccy: str, ts: pd.Timestamp) -> float:
        if base is None or ccy == base:
            return 1.0
        value = fx_at(ccy, ts)
        if value is None or value != value or value <= 0:
            raise ValueError(f"no {ccy}/{base} exchange rate for {ts:%Y-%m-%d}")
        return float(value)

    lots: dict[str, deque] = defaultdict(deque)  # fifo: [qty, unit_cost, buy_date, unit_cost_base]
    qty: dict[str, float] = defaultdict(float)
    cost: dict[str, float] = defaultdict(float)  # average: total cost of open shares
    cost_b: dict[str, float] = defaultdict(float)  # ... and in the base currency
    realized: dict[str, float] = defaultdict(float)
    real_b: dict[str, float] = defaultdict(float)
    real_price_b: dict[str, float] = defaultdict(float)
    real_fx_b: dict[str, float] = defaultdict(float)
    fees: dict[str, float] = defaultdict(float)
    fees_b: dict[str, float] = defaultdict(float)
    divs: dict[str, float] = defaultdict(float)
    divs_b: dict[str, float] = defaultdict(float)
    ccy_of: dict[str, str] = {}
    first_buy: dict[str, pd.Timestamp] = {}
    sold_rows: list[tuple] = []

    for t in txs:
        if t.side in CASH_SIDES:
            continue
        s, fx = t.symbol, rate(t.currency, t.ts)
        ccy_of[s] = t.currency
        if t.side == "DIVIDEND":
            divs[s] += t.gross - t.fees
            divs_b[s] += (t.gross - t.fees) * fx
            continue
        fees[s] += t.fees
        fees_b[s] += t.fees * fx
        if t.side == "BUY":
            first_buy.setdefault(s, t.ts)
            unit = (t.quantity * t.price + t.fees) / t.quantity
            lots[s].append([t.quantity, unit, t.ts, unit * fx])
            qty[s] += t.quantity
            cost[s] += t.quantity * unit
            cost_b[s] += t.quantity * unit * fx
            continue

        if t.quantity > qty[s] + EPS:
            raise ValueError(f"cannot sell {t.quantity:g} {s} on {t.ts:%Y-%m-%d}: only {qty[s]:g} held")
        proceeds_unit = (t.quantity * t.price - t.fees) / t.quantity

        def record(buy_ts, take, unit_cost, unit_cost_b, held_days):
            gain = (proceeds_unit - unit_cost) * take
            gain_b = (proceeds_unit * take * fx) - unit_cost_b * take
            fx_effect = gain_b - gain * fx  # what the exchange rate added; the rest is the stock's own gain
            sold_rows.append((s, t.ts, buy_ts, take, unit_cost * take, proceeds_unit * take, gain, held_days,
                              gain_b, fx_effect))
            realized[s] += gain
            real_b[s] += gain_b
            real_price_b[s] += gain * fx
            real_fx_b[s] += fx_effect

        if method == "average":
            record(first_buy[s], t.quantity, cost[s] / qty[s], cost_b[s] / qty[s], (t.ts - first_buy[s]).days)
            cost[s] -= cost[s] / qty[s] * t.quantity
            cost_b[s] -= cost_b[s] / qty[s] * t.quantity
        else:
            remaining = t.quantity
            while remaining > EPS:
                lot = lots[s][0]
                take = min(remaining, lot[0])
                record(lot[2], take, lot[1], lot[3], (t.ts - lot[2]).days)
                cost[s] -= lot[1] * take
                cost_b[s] -= lot[3] * take
                lot[0] -= take
                remaining -= take
                if lot[0] <= EPS:
                    lots[s].popleft()
        qty[s] -= t.quantity
        if qty[s] <= EPS:
            qty[s], cost[s], cost_b[s] = 0.0, 0.0, 0.0
            lots[s].clear()

    rows = {
        s: {"quantity": q, "avg_cost": cost[s] / q, "cost_basis": cost[s], "realized_pnl": realized[s],
            "fees_paid": fees[s], "first_buy": first_buy[s], "currency": ccy_of[s], "dividends": divs[s],
            **({"cost_basis_base": cost_b[s], "dividends_base": divs_b[s]} if base else {})}
        for s, q in qty.items() if q > EPS
    }
    columns = POSITION_COLUMNS + (["cost_basis_base", "dividends_base"] if base else [])
    positions = pd.DataFrame.from_dict(rows, orient="index", columns=columns)
    positions.index.name = "symbol"
    symbols = list(ccy_of)

    def per_symbol(values: dict[str, float]) -> pd.Series:
        return pd.Series({s: values.get(s, 0.0) for s in symbols}, dtype="float64")

    real = pd.DataFrame({"realized_pnl": per_symbol(realized), "fees_paid": per_symbol(fees),
                         "dividends": per_symbol(divs), "currency": pd.Series(ccy_of)})
    if base:
        real["realized_pnl_base"] = per_symbol(real_b)
        real["realized_price_base"] = per_symbol(real_price_b)
        real["realized_fx_base"] = per_symbol(real_fx_b)
        real["fees_paid_base"] = per_symbol(fees_b)
        real["dividends_base"] = per_symbol(divs_b)
    real.index.name = "symbol"
    lot_columns = LOT_COLUMNS_BASE if base else LOT_COLUMNS
    lot_frame = pd.DataFrame(sold_rows, columns=LOT_COLUMNS_BASE)[lot_columns]
    return Book(positions.sort_index(), real.sort_index(), lot_frame, base)


class TransactionRepo:
    def __init__(self, state: StateDB):
        self.state = state

    def add(self, tx: Transaction) -> int:
        return self.state.execute(
            "INSERT INTO transactions(portfolio, ts, symbol, side, quantity, price, fees, note, currency) VALUES (?,?,?,?,?,?,?,?,?)",
            (tx.portfolio, tx.ts.strftime("%Y-%m-%d %H:%M:%S"), tx.symbol, tx.side, tx.quantity, tx.price, tx.fees, tx.note,
             tx.currency),
        )

    def list(self, portfolio: str | None = None) -> list[Transaction]:
        sql, args = "SELECT * FROM transactions", []
        if portfolio:
            sql, args = sql + " WHERE portfolio=?", [portfolio]
        return [Transaction(pd.Timestamp(r["ts"]), r["symbol"], r["side"], r["quantity"], r["price"], r["fees"],
                            r["note"], r["portfolio"], r["id"], r["currency"])
                for r in self.state.query(sql + " ORDER BY ts, id", args)]

    def delete(self, tx_id: int) -> None:
        self.state.execute("DELETE FROM transactions WHERE id=?", (tx_id,))

    def portfolios(self) -> list[str]:
        return [r["portfolio"] for r in self.state.query("SELECT DISTINCT portfolio FROM transactions ORDER BY portfolio")]

    def to_frame(self, portfolio: str | None = None) -> pd.DataFrame:
        txs = self.list(portfolio)
        return pd.DataFrame(
            [(t.id, t.ts, t.symbol, t.side, t.quantity, t.price, t.fees, t.currency, t.note, t.portfolio) for t in txs],
            columns=["id", "date", "symbol", "side", "quantity", "price", "fees", "currency", "note", "portfolio"])

    def import_frame(self, df: pd.DataFrame, portfolio: str = "default") -> int:
        """Import a CSV/Excel-like frame; all rows are validated before any is written.

        Required: date, symbol, side, quantity, price. Optional: fees, currency. Cash rows (side DEPOSIT / WITHDRAW)
        may leave the symbol empty and give the amount as quantity (price defaults to 1) in the ``currency`` column.
        """
        names = {c.strip().lower(): c for c in df.columns}

        def pick(*options, required: bool = True):
            for o in options:
                if o in names:
                    return names[o]
            if required:
                raise ValueError(f"missing a '{options[0]}' column (accepted: {', '.join(options)})")
            return None

        c_date = pick("date", "ts", "time", "datetime")
        c_sym = pick("symbol", "ticker")
        c_side = pick("side", "type", "action")
        c_qty = pick("quantity", "qty", "shares", "amount")
        c_px = pick("price", "avg_price", "fill_price", required=False)
        c_fee = pick("fees", "fee", "commission", "tax", required=False)
        c_ccy = pick("currency", "ccy", required=False)
        if c_px is None and not df.empty and not df[c_side].astype(str).str.strip().str.upper().isin(CASH_SIDES).all():
            raise ValueError("missing a 'price' column (accepted: price, avg_price, fill_price)")

        def text(value) -> str:
            return "" if pd.isna(value) else str(value).strip()

        txs, errors = [], []
        for i, row in df.reset_index(drop=True).iterrows():
            try:
                side = text(row[c_side]).upper()
                price = row[c_px] if c_px and pd.notna(row[c_px]) else (1.0 if side in CASH_SIDES else None)
                if price is None:
                    raise ValueError("price is empty")
                txs.append(Transaction(
                    pd.Timestamp(row[c_date]), text(row[c_sym]), side, float(row[c_qty]),
                    float(price), float(row[c_fee]) if c_fee and pd.notna(row[c_fee]) else 0.0,
                    portfolio=portfolio, currency=text(row[c_ccy]) if c_ccy else ""))
            except Exception as exc:
                errors.append(f"row {i + 2}: {exc}")
        if errors:
            raise ValueError("; ".join(errors[:5]) + (f" (+{len(errors) - 5} more)" if len(errors) > 5 else ""))
        build_book([t for t in self.list(portfolio)] + txs)  # surface oversells before writing anything
        for t in txs:
            self.add(t)
        return len(txs)
