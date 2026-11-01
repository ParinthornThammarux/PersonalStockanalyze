"""Turn target weights into concrete buy/sell orders."""

from __future__ import annotations

import numpy as np
import pandas as pd

ORDER_COLUMNS = ["symbol", "side", "shares", "price", "value", "fees", "current_weight", "target_weight", "new_weight"]


def rebalance_orders(
    holdings: pd.Series,
    prices: pd.Series,
    target_weights: pd.Series,
    cash: float = 0.0,
    *,
    whole_shares: bool = True,
    min_trade_value: float = 0.0,
    fee_bps: float = 0.0,
) -> pd.DataFrame:
    """Orders that move ``holdings`` toward ``target_weights`` of total equity (holdings + cash).

    Sells are always executed in full; buys are scaled back pro-rata if the cash raised by
    selling (plus existing cash) cannot pay for them including fees. Symbols not in
    ``target_weights`` are treated as target 0 (sold). ``new_weight`` is what you actually end
    up with after rounding to whole shares.
    """
    symbols = sorted(set(holdings.index) | set(target_weights.index))
    qty = holdings.reindex(symbols).fillna(0.0).astype(float)
    px = prices.reindex(symbols).astype(float)
    if px.isna().any():
        raise ValueError(f"missing prices for: {', '.join(px.index[px.isna()])}")
    target = target_weights.reindex(symbols).fillna(0.0).astype(float)
    if target.sum() > 1 + 1e-9 or (target < 0).any():
        raise ValueError("target weights must be non-negative and sum to at most 1")

    equity = float((qty * px).sum() + cash)
    current_value = qty * px
    delta_shares = (target * equity - current_value) / px
    delta_shares = np.trunc(delta_shares) if whole_shares else delta_shares
    delta_shares[(delta_shares.abs() * px) < max(min_trade_value, 1e-12)] = 0.0
    delta_shares = delta_shares.clip(lower=-qty)  # cannot sell more than held

    fee = fee_bps / 1e4
    sells = (-delta_shares.clip(upper=0)) * px
    buys_wanted = delta_shares.clip(lower=0) * px
    available = cash + float(sells.sum() * (1 - fee))
    need = float(buys_wanted.sum() * (1 + fee))
    if need > available + 1e-9 and need > 0:
        scale = max(available, 0.0) / need
        delta_shares = delta_shares.where(delta_shares < 0, delta_shares * scale)
        delta_shares = np.trunc(delta_shares).where(delta_shares > 0, delta_shares) if whole_shares else delta_shares

    value = delta_shares.abs() * px
    fees = value * fee
    new_qty = qty + delta_shares
    new_equity = float((new_qty * px).sum()) or 1.0
    orders = pd.DataFrame({
        "symbol": symbols,
        "side": np.where(delta_shares > 0, "BUY", np.where(delta_shares < 0, "SELL", "HOLD")),
        "shares": delta_shares.abs().to_numpy(), "price": px.to_numpy(), "value": value.to_numpy(),
        "fees": fees.to_numpy(), "current_weight": (current_value / equity).to_numpy() if equity else 0.0,
        "target_weight": target.to_numpy(), "new_weight": (new_qty * px / new_equity).to_numpy(),
    }, columns=ORDER_COLUMNS)
    orders = orders[orders["side"] != "HOLD"]
    return orders.sort_values(["side", "value"], ascending=[False, False], ignore_index=True)  # SELLs first
