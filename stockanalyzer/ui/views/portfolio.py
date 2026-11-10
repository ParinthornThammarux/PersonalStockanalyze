"""Portfolio & Risk: transactions, P&L in one currency, what you earned, risk, optimisation and rebalancing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from stockanalyzer.analysis.risk import metrics as risk
from stockanalyzer.data import events as ev
from stockanalyzer.domain.assets import benchmark_for, currency_of, lot_size
from stockanalyzer.portfolio import currency as ccy
from stockanalyzer.portfolio import dividends as dv
from stockanalyzer.portfolio import optimize as opt
from stockanalyzer.portfolio import returns as ret
from stockanalyzer.portfolio import valuation as val
from stockanalyzer.portfolio.ledger import TRADE_SIDES, Transaction, build_book
from stockanalyzer.portfolio.rebalance import rebalance_orders
from stockanalyzer.ui import charts
from stockanalyzer.ui import data as ui_data
from stockanalyzer.ui.context import services
from stockanalyzer.ui.format import money, pct
from stockanalyzer.ui.widgets import show_chart, show_table

TEMPLATE = (
    "date,symbol,side,quantity,price,fees,currency\n"
    "2024-01-10,,DEPOSIT,200000,,,THB\n"
    "2024-01-15,AAPL,BUY,10,185.20,1.0,USD\n"
    "2024-03-04,PTT.BK,BUY,1000,32.50,50,THB\n"
    "2024-04-20,PTT.BK,DIVIDEND,1000,1.50,15,THB\n"
    "2024-06-03,AAPL,SELL,4,194.10,1.0,USD\n"
)
KINDS = {"Buy": "BUY", "Sell": "SELL", "Dividend": "DIVIDEND", "Deposit": "DEPOSIT", "Withdraw": "WITHDRAW"}
BASE_CHOICES = ["THB", "USD", "EUR", "SGD", "JPY", "GBP", "HKD"]


def _last_prices(symbols: list[str]) -> dict[str, float]:
    out = {}
    for s in symbols:
        df = ui_data.prices(s, "1mo", "1d", False)
        if not df.empty:
            out[s] = float(df["close"].iloc[-1])
    return out


# -- transactions ---------------------------------------------------------------------------------
def _new_transaction(kind: str, pf: str, base: str) -> tuple[Transaction | None, bool]:
    """The entry form for one kind of row; returns (the transaction, submitted)."""
    k = kind.lower()
    with st.form(f"tx_form_{k}", clear_on_submit=True):
        if kind in ("Buy", "Sell"):
            c = st.columns([1.2, 1.2, 1, 1, 1, 1])
            day = c[0].date_input("Date", key=f"tx_{k}_day")
            symbol = c[1].text_input("Symbol", key=f"tx_{k}_sym").strip().upper()
            qty = c[2].number_input("Quantity", min_value=0.0, value=0.0, format="%.4f", key=f"tx_{k}_qty")
            price = c[3].number_input("Price", min_value=0.0, value=0.0, format="%.4f", key=f"tx_{k}_px")
            fees = c[4].number_input("Fees", min_value=0.0, value=0.0, format="%.2f", key=f"tx_{k}_fee")
            currency = c[5].text_input("Currency", key=f"tx_{k}_ccy", placeholder="from symbol",
                                       help="Leave empty to use the symbol's own: THB for .BK, USD otherwise.")
        elif kind == "Dividend":
            c = st.columns([1.2, 1.2, 1, 1, 1, 1])
            day = c[0].date_input("Date received", key="tx_div_day")
            symbol = c[1].text_input("Symbol", key="tx_div_sym").strip().upper()
            qty = c[2].number_input("Shares", min_value=0.0, value=0.0, format="%.4f", key="tx_div_qty")
            price = c[3].number_input("Per share", min_value=0.0, value=0.0, format="%.4f", key="tx_div_px")
            fees = c[4].number_input("Tax withheld (total)", min_value=0.0, value=0.0, format="%.2f", key="tx_div_fee")
            currency = c[5].text_input("Currency", key="tx_div_ccy", placeholder="from symbol")
        else:
            c = st.columns([1.2, 1, 1])
            day = c[0].date_input("Date", key=f"tx_{k}_day")
            qty = c[1].number_input("Amount", min_value=0.0, value=0.0, format="%.2f", key=f"tx_{k}_amount")
            currency = c[2].text_input("Currency", value=base, key=f"tx_{k}_ccy")
            symbol, price, fees = "", 1.0, 0.0
        note = st.text_input("Note", key=f"tx_{k}_note")
        submitted = st.form_submit_button(f"Add {k}", type="primary")
    if not submitted:
        return None, False
    side = KINDS[kind]
    return Transaction(pd.Timestamp(day), symbol, side, qty, price, fees, note, pf, None, currency), True


def _transactions_tab(pf: str, txs: list[Transaction], base: str) -> None:
    sv = services()
    repo = sv.transactions
    kind = st.radio("Record a", list(KINDS), horizontal=True, key="tx_kind",
                    help="Dividends and deposits / withdrawals make the money-weighted return (XIRR) and the cash balance "
                         "exact. Without any deposit, every purchase counts as money you put in.")
    try:
        tx, submitted = _new_transaction(kind, pf, base)
        if submitted:
            build_book([*txs, tx])  # rejects overselling before anything is stored
            repo.add(tx)
            st.rerun()
    except ValueError as exc:
        st.error(str(exc))

    up = st.file_uploader("Import CSV / Excel", type=["csv", "xlsx"], key="tx_upload")
    st.download_button("Download template", TEMPLATE, file_name="transactions_template.csv", mime="text/csv")
    if up is not None:
        raw = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
        show_table(raw.head(10))
        if st.button(f"Import {len(raw)} rows into '{pf}'"):
            try:
                st.success(f"Imported {repo.import_frame(raw, pf)} transactions")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    frame = repo.to_frame(pf)
    if frame.empty:
        st.caption("No transactions yet.")
        return
    shown = frame.iloc[::-1].reset_index(drop=True)
    event = st.dataframe(shown, hide_index=True, width="stretch", on_select="rerun", selection_mode="multi-row", key="tx_table")
    chosen = [int(shown.loc[i, "id"]) for i in event.selection.rows] if event and event.selection.rows else []
    a, b = st.columns([1, 4])
    if a.button(f"Delete {len(chosen)} selected" if chosen else "Delete selected", disabled=not chosen, key="tx_delete"):
        for tx_id in chosen:
            repo.delete(tx_id)
        st.rerun()
    b.download_button("Download transactions (CSV)", frame.to_csv(index=False).encode(), file_name=f"{pf}_transactions.csv",
                      mime="text/csv")
    if not chosen:
        b.caption("Click the box at the start of a row to select it for deletion.")


# -- overview -------------------------------------------------------------------------------------
def _overview_tab(book, marked, cash: float | None, base: str) -> None:
    s = val.summary(marked, book)
    k = st.columns(6)
    k[0].metric(f"Market value ({base})", money(s["market_value"]))
    k[1].metric(f"Cash ({base})", money(cash) if cash is not None else "-",
                help="Tracked when the ledger has deposits or withdrawals; without them it is not known.")
    k[2].metric("Unrealised P&L", money(s["unrealized_pnl"]), pct(s["unrealized_pct"], signed=True))
    k[3].metric("Realised P&L", money(s["realized_pnl"]))
    k[4].metric("Dividends", money(s["dividends"]))
    k[5].metric("Total P&L", money(s["total_pnl"]), help=f"Unrealised + realised + dividends, in {base}. "
                                                         f"Fees paid ({money(s['fees_paid'])}) are already inside it.")
    if cash is not None and cash < -0.005 * max(s["market_value"], 1.0):
        st.warning(f"Purchases exceed deposits by {money(-cash)} {base}: a deposit is missing from the ledger (or you borrowed).")

    view = marked.copy()
    view["realized"] = book.realized["realized_pnl_base"].reindex(view.index)
    view["dividends"] = book.realized["dividends_base"].reindex(view.index)
    cols = ["currency", "quantity", "avg_cost", "price", "market_value", "unrealized_pnl", "unrealized_pct", "unrealized_fx",
            "weight", "realized", "dividends", "asset_class"] + [c for c in ("name", "industry") if c in view.columns]
    show_table(view[cols].rename(columns={"market_value": f"value ({base})", "unrealized_pnl": f"unrealised ({base})",
                                          "unrealized_fx": "of which currency", "realized": f"realised ({base})",
                                          "dividends": f"dividends ({base})"})
               .style.format({"quantity": "{:,.2f}", "avg_cost": "{:,.2f}", "price": "{:,.2f}", f"value ({base})": "{:,.0f}",
                              f"unrealised ({base})": "{:,.0f}", "unrealized_pct": "{:+.1%}", "of which currency": "{:,.0f}",
                              "weight": "{:.1%}", f"realised ({base})": "{:,.0f}", f"dividends ({base})": "{:,.0f}"}, na_rep="-"))
    st.caption(f"Average cost and price are in each stock's own currency, like on a broker statement; values and P&L are in {base}, "
               "with each purchase converted at the exchange rate of its day.")
    a, b, c, d = st.columns(4)
    with a:
        show_chart(charts.bar_figure(val.allocation(marked, "symbol"), title="By holding"))
    with b:
        show_chart(charts.bar_figure(val.allocation(marked, "asset_class"), title="By asset class"))
    with c:
        if "industry" in marked:
            show_chart(charts.bar_figure(val.allocation(marked, "industry"), title="By industry"))
    with d:
        show_chart(charts.bar_figure(val.allocation(marked, "currency"), title="By currency"))
    if not book.lots.empty:
        with st.expander("Realised lots"):
            show_table(book.lots.assign(sell_date=book.lots["sell_date"].dt.date, buy_date=book.lots["buy_date"].dt.date)
                       .iloc[::-1], hide_index=True)


# -- returns --------------------------------------------------------------------------------------
def _dividends_block(pf: str, txs: list[Transaction], traded: list[str], base: str) -> None:
    sv = services()
    recorded = [t for t in txs if t.side == "DIVIDEND"]
    st.markdown("**Dividends**")
    if recorded:
        show_table(pd.DataFrame([{"date": t.ts.date(), "symbol": t.symbol, "shares": t.quantity, "per share": t.price,
                                  "tax withheld": t.fees, "net": t.gross - t.fees, "currency": t.currency, "note": t.note}
                                 for t in recorded]).iloc[::-1], hide_index=True)
    else:
        st.caption("No dividends recorded yet.")
    est = dv.estimated_dividends(txs, {s: ui_data.dividends(s) for s in traded if ev.has_company_events(s)})
    missing = est[~est["recorded"]] if not est.empty else est
    if missing.empty:
        if not est.empty:
            st.caption("Every ex-dividend date of your holdings is covered by a recorded dividend.")
        return
    st.warning(f"{len(missing)} dividend(s) went ex while you held the shares but are not in the ledger, so the returns above "
               "leave them out.")
    with st.expander("Dividends worked out from the ex-dividend dates", expanded=True):
        show_table(missing.assign(ex_date=missing["ex_date"].dt.date).drop(columns="recorded")
                   .rename(columns={"per_share": "per share"}).style.format({"shares": "{:,.2f}", "per share": "{:,.4f}",
                                                                              "gross": "{:,.2f}"}), hide_index=True)
        tax = st.number_input("Tax withheld (% of gross)", 0.0, 50.0, 0.0, 1.0, key="div_tax",
                              help="Check your broker's statement: Thai and US withholding rates differ, and you can edit the "
                                   "rows afterwards.")
        if st.button(f"Record these {len(missing)} dividends", key="div_record"):
            for t in dv.as_transactions(missing, pf, tax / 100):
                sv.transactions.add(t)
            st.rerun()
        st.caption("Dated on the ex-date (the payment comes a few weeks later); amounts are before tax unless you set a rate.")


def _returns_tab(pf, txs, book, marked, cash, base, fx_at, bench_symbol) -> None:
    traded = sorted({t.symbol for t in txs if t.side in TRADE_SIDES})
    cash_mode = cash is not None
    asof = pd.Timestamp.today().normalize()
    value_now = float(marked["market_value"].sum()) + (cash or 0.0)
    flows = ret.investor_flows(txs, fx_at, base)
    m = ret.money_summary(flows, value_now, asof)

    st.markdown(f"**What you earned** · money-weighted, in {base}")
    k = st.columns(5)
    k[0].metric("XIRR (per year)", pct(m.xirr, signed=True) if m.xirr == m.xirr else "-",
                help="The yearly rate that makes your deposits / purchases, withdrawals / sales / dividends and today's value "
                     "balance out. It depends on when you put money in.")
    k[1].metric("Money multiple", f"{m.multiple:.2f}x" if m.multiple == m.multiple else "-",
                help="(Money taken out + value today) / money put in.")
    k[2].metric(f"Put in ({base})", money(m.put_in))
    k[3].metric(f"Value today ({base})", money(m.value))
    k[4].metric(f"Gain ({base})", money(m.gain))
    st.caption(("Money in = your deposits and withdrawals; buying and selling happen inside the account." if cash_mode else
                "No deposits recorded, so every purchase counts as money in, and every sale and dividend as money out.")
               + (f" Held since {m.first:%d %b %Y} ({m.years:.1f} years)." if m.first is not None else ""))
    if m.first is not None and m.years < 1:
        st.warning("Less than a year of history: a yearly rate extrapolated from a few weeks or months can look enormous. "
                   "Read the money multiple and the gain instead.")

    st.markdown("**Where the profit came from**")
    parts = ret.attribution(marked, book.realized)
    left, right = st.columns([3, 2])
    with left:
        total = float(parts["amount"].sum())
        show_table(parts.assign(share=parts["amount"] / total if total else np.nan)
                   .rename(columns={"amount": f"amount ({base})"}).style.format({f"amount ({base})": "{:,.0f}", "share": "{:.0%}"}, na_rep="-"))
        st.caption(f"Total {money(total)} {base}. 'Currency' is what the exchange rate added or took away: a Thai stock held by a "
                   f"{base} investor earns its own gain plus (or minus) the baht moving against {base}.")
    with right:
        show_chart(charts.bar_figure(parts["amount"], title=f"Profit by source ({base})", fmt=",.0f", sort=False))

    # time-weighted: how good the holdings were, regardless of when money came in
    close = ui_data.base_close_matrix(tuple(traded), "5y", base)
    series = val.portfolio_series(txs, close) if not close.empty else None
    rets = series["ret"].dropna() if series is not None else pd.Series(dtype=float)
    st.markdown(f"**How good the holdings were** · time-weighted, in {base}")
    if len(rets) < 30:
        st.info("Needs ~30 trading days of holdings history.")
    else:
        bench_close = ui_data.base_close_matrix((bench_symbol,), "5y", base)
        bench = risk.to_returns(bench_close[bench_symbol]).reindex(rets.index).dropna() if bench_symbol in bench_close else None
        twr, bench_total = risk.total_return(rets), risk.total_return(bench) if bench is not None and len(bench) else float("nan")
        c = st.columns(4)
        c[0].metric("Time-weighted return", pct(twr, signed=True))
        c[1].metric("Per year", pct(risk.annualized_return(rets), signed=True))
        c[2].metric(f"{bench_symbol} in {base}", pct(bench_total, signed=True))
        c[3].metric("Difference", pct(twr - bench_total, signed=True) if bench_total == bench_total else "-")
        curve = pd.DataFrame({"Portfolio": (1 + rets).cumprod()})
        if bench is not None and len(bench):
            curve[f"{bench_symbol} in {base}"] = (1 + bench).cumprod()
        show_chart(charts.line_figure(curve, title=f"Growth of 1 {base}, same start", rebase=True, height=320))
        st.caption("Prices are dividend-adjusted, so dividends count as reinvested here - the recorded dividends are not added a second "
                   "time. The two measures differ when contributions are large or badly timed: the time-weighted return judges the "
                   "holdings, the XIRR judges your timing.")
    _dividends_block(pf, txs, traded, base)


# -- risk -----------------------------------------------------------------------------------------
def _risk_tab(txs, marked, base: str, bench_symbol: str) -> None:
    symbols = list(marked.index)
    traded = sorted({t.symbol for t in txs if t.side in TRADE_SIDES})
    close = ui_data.base_close_matrix(tuple(traded), "5y", base)
    series = val.portfolio_series(txs, close)
    rets = series["ret"].dropna()
    if len(rets) < 30:
        st.info("Need at least ~30 trading days of holdings history for risk statistics.")
        return
    bench_close = ui_data.base_close_matrix((bench_symbol,), "5y", base)
    bench = risk.to_returns(bench_close[bench_symbol]) if bench_symbol in bench_close else None
    s = risk.summary(rets, bench)

    k = st.columns(6)
    k[0].metric("CAGR", pct(s["cagr"]))
    k[1].metric("Volatility", pct(s["volatility"]))
    k[2].metric("Sharpe (rf=0)", f"{s['sharpe']:.2f}")
    k[3].metric("Max drawdown", pct(s["max_drawdown"]))
    k[4].metric("VaR / CVaR 95% (1d)", f"{pct(s['var_95'])} / {pct(s['cvar_95'])}")
    k[5].metric(f"Beta vs {bench_symbol}", f"{s['vs_benchmark_beta']:.2f}" if "vs_benchmark_beta" in s else "-")
    st.caption(f"Time-weighted returns of the current and past holdings in {base}, so exchange-rate moves are part of the risk; "
               "contributions do not distort them.")

    dd = risk.drawdown_series(rets)
    show_chart(charts.line_figure(pd.DataFrame({"Portfolio value (rebased)": dd["equity"] * 100}), title="Growth of 100", height=300))

    asset_rets = risk.to_returns(close[[c for c in symbols if c in close.columns]]).dropna(how="all")
    if len(asset_rets.columns) >= 2:
        a, b = st.columns(2)
        with a:
            show_chart(charts.heatmap_figure(risk.correlation_matrix(asset_rets.tail(504)), title=f"Correlation (2y daily, in {base})"))
        with b:
            w = marked["weight"].fillna(0).reindex(asset_rets.columns).fillna(0)
            w = w[w > 0]
            rc = risk.risk_contributions(w / w.sum(), asset_rets[w.index].dropna(axis=0, how="any").tail(504))
            show_chart(charts.bar_figure(rc["risk_contribution"], title="Share of portfolio risk"))
            st.caption("A holding whose risk share is far above its weight is driving the portfolio's volatility.")

    st.markdown("**Stress tests**")
    scen = risk.replay_scenarios(rets)
    c1, c2 = st.columns(2)
    with c1:
        if scen.empty:
            st.caption("Your holdings history does not reach any of the historical crisis windows.")
        else:
            show_table(scen.assign(**{"return": scen["return"] * 100, "worst_day": scen["worst_day"] * 100})
                       .style.format({"return": "{:+.1f}%", "worst_day": "{:+.1f}%"}), hide_index=True)
    with c2:
        move = st.slider("Market shock", -40, 0, -20, 5, format="%d%%") / 100
        betas = {}
        if bench is not None:
            for sym in symbols:
                r = asset_rets[sym].dropna() if sym in asset_rets else pd.Series(dtype=float)
                betas[sym] = risk.beta_alpha(r, bench)["beta"] if len(r) > 60 else 1.0
        impact = risk.beta_shock(marked["weight"].fillna(0), pd.Series(betas, dtype=float), move)
        st.metric(f"Estimated impact of a {move:.0%} {bench_symbol} move", pct(impact, signed=True),
                  help="First-order estimate: sum of weight x beta x market move. Ignores correlation breakdowns.")

    store = services().md.store.read_many(symbols, last_n=20, columns=["close", "volume"])
    if not store.empty:
        adv_local = store.assign(dv=store["close"] * store["volume"]).groupby("symbol")["dv"].mean()
        adv = adv_local * marked["fx"].reindex(adv_local.index)  # to the base currency, like the position values
        days = risk.days_to_liquidate(marked["market_value"], adv)
        with st.expander("Liquidity - days to exit at 20% of average daily volume"):
            show_table(days.to_frame().style.format("{:.2f}", na_rep="n/a"))


# -- optimise -------------------------------------------------------------------------------------
def _optimize_tab(marked, base: str, cash_default: float = 0.0) -> None:
    held = list(marked.index)
    extra = st.text_input("Extra candidate symbols (optional, comma separated)", "", key="opt_extra")
    universe = list(dict.fromkeys(held + [s.strip().upper() for s in extra.split(",") if s.strip()]))
    c1, c2, c3, c4 = st.columns(4)
    method = c1.selectbox("Method", opt.METHODS, index=2, format_func=lambda m: m.replace("_", " ").title(), key="opt_method")
    lookback = c2.selectbox("Lookback", ["1y", "2y", "3y", "5y"], index=2, key="opt_lb")
    max_w = c3.slider("Max weight per asset", 0.1, 1.0, 0.4, 0.05, key="opt_max")
    rf = c4.number_input("Risk-free rate", 0.0, 0.2, 0.04, 0.005, key="opt_rf")
    if len(universe) < 2:
        st.info("Add at least two symbols (holdings or candidates).")
        return
    max_w = max(max_w, 1.0 / len(universe))
    close = ui_data.base_close_matrix(tuple(universe), lookback, base).dropna(axis=1, thresh=100).dropna()
    if close.shape[1] < 2:
        st.error("Not enough overlapping price history (or no exchange rate for some currency).")
        return
    returns = risk.to_returns(close)
    try:
        res = opt.optimize(returns, method, rf=rf, max_weight=max_w)
    except ValueError as exc:
        st.error(str(exc))
        return
    current = (marked["weight"].reindex(res.weights.index).fillna(0.0))
    k = st.columns(3)
    k[0].metric("Expected return (shrunk)", pct(res.expected_return))
    k[1].metric("Volatility", pct(res.volatility))
    k[2].metric("Sharpe", f"{res.sharpe:.2f}")
    st.caption(f"Returns and risk are in {base}, so a Thai stock's exchange-rate moves count.")
    cmp = pd.DataFrame({"current": current, "optimised": res.weights}).sort_values("optimised", ascending=False)
    show_table(cmp.style.format("{:.1%}"))
    if method in ("min_variance", "max_sharpe"):
        ef = opt.efficient_frontier(returns, 20, rf=rf, max_weight=max_w)
        if not ef.empty:
            import plotly.graph_objects as go
            fig = go.Figure(go.Scatter(x=ef["volatility"] * 100, y=ef["return"] * 100, mode="lines+markers", name="Efficient frontier"))
            fig.add_trace(go.Scatter(x=[res.volatility * 100], y=[res.expected_return * 100], mode="markers", name=method,
                                     marker=dict(size=13, color="#ef5350")))
            fig.update_layout(height=340, xaxis_title="Volatility %", yaxis_title="Expected return %", margin=dict(l=10, r=10, t=20, b=10))
            show_chart(fig)
    st.caption("Expected returns come from history and are shrunk halfway to their average; treat max-Sharpe weights as fragile. "
               "Min-variance and risk-parity depend only on risk estimates, which are far more stable.")

    st.markdown("**Rebalance orders**")
    cash = st.number_input(f"Cash available ({base})", 0.0, 1e12, cash_default, 100.0, key="opt_cash")
    a, b = st.columns(2)
    whole = a.checkbox("Whole shares only", value=True, key="opt_whole")
    lots = b.checkbox("Round Thai orders to 100-share board lots", value=True, key="opt_lots", disabled=not whole)
    local = _last_prices(list(res.weights.index))
    ccy_of = {s: currency_of(s) for s in res.weights.index}
    fx_now = ccy.latest(ui_data.fx_series(tuple(sorted({*ccy_of.values(), base})), base))  # candidates may use other currencies
    rates = {s: fx_now.get(ccy_of[s]) for s in res.weights.index}
    if any(r is None or r != r for r in rates.values()) or any(s not in local for s in res.weights.index):
        st.error("A price or an exchange rate is missing for: " + ", ".join(s for s in res.weights.index
                                                                         if s not in local or rates[s] is None or rates[s] != rates[s]))
        return
    prices_base = pd.Series({s: local[s] * rates[s] for s in res.weights.index})
    qty = marked["quantity"].reindex(res.weights.index).fillna(0.0)
    try:
        orders = rebalance_orders(qty, prices_base, res.weights, cash, whole_shares=whole, fee_bps=2,
                                  lot_sizes=pd.Series({s: lot_size(s) for s in res.weights.index}) if lots else None)
    except ValueError as exc:
        st.error(str(exc))
        return
    if orders.empty:
        st.success("Already at target.")
    else:
        orders = orders.assign(currency=orders["symbol"].map(ccy_of), price_local=orders["symbol"].map(local))
        orders = orders.rename(columns={"price": f"price ({base})", "value": f"value ({base})", "fees": f"fees ({base})"})
        show_table(orders.style.format({"shares": "{:,.2f}", f"price ({base})": "{:,.2f}", "price_local": "{:,.2f}",
                                        f"value ({base})": "{:,.0f}", f"fees ({base})": "{:,.2f}", "current_weight": "{:.1%}",
                                        "target_weight": "{:.1%}", "new_weight": "{:.1%}"}), hide_index=True)
        st.download_button("Download orders (CSV)", orders.to_csv(index=False).encode(), file_name="rebalance_orders.csv", mime="text/csv")
    st.caption("Orders are suggestions computed from the latest stored prices; check them against your broker before trading.")


def _default_base(txs: list[Transaction]) -> str:
    return "THB" if not txs or any(t.currency == "THB" for t in txs) else "USD"


def render() -> None:
    st.title("💼 Portfolio & Risk")
    sv = services()
    repo = sv.transactions
    names = repo.portfolios() or ["default"]
    with st.sidebar:
        choice = st.selectbox("Portfolio", [*names, "+ new portfolio"], key="pf_choice")
        pf = st.text_input("New portfolio name", key="pf_new").strip() if choice == "+ new portfolio" else choice
        method = st.radio("Cost basis", ["average", "fifo"], format_func=lambda m: {"average": "Average cost", "fifo": "FIFO lots"}[m], key="pf_method")
    if not pf:
        st.info("Name the new portfolio in the sidebar.")
        return

    txs = repo.list(pf)
    with st.sidebar:
        used = sorted({t.currency for t in txs} | {"THB", "USD"})
        options = list(dict.fromkeys([*BASE_CHOICES, *used]))
        base = st.selectbox("Show values in", options, index=options.index(_default_base(txs)), key="pf_base",
                            help="Every holding is converted into this currency, each purchase at the exchange rate of its day.")
        bench_symbol = st.text_input("Benchmark", benchmark_for(next((t.symbol for t in txs if t.side in TRADE_SIDES), "SPY")),
                                     key="pf_bench").strip().upper() or "SPY"

    tab_tx, tab_ov, tab_ret, tab_risk, tab_opt = st.tabs(["Transactions", "Overview", "Returns", "Risk", "Optimise & rebalance"])
    with tab_tx:
        _transactions_tab(pf, txs, base)
    pending = (tab_ov, tab_ret, tab_risk, tab_opt)
    if not txs:
        for t in pending:
            with t:
                st.info("Add transactions first (form or CSV import on the Transactions tab).")
        return

    table = ui_data.fx_series(tuple(sorted({t.currency for t in txs} | {base})), base)
    no_rate = ccy.missing(table)
    if no_rate:
        for t in pending:
            with t:
                st.error("No exchange rate to " + base + " for: " + ", ".join(no_rate) + ". Pick another base currency, or fix the "
                         "currency of those transactions (Yahoo needs a pair such as USDTHB=X).")
        return
    fx_at, fx_now = ccy.lookup(table), ccy.latest(table)
    try:
        book = build_book(txs, method, base=base, fx_at=fx_at)
    except ValueError as exc:
        for t in pending:
            with t:
                st.error(str(exc))
        return
    cash = ret.cash_balance(txs, fx_at, base) if ret.has_cash_flows(txs) else None
    if book.positions.empty:
        for t in pending:
            with t:
                st.info("No open positions." + (f" Cash: {money(cash)} {base}." if cash is not None else ""))
        return

    prices = _last_prices(list(book.positions.index))
    meta = ui_data.universe_table().drop_duplicates("symbol").set_index("symbol")[["name", "industry"]]
    marked = val.mark_to_market(book.positions, prices, meta, fx=fx_now)
    if marked["price"].isna().any():
        st.warning("No price for: " + ", ".join(marked.index[marked["price"].isna()]) + " - they are excluded from totals.")
    with tab_ov:
        _overview_tab(book, marked, cash, base)
    with tab_ret:
        _returns_tab(pf, txs, book, marked, cash, base, fx_at, bench_symbol)
    with tab_risk:
        _risk_tab(txs, marked.dropna(subset=["price"]), base, bench_symbol)
    with tab_opt:
        _optimize_tab(marked.dropna(subset=["price"]), base)
