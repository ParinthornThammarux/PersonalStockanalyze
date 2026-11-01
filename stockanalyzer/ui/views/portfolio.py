"""Portfolio & Risk: transactions, P&L, allocation, risk analytics, optimisation and rebalancing."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from stockanalyzer.analysis.risk import metrics as risk
from stockanalyzer.portfolio import optimize as opt
from stockanalyzer.portfolio import valuation as val
from stockanalyzer.portfolio.ledger import Transaction, build_book
from stockanalyzer.portfolio.rebalance import rebalance_orders
from stockanalyzer.ui import charts
from stockanalyzer.ui import data as ui_data
from stockanalyzer.ui.context import services
from stockanalyzer.ui.format import money, pct
from stockanalyzer.ui.widgets import show_chart, show_table

TEMPLATE = "date,symbol,side,quantity,price,fees\n2024-01-15,AAPL,BUY,10,185.20,1.0\n2024-06-03,AAPL,SELL,4,194.10,1.0\n"


def _last_prices(symbols: list[str]) -> dict[str, float]:
    out = {}
    for s in symbols:
        df = ui_data.prices(s, "1mo", "1d", False)
        if not df.empty:
            out[s] = float(df["close"].iloc[-1])
    return out


def _transactions_tab(pf: str, txs: list[Transaction]) -> None:
    sv = services()
    repo = sv.transactions
    with st.form("tx_form", clear_on_submit=True):
        c = st.columns([1.2, 1.2, 1, 1, 1, 1])
        day = c[0].date_input("Date")
        symbol = c[1].text_input("Symbol").strip().upper()
        side = c[2].selectbox("Side", ["BUY", "SELL"])
        qty = c[3].number_input("Quantity", min_value=0.0, value=0.0, format="%.4f")
        price = c[4].number_input("Price", min_value=0.0, value=0.0, format="%.4f")
        fees = c[5].number_input("Fees", min_value=0.0, value=0.0, format="%.2f")
        note = st.text_input("Note")
        submitted = st.form_submit_button("Add transaction", type="primary")
    if submitted:
        try:
            tx = Transaction(pd.Timestamp(day), symbol, side, qty, price, fees, note, pf)
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
    show_table(frame.iloc[::-1], hide_index=True)
    gone = st.selectbox("Delete transaction id", frame["id"].tolist(), index=None, placeholder="choose id", key="tx_del")
    if gone is not None and st.button("Delete"):
        repo.delete(int(gone))
        st.rerun()


def _overview_tab(book, marked, meta) -> None:
    s = val.summary(marked, book)
    k = st.columns(5)
    k[0].metric("Market value", money(s["market_value"]))
    k[1].metric("Unrealised P&L", money(s["unrealized_pnl"]), pct(s["unrealized_pct"], signed=True))
    k[2].metric("Realised P&L", money(s["realized_pnl"]))
    k[3].metric("Total P&L", money(s["total_pnl"]))
    k[4].metric("Fees paid", money(s["fees_paid"]))
    cols = ["quantity", "avg_cost", "price", "market_value", "unrealized_pnl", "unrealized_pct", "weight", "realized_pnl", "asset_class"]
    cols += [c for c in ("name", "industry") if c in marked.columns]
    show_table(marked[cols].style.format({"quantity": "{:,.2f}", "avg_cost": "{:,.2f}", "price": "{:,.2f}", "market_value": "{:,.0f}",
                                           "unrealized_pnl": "{:,.0f}", "unrealized_pct": "{:+.1%}", "weight": "{:.1%}",
                                           "realized_pnl": "{:,.0f}"}, na_rep="-"))
    a, b, c = st.columns(3)
    with a:
        show_chart(charts.bar_figure(val.allocation(marked, "symbol"), title="By holding"))
    with b:
        show_chart(charts.bar_figure(val.allocation(marked, "asset_class"), title="By asset class"))
    with c:
        if "industry" in marked:
            show_chart(charts.bar_figure(val.allocation(marked, "industry"), title="By industry"))
    if not book.lots.empty:
        with st.expander("Realised lots"):
            show_table(book.lots.assign(sell_date=book.lots["sell_date"].dt.date, buy_date=book.lots["buy_date"].dt.date)
                       .iloc[::-1], hide_index=True)


def _risk_tab(txs, marked, bench_symbol: str) -> None:
    symbols = list(marked.index)
    close = ui_data.close_matrix(tuple(symbols), "5y")
    series = val.portfolio_series(txs, close)
    rets = series["ret"].dropna()
    if len(rets) < 30:
        st.info("Need at least ~30 trading days of holdings history for risk statistics.")
        return
    bench_px = ui_data.prices(bench_symbol, "5y")
    bench = risk.to_returns(bench_px["close"]) if not bench_px.empty else None
    s = risk.summary(rets, bench)

    k = st.columns(6)
    k[0].metric("CAGR", pct(s["cagr"]))
    k[1].metric("Volatility", pct(s["volatility"]))
    k[2].metric("Sharpe (rf=0)", f"{s['sharpe']:.2f}")
    k[3].metric("Max drawdown", pct(s["max_drawdown"]))
    k[4].metric("VaR / CVaR 95% (1d)", f"{pct(s['var_95'])} / {pct(s['cvar_95'])}")
    k[5].metric(f"Beta vs {bench_symbol}", f"{s['vs_benchmark_beta']:.2f}" if "vs_benchmark_beta" in s else "-")
    st.caption("Time-weighted returns of the current and past holdings; contributions do not distort them.")

    dd = risk.drawdown_series(rets)
    show_chart(charts.line_figure(pd.DataFrame({"Portfolio value (rebased)": dd["equity"] * 100}), title="Growth of 100", height=300))

    asset_rets = risk.to_returns(close).dropna(how="all")
    if len(symbols) >= 2:
        a, b = st.columns(2)
        with a:
            show_chart(charts.heatmap_figure(risk.correlation_matrix(asset_rets.tail(504)), title="Correlation (2y daily)"))
        with b:
            w = marked["weight"].fillna(0)
            w = w[w > 0]
            rc = risk.risk_contributions(w / w.sum(), asset_rets.dropna(axis=0, how="any").tail(504))
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
        adv = (store.assign(dv=store["close"] * store["volume"]).groupby("symbol")["dv"].mean())
        days = risk.days_to_liquidate(marked["market_value"], adv)
        with st.expander("Liquidity - days to exit at 20% of average daily volume"):
            show_table(days.to_frame().style.format("{:.2f}", na_rep="n/a"))


def _optimize_tab(marked, cash_default: float = 0.0) -> None:
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
    close = ui_data.close_matrix(tuple(universe), lookback).dropna(axis=1, thresh=100).dropna()
    if close.shape[1] < 2:
        st.error("Not enough overlapping price history.")
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
    cash = st.number_input("Cash available", 0.0, 1e10, cash_default, 100.0, key="opt_cash")
    whole = st.checkbox("Whole shares only", value=True, key="opt_whole")
    prices = _last_prices(list(res.weights.index))
    qty = marked["quantity"].reindex(res.weights.index).fillna(0.0)
    try:
        orders = rebalance_orders(qty, pd.Series(prices), res.weights, cash, whole_shares=whole, fee_bps=2)
    except ValueError as exc:
        st.error(str(exc))
        return
    if orders.empty:
        st.success("Already at target.")
    else:
        show_table(orders.style.format({"shares": "{:,.2f}", "price": "{:,.2f}", "value": "{:,.0f}", "fees": "{:,.2f}",
                                        "current_weight": "{:.1%}", "target_weight": "{:.1%}", "new_weight": "{:.1%}"}), hide_index=True)
        st.download_button("Download orders (CSV)", orders.to_csv(index=False).encode(), file_name="rebalance_orders.csv", mime="text/csv")
    st.caption("Orders are suggestions computed from the latest stored prices; check them against your broker before trading.")


def render() -> None:
    st.title("💼 Portfolio & Risk")
    sv = services()
    repo = sv.transactions
    names = repo.portfolios() or ["default"]
    with st.sidebar:
        choice = st.selectbox("Portfolio", [*names, "+ new portfolio"], key="pf_choice")
        pf = st.text_input("New portfolio name", key="pf_new").strip() if choice == "+ new portfolio" else choice
        method = st.radio("Cost basis", ["average", "fifo"], format_func=lambda m: {"average": "Average cost", "fifo": "FIFO lots"}[m], key="pf_method")
        bench_symbol = st.text_input("Benchmark", "SPY", key="pf_bench").strip().upper() or "SPY"
    if not pf:
        st.info("Name the new portfolio in the sidebar.")
        return

    txs = repo.list(pf)
    tab_tx, tab_ov, tab_risk, tab_opt = st.tabs(["Transactions", "Overview", "Risk", "Optimise & rebalance"])
    with tab_tx:
        _transactions_tab(pf, txs)
    if not txs:
        for t in (tab_ov, tab_risk, tab_opt):
            with t:
                st.info("Add transactions first (form or CSV import on the Transactions tab).")
        return

    book = build_book(txs, method)
    if book.positions.empty:
        for t in (tab_ov, tab_risk, tab_opt):
            with t:
                st.info("No open positions.")
        return
    prices = _last_prices(list(book.positions.index))
    meta = ui_data.universe_table().drop_duplicates("symbol").set_index("symbol")[["name", "industry"]]
    marked = val.mark_to_market(book.positions, prices, meta)
    if marked["price"].isna().any():
        st.warning("No price for: " + ", ".join(marked.index[marked["price"].isna()]) + " - they are excluded from totals.")
    with tab_ov:
        _overview_tab(book, marked, meta)
    with tab_risk:
        _risk_tab(txs, marked.dropna(subset=["price"]), bench_symbol)
    with tab_opt:
        _optimize_tab(marked.dropna(subset=["price"]))
