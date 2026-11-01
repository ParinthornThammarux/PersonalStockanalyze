"""Backtest: single-asset strategies with costs, parameter sweeps, and multi-asset momentum rotation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from stockanalyzer.analysis.risk import metrics as risk
from stockanalyzer.backtest import strategies as S
from stockanalyzer.backtest.engine import BacktestConfig, run_backtest, run_portfolio_backtest
from stockanalyzer.data import universe as uni
from stockanalyzer.ui import charts
from stockanalyzer.ui import data as ui_data
from stockanalyzer.ui.context import services
from stockanalyzer.ui.format import heat, pct
from stockanalyzer.ui.widgets import param_inputs, show_chart, show_table, symbol_picker

PERIODS = ["2y", "5y", "10y", "max"]

LABELS = [("total_return", "Total return", "pct"), ("cagr", "CAGR", "pct"), ("volatility", "Volatility", "pct"),
          ("sharpe", "Sharpe", "num"), ("sortino", "Sortino", "num"), ("max_drawdown", "Max drawdown", "pct"),
          ("calmar", "Calmar", "num")]


def _comparison(m: dict) -> pd.DataFrame:
    bh = {"total_return": m["buy_hold_return"], "cagr": m["buy_hold_cagr"], "sharpe": m["buy_hold_sharpe"],
          "max_drawdown": m["buy_hold_max_drawdown"]}
    rows = []
    for key, label, kind in LABELS:
        f = (lambda v: pct(v)) if kind == "pct" else (lambda v: "-" if v is None or v != v else f"{v:.2f}")
        rows.append({"metric": label, "strategy": f(m.get(key)), "buy & hold": f(bh[key]) if key in bh else ""})
    return pd.DataFrame(rows).set_index("metric")


def _single_tab() -> None:
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        symbol = symbol_picker(container=c1)
        period = st.selectbox("History", PERIODS, index=2, key="bt_period")
    with c2:
        name = st.selectbox("Strategy", list(S.STRATEGIES), format_func=lambda n: S.STRATEGIES[n].label, key="bt_strategy")
        strat = S.STRATEGIES[name]
        st.caption(strat.description)
    with c3:
        commission = st.number_input("Commission (bps per side)", 0.0, 100.0, 1.0, 0.5, key="bt_comm")
        slippage = st.number_input("Slippage (bps per side)", 0.0, 100.0, 2.0, 0.5, key="bt_slip")
    params = {}
    if strat.params:
        with st.expander("Strategy parameters", expanded=True):
            cols = st.columns(len(strat.params))
            for col, p in zip(cols, strat.params):
                params.update(param_inputs((p,), f"bt_{name}", container=col))
    o1, o2, o3 = st.columns(3)
    execution = o1.radio("Execution", ["next_open", "close"], horizontal=True, key="bt_exec",
                         help="next_open: trade at the next bar's open (realistic). close: trade at the signal bar's close (optimistic).")
    allow_short = o2.checkbox("Allow short", value=False, key="bt_short")
    capital = o3.number_input("Initial capital", 1000.0, 1e9, 100_000.0, 1000.0, key="bt_cap")
    if not symbol:
        return

    df = ui_data.require_prices(symbol, period)
    cfg = BacktestConfig(initial_capital=capital, commission_bps=commission, slippage_bps=slippage,
                         execution=execution, allow_short=allow_short)
    try:
        res = S.run_strategy(df, name, params, cfg)
    except ValueError as exc:
        st.error(str(exc))
        return
    m = res.metrics

    k = st.columns(5)
    k[0].metric("Final equity", f"{m['final_equity']:,.0f}", pct(m["total_return"], signed=True))
    k[1].metric("CAGR", pct(m["cagr"]), f"B&H {pct(m['buy_hold_cagr'])}", delta_color="off")
    k[2].metric("Sharpe", f"{m['sharpe']:.2f}", f"B&H {m['buy_hold_sharpe']:.2f}", delta_color="off")
    k[3].metric("Max drawdown", pct(m["max_drawdown"]), f"B&H {pct(m['buy_hold_max_drawdown'])}", delta_color="off")
    k[4].metric("Time in market", pct(m["exposure"], 0), f"{int(m['n_trades'])} trades", delta_color="off")
    if m["buy_hold_sharpe"] > m["sharpe"] and name != "buy_hold":
        st.warning("This strategy did not beat simply holding the asset on a risk-adjusted basis over this period.")

    dd = risk.drawdown_series(res.returns)["drawdown"]
    show_chart(charts.equity_figure(res.equity, res.benchmark_equity, dd, strat.label))

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Performance**")
        show_table(_comparison(m))
        st.caption(f"Costs paid: {m['total_costs'] * 100:.2f}% of capital · turnover {m['turnover_per_year']:.1f}x per year")
    with right:
        st.markdown("**Trades**")
        if res.trades.empty:
            st.caption("No trades.")
        else:
            st.caption(f"Win rate {pct(m['win_rate'], 0)} · avg trade {pct(m['avg_trade'])} · profit factor {m['profit_factor']:.2f} · "
                       f"avg {m['avg_bars_held']:.0f} bars")
            show_table(res.trades.assign(**{"return": res.trades["return"] * 100}).iloc[::-1].head(100)
                       .style.format({"return": "{:+.1f}%", "entry_price": "{:,.2f}", "exit_price": "{:,.2f}"}), hide_index=True)

    with st.expander("Calendar returns of the strategy"):
        table = risk.monthly_table(res.returns) * 100
        show_table(heat(table.style.format("{:+.1f}", na_rep=""), vmin=-8, vmax=8))

    with st.expander("Parameter sweep (in-sample vs out-of-sample)"):
        if not strat.params:
            st.caption("This strategy has no parameters.")
            return
        picked = st.multiselect("Parameters to sweep", [p.name for p in strat.params], default=[strat.params[0].name][:1], max_selections=2)
        grid = {}
        for p in strat.params:
            if p.name in picked:
                a, b, n = st.columns(3)
                lo = a.number_input(f"{p.name} from", float(p.min), float(p.max), float(p.default) * 0.5 if p.default * 0.5 >= p.min else float(p.min), key=f"sw_lo_{p.name}")
                hi = b.number_input(f"{p.name} to", float(p.min), float(p.max), min(float(p.default) * 1.5, float(p.max)), key=f"sw_hi_{p.name}")
                pts = int(n.number_input(f"{p.name} points", 2, 8, 5, key=f"sw_n_{p.name}"))
                vals = np.linspace(lo, hi, pts)
                grid[p.name] = [int(round(v)) for v in vals] if p.is_int else [round(float(v), 3) for v in vals]
                grid[p.name] = sorted(set(grid[p.name]))
        split = st.slider("In-sample share of history", 0.5, 0.9, 0.7, 0.05)
        if grid and st.button("Run sweep"):
            fixed = {k: v for k, v in params.items() if k not in grid}
            with st.spinner("Running backtests ..."):
                out = S.sweep(df, name, {**{k: [v] for k, v in fixed.items()}, **grid}, cfg, split=split)
            if out.empty:
                st.info("No valid combinations (check fast < slow).")
            else:
                show_table(out.style.format({c: "{:.2f}" for c in out.columns if c.startswith("sharpe")} |
                                            {c: "{:.1%}" for c in out.columns if c.startswith("cagr") or c in ("max_drawdown", "total_return")}),
                           hide_index=True)
                best = out.iloc[0]
                st.caption(f"Best in-sample Sharpe {best['sharpe_in']:.2f} -> out-of-sample {best['sharpe_out']:.2f}. "
                           "A big drop is the price of picking parameters on the same data you evaluate on.")
                if len(grid) == 2:
                    a, b = list(grid)
                    pivot = out.pivot(index=a, columns=b, values="sharpe_out")
                    show_chart(charts.heatmap_figure(pivot, title="Out-of-sample Sharpe", zmin=-1, zmax=2, colorscale="RdYlGn", fmt=".2f"))


def _rotation_tab() -> None:
    sv = services()
    src = st.radio("Universe", ["Cross-asset presets", "Watchlist", "Custom list"], horizontal=True, key="rot_src")
    if src == "Cross-asset presets":
        symbols = [s for s in uni.cross_asset_universe()["symbol"] if s not in ("^VIX", "^TNX")]
    elif src == "Watchlist":
        names = sv.watchlists.names()
        symbols = sv.watchlists.symbols(st.selectbox("List", names, key="rot_list")) if names else []
    else:
        symbols = [s.strip().upper() for s in st.text_input("Symbols (comma separated)", "SPY, QQQ, TLT, GLD, EEM, IWM, VNQ", key="rot_custom").split(",") if s.strip()]
    c1, c2, c3, c4 = st.columns(4)
    top_n = c1.slider("Hold top N", 1, 10, 3, key="rot_n")
    lookback = c2.slider("Momentum lookback (bars)", 21, 252, 126, key="rot_lb")
    skip = c3.slider("Skip recent bars", 0, 42, 21, key="rot_skip")
    rebalance = c4.selectbox("Rebalance", ["ME", "QE", "W-FRI"], format_func={"ME": "Monthly", "QE": "Quarterly", "W-FRI": "Weekly"}.get, key="rot_reb")
    period = st.selectbox("History", PERIODS, index=2, key="rot_period")
    cost = st.number_input("Cost (bps per unit of turnover)", 0.0, 100.0, 5.0, key="rot_cost")

    if len(symbols) < 3:
        st.info("Pick at least 3 symbols.")
        return
    if not st.button("Run rotation backtest", type="primary", key="rot_run"):
        return
    with st.spinner(f"Loading {len(symbols)} symbols ..."):
        close = ui_data.close_matrix(tuple(symbols), period).dropna(how="all")
    close = close.dropna(axis=1, thresh=int(len(close) * 0.5))
    if close.shape[1] < 3:
        st.error("Not enough symbols with history. Update prices in Data Manager first.")
        return
    weights = S.momentum_rotation_weights(close, top_n, lookback, skip, rebalance)
    cfg = BacktestConfig(commission_bps=cost, slippage_bps=0)
    strat = run_portfolio_backtest(close, weights, cfg)
    eq_w = weights.copy()
    eq_w[eq_w.notna().any(axis=1)] = 1.0 / close.shape[1]
    bench = run_portfolio_backtest(close, eq_w, cfg)
    m, bm = strat.metrics, bench.metrics

    k = st.columns(4)
    k[0].metric("CAGR", pct(m["cagr"]), f"equal-weight {pct(bm['cagr'])}", delta_color="off")
    k[1].metric("Sharpe", f"{m['sharpe']:.2f}", f"equal-weight {bm['sharpe']:.2f}", delta_color="off")
    k[2].metric("Max drawdown", pct(m["max_drawdown"]), f"equal-weight {pct(bm['max_drawdown'])}", delta_color="off")
    k[3].metric("Turnover / year", f"{m['turnover_per_year']:.1f}x", f"{int(m['rebalances'])} rebalances", delta_color="off")
    show_chart(charts.equity_figure(strat.equity, bench.equity, risk.drawdown_series(strat.returns)["drawdown"], "Momentum rotation"))
    held = strat.weights.iloc[-1]
    held = held[held > 0].sort_values(ascending=False)
    st.markdown("**Currently held:** " + (", ".join(f"{s} {w:.0%}" for s, w in held.items()) or "cash"))
    st.caption("Benchmark = the same assets equal-weighted and rebalanced on the same dates. Survivorship note: the universe is "
               "today's list, so past results flatter the strategy.")


def render() -> None:
    st.title("🧪 Backtest")
    st.caption("Signals use only information up to each bar's close and trade no earlier than the next bar. "
               "Prices are split/dividend adjusted. Results are history, not a forecast.")
    tab_single, tab_rot = st.tabs(["Single-asset strategy", "Momentum rotation"])
    with tab_single:
        _single_tab()
    with tab_rot:
        _rotation_tab()
