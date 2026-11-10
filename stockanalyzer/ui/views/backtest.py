"""Backtest: one strategy on one or several assets held at once (costs, weights, dividends, parameter sweeps) and momentum rotation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from stockanalyzer.analysis.risk import metrics as risk
from stockanalyzer.backtest import strategies as S
from stockanalyzer.backtest.engine import BacktestConfig, run_backtest, run_portfolio_backtest
from stockanalyzer.data import pools as stock_pools
from stockanalyzer.data import sectors
from stockanalyzer.data import universe as uni
from stockanalyzer.domain.assets import currency_of
from stockanalyzer.ui import charts
from stockanalyzer.ui import data as ui_data
from stockanalyzer.ui.context import services
from stockanalyzer.ui.format import heat, pct
from stockanalyzer.ui.widgets import param_inputs, show_chart, show_table

PERIODS = ["2y", "5y", "10y", "max"]
UNIVERSES = ["Cross-asset presets", "US sector ETFs", "Asset classes", "Stock index", "US by market cap", "Watchlist", "Custom list"]
MAX_ASSETS = 100
MIN_HISTORY = 0.8  # share of the longest history an asset needs to stay in a multi-asset test
REBALANCE = {None: "Only when a signal changes", "ME": "Also monthly", "QE": "Also quarterly"}

LABELS = [("total_return", "Total return", "pct"), ("cagr", "CAGR", "pct"), ("volatility", "Volatility", "pct"),
          ("sharpe", "Sharpe", "num"), ("sortino", "Sortino", "num"), ("max_drawdown", "Max drawdown", "pct"),
          ("calmar", "Calmar", "num")]


def _comparison(m: dict, bm: dict, bench_label: str) -> pd.DataFrame:
    rows = []
    for key, label, kind in LABELS:
        f = (lambda v: pct(v)) if kind == "pct" else (lambda v: "-" if v is None or v != v else f"{v:.2f}")
        rows.append({"metric": label, "strategy": f(m.get(key)), bench_label: f(bm.get(key))})
    return pd.DataFrame(rows).set_index("metric")


def _asset_table(res: S.MultiStrategyResult) -> pd.DataFrame:
    """One row per asset: what it added to the portfolio and how its signal did on its own."""
    pf = res.portfolio
    pnl, paid = pf.pnl_by_asset(), pf.dividends.sum()
    rows = []
    for sym, a in res.assets.items():
        m = a.metrics
        rows.append({"symbol": sym, "contribution": pnl[sym] / pf.config.initial_capital, "dividends": paid[sym],
                     "avg weight": pf.weights[sym].mean(), "time in market": m["exposure"],
                     "alone": res.alone.loc[sym, "alone"], "buy & hold": res.alone.loc[sym, "buy_hold"],
                     "trades": int(m["n_trades"]), "win rate": m["win_rate"]})
    return pd.DataFrame(rows).set_index("symbol").sort_values("contribution", ascending=False)


def _weights_inputs(symbols: list[str]) -> pd.Series:
    """Relative weight per symbol (2 / 1 / 1 = 50% / 25% / 25%). Stops the page if all are zero."""
    cols = st.columns(min(len(symbols), 6))
    raw = {sym: cols[i % len(cols)].number_input(sym, 0.0, 1000.0, 1.0, 1.0, key=f"bt_w_{sym}") for i, sym in enumerate(symbols)}
    total = sum(raw.values())
    if total <= 0:
        st.error("Give at least one asset a weight above zero.")
        st.stop()
    st.caption("Shares: " + " · ".join(f"{s} {v / total:.0%}" for s, v in raw.items()))
    return pd.Series(raw)


def _load_frames(symbols: list[str], period: str) -> dict[str, pd.DataFrame]:
    """Stored (not dividend-adjusted) prices per symbol; the ones without data are reported with name matches."""
    frames, missing = {}, []
    with st.spinner(f"Loading {len(symbols)} symbols ..."):
        ui_data.prefetch(symbols)
        for sym in symbols:
            df = ui_data.prices(sym, period, "1d", False, live=False)  # finished bars only: a backtest has no use for today's partial one
            if df.empty:
                missing.append(sym)
            else:
                frames[sym] = df
    if missing:
        table = ui_data.universe_table()
        hints = []
        for sym in missing:
            hits = uni.search(table, sym, limit=3)
            hints.append(sym + (f" (did you mean {', '.join(hits['symbol'])}?)" if not hits.empty else ""))
        st.warning("No price data, left out: " + "; ".join(hints))
    return frames


def _strategy_tab() -> None:
    symbols = _universe_picker("bt", default="Custom list", custom=st.session_state.get("symbol", "AAPL"))
    if not symbols:
        st.info("Enter at least one symbol.")
        return
    if len(symbols) > MAX_ASSETS:
        st.warning(f"Using the first {MAX_ASSETS} of {len(symbols)} symbols.")
        symbols = symbols[:MAX_ASSETS]
    multi = len(symbols) > 1

    c1, c2, c3 = st.columns(3)
    with c1:
        period = st.selectbox("History", PERIODS, index=2, key="bt_period")
        capital = st.number_input("Initial capital", 1000.0, 1e9, 100_000.0, 1000.0, key="bt_cap")
    with c2:
        name = st.selectbox("Strategy", list(S.STRATEGIES), format_func=lambda n: S.STRATEGIES[n].label, key="bt_strategy")
        strat = S.STRATEGIES[name]
        st.caption(strat.description + (" Applied to every asset separately." if multi else ""))
        execution = st.radio("Execution", ["next_open", "close"], horizontal=True, key="bt_exec",
                             help="next_open: trade at the next bar's open (realistic). close: trade at the signal bar's close (optimistic).")
    with c3:
        commission = st.number_input("Commission (bps per side)", 0.0, 100.0, 1.0, 0.5, key="bt_comm")
        slippage = st.number_input("Slippage (bps per side)", 0.0, 100.0, 2.0, 0.5, key="bt_slip")
        dividends = st.radio("Dividends", ["Reinvest", "Take as cash"], horizontal=True, key="bt_div",
                             help="Reinvest: bought back on the ex-date, so the result is total return. Take as cash: the price return "
                                  "plus the dividends collected as cash that is never invested (no interest, no compounding).")
    params = {}
    if strat.params:
        with st.expander("Strategy parameters", expanded=True):
            for col, p in zip(st.columns(len(strat.params)), strat.params):
                params.update(param_inputs((p,), f"bt_{name}", container=col))
    allocation, rebalance, weights, trim = "slots", None, None, False
    if multi:
        a1, a2, a3, a4 = st.columns(4)
        allocation = a1.selectbox("Capital allocation", list(S.ALLOCATIONS), format_func=lambda a: S.ALLOCATIONS[a][0], key="bt_alloc")
        a1.caption(S.ALLOCATIONS[allocation][1])
        rebalance = a2.selectbox("Re-set weights", list(REBALANCE), format_func=REBALANCE.get, key="bt_reb",
                                 help="Between re-sets the weights drift with prices; every re-set trades back to the allocation and pays costs.")
        trim = a4.checkbox("Leave out short histories", value=True, key="bt_trim",
                           help=f"All assets are tested over the period they ALL have prices, so one recent listing would shorten the "
                                f"test for the rest. This leaves out assets with less than {MIN_HISTORY:.0%} of the longest history.")
        if a3.radio("Asset weights", ["Equal", "Custom"], horizontal=True, key="bt_wmode") == "Custom":
            with st.expander("Asset weights (relative)", expanded=True):
                weights = _weights_inputs(symbols)

    frames = _load_frames(symbols, period)
    if not frames:
        st.error("None of these symbols has prices. Update them in Data Manager or check the tickers.")
        return
    currencies = sorted({currency_of(sym) for sym in frames})
    if len(currencies) > 1:
        st.info(f"These assets are quoted in {', '.join(currencies)}. Each is tested in its own currency and the results are added up "
                "as if they were one, so exchange-rate moves (and the cost of converting) are not part of the result.")
    if trim and len(frames) > 1:
        frames, short = S.drop_short_histories(frames, MIN_HISTORY)
        if short:
            st.info(f"Left out for a short history (under {MIN_HISTORY:.0%} of the longest): "
                    + ", ".join(f"{s} (from {d:%Y-%m-%d})" for s, d in short.items())
                    + ". Untick 'Leave out short histories' to keep them and test over a shorter period instead.")
    cfg = BacktestConfig(initial_capital=capital, commission_bps=commission, slippage_bps=slippage, execution=execution,
                         reinvest_dividends=dividends == "Reinvest")
    try:
        res = S.run_multi_strategy(frames, name, params, cfg, allocation, rebalance,
                                   None if weights is None else weights[list(frames)])
    except ValueError as exc:
        st.error(str(exc))
        return
    n = len(frames)
    pf, bench = res.portfolio, res.benchmark
    m, bm = pf.metrics, bench.metrics
    bench_label = "buy & hold" if n == 1 else "buy & hold (same weights)"

    first = pf.equity.index[0]
    if n > 1 and (first - min(df.index[0] for df in frames.values())).days > 30:
        youngest = max(frames, key=lambda s: frames[s].index[0])
        st.info(f"History is limited to {first:%Y-%m-%d} by {youngest}, the youngest of the {n} assets, "
                f"so that every asset can be held for the whole test.")

    k = st.columns(6)
    k[0].metric("Final equity", f"{m['final_equity']:,.0f}", pct(m["total_return"], signed=True))
    k[1].metric("CAGR", pct(m["cagr"]), f"B&H {pct(bm['cagr'])}", delta_color="off")
    k[2].metric("Sharpe", f"{m['sharpe']:.2f}", f"B&H {bm['sharpe']:.2f}", delta_color="off")
    k[3].metric("Max drawdown", pct(m["max_drawdown"]), f"B&H {pct(bm['max_drawdown'])}", delta_color="off")
    k[4].metric("Invested" if n > 1 else "Time in market", pct(m["exposure"], 0),
                f"{m['avg_positions']:.1f} of {n} assets held on average" if n > 1 else f"{int(res.assets[next(iter(frames))].metrics['n_trades'])} trades",
                delta_color="off")
    k[5].metric("Dividends", f"{m['dividends_received']:,.0f}",
                f"{pct(m['dividend_yield'])} a year · {'reinvested' if cfg.reinvest_dividends else 'paid out as cash'}", delta_color="off")
    if bm["sharpe"] > m["sharpe"] and name != "buy_hold":
        st.warning("This strategy did not beat simply holding "
                   + ("the asset" if n == 1 else "all these assets") + " on a risk-adjusted basis over this period.")

    dd = risk.drawdown_series(pf.returns)["drawdown"]
    title = strat.label if n == 1 else f"{strat.label} x {n} assets"
    show_chart(charts.equity_figure(pf.equity, bench.equity, dd, title, "Buy & hold"))

    if n > 1:
        st.markdown("**What was held**")
        show_chart(charts.weights_figure(pf.weights))
        now = res.weights.iloc[-1]
        now = now[now.abs() > 1e-9].sort_values(ascending=False)
        st.markdown("**Signalled now" + (" (trades at the next open)" if execution == "next_open" else "") + ":** "
                    + (", ".join(f"{s} {w:.0%}" for s, w in now.items()) + (f", cash {1 - now.sum():.0%}" if now.sum() < 0.999 else "")
                       if len(now) else "all cash"))
        st.markdown("**By asset**")
        show_table(_asset_table(res).style.format(
            {"contribution": "{:+.1%}", "dividends": "{:,.0f}", "avg weight": "{:.0%}", "time in market": "{:.0%}",
             "alone": "{:+.1%}", "buy & hold": "{:+.1%}", "win rate": "{:.0%}"}, na_rep="-"))
        st.caption(f"Contribution = what the asset added, dividends included, as % of starting capital (they sum to the portfolio's gain "
                   f"before costs; costs took {m['total_costs'] * 100:.2f}% of capital, turnover {m['turnover_per_year']:.1f}x per year, "
                   f"{int(m['rebalances'])} re-sets). 'Alone' = the same signal on that asset with all the capital.")
    else:
        sym, trades = next(iter(frames)), res.assets[next(iter(frames))].trades
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Performance**")
            show_table(_comparison(m, bm, bench_label))
            st.caption(f"Costs paid: {m['total_costs'] * 100:.2f}% of capital · turnover {m['turnover_per_year']:.1f}x per year")
        with right:
            st.markdown("**Trades**")
            if trades.empty:
                st.caption("No trades.")
            else:
                am = res.assets[sym].metrics
                st.caption(f"Win rate {pct(am['win_rate'], 0)} · avg trade {pct(am['avg_trade'])} · profit factor {am['profit_factor']:.2f} · "
                           f"avg {am['avg_bars_held']:.0f} bars" + ("" if cfg.reinvest_dividends else " · trade returns exclude dividends"))
                show_table(trades.assign(**{"return": trades["return"] * 100}).iloc[::-1].head(100)
                           .style.format({"return": "{:+.1f}%", "entry_price": "{:,.2f}", "exit_price": "{:,.2f}"}), hide_index=True)

    paid = pf.dividends.sum(axis=1)
    with st.expander("Dividends received", expanded=bool(paid.sum() > 0)):
        if paid.sum() <= 0:
            st.caption("No dividends were paid on the shares held (or these assets pay none).")
        else:
            show_chart(charts.bar_figure(paid.groupby(paid.index.year).sum(), title="Dividends received per year", horizontal=False,
                                         fmt=",.0f", sort=False, height=300))
            st.caption(f"{paid.sum():,.0f} over the test, {pct(m['dividend_yield'])} a year of the average equity. " + (
                "Each one bought more shares on its ex-date, so it also appears in the equity curve and compounds."
                if cfg.reinvest_dividends else
                "Each one was paid out to cash that is not invested again (no interest), so it does not compound; "
                "Final equity = shares + that cash.") + " Tax and withholding are not modelled.")

    with st.expander("Calendar returns of the strategy"):
        show_table(heat((risk.monthly_table(pf.returns) * 100).style.format("{:+.1f}", na_rep=""), vmin=-8, vmax=8))

    with st.expander("Parameter sweep (in-sample vs out-of-sample)"):
        if n > 1:
            st.caption("The sweep runs on one symbol at a time: enter a single symbol to use it.")
            return
        if not strat.params:
            st.caption("This strategy has no parameters.")
            return
        df = next(iter(frames.values()))
        picked = st.multiselect("Parameters to sweep", [p.name for p in strat.params], default=[strat.params[0].name][:1], max_selections=2)
        grid = {}
        for p in strat.params:
            if p.name in picked:
                a, b, c = st.columns(3)
                lo = a.number_input(f"{p.name} from", float(p.min), float(p.max), float(p.default) * 0.5 if p.default * 0.5 >= p.min else float(p.min), key=f"sw_lo_{p.name}")
                hi = b.number_input(f"{p.name} to", float(p.min), float(p.max), min(float(p.default) * 1.5, float(p.max)), key=f"sw_hi_{p.name}")
                pts = int(c.number_input(f"{p.name} points", 2, 8, 5, key=f"sw_n_{p.name}"))
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
                           "A big drop is the price of picking parameters on the same data you evaluate on. "
                           "The sweep always ranks total return with dividends reinvested.")
                if len(grid) == 2:
                    a, b = list(grid)
                    pivot = out.pivot(index=a, columns=b, values="sharpe_out")
                    show_chart(charts.heatmap_figure(pivot, title="Out-of-sample Sharpe", zmin=-1, zmax=2, colorscale="RdYlGn", fmt=".2f"))


def _count_slider(label: str, available: int, key: str, default: int, limit: int) -> int:
    """How many symbols to take out of ``available``: a slider (2..limit), or all of them when there are barely any."""
    top = min(available, limit)
    if top <= 2:
        return top
    return st.slider(label, 2, top, min(default, top), key=key)


def _stock_pool_symbols(prefix: str, limit: int) -> list[str]:
    pools = stock_pools.load_pools()
    c1, c2 = st.columns([2, 1])
    key = c1.selectbox("Index", list(pools), format_func=lambda k: f"{pools[k].label} ({len(pools[k])} stocks)", key=f"{prefix}_pool")
    pool = pools[key]
    n = _count_slider("Use the largest N", len(pool), f"{prefix}_pool_n_{key}", 10, limit)
    symbols = stock_pools.largest(pool.symbols, n)
    c2.caption(f"{pool.description}. {pool.source}, {pool.as_of}: a snapshot, indexes change every few months. "
               "Largest = by market cap in the packaged US list (Thai stocks: in list order). **Today's members only**: stocks "
               "that left the index along the way are missing, so a backtest over them flatters the past.")
    return symbols


def _us_market_cap_symbols(prefix: str, limit: int) -> list[str]:
    counts = stock_pools.industries()
    c1, c2 = st.columns([2, 1])
    industry = c1.selectbox("Industry", [None, *counts], format_func=lambda i: f"All industries ({sum(counts.values()):,} stocks)" if i is None
                            else f"{i} ({counts[i]})", key=f"{prefix}_ind")
    available = sum(counts.values()) if industry is None else counts[industry]
    n = _count_slider("Use the largest N", available, f"{prefix}_mcap_n_{industry}", 20, limit)
    c2.caption("US-listed stocks ranked by market cap in the packaged list (a snapshot; SPAC shells left out). **Ranked by "
               "today's size**, so the past looks better than it was: the winners are already in the list.")
    return stock_pools.us_by_market_cap(n, industry)


def _universe_picker(prefix: str, *, default: str, custom: str, limit: int = MAX_ASSETS) -> list[str]:
    """Symbols from a preset, an index, the biggest US stocks, a watchlist or a typed list (keys ``{prefix}_src`` ...)."""
    src = st.radio("Universe", UNIVERSES, index=UNIVERSES.index(default), horizontal=True, key=f"{prefix}_src")
    if src == "Cross-asset presets":
        symbols = [s for s in uni.cross_asset_universe()["symbol"] if s not in ("^VIX", "^TNX")]
    elif src in ("US sector ETFs", "Asset classes"):
        symbols = list(sectors.UNIVERSES["us_sectors" if src == "US sector ETFs" else "asset_classes"].members)
    elif src == "Stock index":
        symbols = _stock_pool_symbols(prefix, limit)
    elif src == "US by market cap":
        symbols = _us_market_cap_symbols(prefix, limit)
    elif src == "Watchlist":
        sv = services()
        names = sv.watchlists.names()
        symbols = sv.watchlists.symbols(st.selectbox("List", names, key=f"{prefix}_list")) if names else []
    else:
        typed = st.text_input("Symbols (comma separated)", custom, key=f"{prefix}_custom")
        return list(dict.fromkeys(s.strip().upper() for s in typed.split(",") if s.strip()))
    if src in ("Stock index", "US by market cap", "Watchlist") and symbols:
        have = ui_data.stored_count(symbols)
        if have < len(symbols):
            st.caption(f"{have} of {len(symbols)} are already in your price store; the other {len(symbols) - have} are downloaded "
                       "when the test runs (the first run takes a while).")
    return symbols


def _rotation_tab() -> None:
    symbols = _universe_picker("rot", default="Cross-asset presets", custom="SPY, QQQ, TLT, GLD, EEM, IWM, VNQ")
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
        ui_data.prefetch(symbols)
        close = ui_data.close_matrix(tuple(symbols), period, live=False).dropna(how="all")
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
               "Prices are split-adjusted; dividends are reinvested or taken as cash, as you choose. Results are history, not a forecast.")
    tab_strategy, tab_rot = st.tabs(["Strategy (one or many assets)", "Momentum rotation"])
    with tab_strategy:
        _strategy_tab()
    with tab_rot:
        _rotation_tab()
