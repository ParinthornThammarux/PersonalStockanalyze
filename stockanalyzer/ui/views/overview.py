"""Stock Overview: quote, chart, trend/signals, performance and risk for one symbol."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from stockanalyzer.analysis.risk import metrics as risk
from stockanalyzer.analysis.technical import indicators as ind
from stockanalyzer.analysis.technical import signals
from stockanalyzer.analysis.fundamental.valuation import snapshot_from_info
from stockanalyzer.domain.assets import AssetClass, infer_asset_class
from stockanalyzer.ui import charts
from stockanalyzer.ui import data as ui_data
from stockanalyzer.ui.context import services
from stockanalyzer.ui.format import heat, money, pct
from stockanalyzer.ui.widgets import period_and_interval, show_chart, show_table, symbol_picker

HORIZONS = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260}


def _quote_row(df: pd.DataFrame, daily: pd.DataFrame, symbol: str) -> None:
    last, prev = float(df["close"].iloc[-1]), float(df["close"].iloc[-2]) if len(df) > 1 else float("nan")
    year = daily.tail(252)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Last", f"{last:,.2f}", f"{(last / prev - 1) * 100:+.2f}%" if prev == prev else None)
    c2.metric("52w high", f"{year['high'].max():,.2f}", f"{(last / year['high'].max() - 1) * 100:.1f}%")
    c3.metric("52w low", f"{year['low'].min():,.2f}", f"{(last / year['low'].min() - 1) * 100:+.1f}%")
    if infer_asset_class(symbol) is AssetClass.EQUITY:
        snap = snapshot_from_info(ui_data.info(symbol))
        c4.metric("Market cap", money(snap.get("market_cap")))
        c5.metric("P/E (ttm)", f"{snap['pe']:.1f}" if snap.get("pe") else "-")
        if snap.get("sector"):
            st.caption(f"{snap.get('name') or symbol} · {snap['sector']} · {snap.get('industry', '')}")
    else:
        c4.metric("Asset class", infer_asset_class(symbol).value.title())
        c5.metric("Bars loaded", f"{len(df):,}")


def render() -> None:
    st.title("🔎 Stock Overview")
    symbol = symbol_picker()
    period, interval = period_and_interval()
    if not symbol:
        st.info("Enter a symbol in the sidebar.")
        return

    df = ui_data.require_prices(symbol, period, interval)
    daily = df if interval == "1d" else ui_data.require_prices(symbol, "2y", "1d")
    sv = services()

    _quote_row(df, daily, symbol)

    close = df["close"]
    overlays = [(f"SMA {n}", ind.sma(close, n)) for n in (20, 50, 200) if len(df) > n]
    skip = not infer_asset_class(symbol).trades_weekends
    show_chart(charts.price_figure(df, title=f"{symbol} ({interval})", overlays=overlays, skip_weekends=skip))

    tab_trend, tab_perf, tab_data = st.tabs(["Trend & signals", "Performance & risk", "Data"])

    with tab_trend:
        if len(daily) < 60:
            st.info("Need at least 60 daily bars for trend analysis.")
        else:
            t = signals.trend_state(daily)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trend", t["label"], f"score {t['score']:+d}")
            c2.metric("ADX(14)", f"{t['adx']:.0f}" if t["adx"] else "-", "trending" if t["trending"] else "weak trend")
            c3.metric("60-day slope (annualised)", f"{t['slope_annual_pct']:+.0f}%")
            c4.metric("Trend fit R²", f"{t['r2']:.2f}")
            st.caption(" · ".join(t["notes"]))
            st.markdown("**Recent signals** (last 250 bars)")
            log = signals.event_log(daily)
            show_table(log.head(15), hide_index=True)

    with tab_perf:
        full = ui_data.prices(symbol, "max", "1d")
        if len(full) < 30:
            st.info("Not enough history.")
        else:
            last = full["close"].iloc[-1]
            rows = {k: last / full["close"].iloc[-n - 1] - 1 for k, n in HORIZONS.items() if len(full) > n}
            ytd = full.loc[full.index >= pd.Timestamp(full.index[-1].year, 1, 1), "close"]
            rows = {**{"YTD": last / ytd.iloc[0] - 1 if len(ytd) else np.nan}, **rows}
            st.markdown("**Returns**")
            st.dataframe(pd.DataFrame([{k: v * 100 for k, v in rows.items()}], index=["%"]).style.format("{:+.1f}%"),
                         width="stretch")
            rets = risk.to_returns(full["close"])
            bench = None
            if symbol != "SPY" and infer_asset_class(symbol) in (AssetClass.EQUITY, AssetClass.ETF):
                spy = ui_data.prices("SPY", "max", "1d")
                bench = risk.to_returns(spy["close"]) if not spy.empty else None
            s = risk.summary(rets.tail(252), bench.tail(252) if bench is not None else None)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Volatility (1y)", pct(s["volatility"]))
            c2.metric("Sharpe (1y, rf=0)", f"{s['sharpe']:.2f}")
            c3.metric("Max drawdown (1y)", pct(s["max_drawdown"]))
            c4.metric("VaR 95% (1 day)", pct(s["var_95"]))
            c5.metric("Beta vs SPY", f"{s['vs_benchmark_beta']:.2f}" if "vs_benchmark_beta" in s else "-")
            dd = risk.drawdown_series(rets)
            fig = charts.line_figure(pd.DataFrame({"Drawdown %": dd["drawdown"] * 100}), title="Drawdown from peak (full history)", height=260)
            show_chart(fig)
            with st.expander("Calendar returns"):
                table = risk.monthly_table(rets) * 100
                show_table(heat(table.tail(10).style.format("{:+.1f}", na_rep=""), vmin=-10, vmax=10))

    with tab_data:
        show_table(df.tail(250).iloc[::-1])
        st.download_button("Download CSV", df.to_csv().encode(), file_name=f"{symbol}_{interval}.csv", mime="text/csv")

    st.divider()
    lists = sv.watchlists.names() or ["Favorites"]
    a, b = st.columns([2, 1])
    target = a.selectbox("Add to watchlist", lists, key="ov_wl")
    if b.button(f"Add {symbol}", key="ov_add", width="stretch"):
        sv.watchlists.add(target, symbol)
        st.toast(f"{symbol} added to {target}")
