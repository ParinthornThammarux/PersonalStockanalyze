"""Support/Resistance zones: ATR-adaptive zones from confirmed pivots (ported from Custom Analysis)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from stockanalyzer.analysis.technical import indicators as ind
from stockanalyzer.analysis.technical import sr_zones
from stockanalyzer.domain.assets import infer_asset_class
from stockanalyzer.ui import charts
from stockanalyzer.ui import data as ui_data
from stockanalyzer.ui.widgets import period_and_interval, show_chart, show_table, symbol_picker


def render() -> None:
    st.title("🧱 Support / Resistance Zones")
    st.caption("Pivots confirmed by N bars on each side are clustered into zones no taller than the ATR "
               "multiple you choose. Zones are drawn from their first pivot to the latest bar and recoloured "
               "by where price is now: above = support, below = resistance.")
    symbol = symbol_picker()
    period, interval = period_and_interval(default_period="1y")

    st.sidebar.markdown("### Zone logic")
    lookback = st.sidebar.slider("Bars to analyse", 100, 1200, 300, step=50)
    strength = st.sidebar.slider("Pivot strength (bars each side)", 2, 12, 4)
    min_touches = st.sidebar.slider("Min pivots per zone", 2, 8, 3)
    atr_period = st.sidebar.slider("ATR period", 5, 50, 14)
    atr_mult = st.sidebar.slider("Zone height (x ATR)", 0.3, 3.0, 1.0, step=0.1)
    max_zones = st.sidebar.slider("Max zones per side", 1, 10, 6)
    use_hl = st.sidebar.checkbox("Use high/low for pivots", value=True)
    show_ma = st.sidebar.checkbox("Moving averages (20/50/200)", value=True)
    show_fib = st.sidebar.checkbox("Fibonacci retracement", value=False)
    with st.sidebar.expander("Colours"):
        support = st.color_picker("Support", "#3B82F6")
        resistance = st.color_picker("Resistance", "#EF4444")
        neutral = st.color_picker("In zone", "#A3A3A3")
        alpha = st.slider("Fill opacity", 0.05, 0.6, 0.2, step=0.05)

    if not symbol:
        st.info("Enter a symbol in the sidebar.")
        return
    df = ui_data.require_prices(symbol, period, interval)
    if len(df) < 60:
        st.warning(f"Only {len(df)} bars - choose a longer period or finer bars.")
        return

    zones = sr_zones.build_zones(df, lookback=lookback, strength=strength, atr_period=atr_period, atr_mult=atr_mult,
                                 min_touches=min_touches, use_hl=use_hl, max_per_side=max_zones)
    overlays = [(f"MA{n}", ind.sma(df["close"], n)) for n in (20, 50, 200) if show_ma and len(df) > n]
    fib = sr_zones.fibonacci_levels(df, min(300, len(df))) if show_fib else None
    fig = charts.price_figure(df, title=f"{symbol} - zones ({interval})", overlays=overlays, volume=False, zones=zones,
                              fib=fib, zone_colors=(support, resistance, neutral), zone_alpha=alpha, height=650,
                              skip_weekends=not infer_asset_class(symbol).trades_weekends)
    show_chart(fig)

    last = float(df["close"].iloc[-1])
    atr = float(ind.atr(df["high"], df["low"], df["close"], atr_period).iloc[-1])
    near = sr_zones.nearest_levels(zones, last)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last", f"{last:,.2f}")
    c2.metric(f"ATR({atr_period})", f"{atr:,.2f}", f"{atr / last * 100:.1f}% of price", delta_color="off")
    sup, res = near["support"], near["resistance"]
    c3.metric("Nearest support", f"{sup.high:,.2f}" if sup else "-", f"-{near['support_dist_pct']:.1f}%" if sup else None, delta_color="off")
    c4.metric("Nearest resistance", f"{res.low:,.2f}" if res else "-", f"+{near['resistance_dist_pct']:.1f}%" if res else None, delta_color="off")
    if near["inside"] is not None:
        st.info(f"Price is inside a {near['inside'].side} zone ({near['inside'].low:,.2f} - {near['inside'].high:,.2f}).")

    if zones:
        rows = [{"role now": z.role(last), "formed as": z.side, "low": z.low, "high": z.high, "pivots": z.touches,
                 "first": df.index[min(z.first_idx, len(df) - 1)], "last pivot": df.index[min(z.last_idx, len(df) - 1)]}
                for z in sorted(zones, key=lambda z: z.high, reverse=True)]
        show_table(pd.DataFrame(rows), hide_index=True)
    else:
        st.info("No zones met the criteria - try fewer required pivots, a larger zone height or a longer lookback.")
    st.caption("Pivots need `strength` bars after them to be confirmed, so the newest zones lag price by that many bars. "
               "Zones are descriptive, not predictive; test any rule built on them in **Backtest**.")
