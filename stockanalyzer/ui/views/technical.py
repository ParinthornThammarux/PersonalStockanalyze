"""Technical Analysis: any combination of registry indicators on one chart."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from stockanalyzer.analysis.technical import indicators as ind
from stockanalyzer.analysis.technical import patterns, registry, signals
from stockanalyzer.domain.assets import infer_asset_class
from stockanalyzer.ui import charts
from stockanalyzer.ui import data as ui_data
from stockanalyzer.ui.widgets import param_inputs, period_and_interval, show_chart, show_table, symbol_picker

MA_CHOICES = [5, 10, 20, 50, 100, 200]
DEFAULT_PANES = ["rsi", "macd"]


def _is_bar(column: str) -> bool:
    return "hist" in column


def render() -> None:
    st.title("📈 Technical Analysis")
    symbol = symbol_picker()
    period, interval = period_and_interval()

    st.sidebar.markdown("### Overlays")
    sma_periods = st.sidebar.multiselect("SMA periods", MA_CHOICES, default=[50, 200])
    ema_periods = st.sidebar.multiselect("EMA periods", MA_CHOICES, default=[])
    overlay_specs = [s for s in registry.REGISTRY.values() if s.pane == "price" and s.name not in ("sma", "ema")]
    chosen_overlays = st.sidebar.multiselect("Other overlays", [s.name for s in overlay_specs],
                                             format_func=lambda n: registry.get(n).label)
    st.sidebar.markdown("### Panels")
    pane_specs = [s for s in registry.REGISTRY.values() if s.pane == "separate"]
    chosen_panes = st.sidebar.multiselect("Indicators", [s.name for s in pane_specs], default=DEFAULT_PANES,
                                          format_func=lambda n: f"{registry.get(n).label} ({registry.get(n).category})")
    show_volume = st.sidebar.checkbox("Volume", value=True)
    show_patterns = st.sidebar.checkbox("Candlestick patterns", value=False)

    if not symbol:
        st.info("Enter a symbol in the sidebar.")
        return
    df = ui_data.require_prices(symbol, period, interval)
    close = df["close"]

    params: dict[str, dict] = {}
    with st.sidebar.expander("Indicator parameters"):
        for name in [*chosen_overlays, *chosen_panes]:
            spec = registry.get(name)
            if spec.params:
                st.markdown(f"**{spec.label}**")
                params[name] = param_inputs(spec.params, f"tp_{name}", container=st)

    overlays = [(f"SMA {n}", ind.sma(close, n)) for n in sma_periods] + \
               [(f"EMA {n}", ind.ema(close, n)) for n in ema_periods]
    for name in chosen_overlays:
        spec = registry.get(name)
        out = spec.compute(df, **params.get(name, {}))
        overlays += [(c, out[c]) for c in spec.plot_columns(params.get(name))]

    panes = []
    for name in chosen_panes:
        spec = registry.get(name)
        out = spec.compute(df, **params.get(name, {}))
        cols = spec.plot_columns(params.get(name)) if spec.plot else list(out.columns)
        panes.append(charts.Pane(spec.label, [(c, out[c]) for c in cols], spec.levels,
                                 bars=[c for c in cols if _is_bar(c)]))

    markers = {}
    if show_patterns:
        pat = patterns.detect_patterns(df)
        markers = {"Hammer": (pat["hammer"] > 0).astype(int), "Bullish engulfing": (pat["engulfing"] > 0).astype(int),
                   "Shooting star": -(pat["shooting_star"] < 0).astype(int), "Bearish engulfing": -(pat["engulfing"] < 0).astype(int)}

    fig = charts.price_figure(df, title=f"{symbol} ({interval})", overlays=overlays, panes=panes, volume=show_volume,
                              markers=markers, skip_weekends=not infer_asset_class(symbol).trades_weekends)
    show_chart(fig)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Latest indicator values**")
        rows = {}
        for name in [*chosen_overlays, *chosen_panes]:
            out = registry.get(name).compute(df, **params.get(name, {}))
            rows.update({c: float(out[c].iloc[-1]) for c in out.columns})
        if rows:
            show_table(pd.Series(rows, name="value").to_frame().style.format("{:,.3f}"))
        else:
            st.caption("Pick indicators in the sidebar.")
    with c2:
        st.markdown("**Signals (daily bars)**")
        daily = df if interval == "1d" else ui_data.prices(symbol, "2y", "1d")
        if len(daily) >= 60:
            show_table(signals.event_log(daily).head(12), hide_index=True)
        else:
            st.caption("Needs 60+ daily bars.")
