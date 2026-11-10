"""Markets & Macro: cross-asset performance, correlations and FRED macro series."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from stockanalyzer.analysis.risk import metrics as risk
from stockanalyzer.data import universe as uni
from stockanalyzer.ui import charts
from stockanalyzer.ui import data as ui_data
from stockanalyzer.ui.context import services
from stockanalyzer.ui.format import heat
from stockanalyzer.ui.views import pairs
from stockanalyzer.ui.widgets import show_chart, show_table

WINDOWS = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504}
DEFAULT_CORR = ["^GSPC", "QQQ", "TLT", "GLD", "GC=F", "CL=F", "DX-Y.NYB", "BTC-USD", "EEM", "VNQ"]


def _performance(closes: pd.DataFrame, names: dict[str, str]) -> pd.DataFrame:
    rows = {}
    for s in closes.columns:
        c = closes[s].dropna()
        if len(c) < 6:
            continue
        ytd = c[c.index >= pd.Timestamp(c.index[-1].year, 1, 1)]
        rows[s] = {"name": names.get(s, s), "last": c.iloc[-1],
                   "1D": c.iloc[-1] / c.iloc[-2] - 1, "1W": c.iloc[-1] / c.iloc[-6] - 1,
                   "1M": c.iloc[-1] / c.iloc[-22] - 1 if len(c) > 21 else float("nan"),
                   "3M": c.iloc[-1] / c.iloc[-64] - 1 if len(c) > 63 else float("nan"),
                   "6M": c.iloc[-1] / c.iloc[-127] - 1 if len(c) > 126 else float("nan"),
                   "YTD": c.iloc[-1] / ytd.iloc[0] - 1 if len(ytd) else float("nan"),
                   "1Y": c.iloc[-1] / c.iloc[-253] - 1 if len(c) > 252 else float("nan")}
    return pd.DataFrame(rows).T


def render() -> None:
    st.title("🌍 Markets & Macro")
    sv = services()
    cross = uni.cross_asset_universe()
    names = dict(zip(cross["symbol"], cross["name"]))
    classes = dict(zip(cross["symbol"], cross["asset_class"]))
    closes = ui_data.stored_closes(tuple(cross["symbol"]), 700)

    if closes.empty or closes.shape[1] < 5:
        st.info("Cross-asset data is not in the local store yet.")
        if st.button("Download indices, commodities, FX, crypto and ETFs", type="primary"):
            with st.spinner("Downloading ..."):
                report = sv.md.update(list(cross["symbol"]))
            st.success(report.summary())
            ui_data.stored_closes.clear()
            st.rerun()
        return

    tab_perf, tab_corr, tab_pairs, tab_macro = st.tabs(["Performance", "Correlations", "Pairs", "Macro (FRED)"])

    with tab_perf:
        perf = _performance(closes, names)
        perf["class"] = perf.index.map(classes)
        for cls in ["index", "rate", "commodity", "fx", "crypto", "etf"]:
            part = perf[perf["class"] == cls].drop(columns="class")
            if part.empty:
                continue
            st.markdown(f"**{cls.upper()}**")
            pcts = ["1D", "1W", "1M", "3M", "6M", "YTD", "1Y"]
            show_table(heat(part.style.format({**{c: "{:+.1%}" for c in pcts}, "last": "{:,.2f}"}, na_rep="-"),
                            subset=pcts, vmin=-0.15, vmax=0.15))
        st.caption(f"As of {closes.index[-1]:%Y-%m-%d}. Stale or missing rows can be refreshed in Data Manager.")
        picks = st.multiselect("Compare (rebased to 100)", list(closes.columns), default=[s for s in ("^GSPC", "GC=F", "BTC-USD", "DX-Y.NYB") if s in closes],
                               format_func=lambda s: f"{s} - {names.get(s, '')}", key="mk_cmp")
        horizon = st.select_slider("Window", list(WINDOWS), value="1Y", key="mk_win")
        if picks:
            show_chart(charts.line_figure(closes[picks].tail(WINDOWS[horizon] + 1), rebase=True))

    with tab_corr:
        picks = st.multiselect("Assets", list(closes.columns), default=[s for s in DEFAULT_CORR if s in closes], key="mk_corr",
                               format_func=lambda s: f"{s} - {names.get(s, '')}")
        c1, c2 = st.columns(2)
        win = c1.select_slider("Window", list(WINDOWS), value="1Y", key="mk_corr_win")
        method = c2.radio("Method", ["pearson", "spearman"], horizontal=True, key="mk_corr_m")
        if len(picks) >= 2:
            rets = risk.to_returns(closes[picks]).tail(WINDOWS[win]).dropna(how="any")
            show_chart(charts.heatmap_figure(risk.correlation_matrix(rets, method), title=f"Daily return correlation ({win})"))
            a, b = st.columns(2)
            first = a.selectbox("Rolling correlation: A", picks, index=0, key="mk_ra")
            second = b.selectbox("B", picks, index=min(1, len(picks) - 1), key="mk_rb")
            if first != second:
                allr = risk.to_returns(closes[[first, second]]).dropna()
                roll = risk.rolling_correlation(allr[first], allr[second], 60).dropna()
                show_chart(charts.line_figure(roll.to_frame("60-day correlation"), height=260))
            st.caption("Correlations move: diversification that works in calm markets often disappears in a sell-off.")

    with tab_pairs:
        pairs.render_panel()

    with tab_macro:
        series = uni.MACRO_SERIES
        stored = {s for s in series if sv.md.store.exists(s)}
        if len(stored) < len(series):
            if st.button("Download macro series from FRED", key="mk_fred"):
                with st.spinner("Downloading ..."):
                    st.success(sv.md.update(list(series), start="2000-01-01").summary())
                ui_data.stored_closes.clear()
                st.rerun()
        if stored:
            picks = st.multiselect("Series", sorted(stored), default=[s for s in ("FRED:DGS10", "FRED:DGS2") if s in stored],
                                   format_func=lambda s: f"{series[s]} ({s.split(':')[1]})", key="mk_macro")
            macro = ui_data.stored_closes(tuple(sorted(stored)), 6000)
            if picks:
                show_chart(charts.line_figure(macro[picks].dropna(how="all").tail(2600), height=380))
            if "FRED:T10Y2Y" in macro:
                spread = macro["FRED:T10Y2Y"].dropna()
                st.metric("10Y-2Y yield spread", f"{spread.iloc[-1]:+.2f} pts", "inverted" if spread.iloc[-1] < 0 else "normal", delta_color="off")
            st.caption("Macro series keep their own release calendar (monthly/daily); they are stored with the same incremental updater as prices.")
