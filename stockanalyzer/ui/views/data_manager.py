"""Data Manager: fill and refresh the local store, features and fundamentals."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from stockanalyzer.data import universe as uni
from stockanalyzer.pipelines.nightly import PipelineOptions, resolve_symbols
from stockanalyzer.screener.features import compute_universe_features
from stockanalyzer.ui import data as ui_data
from stockanalyzer.ui.context import services
from stockanalyzer.ui.widgets import show_table

UI_LIMIT = 1500  # symbols per click; bigger refreshes belong in the CLI / cron


def _progress_cb(bar):
    def cb(done, total, message):
        bar.progress(min(done / max(total, 1), 1.0), text=f"{done}/{total} · {message}")
    return cb


def _update_tab() -> None:
    sv = services()
    st.markdown("Choose what to keep up to date. Only symbols behind the latest completed session are downloaded, "
                "split/dividend revisions are detected, and symbols Yahoo has no data for are remembered for 7 days.")
    mode = st.radio("Stocks", ["Largest US companies", "Only watchlists & extras"], horizontal=True, key="dm_mode")
    limit = None
    if mode == "Largest US companies":
        limit = st.slider("How many (by market cap)", 25, UI_LIMIT, 100, 25, key="dm_limit")
    c1, c2, c3, c4 = st.columns(4)
    cross = c1.checkbox("Cross-asset presets", value=True, key="dm_cross", help="Indices, commodities, FX, crypto, key ETFs")
    sect = c2.checkbox("Sector rotation set", value=True, key="dm_sectors",
                       help="US sector ETFs, asset-class ETFs and the stocks of the Thai sector baskets (~75 symbols)")
    macro = c3.checkbox("FRED macro series", value=False, key="dm_macro")
    wl = c4.checkbox("Watchlist symbols", value=True, key="dm_wl")
    extra = st.text_input("Extra symbols (comma separated)", "", key="dm_extra")
    start = st.date_input("History start for new symbols", pd.Timestamp.today() - pd.DateOffset(years=10), key="dm_start")
    force = st.checkbox("Force re-download", value=False, key="dm_force")
    recompute = st.checkbox("Recompute screener features afterwards", value=True, key="dm_feat")

    opts = PipelineOptions(universe="us" if mode == "Largest US companies" else "none", limit=limit,
                           symbols=[s.strip().upper() for s in extra.split(",") if s.strip()],
                           include_cross_assets=cross, include_sectors=sect, include_macro=macro, include_watchlists=wl)
    symbols = resolve_symbols(opts, sv.md)
    st.caption(f"{len(symbols):,} symbols selected.")
    if st.button("Update now", type="primary", disabled=not symbols, key="dm_run"):
        bar = st.progress(0.0, text="starting ...")
        report = sv.md.update(symbols, start=str(start), force=force, progress=_progress_cb(bar))
        bar.empty()
        (st.error if report.aborted else st.success)(report.summary())
        if report.failed:
            show_table(pd.DataFrame(report.failed.items(), columns=["symbol", "error"]).head(50), hide_index=True)
        if report.no_data:
            st.caption("No data: " + ", ".join(report.no_data[:40]) + (" ..." if len(report.no_data) > 40 else ""))
        if recompute:
            with st.spinner("Computing features ..."):
                frame = compute_universe_features(sv.md.store, workers=0)
                if not frame.empty:
                    sv.features.save(frame)
                    st.success(f"Features computed for {len(frame):,} symbols.")
        ui_data.store_overview.clear()
        ui_data.stored_closes.clear()
        ui_data.rotation_data.clear()


def _fundamentals_tab() -> None:
    sv = services()
    cached = len(sv.fundamentals.snapshots_frame())
    st.markdown(f"**{cached:,}** fundamental snapshots cached (refreshed after 7 days). They add P/E, PEG, ROE, margins, "
                "FCF yield ... to the Screener so value and quality screens work.")
    stored = set(sv.md.store.symbols())
    stocks = uni.load_us_stocks().sort_values("market_cap", ascending=False)
    candidates = [s for s in stocks["symbol"] if s in stored]
    n = st.slider("Refresh up to N symbols (largest first)", 10, 500, 100, 10, key="dm_fn")
    st.caption("Each snapshot is one Yahoo request (~0.5 s with 4 workers in parallel). For thousands, use the CLI.")
    if st.button("Refresh fundamentals", disabled=not candidates, key="dm_fund"):
        bar = st.progress(0.0, text="starting ...")
        report = sv.fundamentals.refresh(candidates, limit=n, workers=4, progress=_progress_cb(bar))
        bar.empty()
        st.success(report.summary())
        ui_data.store_overview.clear()


def _status_tab() -> None:
    sv = services()
    ov = ui_data.store_overview()
    k = st.columns(4)
    k[0].metric("Symbols", f"{ov['symbols']:,}")
    k[1].metric("Rows", f"{ov.get('rows', 0):,}")
    k[2].metric("Size", f"{ov['size_mb']:,.1f} MB")
    k[3].metric("Coverage", f"{ov['first_ts']:%Y} - {ov['last_ts']:%Y-%m-%d}" if ov["symbols"] else "-")
    st.caption(f"Data home: `{sv.settings.home}`")
    problems = sv.state.query("SELECT symbol, status, last_attempt, message FROM update_log WHERE status != 'ok' "
                              "ORDER BY last_attempt DESC LIMIT 200")
    st.markdown(f"**Symbols with problems** ({len(problems)})")
    if problems:
        show_table(pd.DataFrame(problems), hide_index=True)
    else:
        st.caption("None.")
    runs = sv.state.query("SELECT id, started_at, finished_at, status FROM pipeline_runs ORDER BY id DESC LIMIT 10")
    if runs:
        st.markdown("**Recent pipeline runs**")
        show_table(pd.DataFrame(runs), hide_index=True)


def render() -> None:
    st.title("🗄️ Data Manager")
    tab_update, tab_fund, tab_status = st.tabs(["Update prices", "Fundamentals", "Status"])
    with tab_update:
        _update_tab()
    with tab_fund:
        _fundamentals_tab()
    with tab_status:
        _status_tab()
    with st.expander("Run it unattended (recommended for the full universe)"):
        st.code(
            "# once: everything in the packaged US list (~6 min first time, seconds afterwards)\n"
            "python -m stockanalyzer nightly --universe us --fundamentals 300 --workers 4\n\n"
            "# crontab -e   (after the US close in your timezone: 05:30 Tue-Sat in Bangkok)\n"
            "30 5 * * 2-6  cd /path/to/PersonalStockanalyze && python -m stockanalyzer nightly --fundamentals 300",
            language="bash")
