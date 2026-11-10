"""Data Manager: fill and refresh the local store, features and fundamentals."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import streamlit as st

from stockanalyzer.data import universe as uni
from stockanalyzer.pipelines.nightly import PipelineOptions, resolve_symbols
from stockanalyzer.screener.features import compute_universe_features
from stockanalyzer.ui import data as ui_data
from stockanalyzer.ui import loader
from stockanalyzer.ui.context import services
from stockanalyzer.ui.widgets import show_table

UI_LIMIT = 1500  # symbols a "largest US companies" run may ask for; the whole market has its own mode
BACKGROUND_ABOVE = 600  # a download this big takes minutes: run it as a background job instead of inside the page
MODES = ["Largest US companies", f"{loader.WHOLE_MARKET} (all US + all Thai stocks)", "Only watchlists & extras"]
THAI = {"None": "none", "SET50 + sector baskets": "core", "Every share on the SET and mai": "all"}


def _progress_cb(bar):
    def cb(done, total, message):
        bar.progress(min(done / max(total, 1), 1.0), text=f"{done}/{total} · {message}")
    return cb


def _update_tab() -> None:
    sv = services()
    st.markdown("Choose what to keep up to date. Only symbols behind the latest completed session are downloaded, "
                "split/dividend revisions are detected, and symbols Yahoo has no data for are remembered for 7 days.")
    running = loader.job_panel()
    mode = st.radio("Stocks", MODES, horizontal=True, key="dm_mode")
    whole = mode == MODES[1]
    limit = None
    if mode == MODES[0]:
        limit = st.slider("How many (by market cap)", 25, UI_LIMIT, 100, 25, key="dm_limit")
    if whole:
        cov = loader.coverage()
        st.caption(f"{cov.us_listed:,} US and {cov.th_listed:,} Thai stocks are listed ({cov.no_data:,} of them have no prices on Yahoo); "
                   f"{cov.stored:,} are stored now. The first load takes "
                   "10-15 minutes and needs roughly 100 MB per 1,000 stocks of disk; after that a daily run only adds the new days. "
                   "It runs in the background, and if Yahoo throttles it you can start it again - what is stored is kept.")
    c1, c2, c3, c4, c5 = st.columns(5)
    cross = c1.checkbox("Cross-asset presets", value=True, key="dm_cross",
                        help="Indices, commodities, FX (incl. USD/THB), crypto, key ETFs and the SET50 ETF used as the Thai benchmark")
    sect = c2.checkbox("Sector rotation set", value=True, key="dm_sectors",
                       help="US sector ETFs, asset-class ETFs and the stocks of the Thai sector baskets (~75 symbols)")
    chosen_thai = c3.selectbox("Thai stocks", list(THAI), index=1, key="dm_thai", disabled=whole,
                               help="The SET50 and the sector baskets are ~60 stocks; every share is ~900.")
    thai = "all" if whole else THAI[chosen_thai]
    macro = c4.checkbox("FRED macro series", value=False, key="dm_macro")
    wl = c5.checkbox("Watchlist symbols", value=True, key="dm_wl")
    extra = st.text_input("Extra symbols (comma separated)", "", key="dm_extra")
    start = st.date_input("History start for new symbols", pd.Timestamp.today() - pd.DateOffset(years=10), key="dm_start")
    force = st.checkbox("Force re-download", value=False, key="dm_force")
    recompute = st.checkbox("Recompute screener features afterwards", value=True, key="dm_feat")

    opts = PipelineOptions(universe="none" if mode == MODES[2] else "us", limit=limit,
                           symbols=[s.strip().upper() for s in extra.split(",") if s.strip()],
                           include_cross_assets=cross, include_sectors=sect, thai=thai, include_macro=macro,
                           include_watchlists=wl, market_snapshot=whole, start=str(start), force=force)
    symbols = resolve_symbols(opts, sv.md)
    background = len(symbols) > BACKGROUND_ABOVE
    if background:  # a long download is the kind Yahoo throttles: let the job wait and carry on instead of stopping
        opts = replace(opts, rate_limit_waits=loader.PATIENCE)
    st.caption(f"{len(symbols):,} symbols selected" + (" - this runs in the background." if background else "."))
    if st.button("Update now", type="primary", disabled=not symbols or running, key="dm_run"):
        if background:
            loader.start(opts, mode.split(" (")[0], fundamentals=0)
            st.rerun()
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
    snaps = sv.fundamentals.snapshots_frame()
    full = int((snaps["fundamentals"] == "full").sum()) if not snaps.empty else 0
    st.markdown(f"**{len(snaps):,}** stocks have fundamentals cached: **{full:,}** full (refreshed after 7 days: P/E, ROE, margins, "
                f"free cash flow ...) and **{len(snaps) - full:,}** light (P/E, P/B, market cap and yield only).")
    st.markdown("**Whole market, in seconds.** One request returns 250 stocks, so P/E, P/B, market cap and dividend yield of every "
                "US and Thai stock take about a minute. It never replaces a full snapshot.")
    c1, c2 = st.columns(2)
    markets = c1.multiselect("Markets", ["us", "th"], default=["us", "th"], format_func={"us": "United States", "th": "Thailand"}.get,
                             key="dm_snap_markets")
    if c2.button("Fetch P/E, P/B, market cap for the whole market", disabled=not markets, key="dm_snapshot"):
        known = set(uni.load_us_stocks()["symbol"]) | set(uni.load_th_stocks()["symbol"]) | set(sv.md.store.symbols())
        with st.spinner("Asking Yahoo for every listed stock ..."):
            for market in markets:
                try:
                    st.success(f"{'United States' if market == 'us' else 'Thailand'}: "
                               f"{sv.fundamentals.refresh_market(market, only=known).summary()}")
                except Exception as exc:
                    st.error(f"{market}: {exc}")
        ui_data.store_overview.clear()
    st.divider()
    st.markdown("**Full snapshots** - one Yahoo request per stock (measured at ~0.04 s each with 4 in parallel: 400 stocks took 15 s): needed for ROE, margins and "
                "free cash flow screens. For a shortlist, use *Get full fundamentals for these matches* on the Screener.")
    stored = set(sv.md.store.symbols())
    stocks = uni.load_us_stocks().sort_values("market_cap", ascending=False)
    candidates = [s for s in stocks["symbol"] if s in stored]
    thai = sorted(s for s in stored if s.endswith(".BK") and s in set(uni.load_th_stocks()["symbol"]))
    n = st.slider("Refresh up to N US symbols (largest first)", 10, 2000, 100, 10, key="dm_fn")
    with_thai = st.checkbox(f"Also the {len(thai)} Thai stocks in the store", value=bool(thai) and len(thai) <= 150,
                            disabled=not thai, key="dm_fund_th",
                            help="Thai stocks have no size ranking here, so all of them are refreshed rather than cut off by the slider.")
    if st.button("Refresh full fundamentals", disabled=not (candidates or thai), key="dm_fund"):
        bar = st.progress(0.0, text="starting ...")
        us_report = sv.fundamentals.refresh(candidates, limit=n, workers=4, progress=_progress_cb(bar))
        report = us_report
        if with_thai:
            report = sv.fundamentals.refresh(thai, workers=4, progress=_progress_cb(bar))
            st.caption(f"US: {us_report.summary()}")
        bar.empty()
        st.success(report.summary() if not with_thai else f"Thai: {report.summary()}")
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
