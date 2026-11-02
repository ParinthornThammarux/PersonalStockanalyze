"""Sector Rotation: which sectors gain or lose ground against the market, and how that moves over time."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from stockanalyzer.analysis import rotation as rot
from stockanalyzer.data.sectors import UNIVERSES, SectorData
from stockanalyzer.ui import charts
from stockanalyzer.ui import data as ui_data
from stockanalyzer.ui.context import services
from stockanalyzer.ui.format import heat, pct
from stockanalyzer.ui.widgets import show_chart, show_table

PERIODS = ["1 month", "3 months", "6 months", "1 year", "YTD", "Custom"]
FREQUENCIES = {"Month": "ME", "Quarter": "QE", "Week": "W-FRI"}
HEAT_LIMIT = {"ME": 10, "QE": 15, "W-FRI": 5}  # % that saturates the colour scale
HISTORY = {"1 year": 1, "3 years": 3, "5 years": 5, "All": None}


def period_bounds(label: str, end: pd.Timestamp) -> pd.Timestamp:
    """Start of the analysed window for a named period ending at ``end``."""
    if label == "YTD":
        return pd.Timestamp(end.year - 1, 12, 31)
    months = {"1 month": 1, "3 months": 3, "6 months": 6, "1 year": 12}[label]
    return end - pd.DateOffset(months=months)


def period_label(ts: pd.Timestamp, freq: str) -> str:
    if freq == "QE":
        return f"Q{ts.quarter} {ts.year}"
    return ts.strftime("%d %b %y") if freq == "W-FRI" else ts.strftime("%b %Y")


def rank_move(change: float) -> str:
    return "-" if pd.isna(change) or change == 0 else f"{'▲' if change > 0 else '▼'} {abs(change):.0f}"


def missing_data(universe) -> None:
    sv = services()
    st.info(f"No stored prices for **{universe.label}** yet. This downloads {len(universe.symbols)} symbols "
            f"(10 years of daily bars, a few seconds); the nightly job keeps them fresh afterwards.")
    if st.button(f"Download {universe.label}", type="primary", key="rot_download"):
        with st.spinner("Downloading ..."):
            report = sv.md.update(universe.symbols)
        ui_data.rotation_data.clear()
        ui_data.stored_closes.clear()
        if report.aborted:
            st.error(report.summary())
            st.stop()
        if report.failed:
            st.warning(f"{report.summary()} - try again in a few minutes.")
            st.stop()
        st.rerun()


def _regime_at(od: pd.DataFrame, end: pd.Timestamp) -> str:
    known = od.loc[:end, "regime"]
    known = known[known != ""]
    return known.iloc[-1] if len(known) else ""


def _overview_tab(data: SectorData, sb: pd.DataFrame, quads: pd.DataFrame, od: pd.DataFrame,
                  start: pd.Timestamp, end: pd.Timestamp) -> None:
    bench = data.universe.benchmark_name
    bench_ret = rot.window_returns(data.benchmark, start, end)
    leader, laggard = sb.index[0], sb.index[-1]
    k = st.columns(4)
    k[0].metric("Leader", leader, pct(sb.loc[leader, "ret"], signed=True), delta_color="off")
    k[1].metric("Laggard", laggard, pct(sb.loc[laggard, "ret"], signed=True), delta_color="off")
    k[2].metric(f"Beating {bench}", f"{int((sb['rel'] > 0).sum())} of {len(sb)}", f"{bench} {pct(bench_ret, signed=True)}",
                delta_color="off")
    regime = _regime_at(od, end)
    k[3].metric("Risk appetite", regime or "-", help="Offensive vs defensive sectors against their own average - see the "
                "Risk appetite tab.")

    left, right = st.columns([3, 2])
    with left:
        show_chart(charts.bar_figure(sb["rel"], title=f"Return relative to {bench} ({start:%d %b %Y} - {end:%d %b %Y})",
                                     fmt="+.1%"))
    with right:
        moved = sb.dropna(subset=["rank_change"]).sort_values("rank_change", ascending=False)
        gaining, losing = moved[moved["rank_change"] > 0].head(3), moved[moved["rank_change"] < 0].tail(3)[::-1]
        st.markdown("**▲ Gaining ground** (rank vs the window before)")
        for name, r in gaining.iterrows():
            st.markdown(f"- **{name}** #{r['prev_rank']:.0f} → #{r['rank']:.0f} · {pct(r['ret'], signed=True)}")
        if gaining.empty:
            st.caption("No sector climbed.")
        st.markdown("**▼ Losing ground**")
        for name, r in losing.iterrows():
            st.markdown(f"- **{name}** #{r['prev_rank']:.0f} → #{r['rank']:.0f} · {pct(r['ret'], signed=True)}")
        if losing.empty:
            st.caption("No sector fell.")
        if moved.empty:
            st.caption("The window before this one is outside the stored history.")

    members = {name: ", ".join(m.removesuffix(".BK") for m in group) for name, group in data.universe.sectors.items()}
    latest = quads.iloc[-1].replace("", pd.NA) if len(quads) else pd.Series(dtype=object)
    view = pd.DataFrame({
        "Return": sb["ret"], "vs benchmark": sb["rel"], "Window before": sb["prev_ret"], "Rank": sb["rank"],
        "Was": sb["prev_rank"], "Moved": [rank_move(c) for c in sb["rank_change"]],
        "Money flow": sb["flow"], "Flow before": sb["prev_flow"],
        "Quadrant": latest.reindex(sb.index), "Members": [members[n] for n in sb.index],
    }, index=sb.index)
    styled = view.style.format({"Return": "{:+.1%}", "vs benchmark": "{:+.1%}", "Window before": "{:+.1%}", "Rank": "{:.0f}",
                                "Was": "{:.0f}", "Money flow": "{:+.2f}", "Flow before": "{:+.2f}"}, na_rep="-")
    show_table(heat(heat(styled, subset=["vs benchmark"], vmin=-0.1, vmax=0.1), subset=["Money flow", "Flow before"],
                    vmin=-0.2, vmax=0.2))
    st.caption("**Money flow** is Chaikin money flow over the window: volume-weighted buying vs selling pressure, from +1 "
               "(every bar closed at its high on volume) to -1. It is a proxy - Yahoo has no historical fund flows, so "
               "creations and redemptions of an ETF or a stock's institutional buying are not counted. "
               "**Quadrant**: see the Rotation map tab. Past rotation is not a forecast.")
    if any(have < total for have, total in data.coverage.values()):
        short = ", ".join(f"{n} ({h}/{t})" for n, (h, t) in data.coverage.items() if h < t)
        st.caption(f"Some symbols are missing from the local store, so these sectors are partial or absent: {short}. "
                   "Fetch them in Data Manager.")


def _map_tab(data: SectorData, ratio: pd.DataFrame, mom: pd.DataFrame, quads: pd.DataFrame,
             start: pd.Timestamp, end: pd.Timestamp) -> None:
    trail = st.slider("Trail (weeks shown)", 0, 26, 8, key="rot_trail")
    ratio, mom, quads = ratio.loc[:end], mom.loc[:end], quads.loc[:end]
    if ratio.empty:
        st.info("Not enough history for the chosen settings.")
        return
    show_chart(charts.rrg_figure(ratio, mom, trail=trail))
    st.caption("Each line is a sector's path over the last weeks, ending at the large dot. **RS-Ratio** > 100: stronger than "
               f"its own recent norm against {data.universe.benchmark_name}. **RS-Momentum** > 100: that strength is still "
               "improving. A sector normally travels clockwise - Improving → Leading → Weakening → Lagging - so a dot "
               "moving from Weakening toward Lagging is where money is leaving, and one rising out of Lagging into "
               "Improving is where it starts to return.")

    last = quads.iloc[-1].replace("", pd.NA).dropna()
    now = pd.DataFrame({"Quadrant": last, "RS-Ratio": ratio.iloc[-1].reindex(last.index),
                        "RS-Momentum": mom.iloc[-1].reindex(last.index), "Weeks in it": rot.quadrant_age(quads).reindex(last.index)})
    now = now.sort_values(["Quadrant", "RS-Ratio"], key=lambda c: c.map({q: i for i, q in enumerate(rot.QUADRANTS)}) if c.name == "Quadrant" else -c)
    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Where each sector is now**")
        show_table(now.style.format({"RS-Ratio": "{:.1f}", "RS-Momentum": "{:.1f}", "Weeks in it": "{:.0f}"}, na_rep="-"))
    with right:
        changes = rot.quadrant_changes(quads, since=start)
        st.markdown(f"**Rotations since {start:%d %b %Y}** ({len(changes)})")
        if changes.empty:
            st.caption("No sector changed quadrant in this period.")
        else:
            show_table(changes.assign(date=changes["date"].dt.date).rename(columns={"date": "Week ending"}), hide_index=True)


def _timeline_tab(data: SectorData, sb: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> None:
    c1, c2, c3 = st.columns([2, 2, 1])
    freq = FREQUENCIES[c1.radio("Period length", list(FREQUENCIES), horizontal=True, key="rot_freq")]
    shown = c2.slider("Periods shown", 4, 36, 12, key="rot_periods")
    relative = c3.toggle("vs benchmark", value=False, key="rot_relative",
                         help="Subtract the benchmark's return in each period instead of showing raw returns.")
    close = data.close.loc[:end]
    pr = rot.periodic_returns(close, freq)
    if relative:
        bpr = rot.periodic_returns(data.benchmark.loc[:end].to_frame("b"), freq)["b"].reindex(pr.index)
        pr = (1 + pr).div(1 + bpr, axis=0) - 1
    pr = pr.tail(shown)
    if pr.empty:
        st.info("Not enough history.")
        return

    limit = HEAT_LIMIT[freq]
    table = (pr * 100).iloc[::-1]
    table.index = [period_label(d, freq) for d in table.index]
    show_chart(charts.heatmap_figure(table, title="Return by period (newest first)", zmin=-limit, zmax=limit,
                                     colorscale="RdYlGn", fmt=".1f", height=max(320, 30 * len(table) + 120)))
    st.caption("The newest period may still be in progress. Read across a row to see who led that period, down a column "
               "to see how one sector's fortunes changed.")

    st.markdown("**Who led, and where the lead moved**")
    lead = rot.leaders(pr).iloc[::-1]
    lead.index = [period_label(d, freq) for d in lead.index]
    show_table(lead)

    ranks = rot.rank_history(pr)
    names = list(ranks.columns)
    pick = st.multiselect("Follow sectors through the ranking", names, default=list(sb.index[:2]), key="rot_follow")
    show_chart(charts.bump_figure(ranks, highlight=pick))

    rel = data.close.loc[start:end].div(data.benchmark.loc[start:end], axis=0)
    show = st.multiselect("Relative strength vs benchmark over the selected period", names,
                          default=[*sb.index[:3], sb.index[-1]], key="rot_rel_lines")
    if show:
        show_chart(charts.line_figure(rel[show], rebase=True, title=f"Relative strength, 100 = {start:%d %b %Y} "
                                      "(rising = gaining on the benchmark)"))


def _appetite_tab(data: SectorData, od: pd.DataFrame, end: pd.Timestamp, window: int, band: float) -> None:
    u = data.universe
    if od.empty:
        st.info("This universe has no offensive / defensive split.")
        return
    od = od.loc[:end]
    cur = od.iloc[-1]
    regime = _regime_at(od, end)
    flips = rot.regime_changes(od["regime"])
    since = flips["date"].iloc[0] if len(flips) else None
    k = st.columns(3)
    k[0].metric("Regime", regime or "-", f"since {since:%d %b %Y}" if since is not None else None, delta_color="off")
    k[1].metric("Ratio vs its average", pct(cur["ratio"] / cur["trend"] - 1, signed=True) if pd.notna(cur["trend"]) else "-",
                help=f"Offensive ÷ defensive against its {window}-day average. The regime only flips after a move of "
                     f"more than {band:.1%} through the average.")
    k[2].metric("Offensive ÷ defensive, 1-year change",
                pct(od["ratio"].iloc[-1] / od["ratio"].loc[:end - pd.DateOffset(years=1)].iloc[-1] - 1, signed=True)
                if len(od.loc[:end - pd.DateOffset(years=1)]) else "-", delta_color="off")

    years = HISTORY[st.radio("History", list(HISTORY), index=1, horizontal=True, key="rot_history")]
    view = od if years is None else od.loc[end - pd.DateOffset(years=years):]
    show_chart(charts.line_figure(pd.DataFrame({"Offensive ÷ defensive": view["ratio"], f"{window}-day average": view["trend"]}),
                                  height=360))
    st.caption(f"**Offensive**: {', '.join(u.offensive)}.  \n**Defensive**: {', '.join(u.defensive)}.")
    if len(flips):
        st.markdown("**Regime changes**")
        show_table(flips.head(12).assign(date=flips["date"].head(12).dt.date)
                   .rename(columns={"date": "Date", "to": "Switched to", "after_days": "Previous regime lasted (days)"}),
                   hide_index=True)
    st.caption("Rising ratio = money favouring risk over safety. Both groups are equal-weight, rebalanced daily.")


def render() -> None:
    st.title("🔄 Sector Rotation")
    st.caption("Where relative strength - and with it money - is moving between sectors, and how that changed over time. "
               "Everything is computed from the local price store, so any period can be replayed.")
    with st.sidebar:
        st.markdown("### Rotation")
        key = st.radio("Universe", list(UNIVERSES), format_func=lambda k: UNIVERSES[k].label, key="rot_universe")
    universe = UNIVERSES[key]
    data = ui_data.rotation_data(key)
    if data is None or len(data.close) < 30:
        missing_data(universe)
        return

    end = data.asof
    with st.sidebar:
        period = st.selectbox("Period", PERIODS, index=1, key="rot_period")
        if period == "Custom":
            lo = max(data.first.date(), (end - pd.DateOffset(months=3)).date())
            picked = st.date_input("From - to", (lo, end.date()), min_value=data.first.date(), max_value=end.date(), key="rot_range")
            if len(picked) != 2:
                st.info("Pick an end date for the period.")
                return
            start, end = (pd.Timestamp(d) for d in picked)
        else:
            start = period_bounds(period, end)
        with st.expander("Settings"):
            smooth = st.slider("RS-Ratio smoothing (weeks)", 2, 12, rot.RRG_SMOOTH, key="rot_smooth")
            trend = st.slider("RS trend window (weeks)", 13, 78, rot.RRG_TREND, key="rot_trend")
            momentum = st.slider("RS-Momentum window (weeks)", 3, 26, rot.RRG_MOMENTUM, key="rot_momentum")
            window = st.slider("Risk-appetite average (days)", 20, 250, rot.REGIME_WINDOW, 10, key="rot_regime_window")
            band = st.slider("Regime band (%)", 0.0, 3.0, rot.REGIME_BAND * 100, 0.25, key="rot_regime_band") / 100
    if start >= end or start < data.first:
        st.error(f"The period must lie within the stored history ({data.first:%d %b %Y} - {data.asof:%d %b %Y}) and start before it ends.")
        return

    st.caption(f"{universe.label} · benchmark {universe.benchmark_name} · {start:%d %b %Y} → {end:%d %b %Y} · "
               f"data as of {data.asof:%d %b %Y}")
    if universe.note:
        with st.expander("About this universe"):
            st.markdown(universe.note)
    sb = data.scoreboard(start, end)
    if sb.empty:
        st.warning("No sector has prices in this period.")
        return
    close = data.close.loc[:end]
    ratio, mom = rot.rrg(close, data.benchmark.loc[:end], trend=trend, momentum=momentum, smooth=smooth)
    quads = rot.quadrants(ratio, mom)
    od = rot.offensive_defensive(close, universe.offensive, universe.defensive, window, band)

    tab_over, tab_map, tab_time, tab_risk = st.tabs(["Where money is going", "Rotation map", "Timeline", "Risk appetite"])
    with tab_over:
        _overview_tab(data, sb, quads, od, start, end)
    with tab_map:
        _map_tab(data, ratio, mom, quads, start, end)
    with tab_time:
        _timeline_tab(data, sb, start, end)
    with tab_risk:
        _appetite_tab(data, od, end, window, band)
