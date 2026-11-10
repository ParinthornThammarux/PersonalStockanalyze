"""Market Breadth: is the whole market moving, or only the index? Participation, new highs and lows, advance / decline,
the average stock against the index, sector participation and what the index did after each breadth regime."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from stockanalyzer.analysis import breadth as calc
from stockanalyzer.data.breadth import MARKETS
from stockanalyzer.ui import charts
from stockanalyzer.ui import data as ui_data
from stockanalyzer.ui.context import services
from stockanalyzer.ui.format import heat, pct
from stockanalyzer.ui.widgets import show_chart, show_table

HISTORY = {"1 year": 1, "3 years": 3, "5 years": 5, "All": None}
TONE = {"good": "🟢", "bad": "🔴", "neutral": "⚪"}
SURVIVORSHIP = ("Only stocks listed today are in the store, so delisted companies - often the ones that fell - are missing "
                "from the past: older readings lean a little positive.")


def _window(ind: pd.DataFrame, years: int | None) -> pd.DataFrame:
    return ind if years is None else ind.loc[ind.index[-1] - pd.DateOffset(years=years):]


def _no_data(market) -> None:
    st.info(f"No stored prices for **{market.label}** yet. Breadth counts every stock in the market, so it needs the whole "
            "market in the local store: load it under **Data Manager → Whole market** (10-15 minutes the first time, then "
            "the nightly job keeps it fresh).")


def _health_tab(b: calc.Breadth, view: pd.DataFrame, market) -> None:
    regime, since = b.regime_since()
    p200, p200_m = b.at("pct_above_200"), b.at("pct_above_200", 21)
    p50, p50_m = b.at("pct_above_50"), b.at("pct_above_50", 21)
    nh, nl = b.ind["new_highs"].iloc[-10:].mean(), b.ind["new_lows"].iloc[-10:].mean()
    k = st.columns(5)
    k[0].metric("Breadth regime", regime or "-", f"since {since:%d %b %Y}" if since is not None else None, delta_color="off",
                help="Index above / below its 200-day average × most stocks above / below theirs. See the last tab for what "
                     "each regime meant for the index afterwards.")
    k[1].metric("Above 200-day average", pct(p200, 0), f"{(p200 - p200_m) * 100:+.0f} pts in a month" if p200_m == p200_m else None)
    k[2].metric("Above 50-day average", pct(p50, 0), f"{(p50 - p50_m) * 100:+.0f} pts in a month" if p50_m == p50_m else None)
    k[3].metric("New highs / lows (10-day avg)", f"{nh:.0f} / {nl:.0f}")
    k[4].metric("Advancing last day", pct(b.at("ad_ratio"), 0), f"{b.at('adv'):.0f} up · {b.at('dec'):.0f} down", delta_color="off")
    if regime:
        st.caption(f"**{regime}**: {calc.REGIME_TEXT[regime]}.")

    st.markdown("##### What the breadth says")
    for r in b.readings():
        st.markdown(f"{TONE[r.tone]} {r.text}")

    top = pd.DataFrame({market.benchmark_name if b.has_benchmark else "Average stock (equal weight)":
                        view["index"] if b.has_benchmark else view["ew_level"]})
    if b.has_benchmark:
        top["200-day average"] = view["index_sma200"]
    bottom = pd.DataFrame({"Above 200-day avg": view["pct_above_200"], "Above 50-day avg": view["pct_above_50"]})
    show_chart(charts.stacked_figure(top, bottom, bottom_title="share of stocks", bottom_levels=(0.5,), bottom_format=".0%"))
    st.caption("Lower panel: share of stocks closing above their own 50- and 200-day averages. A rising index with a falling "
               "lower line means fewer stocks are carrying it. " + SURVIVORSHIP)


def _highs_tab(b: calc.Breadth, view: pd.DataFrame) -> None:
    show_chart(charts.highs_lows_figure(view["new_highs"], view["new_lows"], view["net_highs_10"]))
    st.caption("A new high (low) is a close at the highest (lowest) level of the last 252 trading days, closing prices only. "
               "Expanding new highs confirm a rally; new lows that keep growing while the index rises are a warning.")
    hl = view["high_low_index"].dropna()
    if not hl.empty:
        show_chart(charts.line_figure((hl * 100).to_frame("Highs ÷ (highs + lows), 10-day average, %"), height=300))
        st.caption("Above 50: more highs than lows. Readings near 100 come with strong advances, near 0 with selloffs.")


def _ad_tab(b: calc.Breadth, view: pd.DataFrame, market) -> None:
    level = view["index"] if b.has_benchmark else view["ew_level"]
    top = pd.DataFrame({market.benchmark_name if b.has_benchmark else "Average stock": level})
    bottom = pd.DataFrame({"Advance / decline line": view["ad_line"] - view["ad_line"].iloc[0]})
    show_chart(charts.stacked_figure(top, bottom, bottom_title="cumulative advances − declines"))
    st.caption("The A/D line adds up advancing minus declining stocks each day. When the index makes a new high and the A/D "
               "line does not, the rally rests on fewer stocks.")

    osc = pd.DataFrame({"McClellan oscillator": view["mcclellan"]})
    show_chart(charts.stacked_figure(pd.DataFrame({"Summation index": view["summation"]}), osc, bottom_bars=["McClellan oscillator"],
                                     bottom_levels=(0,), height=460))
    st.caption("McClellan oscillator: the 19-day minus the 39-day exponential average of (advances − declines) ÷ (advances + "
               "declines) × 1000. Above zero, breadth momentum is up; the summation index adds it up into a longer trend.")

    thrusts = b.thrusts
    st.markdown(f"##### Breadth thrusts ({len(thrusts)} since {b.first:%b %Y})")
    if len(thrusts) == 0:
        st.caption("None in the stored history.")
        return
    outcome = calc.event_outcomes(b.ind, thrusts)
    show_table(outcome.iloc[::-1].rename(index=lambda d: d.strftime("%d %b %Y")).style.format("{:+.1%}", na_rep="not yet"))
    st.caption(f"Martin Zweig's thrust: the {calc.THRUST_DAYS}-day exponential average of the advancing share rises from under "
               f"{calc.THRUST_LOW:.0%} to over {calc.THRUST_HIGH:.1%} within {calc.THRUST_DAYS} days. The columns are the "
               f"{'index' if b.has_benchmark else 'average stock'}'s return after each one. A handful of events is an anecdote, "
               "not a statistic, and Zweig defined it on the whole NYSE, not this stock list.")


def _average_tab(b: calc.Breadth, view: pd.DataFrame, market) -> None:
    if not b.has_benchmark:
        st.info(f"{market.benchmark_name} is not in the store, so there is no index to compare with.")
        return
    top = pd.DataFrame({"Average stock": view["avg_return_3m"], market.benchmark_name: view["index_return_3m"]}).dropna(how="all")
    beat = view[["beat_share_3m"]].rename(columns={"beat_share_3m": "Stocks beating the index"})
    if top.dropna().empty:
        st.info("Not enough overlapping history.")
        return
    show_chart(charts.stacked_figure(top, beat, top_title="return over the last 3 months", bottom_title="share of stocks",
                                     bottom_levels=(0.5,), bottom_format=".0%"))
    avg3, ix3, share = b.at("avg_return_3m"), b.at("index_return_3m"), b.at("beat_share_3m")
    st.caption(f"Top: the 3-month return of the average stock (each stock's own return, averaged) and of {market.benchmark_name}, both "
               f"with dividends; now {avg3:+.1%} vs {ix3:+.1%}. Bottom: the share of stocks that beat the index over those 3 months, "
               f"now {share:.0%}. Below 50% the index is doing better than most of its members - its largest companies are doing "
               "the work. The average stock includes small companies, so part of any gap is the size effect. " + SURVIVORSHIP)
    st.caption("Why not a long-run 'equal weight vs index' line? Compounding a daily-rebalanced average of thousands of small stocks "
               "over years is flattered by bid-ask bounce and by the missing delisted companies, so it would overstate the average stock.")


def _sector_tab(b: calc.Breadth, key: str, floor: float) -> None:
    level = "sector"
    if key == "us":
        level = st.radio("Group by", ["sector", "industry"], horizontal=True, format_func=str.capitalize, key="br_level")
    table, placed, counted = ui_data.breadth_groups(key, floor, b.asof, level)
    if table.empty:
        st.info("No stock could be placed in a group.")
        return
    show_chart(charts.bar_figure(table["pct_above_200"], title=f"Share above the 200-day average, {b.asof:%d %b %Y}", fmt=".0%"))
    view = table.rename(columns={"stocks": "Stocks", "pct_above_50": "Above 50-day", "pct_above_200": "Above 200-day",
                                 "pct_above_200_before": "Above 200-day a month ago", "change_200": "Change (pts)",
                                 "new_highs": "New highs", "new_lows": "New lows", "median_return_month": "Median 1-month return"})
    view["Change (pts)"] = view["Change (pts)"] * 100
    styled = view.style.format({"Above 50-day": "{:.0%}", "Above 200-day": "{:.0%}", "Above 200-day a month ago": "{:.0%}",
                                "Change (pts)": "{:+.0f}", "Median 1-month return": "{:+.1%}"}, na_rep="-")
    show_table(heat(heat(styled, subset=["Change (pts)"], vmin=-30, vmax=30), subset=["Median 1-month return"], vmin=-0.1, vmax=0.1))
    note = f"{placed:,} of the {counted:,} stocks counted on {b.asof:%d %b %Y} have a group."
    if key == "th":
        note += (" Thai shares carry an industry only once their full fundamentals have been fetched, so the large-cap sector "
                 "baskets from Sector Rotation stand in for the rest.")
    st.caption(note + " Groups with fewer than " + ("3" if level == "sector" else "5") + " stocks are left out.")


def _regime_tab(b: calc.Breadth, view: pd.DataFrame, market) -> None:
    level = view["index"] if b.has_benchmark else view["ew_level"]
    show_chart(charts.regime_figure(level.dropna(), view["regime"], name=market.benchmark_name if b.has_benchmark else "Average stock"))
    outcomes = calc.regime_outcomes(b.ind)
    cols = [c for c in outcomes.columns if c.startswith(("avg", "positive"))]
    shown = outcomes[["days", "episodes", *cols]].rename(columns=lambda c: c.replace("avg", "Average").replace("positive", "Up share"))
    fmt = {c: "{:+.1%}" if c.startswith("Average") else "{:.0%}" for c in shown.columns if c not in ("days", "episodes")}
    show_table(shown.style.format(fmt, na_rep="-"))
    st.caption(f"What the {'index' if b.has_benchmark else 'average stock'} did next, from every day in each regime since "
               f"{b.first:%b %Y} (dividends included). The windows overlap, so **episodes** - separate stretches in a regime - is the "
               "real sample size, and a few dozen episodes over one decade cannot prove a rule. Compare each row with "
               "**All days**: a regime only matters if it differs clearly from the average. " + SURVIVORSHIP)
    with st.expander("How the regime is decided"):
        st.markdown(
            f"- **Index trend**: above its 200-day average (it must cross by more than {calc.INDEX_BAND:.0%} to switch).\n"
            f"- **Participation**: more than half the stocks above their own 200-day average (switches beyond "
            f"{calc.BROAD_LINE - calc.BREADTH_BAND:.0%} / {calc.BROAD_LINE + calc.BREADTH_BAND:.0%}).\n"
            + "".join(f"- **{name}**: {text}.\n" for name, text in calc.REGIME_TEXT.items()))


def render() -> None:
    st.title("📊 Market Breadth")
    st.caption("Is the whole market moving, or only the index? Counts every stock in the local store, day by day: how many are "
               "above their averages, at new highs or lows, rising or falling.")
    with st.sidebar:
        st.markdown("### Breadth")
        key = st.radio("Market", list(MARKETS), format_func=lambda k: MARKETS[k].label, key="br_market")
        market = MARKETS[key]
        labels = [label for label, _ in market.floors]
        choice = st.selectbox("Stocks counted", labels, index=market.default_floor, key=f"br_floor_{key}",
                              help="Average traded value over the last 20 days, judged each day with what was known then. Thinly "
                                   "traded stocks print stale prices that count as 'unchanged' and add noise.")
        floor = dict(market.floors)[choice]
        years = HISTORY[st.radio("History", list(HISTORY), index=1, horizontal=True, key="br_history")]

    with st.spinner(f"Counting {market.label} ..."):
        b = ui_data.breadth(key, floor)
    if b is None:
        _no_data(market)
        return
    if not b.has_benchmark:
        st.warning(f"{market.benchmark_name} is not in the store, so the average stock stands in for the index.")
        if st.button(f"Download {market.benchmark}", key="br_bench"):
            with st.spinner("Downloading ..."):
                services().md.update([market.benchmark])
            ui_data.breadth.clear()
            st.rerun()
    view = _window(b.ind, years)
    st.caption(f"{market.label} · {choice.lower()} · {b.at('stocks'):,.0f} stocks counted on {b.asof:%d %b %Y} · history from "
               f"{b.first:%b %Y}")

    tabs = st.tabs(["Health now", "New highs & lows", "Advance / decline", "Average stock vs index", "Sectors",
                    "Does breadth matter?"])
    with tabs[0]:
        _health_tab(b, view, market)
    with tabs[1]:
        _highs_tab(b, view)
    with tabs[2]:
        _ad_tab(b, view, market)
    with tabs[3]:
        _average_tab(b, view, market)
    with tabs[4]:
        _sector_tab(b, key, floor)
    with tabs[5]:
        _regime_tab(b, view, market)
