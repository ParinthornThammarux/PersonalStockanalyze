"""Loading the whole market from the UI: how much of it is stored, and the background job that stores the rest."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import streamlit as st

from stockanalyzer.data import universe as uni
from stockanalyzer.pipelines.jobs import JobState, Jobs, nightly_args
from stockanalyzer.pipelines.nightly import PipelineOptions
from stockanalyzer.ui.context import services

WHOLE_MARKET = "Whole market"
PATIENCE = 8  # waits of 5 minutes each when Yahoo throttles a long download: up to 40 minutes before giving up
STEP_LABELS = {
    "resolve": "Choosing the symbols", "update": "Downloading prices", "update_prices": "Downloading prices",
    "features": "Computing screener features", "market_snapshot": "P/E, P/B, market caps and yields",
    "fundamentals": "Full fundamentals", "events": "Earnings and dividend calendar", "alerts": "Checking alerts",
}


@dataclass(frozen=True)
class Coverage:
    us_listed: int
    us_stored: int
    th_listed: int
    th_stored: int
    no_data: int = 0  # listed tickers Yahoo has no prices for (delisted, renamed, never traded): nothing to load

    @property
    def available(self) -> int:
        return self.us_listed + self.th_listed - self.no_data

    @property
    def stored(self) -> int:
        return self.us_stored + self.th_stored

    @property
    def share(self) -> float:
        return min(self.stored / self.available, 1.0) if self.available > 0 else 1.0


@st.cache_data(show_spinner=False)
def _listed() -> tuple[frozenset[str], frozenset[str]]:
    us = uni.load_us_stocks()
    return frozenset(us.loc[us["industry"] != uni.SPAC_INDUSTRY, "symbol"]), frozenset(uni.load_th_stocks()["symbol"])


def coverage() -> Coverage:
    """How many of the listed US and Thai stocks have prices in the local store (no network).

    Tickers Yahoo answered with no data are counted out: they cannot be loaded, so they must not keep the offer to load more alive.
    """
    sv = services()
    us, th = _listed()
    stored = set(sv.md.store.symbols())
    dead = {r["symbol"] for r in sv.state.query("SELECT symbol FROM update_log WHERE interval='1d' AND status='no_data'")}
    return Coverage(len(us), len(stored & us), len(th), len(stored & th), len(((us | th) & dead) - stored))


def whole_market_options() -> PipelineOptions:
    return PipelineOptions(universe="us", thai="all", include_cross_assets=True, include_sectors=True, market_snapshot=True,
                           rate_limit_waits=PATIENCE)


def jobs() -> Jobs:
    return Jobs(services().settings)


def start(opts: PipelineOptions, title: str, *, fundamentals: int = 0) -> None:
    try:
        jobs().start(nightly_args(opts, fundamentals=fundamentals, events=True), title=title)
    except RuntimeError as exc:
        st.error(str(exc))


def _elapsed(since: str) -> str:
    seconds = max(int((datetime.now(timezone.utc).replace(tzinfo=None) - datetime.fromisoformat(since)).total_seconds()), 0)
    return f"{seconds // 60}m {seconds % 60:02d}s"


def _describe(state: JobState) -> str:
    label = STEP_LABELS.get(state.step, state.step or "starting")
    count = f" · {state.done:,} of {state.total:,}" if state.total else ""
    return f"{state.title or 'Pipeline'} · {label}{count} · {_elapsed(state.started_at)}"


@st.fragment(run_every=2)
def _running_panel() -> None:
    state = jobs().status()
    if state is None or not state.running:
        st.rerun()  # the job ended: redraw the whole page so tables and counts pick up what it stored
        return
    st.progress(state.fraction, text=_describe(state))
    c1, c2 = st.columns([1, 5])
    if c1.button("Stop", key="job_stop", help="What is already stored stays; the next run carries on from there."):
        jobs().stop()
        st.rerun()
    c2.caption("It runs on its own: you can leave this page or close the browser. "
               + (f"Last message: {state.message}" if state.message and state.message != "running" else ""))
    with st.expander("Log"):
        st.code(jobs().log_tail() or "(nothing yet)", language="text")


def job_panel() -> bool:
    """Progress of the background job, or the result of the last one. True while one is running."""
    state = jobs().status()
    if state is None:
        return False
    if state.running:
        _running_panel()
        return True
    if state.interrupted:
        st.warning(f"The last job ({state.title or 'pipeline'}, started {state.started_at} UTC) was stopped before it finished. "
                   "Start it again to carry on: everything already stored is kept.")
    else:
        lines = [f"**{STEP_LABELS.get(k, k)}** - " + (f"FAILED: {v['error']}" if v.get("error") else v.get("summary", "ok"))
                 for k, v in state.summary.items()]
        (st.success if state.ok else st.error)(f"{state.title or 'Pipeline'} finished {state.finished_at} UTC")
        if lines:
            st.markdown("\n\n".join(lines))
    return False


def coverage_banner(cov: Coverage, *, key: str) -> None:
    """Tell how much of the market is searchable, and offer to load the rest."""
    running = job_panel()
    if cov.share >= 0.97 or running:
        return
    gone = f" ({cov.no_data:,} more listed tickers have no prices on Yahoo - delisted or renamed - and cannot be loaded)" if cov.no_data else ""
    st.info(f"Only **{cov.stored:,} of {cov.available:,}** stocks that Yahoo has prices for are stored ({cov.us_stored:,} US, "
            f"{cov.th_stored:,} Thai){gone}, so the screener searches {cov.share:.0%} of the market. "
            "Loading the rest takes about 10-15 minutes the first time (prices of every stock, then P/E, P/B and market cap for "
            "all of them) and runs in the background.")
    if st.button("Load the whole market", key=key, type="primary"):
        start(whole_market_options(), WHOLE_MARKET)
        st.rerun()
