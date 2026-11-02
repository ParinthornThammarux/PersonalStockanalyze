"""Cached data access for the pages (``st.cache_data`` keeps slider moves from re-fetching)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st

from stockanalyzer.analysis.fundamental import bands
from stockanalyzer.data import fx as fx_data
from stockanalyzer.data import sectors as sector_data
from stockanalyzer.data import universe as uni
from stockanalyzer.data.providers.news import newest_first
from stockanalyzer.domain.financials import FinancialStatements
from stockanalyzer.ui.context import services


@st.cache_data(ttl=120, show_spinner=False)
def prices(symbol: str, period: str | None, interval: str = "1d", adjusted: bool = True, live: bool = True) -> pd.DataFrame:
    return services().md.prices(symbol, period=period, interval=interval, adjusted=adjusted, live=live)


@st.cache_data(ttl=600, show_spinner=False)
def close_matrix(symbols: tuple[str, ...], period: str | None = "2y", live: bool = True) -> pd.DataFrame:
    return services().md.close_matrix(symbols, period=period, live=live)


def prefetch(symbols: list[str] | tuple[str, ...]) -> None:
    """Download what the store is missing for ``symbols`` in batches (fresh ones are skipped).

    Loading a big pool one symbol at a time would be one slow request each.
    """
    services().md.update(list(symbols))


def stored_count(symbols: list[str] | tuple[str, ...]) -> int:
    """How many of ``symbols`` already have prices in the local store."""
    return len(services().md.store.last_timestamps(list(symbols)))


@st.cache_data(ttl=300, show_spinner=False)
def stored_closes(symbols: tuple[str, ...], last_n: int = 520) -> pd.DataFrame:
    """Adjusted closes straight from the local store (never touches the network)."""
    long = services().md.store.read_many(list(symbols), last_n=last_n, columns=["adj_close"])
    if long.empty:
        return pd.DataFrame()
    return long.pivot(index="ts", columns="symbol", values="adj_close").sort_index().ffill()


@st.cache_data(ttl=60, show_spinner=False)
def store_overview() -> dict:
    sv = services()
    stats = sv.md.store.stats(deep=True)
    latest = sv.features.dates()
    runs = sv.state.query("SELECT started_at, finished_at, status FROM pipeline_runs ORDER BY id DESC LIMIT 1")
    return {**stats, "features_asof": latest[-1] if latest else None, "last_run": runs[0] if runs else None,
            "alerts_active": len(sv.alerts.list(active_only=True)), "fundamentals": len(sv.fundamentals.snapshots_frame())}


@st.cache_data(ttl=300, show_spinner=False)
def rotation_data(key: str) -> sector_data.SectorData | None:
    """Sector levels of one rotation universe, straight from the local store (never touches the network)."""
    return sector_data.load_sector_data(services().md.store, sector_data.UNIVERSES[key])


@st.cache_data(ttl=3600, show_spinner=False)
def statements(symbol: str, quarterly: bool = False) -> FinancialStatements:
    """Yahoo's periods plus every one archived earlier, per-share figures restated for splits since."""
    sv = services()
    return sv.fundamentals.statements(symbol, quarterly=quarterly, splits=sv.events.splits(symbol))


@st.cache_data(ttl=3600, show_spinner=False)
def report_days(symbol: str) -> list[pd.Timestamp]:
    """Days the company has reported earnings (cached calendar), used to date when each period became public."""
    events = services().events
    events.refresh([symbol])
    history = events.earnings_history(symbol)
    return [] if history.empty else history.loc[history["eps_actual"].notna(), "day"].tolist()


@st.cache_data(ttl=3600, show_spinner=False)
def dividends(symbol: str) -> pd.Series:
    return services().events.dividends(symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def valuation_history(symbol: str, fx: float = 1.0) -> pd.DataFrame:
    """Daily close, TTM EPS, book value per share, P/E, P/B and EV/EBITDA as known on each day."""
    close = prices(symbol, "max", "1d", False)["close"]
    return bands.valuation_history(close, statements(symbol, False), statements(symbol, True), report_days(symbol), fx)


@st.cache_data(ttl=3600, show_spinner=False)
def fx_rate(from_ccy: str, to_ccy: str) -> float | None:
    return fx_data.fx_rate(services().md, from_ccy, to_ccy)


@st.cache_data(ttl=3600, show_spinner=False)
def info(symbol: str) -> dict:
    return services().fundamentals.info(symbol)


@st.cache_data(show_spinner=False)
def universe_table() -> pd.DataFrame:
    return pd.concat([uni.load_us_stocks(), uni.cross_asset_universe()], ignore_index=True)


@st.cache_data(show_spinner=False)
def symbol_names() -> dict[str, str]:
    table = universe_table().drop_duplicates("symbol")
    return dict(zip(table["symbol"], table["name"]))


@st.cache_data(ttl=1800, show_spinner=False)
def headlines(symbols: tuple[str, ...], limit: int = 15) -> tuple[list[dict], dict[str, str]]:
    """Headlines of several symbols fetched in parallel, newest first, and ``{symbol: error}`` for the failures."""
    sv = services()
    items: list[dict] = []
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(sv.events.headlines, s, limit=limit): s for s in symbols}
        for fut, symbol in futures.items():
            try:
                items += fut.result()
            except Exception as exc:  # one source down must not hide the others
                errors[symbol] = str(exc)
    return newest_first(items), errors


@st.cache_data(ttl=1800, show_spinner=False)
def news_search(query: str, lang: str, limit: int = 30) -> list[dict]:
    return services().events.search_news(query, lang=lang, limit=limit)


def no_data_message(symbol: str) -> str:
    reason = services().md.last_error(symbol)
    return f"No price data for **{symbol}**" + (f" ({reason})" if reason else ". Check the ticker (Yahoo format, e.g. BRK-B, PTT.BK).")


def require_prices(symbol: str, period: str | None, interval: str = "1d", adjusted: bool = True) -> pd.DataFrame:
    """Load prices or stop the page with a readable message."""
    with st.spinner(f"Loading {symbol} ..."):
        df = prices(symbol, period, interval, adjusted)
    if df.empty:
        st.error(no_data_message(symbol))
        st.stop()
    return df
