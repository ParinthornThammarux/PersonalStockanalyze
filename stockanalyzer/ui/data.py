"""Cached data access for the pages (``st.cache_data`` keeps slider moves from re-fetching)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st

from stockanalyzer.analysis.fundamental import bands
from stockanalyzer.data import breadth as breadth_data
from stockanalyzer.data import dividend_events
from stockanalyzer.data import fx as fx_data
from stockanalyzer.data import sectors as sector_data
from stockanalyzer.data import universe as uni
from stockanalyzer.data.providers.base import ProviderError
from stockanalyzer.data.providers.news import newest_first
from stockanalyzer.domain.estimates import Estimates
from stockanalyzer.domain.financials import FinancialStatements
from stockanalyzer.portfolio import currency as ccy
from stockanalyzer.ui.context import services


@st.cache_data(ttl=120, show_spinner=False)
def prices(symbol: str, period: str | None, interval: str = "1d", adjusted: bool = True, live: bool = True) -> pd.DataFrame:
    return services().md.prices(symbol, period=period, interval=interval, adjusted=adjusted, live=live)


@st.cache_data(ttl=600, show_spinner=False)
def close_matrix(symbols: tuple[str, ...], period: str | None = "2y", live: bool = True) -> pd.DataFrame:
    return services().md.close_matrix(symbols, period=period, live=live)


@st.cache_data(ttl=3600, show_spinner=False)
def fx_series(currencies: tuple[str, ...], base: str) -> dict:
    """Daily exchange rates into ``base`` for each currency (``None`` for a pair Yahoo does not have)."""
    return ccy.fx_table(services().md, currencies, base)


@st.cache_data(ttl=600, show_spinner=False)
def base_close_matrix(symbols: tuple[str, ...], period: str | None, base: str) -> pd.DataFrame:
    """Adjusted closes converted into ``base`` day by day; a symbol whose currency has no rate is left out."""
    close = close_matrix(symbols, period, False)
    return ccy.to_base(close, fx_series(tuple(ccy.currencies_of(close.columns)), base)) if not close.empty else close


def prefetch(symbols: list[str] | tuple[str, ...]) -> None:
    """Download what the store is missing for ``symbols`` in batches (fresh ones are skipped).

    Loading a big pool one symbol at a time would be one slow request each.
    """
    services().md.update(list(symbols))


def stored_count(symbols: list[str] | tuple[str, ...]) -> int:
    """How many of ``symbols`` already have prices in the local store."""
    return len(services().md.store.last_timestamps(list(symbols)))


@st.cache_data(ttl=300, show_spinner=False)
def stored_closes(symbols: tuple[str, ...], last_n: int = 520, ffill: bool = True) -> pd.DataFrame:
    """Adjusted closes straight from the local store (never touches the network).

    ``ffill=False`` keeps NaN on the days a symbol did not trade (what correlation and cointegration need).
    """
    long = services().md.store.read_many(list(symbols), last_n=last_n, columns=["adj_close"])
    if long.empty:
        return pd.DataFrame()
    wide = long.pivot(index="ts", columns="symbol", values="adj_close").sort_index()
    return wide.ffill() if ffill else wide


@st.cache_data(ttl=300, show_spinner=False)
def stored_symbols() -> tuple[str, ...]:
    """Every symbol that has daily prices in the local store."""
    return tuple(services().md.store.symbols())


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


@st.cache_data(ttl=600, show_spinner=False)
def breadth(key: str, min_traded: float = 0.0):
    """Daily breadth of one market straight from the local store (``None`` when it holds no prices for that market)."""
    return breadth_data.load(services().md.store, key, min_traded=min_traded)


@st.cache_data(ttl=600, show_spinner=False)
def breadth_groups(key: str, min_traded: float, asof: pd.Timestamp, level: str = "sector") -> tuple[pd.DataFrame, int, int]:
    """Participation per sector (or Yahoo industry) on ``asof``, how many counted stocks have a group, and how many were counted."""
    sv = services()
    table = breadth_data.market_universe(key)
    snaps = sv.fundamentals.snapshots_frame()
    labels = breadth_data.industries(table, snaps) if level == "industry" else breadth_data.groups(key, table, snaps)
    return breadth_data.group_table(sv.md.store, key, asof, labels, min_traded=min_traded, min_stocks=3 if level == "sector" else 5)


@st.cache_data(ttl=3600, show_spinner=False)
def xd_events(symbol: str) -> pd.DataFrame:
    """Every ex-dividend day of one stock in the local store, with the prices around it."""
    return dividend_events.xd_events(services().md.store, [symbol])


@st.cache_data(ttl=3600, show_spinner=False)
def market_xd_events(key: str) -> pd.DataFrame:
    """Every ex-dividend day of every stock of one market in the local store (a second or two for the whole US)."""
    return dividend_events.xd_events(services().md.store, breadth_data.market_universe(key)["symbol"].tolist())


@st.cache_data(ttl=3600, show_spinner=False)
def benchmark_close(symbol: str) -> pd.Series:
    return breadth_data.benchmark_close(services().md.store, symbol)


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


@st.cache_data(ttl=3600, show_spinner=False)
def _estimates(symbol: str) -> Estimates:
    return services().fundamentals.estimates(symbol)  # raises when Yahoo fails: a failure is never cached


def estimates(symbol: str) -> Estimates:
    """Analyst consensus (cached for an hour, stored once a day); an empty ``Estimates`` when there is none or Yahoo failed."""
    try:
        return _estimates(symbol)
    except ProviderError:
        return Estimates(symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def estimates_history(symbol: str) -> list[tuple[str, Estimates]]:
    """Every daily consensus snapshot stored for ``symbol`` (it grows by one for each day the page is opened)."""
    return services().fundamentals.estimates_history(symbol)


@st.cache_data(show_spinner=False)
def universe_table() -> pd.DataFrame:
    return pd.concat([uni.load_us_stocks(), uni.load_th_stocks(), uni.cross_asset_universe()], ignore_index=True)


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
