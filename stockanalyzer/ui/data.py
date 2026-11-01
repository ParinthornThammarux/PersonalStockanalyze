"""Cached data access for the pages (``st.cache_data`` keeps slider moves from re-fetching)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from stockanalyzer.data import fx as fx_data
from stockanalyzer.data import universe as uni
from stockanalyzer.domain.financials import FinancialStatements
from stockanalyzer.ui.context import services


@st.cache_data(ttl=120, show_spinner=False)
def prices(symbol: str, period: str | None, interval: str = "1d", adjusted: bool = True) -> pd.DataFrame:
    return services().md.prices(symbol, period=period, interval=interval, adjusted=adjusted)


@st.cache_data(ttl=600, show_spinner=False)
def close_matrix(symbols: tuple[str, ...], period: str | None = "2y") -> pd.DataFrame:
    return services().md.close_matrix(symbols, period=period)


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


@st.cache_data(ttl=3600, show_spinner=False)
def statements(symbol: str, quarterly: bool = False) -> FinancialStatements:
    return services().fundamentals.statements(symbol, quarterly=quarterly)


@st.cache_data(ttl=3600, show_spinner=False)
def fx_rate(from_ccy: str, to_ccy: str) -> float | None:
    return fx_data.fx_rate(services().md, from_ccy, to_ccy)


@st.cache_data(ttl=3600, show_spinner=False)
def info(symbol: str) -> dict:
    return services().fundamentals.info(symbol)


@st.cache_data(show_spinner=False)
def universe_table() -> pd.DataFrame:
    return pd.concat([uni.load_us_stocks(), uni.cross_asset_universe()], ignore_index=True)


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
