"""Home: what is in the local data store, and what to do next."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from stockanalyzer import __version__
from stockanalyzer.alerts.watchlist import FAVORITES
from stockanalyzer.ui import data as ui_data
from stockanalyzer.ui.context import services
from stockanalyzer.ui.format import pct
from stockanalyzer.ui.widgets import show_table


def render() -> None:
    sv = services()
    st.title("📈 Stock Analyzer")
    st.caption(f"v{__version__} · multi-asset analysis: data store → technical & fundamental analysis → "
               "screening → backtesting → portfolio risk")

    ov = ui_data.store_overview()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Symbols stored", f"{ov['symbols']:,}")
    c2.metric("Latest price date", f"{ov['last_ts']:%Y-%m-%d}" if ov["symbols"] else "-")
    c3.metric("Feature snapshot", ov["features_asof"] or "none")
    c4.metric("Active alerts", ov["alerts_active"])

    if not ov["symbols"]:
        st.info(
            "The local data store is empty. Open **Data Manager** and update a universe "
            "(e.g. the 100 largest US companies + cross-asset presets takes about 10 seconds), "
            "or look up any symbol on **Stock Overview** - it is downloaded and cached on first use."
        )
        st.code("python -m stockanalyzer nightly --limit 500", language="bash")
    elif ov["features_asof"] is None:
        st.warning("Prices are stored but no feature snapshot exists yet. Open **Screener** and compute it, "
                   "or run `python -m stockanalyzer features`.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Watchlist snapshot")
        symbols = sv.watchlists.symbols(FAVORITES) or sv.watchlists.symbols()
        if symbols:
            closes = ui_data.stored_closes(tuple(symbols), 300)
            rows = []
            for s in symbols:
                if s in closes and closes[s].dropna().size > 21:
                    c = closes[s].dropna()
                    rows.append({"symbol": s, "last": c.iloc[-1], "1d %": (c.iloc[-1] / c.iloc[-2] - 1) * 100,
                                 "1m %": (c.iloc[-1] / c.iloc[-22] - 1) * 100,
                                 "3m %": (c.iloc[-1] / c.iloc[-64] - 1) * 100 if len(c) > 63 else float("nan")})
                else:
                    rows.append({"symbol": s, "last": float("nan")})
            show_table(pd.DataFrame(rows).set_index("symbol").style.format("{:,.2f}", na_rep="-"))
        else:
            st.caption("No watchlists yet - add symbols on the **Watchlists & Alerts** page.")
    with right:
        st.subheader("Recent alerts")
        events = sv.alerts.events(8)
        if events:
            show_table(pd.DataFrame(events)[["triggered_at", "symbol", "message"]], hide_index=True)
        else:
            st.caption("Nothing has fired yet.")

    run = ov["last_run"]
    if run:
        st.caption(f"Last pipeline run: {run['started_at']} UTC - {run['status']}")

    with st.expander("How the pieces fit together"):
        st.markdown(
            """
- **Data Manager / `nightly`** keeps a Parquet price store fresh (incremental, split-aware) and builds the feature table.
- **Screener** filters and ranks the whole universe from that table in milliseconds.
- **Overview, Technical, Zones, Fundamentals** are deep dives on one symbol.
- **Backtest** tests rule-based strategies with costs and no look-ahead; **Portfolio** tracks holdings, risk and rebalancing.
            """
        )
