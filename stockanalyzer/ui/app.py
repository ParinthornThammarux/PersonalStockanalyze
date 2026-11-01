"""Streamlit application: page registry and navigation."""

from __future__ import annotations

import streamlit as st

from stockanalyzer.ui import nav
from stockanalyzer.ui.views import (
    backtest,
    data_manager,
    fundamentals,
    home,
    markets,
    overview,
    portfolio,
    screener,
    technical,
    watchlist,
    zones,
)

# key -> (module, title, icon, section)
PAGE_SPECS = {
    "home": (home, "Home", "🏠", "Start"),
    "overview": (overview, "Stock Overview", "🔎", "Analyse"),
    "technical": (technical, "Technical Analysis", "📈", "Analyse"),
    "zones": (zones, "Support / Resistance", "🧱", "Analyse"),
    "fundamentals": (fundamentals, "Fundamentals & Valuation", "📑", "Analyse"),
    "screener": (screener, "Screener", "🧮", "Scale"),
    "watchlist": (watchlist, "Watchlists & Alerts", "⭐", "Scale"),
    "markets": (markets, "Markets & Macro", "🌍", "Scale"),
    "backtest": (backtest, "Backtest", "🧪", "Test"),
    "portfolio": (portfolio, "Portfolio & Risk", "💼", "Test"),
    "data": (data_manager, "Data Manager", "🗄️", "System"),
}


def build_navigation():
    sections: dict[str, list] = {}
    for key, (module, title, icon, section) in PAGE_SPECS.items():
        page = st.Page(module.render, title=title, icon=icon, url_path=key, default=key == "home")
        nav.PAGES[key] = page
        sections.setdefault(section, []).append(page)
    return st.navigation(sections)


def main() -> None:
    st.set_page_config(page_title="Stock Analyzer", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
    build_navigation().run()
