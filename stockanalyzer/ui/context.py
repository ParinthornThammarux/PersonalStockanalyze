"""Shared services for the Streamlit pages (one set per server process)."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from stockanalyzer.alerts.alerts import AlertManager
from stockanalyzer.alerts.watchlist import WatchlistRepo
from stockanalyzer.config import Settings, get_settings
from stockanalyzer.data.events import EventsService
from stockanalyzer.data.fundamentals import FundamentalsService
from stockanalyzer.data.service import MarketData
from stockanalyzer.portfolio.ledger import TransactionRepo
from stockanalyzer.screener.store import FeatureStore
from stockanalyzer.storage.state import StateDB


@dataclass
class Services:
    settings: Settings
    state: StateDB
    md: MarketData
    fundamentals: FundamentalsService
    events: EventsService | None = None  # built over the same store and state when not given

    def __post_init__(self):
        if self.events is None:
            self.events = EventsService(store=self.md.store, state=self.state, settings=self.settings)

    @property
    def watchlists(self) -> WatchlistRepo:
        return WatchlistRepo(self.state)

    @property
    def alerts(self) -> AlertManager:
        return AlertManager(self.state)

    @property
    def transactions(self) -> TransactionRepo:
        return TransactionRepo(self.state)

    @property
    def features(self) -> FeatureStore:
        return FeatureStore(self.settings.features_dir)


_override: Services | None = None


def set_services(services: Services | None) -> None:
    """Inject services (tests, or embedding the pages in another app)."""
    global _override
    _override = services


def build_services(settings: Settings | None = None) -> Services:
    settings = (settings or get_settings()).ensure_dirs()
    state = StateDB(settings.state_db)
    return Services(settings, state, MarketData(settings=settings, state=state),
                    FundamentalsService(settings=settings, state=state))


@st.cache_resource(show_spinner=False)
def _default_services() -> Services:
    return build_services()


def services() -> Services:
    return _override if _override is not None else _default_services()
