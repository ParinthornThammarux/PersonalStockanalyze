"""The scheduled job: update prices -> compute features -> refresh fundamentals -> events -> check alerts.

Every step is isolated: a failure is recorded in the result and the run continues, so one
flaky provider call never leaves the screener with yesterday's features *and* no alerts.
Run it with ``python -m stockanalyzer nightly`` (see README for a cron line).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from stockanalyzer.alerts.alerts import AlertManager
from stockanalyzer.alerts.watchlist import WatchlistRepo
from stockanalyzer.data import pools, sectors, universe as uni
from stockanalyzer.data.events import EventsService
from stockanalyzer.data.fundamentals import FundamentalsService
from stockanalyzer.data.providers.base import RateLimitError
from stockanalyzer.data.service import MarketData
from stockanalyzer.portfolio.ledger import TransactionRepo, build_book
from stockanalyzer.screener.features import compute_universe_features
from stockanalyzer.screener.store import FeatureStore
from stockanalyzer.storage.state import StateDB, to_iso

log = logging.getLogger(__name__)

MAX_FAILED_SHARE = 0.5  # update step counts as failed when this share of attempted symbols errored


@dataclass
class PipelineOptions:
    universe: str = "us"  # "us" = packaged US list, "none" = only the extras below
    min_market_cap: float | None = None
    exclude_spacs: bool = True
    limit: int | None = None  # keep the N largest by market cap
    symbols: list[str] = field(default_factory=list)
    include_cross_assets: bool = True
    include_sectors: bool = False  # sector ETFs, asset-class ETFs and the Thai sector baskets for Sector Rotation
    thai: str = "none"  # "core" = SET50 + the sector baskets, "all" = every share on the SET and mai (th_stocks.json)
    market_snapshot: bool = False  # P/E, P/B, market cap, yield of the whole market in a few requests (light fundamentals)
    rate_limit_waits: int = 0  # when Yahoo throttles: wait and carry on this many times instead of giving up
    rate_limit_wait_minutes: float = 5.0
    include_watchlists: bool = True
    include_macro: bool = False
    benchmark: str = "SPY"
    start: str | None = None  # history start for symbols new to the store
    force: bool = False
    update_prices: bool = True
    compute_features: bool = True
    fundamentals_limit: int = 0  # refresh at most N stale snapshots per run (0 = skip)
    refresh_events: bool = False  # earnings/dividend dates of tracked symbols + the economic calendar
    check_alerts: bool = True
    workers: int = 0
    chunk_size: int = 100


@dataclass
class PipelineResult:
    steps: dict[str, dict] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""

    @property
    def ok(self) -> bool:
        return all(not s.get("error") for s in self.steps.values())

    def summary(self) -> str:
        lines = []
        for name, info in self.steps.items():
            lines.append(f"{name}: " + (f"FAILED - {info['error']}" if info.get("error") else info.get("summary", "ok")))
        return "\n".join(lines)


def tracked_symbols(state: StateDB) -> list[str]:
    """Symbols the user follows: every watchlist plus the open positions of every portfolio."""
    symbols = WatchlistRepo(state).symbols()
    repo = TransactionRepo(state)
    for pf in repo.portfolios():
        try:
            symbols += list(build_book(repo.list(pf)).positions.index)
        except ValueError:  # an inconsistent ledger must not hide the watchlists
            log.warning("portfolio %r cannot be replayed; its holdings are not tracked", pf)
    return list(dict.fromkeys(symbols))


THAI_CHOICES = ("none", "core", "all")


def thai_core() -> list[str]:
    """The SET50 and every member of the Thai sector baskets: the Thai stocks the app tracks by default."""
    return list(dict.fromkeys([*pools.load_pools()["set50"].symbols, *sectors.THAI_SECTORS.members]))


def resolve_symbols(opts: PipelineOptions, md: MarketData) -> list[str]:
    """Everything the run should keep fresh, benchmark and extras first."""
    symbols: list[str] = [opts.benchmark] if opts.benchmark else []
    symbols += [s.upper() for s in opts.symbols]
    if opts.include_cross_assets:
        symbols += list(uni.cross_asset_universe()["symbol"])
    if opts.include_sectors:
        symbols += sectors.all_symbols()
    if opts.thai not in THAI_CHOICES:
        raise ValueError(f"thai must be one of {', '.join(THAI_CHOICES)}, got {opts.thai!r}")
    if opts.thai == "core":
        symbols += thai_core()
    elif opts.thai == "all":
        symbols += list(uni.load_th_stocks()["symbol"])
    if opts.include_macro:
        symbols += list(uni.MACRO_SERIES)
    if opts.include_watchlists:
        symbols += WatchlistRepo(md.state).symbols()
    if opts.universe == "us":
        stocks = uni.filter_universe(
            uni.load_us_stocks(), min_market_cap=opts.min_market_cap,
            exclude_spacs=opts.exclude_spacs, limit=opts.limit,
        )
        symbols += list(stocks["symbol"])
    return list(dict.fromkeys(symbols))


def run_pipeline(
    opts: PipelineOptions | None = None,
    md: MarketData | None = None,
    fundamentals: FundamentalsService | None = None,
    progress: Callable[[str, int, int, str], None] | None = None,
    events: EventsService | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> PipelineResult:
    opts = opts or PipelineOptions()
    md = md or MarketData()
    result = PipelineResult(started_at=to_iso(datetime.now(timezone.utc)))
    run_id = md.state.execute(
        "INSERT INTO pipeline_runs(started_at, status) VALUES (?, 'running')", (result.started_at,))
    symbols: list[str] = []

    def step(name: str, fn: Callable[[], str]) -> None:
        t0 = time.monotonic()
        if progress:
            progress(name, 0, 0, "running")
        try:
            summary = fn()
            result.steps[name] = {"summary": f"{summary} ({time.monotonic() - t0:.1f}s)"}
        except Exception as exc:
            log.exception("pipeline step %s failed", name)
            result.steps[name] = {"error": f"{type(exc).__name__}: {exc}"}

    def resolve() -> str:
        nonlocal symbols
        symbols = resolve_symbols(opts, md)
        return f"{len(symbols)} symbols"

    step("resolve", resolve)

    def wait_for_yahoo(what: str, attempt: int) -> None:
        minutes = opts.rate_limit_wait_minutes
        if progress:
            progress("update", 0, 0, f"Yahoo is rate limiting {what} - waiting {minutes:g} min, then carrying on "
                                     f"({attempt}/{opts.rate_limit_waits})")
        log.warning("rate limited; waiting %g min (%d/%d)", minutes, attempt, opts.rate_limit_waits)
        sleep(minutes * 60)

    if opts.update_prices and symbols:
        def update() -> str:
            rounds = []
            for attempt in range(opts.rate_limit_waits + 1):
                report = md.update(
                    symbols, start=opts.start, force=opts.force,
                    progress=(lambda d, t, m: progress("update", d, t, m)) if progress else None,
                )
                rounds.append(report.summary())
                if not report.aborted or attempt == opts.rate_limit_waits:
                    break
                wait_for_yahoo("price downloads", attempt + 1)  # what is stored stays; the next round skips it
            attempted = report.updated + len(report.failed)
            # a few failed tickers are normal; most of them failing means the provider is down
            if report.aborted or (attempted and len(report.failed) / attempted >= MAX_FAILED_SHARE):
                raise RuntimeError("; ".join(rounds))
            return "; ".join(rounds)

        step("update_prices", update)

    if opts.compute_features:
        def features() -> str:
            frame = compute_universe_features(
                md.store, None, benchmark=opts.benchmark, workers=opts.workers or max(1, (os.cpu_count() or 2) - 1),
            )
            if frame.empty:
                raise RuntimeError("no price data in the store - run an update first")
            FeatureStore(md.settings.features_dir).save(frame)
            return f"{len(frame)} symbols, as of {frame['last_date'].max():%Y-%m-%d}"

        step("features", features)

    if opts.market_snapshot:
        def snapshot() -> str:
            svc = fundamentals or FundamentalsService(settings=md.settings, state=md.state)
            known = set(uni.load_us_stocks()["symbol"]) | set(uni.load_th_stocks()["symbol"]) | set(md.store.symbols())
            markets = (["us"] if opts.universe == "us" else []) + (["th"] if opts.thai != "none" else [])
            if not markets:
                return "no market selected"
            parts = []
            for market in markets:
                for attempt in range(opts.rate_limit_waits + 1):
                    try:
                        parts.append(f"{market}: {svc.refresh_market(market, only=known).summary()}")
                        break
                    except RateLimitError:
                        if attempt == opts.rate_limit_waits:
                            raise
                        wait_for_yahoo("the market snapshot", attempt + 1)
            return "; ".join(parts)

        step("market_snapshot", snapshot)

    if opts.fundamentals_limit > 0:
        def refresh() -> str:
            svc = fundamentals or FundamentalsService(settings=md.settings, state=md.state)
            stocks = [s for s in symbols if s in set(md.store.symbols())]
            return svc.refresh(stocks, limit=opts.fundamentals_limit, workers=4).summary()

        step("fundamentals", refresh)

    if opts.refresh_events:
        def calendar() -> str:
            svc = events or EventsService(store=md.store, state=md.state, settings=md.settings)
            report = svc.refresh(tracked_symbols(md.state))
            return f"{report.summary()}; {svc.refresh_economic()}"

        step("events", calendar)

    if opts.check_alerts:
        def alerts() -> str:
            fired = AlertManager(md.state).evaluate(md.store.read)
            for e in fired:
                log.info("ALERT %s: %s", e.symbol, e.message)
            return f"{len(fired)} alert(s) fired"

        step("alerts", alerts)

    result.finished_at = to_iso(datetime.now(timezone.utc))
    md.state.execute(
        "UPDATE pipeline_runs SET finished_at=?, status=?, summary=? WHERE id=?",
        (result.finished_at, "ok" if result.ok else "failed", json.dumps(result.steps), run_id),
    )
    return result


def last_runs(md: MarketData, n: int = 5) -> pd.DataFrame:
    rows = md.state.query("SELECT id, started_at, finished_at, status FROM pipeline_runs ORDER BY id DESC LIMIT ?", (n,))
    return pd.DataFrame(rows, columns=["id", "started_at", "finished_at", "status"])
