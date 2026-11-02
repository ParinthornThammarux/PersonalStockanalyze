"""Event calendar: earnings, dividends, splits and economic releases, plus news headlines.

Where each piece comes from:

- **Past ex-dividend dates (with amounts) and splits** are read from the price store, so a symbol's
  whole dividend history costs no extra request.
- **Earnings** (past reports with estimate / actual / surprise, and the next scheduled one) and the
  **next ex-dividend / payment dates** come from the events provider and are cached for a day.
- **Economic releases** (rate decisions, CPI, payrolls ...) of every region are cached for half a day,
  a week back and five weeks ahead.
- **Headlines** are fetched live; the UI caches them for a while.

Company events are exchange-local calendar days (a Thai report at 16:00 Bangkok is that Bangkok day).
Economic releases keep their exact UTC time so a page can show them in the viewer's timezone.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Callable, Iterable

import pandas as pd

from stockanalyzer.config import Settings, get_settings
from stockanalyzer.data.fundamentals import RefreshReport
from stockanalyzer.data.market_clock import exchange_timezone
from stockanalyzer.data.providers.base import EARNINGS_COLUMNS, EventsProvider, NewsProvider
from stockanalyzer.data.providers.news import GoogleNewsProvider
from stockanalyzer.data.providers.yfinance_provider import YFinanceProvider
from stockanalyzer.data.store import PriceStore
from stockanalyzer.domain.assets import AssetClass, infer_asset_class
from stockanalyzer.storage.state import StateDB, to_iso

log = logging.getLogger(__name__)

EARNINGS, EX_DIVIDEND, DIVIDEND_PAYMENT, SPLIT, ECONOMIC = "earnings", "ex_dividend", "dividend_payment", "split", "economic"
COMPANY_KINDS = (EARNINGS, EX_DIVIDEND, DIVIDEND_PAYMENT, SPLIT)
ALL_KINDS = (*COMPANY_KINDS, ECONOMIC)
KIND_LABELS = {EARNINGS: "Earnings", EX_DIVIDEND: "Ex-dividend (XD)", DIVIDEND_PAYMENT: "Dividend payment",
               SPLIT: "Stock split", ECONOMIC: "Economic release"}
KIND_ICONS = {EARNINGS: "📊", EX_DIVIDEND: "💰", DIVIDEND_PAYMENT: "💵", SPLIT: "✂️", ECONOMIC: "🏛️"}
PROVIDER_KINDS = (EARNINGS, EX_DIVIDEND, DIVIDEND_PAYMENT)  # the rest come from the price store

COMPANY_TTL_HOURS = 24
ERROR_RETRY_HOURS = 2
ECONOMIC_TTL_HOURS = 12
ECONOMIC_DAYS_BACK, ECONOMIC_DAYS_AHEAD = 7, 35
ECONOMIC_SCOPE = "@economic"  # events_log key of the economic calendar (symbols are the other keys)

REGIONS = {"US": "🇺🇸 United States", "TH": "🇹🇭 Thailand", "CN": "🇨🇳 China", "EU": "🇪🇺 Euro area",
           "JP": "🇯🇵 Japan", "GB": "🇬🇧 United Kingdom", "DE": "🇩🇪 Germany", "IN": "🇮🇳 India",
           "KR": "🇰🇷 South Korea", "HK": "🇭🇰 Hong Kong", "SG": "🇸🇬 Singapore", "AU": "🇦🇺 Australia",
           "CA": "🇨🇦 Canada"}
DEFAULT_REGIONS = ("US", "TH")

# releases that move whole markets; the rest (EIA stocks, regional Fed surveys ...) only on request
KEY_RELEASE = re.compile(
    r"FOMC|Fed Funds|Repo Rate|Rate Decision|Policy Rate|Interest Rate|Refinancing|Deposit (Facility )?Rate"
    r"|Bank Rate|Cash Rate|Loan Prime|\bCPI\b|PPI Final Demand|\bPCE\b|Inflation|Non-Farm Payrolls"
    r"|Unemployment Rate|Initial Jobless|JOLTS|ADP National|\bGDP\b|Retail Sales|ISM Manufacturing PMI"
    r"|ISM N-Mfg PMI|Industrial Prod|Consumer Confidence|U Mich Sentiment|Exports",
    re.IGNORECASE,
)
# sub-series that repeat a key release (index levels, GDP components, regional Fed estimates)
MINOR_RELEASE = re.compile(
    r"Cleveland Fed|Dallas Fed|Rich Fed|Index, N?SA|Deflator|Sales (Advance|Final)|Cons Spending|Ex-Autos"
    r"|Prices (Advance|Final|Fnal)",
    re.IGNORECASE,
)

EVENT_COLUMNS = ["day", "kind", "symbol", "title", "at", "region", "data"]
HISTORY_COLUMNS = ["day", "at", "eps_estimate", "eps_actual", "surprise_pct", "reaction"]


def has_company_events(symbol: str) -> bool:
    """Earnings and dividends exist for stocks and ETFs, not for indices, futures, FX, crypto or FRED series."""
    return ":" not in symbol and infer_asset_class(symbol) in (AssetClass.EQUITY, AssetClass.ETF)


def news_query(symbol: str) -> tuple[str, str]:
    """Search terms and language for a symbol's headlines: Thai news for SET shares, US English otherwise."""
    s = symbol.strip().upper()
    if s.endswith(".BK"):
        return f'"{s.removesuffix(".BK")}" หุ้น', "th"
    return f'"{s}" stock', "en"


def is_key_release(name: str) -> bool:
    return bool(KEY_RELEASE.search(name)) and not MINOR_RELEASE.search(name)


def _clean(value):
    """JSON-safe scalar: NaN -> None, numpy numbers -> float."""
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return value
    return None if math.isnan(x) else x


def _day(value) -> str | None:
    return None if value is None else pd.Timestamp(value).strftime("%Y-%m-%d")


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    df["day"] = pd.to_datetime(df["day"])
    df["at"] = pd.to_datetime(df["at"], utc=True)  # stored as naive ISO text in UTC
    return df.sort_values(["day", "at", "symbol", "kind"], ignore_index=True)


def _compact(value: float) -> str:
    a = abs(value)
    for limit, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if a >= limit:
            return f"{value / limit:,.2f} {suffix}"
    return f"{value:,.0f}"


def describe(kind: str, data: dict) -> str:
    """One-line detail of an event: EPS estimate vs actual, dividend amount, release consensus ..."""
    def num(v) -> str:
        return "-" if v is None else f"{v:,.2f}"

    if kind == EARNINGS:
        parts = []
        if data.get("eps_actual") is not None:
            parts.append(f"EPS {num(data['eps_actual'])} vs est. {num(data.get('eps_estimate'))}")
            if data.get("surprise_pct") is not None:
                parts.append(f"surprise {data['surprise_pct']:+.1f}%")
        elif data.get("eps_estimate") is not None:
            low, high = data.get("eps_low"), data.get("eps_high")
            spread = f" (range {num(low)} to {num(high)})" if low is not None and high is not None and low != high else ""
            parts.append(f"EPS est. {num(data['eps_estimate'])}{spread}")
        if data.get("revenue_estimate"):
            parts.append(f"revenue est. {_compact(data['revenue_estimate'])}")
        if data.get("window_end"):
            parts.append(f"date not confirmed (window to {data['window_end']})")
        return " · ".join(parts)
    if kind == EX_DIVIDEND:
        return f"{data['amount']:,.4g} per share" if data.get("amount") else "amount not in the data yet"
    if kind == SPLIT:
        ratio = data.get("ratio") or 0
        return f"{ratio:g}-for-1" if ratio >= 1 else (f"1-for-{1 / ratio:g} (reverse)" if ratio else "")
    if kind == ECONOMIC:
        parts = [f"for {data['period']}"] if data.get("period") else []
        for key, label in (("actual", "actual"), ("expected", "expected"), ("last", "previous")):
            if data.get(key) is not None:
                parts.append(f"{label} {data[key]:,.2f}")
        return " · ".join(parts)
    return ""


def earnings_reaction(close: pd.Series, day, at=None, tz: str = "America/New_York") -> float:
    """Close-to-close move across a report: the last close before it vs the first close that could react.

    Reports released after midday exchange time (in practice after the close) are first traded the
    next session; earlier ones, and those without a known time, the same day.
    """
    close = close.dropna()
    day = pd.Timestamp(day).normalize()
    after_close = at is not None and not pd.isna(at) and pd.Timestamp(at).tz_convert(tz).hour >= 12
    idx = close.index
    before = close[idx <= day] if after_close else close[idx < day]
    after = close[idx > day] if after_close else close[idx >= day]
    if before.empty or after.empty:
        return float("nan")
    if (day - before.index[-1]).days > 7 or (after.index[0] - day).days > 7:  # a gap in the data, not a reaction
        return float("nan")
    return float(after.iloc[0] / before.iloc[-1] - 1)


class EventsService:
    def __init__(
        self,
        provider: EventsProvider | None = None,
        store: PriceStore | None = None,
        state: StateDB | None = None,
        *,
        news: NewsProvider | None = None,
        settings: Settings | None = None,
        now: Callable[[], datetime] | None = None,
    ):
        self.settings = settings or get_settings()
        self.provider = provider or YFinanceProvider()
        self.news = news or GoogleNewsProvider()
        self.store = store or PriceStore(self.settings.prices_dir)
        self.state = state or StateDB(self.settings.state_db)
        self._now = now or (lambda: datetime.now(timezone.utc))

    def now(self) -> pd.Timestamp:
        return pd.Timestamp(self._now()).tz_convert("UTC")

    def today(self, tz: tzinfo | None = None) -> pd.Timestamp:
        """Today's date in ``tz`` (UTC when omitted) by the service's clock."""
        return pd.Timestamp(self._now().astimezone(tz or timezone.utc).date())

    # -- fetch log ------------------------------------------------------------------------------
    def _log_entries(self) -> dict[str, dict]:
        return {r["scope"]: r for r in self.state.query("SELECT * FROM events_log")}

    def _log(self, scope: str, status: str, message: str = "") -> None:
        self.state.execute(
            "INSERT INTO events_log(scope, fetched_at, status, message) VALUES (?,?,?,?) ON CONFLICT(scope) DO UPDATE "
            "SET fetched_at=excluded.fetched_at, status=excluded.status, message=excluded.message",
            (scope, to_iso(self._now()), status, message[:500]),
        )

    def _is_stale(self, entry: dict | None, max_age_hours: float) -> bool:
        if entry is None:
            return True
        limit = min(max_age_hours, ERROR_RETRY_HOURS) if entry["status"] == "error" else max_age_hours
        fetched = datetime.fromisoformat(entry["fetched_at"]).replace(tzinfo=timezone.utc)
        return self._now() - fetched >= timedelta(hours=limit)

    def last_fetched(self, scope: str) -> str | None:
        entry = self._log_entries().get(scope)
        return entry["fetched_at"] if entry else None

    # -- company events -------------------------------------------------------------------------
    def _fetch_one(self, symbol: str) -> tuple[list[tuple], dict[str, str]]:
        """Provider rows ``(kind, day, at, data)`` for one symbol, and ``{kind: error}`` for calls that failed."""
        tz = exchange_timezone(symbol)
        rows: dict[tuple[str, str], tuple] = {}
        errors: dict[str, str] = {}
        try:
            earnings = self.provider.earnings_dates(symbol)
        except Exception as exc:
            errors[EARNINGS] = f"{type(exc).__name__}: {exc}"
            earnings = pd.DataFrame(columns=EARNINGS_COLUMNS)
        for r in earnings.itertuples(index=False):
            at = pd.Timestamp(r.at)
            new_york = at.tz_convert("America/New_York")
            known = (new_york.hour, new_york.minute) != (0, 0)  # Yahoo uses New York midnight for "time not given"
            day = (at.tz_convert(tz) if known else new_york).strftime("%Y-%m-%d")
            data = {k: _clean(getattr(r, k)) for k in ("eps_estimate", "eps_actual", "surprise_pct")}
            rows[(EARNINGS, day)] = (EARNINGS, day, to_iso(at) if known else None, data)

        try:
            cal = self.provider.calendar(symbol)
        except Exception as exc:
            errors[EX_DIVIDEND] = errors[DIVIDEND_PAYMENT] = f"{type(exc).__name__}: {exc}"
            cal = {}
        if cal.get("earnings"):
            # the calendar's date can be a day off the earnings table's timed entry: enrich that one instead
            target = pd.Timestamp(cal["earnings"][0])
            near = [k for k in rows if k[0] == EARNINGS and abs((pd.Timestamp(k[1]) - target).days) <= 3]
            key = min(near, key=lambda k: abs((pd.Timestamp(k[1]) - target).days)) if near else (EARNINGS, cal["earnings"][0])
            kind, day, at, data = rows.get(key, (EARNINGS, key[1], None, {}))
            data = {**data, **{k: cal[k] for k in ("eps_estimate", "eps_low", "eps_high", "revenue_estimate")
                               if cal.get(k) is not None and data.get(k) is None}}
            if len(cal["earnings"]) > 1 and not near:  # Yahoo gives a window while the date is unconfirmed
                data["window_end"] = cal["earnings"][-1]
            rows[key] = (kind, day, at, data)
        for kind, field_name in ((EX_DIVIDEND, "ex_dividend"), (DIVIDEND_PAYMENT, "dividend_payment")):
            if cal.get(field_name):
                rows[(kind, cal[field_name])] = (kind, cal[field_name], None, {})
        return list(rows.values()), errors

    def _replace(self, symbol: str, rows: list[tuple], kinds: list[str]) -> None:
        with self.state.connect() as conn:
            conn.execute(f"DELETE FROM events WHERE symbol=? AND kind IN ({','.join('?' * len(kinds))})", (symbol, *kinds))
            conn.executemany(
                "INSERT OR REPLACE INTO events(symbol, kind, day, title, region, at, data) VALUES (?,?,?,?,?,?,?)",
                [(symbol, kind, day, KIND_LABELS[kind], "", at, json.dumps(data)) for kind, day, at, data in rows],
            )

    def refresh(
        self,
        symbols: Iterable[str],
        *,
        max_age_hours: float = COMPANY_TTL_HOURS,
        workers: int = 4,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> RefreshReport:
        """Fetch earnings and dividend dates for stale symbols in parallel (others are served from the cache)."""
        t0 = time.monotonic()
        syms = [s for s in dict.fromkeys(x.strip().upper() for x in symbols) if has_company_events(s)]
        report = RefreshReport(requested=len(syms))
        logged = self._log_entries()
        stale = [s for s in syms if self._is_stale(logged.get(s), max_age_hours)]
        report.fresh = len(syms) - len(stale)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(self._fetch_one, s): s for s in stale}
            for done, fut in enumerate(as_completed(futures), start=1):
                symbol = futures[fut]
                try:
                    rows, errors = fut.result()
                except Exception as exc:  # one bad symbol must not stop the batch
                    rows, errors = [], dict.fromkeys(PROVIDER_KINDS, f"{type(exc).__name__}: {exc}")
                if set(errors) >= set(PROVIDER_KINDS):  # nothing came back: keep what is cached, retry sooner
                    report.failed[symbol] = next(iter(errors.values()))
                    self._log(symbol, "error", report.failed[symbol])
                else:
                    self._replace(symbol, rows, [k for k in PROVIDER_KINDS if k not in errors])
                    if rows:
                        report.refreshed += 1
                    else:
                        report.empty.append(symbol)
                    self._log(symbol, "ok" if rows else "empty", "; ".join(dict.fromkeys(errors.values())))
                if progress:
                    progress(done, len(stale), symbol)
        report.elapsed = time.monotonic() - t0
        return report

    def _store_actions(self, symbols: list[str], start: str | None, end: str | None) -> list[dict]:
        """Ex-dividend dates (with amounts) and splits recorded in the price store."""
        stored = [s for s in symbols if self.store.exists(s)]
        if not stored:
            return []
        df = self.store.read_many(stored, start=start, end=end, columns=["dividends", "splits"])
        df = df[(df["dividends"] > 0) | (df["splits"] > 0)]
        rows = []
        for r in df.itertuples(index=False):
            day = r.ts.strftime("%Y-%m-%d")
            if r.dividends > 0:
                rows.append({"day": day, "kind": EX_DIVIDEND, "symbol": r.symbol, "title": KIND_LABELS[EX_DIVIDEND],
                             "at": None, "region": "", "data": {"amount": float(r.dividends)}})
            if r.splits > 0:
                rows.append({"day": day, "kind": SPLIT, "symbol": r.symbol, "title": KIND_LABELS[SPLIT],
                             "at": None, "region": "", "data": {"ratio": float(r.splits)}})
        return rows

    def company_events(self, symbols: Iterable[str], start=None, end=None, *, kinds=COMPANY_KINDS) -> pd.DataFrame:
        """Cached provider events plus the store's dividends and splits; a stored ex-date replaces the provider's."""
        syms = list(dict.fromkeys(s.strip().upper() for s in symbols))
        kinds = [k for k in kinds if k in COMPANY_KINDS]
        if not syms or not kinds:
            return _frame([])
        lo, hi = _day(start), _day(end)
        sql = (f"SELECT * FROM events WHERE symbol IN ({','.join('?' * len(syms))}) "
               f"AND kind IN ({','.join('?' * len(kinds))})")
        args: list = [*syms, *kinds]
        if lo:
            sql, args = sql + " AND day >= ?", [*args, lo]
        if hi:
            sql, args = sql + " AND day <= ?", [*args, hi]
        rows = [{**r, "data": json.loads(r["data"])} for r in self.state.query(sql, args)]
        if EX_DIVIDEND in kinds or SPLIT in kinds:
            stored = [r for r in self._store_actions(syms, lo, hi) if r["kind"] in kinds]
            seen = {(r["symbol"], r["kind"], r["day"]) for r in stored}
            rows = [r for r in rows if (r["symbol"], r["kind"], r["day"]) not in seen] + stored
        return _frame(rows)

    # -- economic releases ----------------------------------------------------------------------
    def refresh_economic(self, *, max_age_hours: float = ECONOMIC_TTL_HOURS) -> str:
        """Re-fetch the economic calendar around today when the cached copy is older than ``max_age_hours``."""
        if not self._is_stale(self._log_entries().get(ECONOMIC_SCOPE), max_age_hours):
            return "economic calendar is fresh"
        start = self.today() - pd.Timedelta(days=ECONOMIC_DAYS_BACK)
        end = self.today() + pd.Timedelta(days=ECONOMIC_DAYS_AHEAD)
        try:
            df = self.provider.economic_events(start, end)
        except Exception as exc:
            self._log(ECONOMIC_SCOPE, "error", f"{type(exc).__name__}: {exc}")
            raise
        rows = [(ECONOMIC, r.at.strftime("%Y-%m-%d"), r.event, r.region, to_iso(r.at),
                 json.dumps({"period": r.period or None, **{k: _clean(getattr(r, k)) for k in ("actual", "expected", "last", "revised")}}))
                for r in df.itertuples(index=False)]
        with self.state.connect() as conn:
            conn.execute("DELETE FROM events WHERE kind=? AND day>=? AND day<=?", (ECONOMIC, _day(start), _day(end)))
            conn.executemany("INSERT OR REPLACE INTO events(symbol, kind, day, title, region, at, data) "
                             "VALUES ('',?,?,?,?,?,?)", rows)
        self._log(ECONOMIC_SCOPE, "ok" if rows else "empty", f"{len(rows)} releases")
        return f"{len(rows)} economic releases {start:%Y-%m-%d}..{end:%Y-%m-%d}"

    def economic_events(self, start=None, end=None, *, regions: Iterable[str] | None = DEFAULT_REGIONS,
                        key_only: bool = True, tz: tzinfo | str | None = None) -> pd.DataFrame:
        """Cached releases; with ``tz`` the ``day`` column is the local day the release happens on."""
        # stored days are UTC dates: read one day either side, then re-bucket into local days
        lo = _day(pd.Timestamp(start) - pd.Timedelta(days=1)) if start is not None else None
        hi = _day(pd.Timestamp(end) + pd.Timedelta(days=1)) if end is not None else None
        sql, args = "SELECT * FROM events WHERE kind=?", [ECONOMIC]
        if lo:
            sql, args = sql + " AND day >= ?", [*args, lo]
        if hi:
            sql, args = sql + " AND day <= ?", [*args, hi]
        wanted = None if regions is None else {r.upper() for r in regions}
        rows = [{**r, "data": json.loads(r["data"])} for r in self.state.query(sql, args)
                if (wanted is None or r["region"] in wanted) and (not key_only or is_key_release(r["title"]))]
        df = _frame(rows)
        if tz is not None and not df.empty:
            df["day"] = df["at"].dt.tz_convert(tz).dt.tz_localize(None).dt.normalize()
        if start is not None:
            df = df[df["day"] >= pd.Timestamp(start).normalize()]
        if end is not None:
            df = df[df["day"] <= pd.Timestamp(end).normalize()]
        return df.sort_values(["day", "at", "region"], ignore_index=True)

    def agenda(self, symbols: Iterable[str], start=None, end=None, *, kinds=ALL_KINDS,
               regions: Iterable[str] | None = DEFAULT_REGIONS, key_only: bool = True,
               tz: tzinfo | str | None = None) -> pd.DataFrame:
        """Company events of ``symbols`` and economic releases of ``regions`` in one date-sorted frame."""
        parts = [self.company_events(symbols, start, end, kinds=kinds)]
        if ECONOMIC in kinds and regions:
            parts.append(self.economic_events(start, end, regions=regions, key_only=key_only, tz=tz))
        parts = [p for p in parts if not p.empty]
        if not parts:
            return _frame([])
        return pd.concat(parts, ignore_index=True).sort_values(["day", "at", "symbol", "kind"], ignore_index=True)

    # -- per-symbol views -----------------------------------------------------------------------
    def earnings_history(self, symbol: str, close: pd.Series | None = None) -> pd.DataFrame:
        """Cached reports newest first; ``reaction`` is the close-to-close move each one caused (needs ``close``)."""
        ev = self.company_events([symbol], kinds=(EARNINGS,))
        if ev.empty:
            return pd.DataFrame(columns=HISTORY_COLUMNS)
        tz = exchange_timezone(symbol)
        rows = [{"day": r.day, "at": r.at, **{k: r.data.get(k) for k in ("eps_estimate", "eps_actual", "surprise_pct")},
                 "reaction": earnings_reaction(close, r.day, r.at, tz) if close is not None else float("nan")}
                for r in ev.itertuples(index=False)]
        out = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
        out[["eps_estimate", "eps_actual", "surprise_pct"]] = out[["eps_estimate", "eps_actual", "surprise_pct"]].astype("float64")
        return out.sort_values("day", ascending=False, ignore_index=True)

    def dividends(self, symbol: str) -> pd.Series:
        """Every ex-dividend amount in the price store, indexed by ex-date."""
        rows = self._store_actions([symbol.strip().upper()], None, None)
        divs = {pd.Timestamp(r["day"]): r["data"]["amount"] for r in rows if r["kind"] == EX_DIVIDEND}
        return pd.Series(divs, dtype="float64", name="dividend").sort_index()

    def splits(self, symbol: str) -> pd.Series:
        """Every split in the price store as ratio by date (10.0 = 10-for-1, 0.1 = 1-for-10)."""
        rows = self._store_actions([symbol.strip().upper()], None, None)
        found = {pd.Timestamp(r["day"]): r["data"]["ratio"] for r in rows if r["kind"] == SPLIT}
        return pd.Series(found, dtype="float64", index=pd.DatetimeIndex(list(found)), name="split").sort_index()

    def dividend_summary(self, symbols: Iterable[str]) -> pd.DataFrame:
        """Per symbol: last ex-date and amount, trailing-12-month total and yield, payments a year, next known dates."""
        syms = [s for s in dict.fromkeys(x.strip().upper() for x in symbols) if has_company_events(s)]
        stored = [s for s in syms if self.store.exists(s)]
        prices = (self.store.read_many(stored, start=self.today() - pd.DateOffset(years=3), columns=["close", "dividends"])
                  if stored else pd.DataFrame(columns=["symbol", "ts", "close", "dividends"]))
        upcoming = self.company_events(syms, start=self.today(), kinds=(EX_DIVIDEND, DIVIDEND_PAYMENT))
        rows = []
        for sym in syms:
            g = prices[prices["symbol"] == sym]
            divs = g[g["dividends"] > 0]
            ttm = divs[divs["ts"] > g["ts"].max() - pd.DateOffset(years=1)] if len(g) else divs
            last_close = float(g["close"].iloc[-1]) if len(g) else float("nan")
            nxt = upcoming[upcoming["symbol"] == sym]
            rows.append({
                "symbol": sym,
                "last_ex_date": divs["ts"].iloc[-1] if len(divs) else pd.NaT,
                "last_amount": float(divs["dividends"].iloc[-1]) if len(divs) else float("nan"),
                "ttm_dividends": float(ttm["dividends"].sum()) if len(divs) else float("nan"),
                "ttm_yield": float(ttm["dividends"].sum()) / last_close if len(divs) and last_close else float("nan"),
                "payments_per_year": len(ttm),
                "next_ex_date": nxt.loc[nxt["kind"] == EX_DIVIDEND, "day"].min(),
                "next_payment": nxt.loc[nxt["kind"] == DIVIDEND_PAYMENT, "day"].min(),
            })
        return pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame()

    # -- news -----------------------------------------------------------------------------------
    def headlines(self, symbol: str, *, limit: int = 15) -> list[dict]:
        """Yahoo's stories tagged with the symbol; Thai news search for SET shares (Yahoo has none) or as a fallback."""
        symbol = symbol.strip().upper()
        items: list[dict] = []
        if not symbol.endswith(".BK"):
            try:
                items = self.provider.news(symbol, limit=limit)
            except Exception as exc:
                log.info("news(%s) from the events provider failed: %s", symbol, exc)
        if not items:
            query, lang = news_query(symbol)
            items = self.news.headlines(query, lang=lang, limit=limit)
        return [{**item, "symbol": symbol} for item in items[:limit]]

    def search_news(self, query: str, *, lang: str = "en", limit: int = 30) -> list[dict]:
        return [{**item, "symbol": ""} for item in self.news.headlines(query, lang=lang, limit=limit)]
