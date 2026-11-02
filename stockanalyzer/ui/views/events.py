"""Calendar & News: earnings, dividends, splits and economic releases for the symbols you follow, plus headlines."""

from __future__ import annotations

import calendar
import html
import re

import pandas as pd
import streamlit as st

from stockanalyzer.config import display_timezone
from stockanalyzer.data import events as ev
from stockanalyzer.pipelines.nightly import tracked_symbols
from stockanalyzer.ui import charts
from stockanalyzer.ui import data as ui_data
from stockanalyzer.ui.context import services
from stockanalyzer.ui.format import heat, pct
from stockanalyzer.ui.widgets import show_chart, show_table

WINDOWS = {"Next 7 days": (0, 7), "Next 30 days": (0, 30), "Next 90 days": (0, 90), "Past 30 days": (-30, 0)}
MAX_CHIPS = 5  # events listed in one month-grid cell before "+N more"
CHIP_COLORS = {ev.EARNINGS: "41,98,255", ev.EX_DIVIDEND: "38,166,154", ev.DIVIDEND_PAYMENT: "38,166,154",
               ev.SPLIT: "255,152,0", ev.ECONOMIC: "128,128,128"}
THAI = re.compile(r"[฀-๿]")
HEADLINE_RELEASE = re.compile(r"FOMC|Fed Funds|Repo Rate|Rate Decision|Policy Rate|Interest Rate|\bCPI\b|Payrolls|\bGDP\b|\bPCE\b",
                              re.IGNORECASE)  # named first in a busy month-grid cell
MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+!$|<>~])")

MONTH_CSS = """
<style>
.sa-wrap { overflow-x: auto; }
.sa-cal { width: 100%; min-width: 560px; border-collapse: collapse; table-layout: fixed; font-size: 0.78rem; }
.sa-cal th { padding: 4px; text-align: left; font-weight: 600; opacity: 0.7; }
.sa-cal td { vertical-align: top; height: 92px; padding: 4px; border: 1px solid rgba(128,128,128,0.25); overflow: hidden; }
.sa-cal td.sa-weekend { background: rgba(128,128,128,0.06); }
.sa-cal td.sa-out { opacity: 0.4; }
.sa-cal td.sa-today { box-shadow: inset 0 0 0 2px rgba(41,98,255,0.7); }
.sa-day { font-weight: 600; margin-bottom: 2px; }
.sa-chip { border-radius: 4px; padding: 1px 4px; margin: 1px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sa-more { opacity: 0.6; }
</style>
"""


# -- formatting helpers ---------------------------------------------------------------------------------
def md_escape(text: str) -> str:
    """Literal text inside st.markdown ('$' would start LaTeX, '[' a link ...)."""
    return MD_SPECIAL.sub(r"\\\1", text)


def when(day: pd.Timestamp, today: pd.Timestamp) -> str:
    d = (pd.Timestamp(day).normalize() - today).days
    if d == 0:
        return "today"
    if d == 1:
        return "tomorrow"
    if d == -1:
        return "yesterday"
    return f"in {d} days" if d > 0 else f"{-d} days ago"


def ago(ts: pd.Timestamp, now: pd.Timestamp) -> str:
    if ts is None or pd.isna(ts):
        return ""
    minutes = max((now - ts).total_seconds() / 60, 0)
    if minutes < 60:
        return f"{minutes:.0f} min ago"
    if minutes < 48 * 60:
        return f"{minutes / 60:.0f} h ago"
    return f"{minutes / 1440:.0f} days ago"


def your_time(at: pd.Series, tz) -> list[str]:
    return [t.tz_convert(tz).strftime("%a %d %b %H:%M") if pd.notna(t) else "" for t in at]


def company_table(df: pd.DataFrame, today: pd.Timestamp, tz) -> pd.DataFrame:
    names = ui_data.symbol_names()
    return pd.DataFrame({
        "Date": df["day"].dt.date, "When": [when(d, today) for d in df["day"]],
        "Event": [f"{ev.KIND_ICONS[k]} {ev.KIND_LABELS[k]}" for k in df["kind"]],
        "Symbol": df["symbol"], "Name": [names.get(s, "") for s in df["symbol"]],
        "Details": [ev.describe(k, d) for k, d in zip(df["kind"], df["data"])],
        "Your time": your_time(df["at"], tz),
    })


def economic_table(df: pd.DataFrame, today: pd.Timestamp, tz) -> pd.DataFrame:
    def field(name: str) -> pd.Series:
        return pd.to_numeric(pd.Series([d.get(name) for d in df["data"]], dtype="object"), errors="coerce")

    return pd.DataFrame({
        "Date": df["day"].dt.date, "When": [when(d, today) for d in df["day"]],
        "Time": [t.tz_convert(tz).strftime("%H:%M") for t in df["at"]],
        "Region": [ev.REGIONS.get(r, r) for r in df["region"]], "Release": df["title"],
        "Period": [d.get("period") or "" for d in df["data"]],
        "Actual": field("actual").to_numpy(), "Expected": field("expected").to_numpy(), "Previous": field("last").to_numpy(),
    })


def agenda_table(df: pd.DataFrame, today: pd.Timestamp, tz) -> pd.DataFrame:
    """Company events and economic releases in one compact table (Home)."""
    econ = df["kind"] == ev.ECONOMIC
    return pd.DataFrame({
        "Date": df["day"].dt.date, "When": [when(d, today) for d in df["day"]],
        "Event": [f"🏛️ {r} {t}" if e else f"{ev.KIND_ICONS[k]} {s} {ev.KIND_LABELS[k]}"
                  for e, k, s, r, t in zip(econ, df["kind"], df["symbol"], df["region"], df["title"])],
        "Details": [ev.describe(k, d) for k, d in zip(df["kind"], df["data"])],
        "Your time": your_time(df["at"], tz),
    })


NUMBER = st.column_config.NumberColumn(format="%.2f")


def chip(kind: str, label: str, tooltip: str) -> str:
    return (f"<div class='sa-chip' style='background: rgba({CHIP_COLORS[kind]},0.18)' "
            f"title='{html.escape(tooltip, quote=True)}'>{html.escape(label)}</div>")


def month_html(weeks: list[list], events: pd.DataFrame, month: int, today: pd.Timestamp, tz) -> str:
    cells: dict = {}
    company = events[events["kind"] != ev.ECONOMIC]
    for r in company.itertuples(index=False):
        tip = f"{r.symbol} · {ev.KIND_LABELS[r.kind]}" + (f" · {ev.describe(r.kind, r.data)}" if ev.describe(r.kind, r.data) else "")
        cells.setdefault(r.day.date(), []).append(chip(r.kind, f"{ev.KIND_ICONS[r.kind]} {r.symbol.removesuffix('.BK')}", tip))
    econ = events[events["kind"] == ev.ECONOMIC]
    for (day, region), g in econ.groupby([econ["day"].dt.date, "region"], sort=True):
        tip = "\n".join(f"{t.tz_convert(tz):%H:%M} {title}" for t, title in zip(g["at"], g["title"]))
        lead = sorted(g["title"], key=lambda t: not HEADLINE_RELEASE.search(t))[0]
        label = f"🏛️ {region} {lead}" if len(g) == 1 else f"🏛️ {region} ({len(g)}): {lead}"
        cells.setdefault(day, []).append(chip(ev.ECONOMIC, label, tip))

    head = "".join(f"<th>{d}</th>" for d in calendar.day_abbr)
    rows = []
    for week in weeks:
        tds = []
        for d in week:
            cls = " ".join(c for c, on in (("sa-out", d.month != month), ("sa-today", d == today.date()),
                                           ("sa-weekend", d.weekday() >= 5)) if on)
            items = cells.get(d, [])
            more = f"<div class='sa-more'>+{len(items) - MAX_CHIPS} more</div>" if len(items) > MAX_CHIPS else ""
            tds.append(f"<td class='{cls}'><div class='sa-day'>{d.day}</div>{''.join(items[:MAX_CHIPS])}{more}</td>")
        rows.append(f"<tr>{''.join(tds)}</tr>")
    return f"{MONTH_CSS}<div class='sa-wrap'><table class='sa-cal'><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


# -- loading ------------------------------------------------------------------------------------------
def load_events(symbols: list[str], *, force: bool = False, economic: bool = True) -> None:
    """Fetch what is missing or more than a day old; symbols cached today cost nothing."""
    sv = services()
    bar = None

    def progress(done: int, total: int, symbol: str) -> None:
        nonlocal bar
        bar = bar or st.progress(0.0)
        bar.progress(done / total, text=f"Loading events {done}/{total} · {symbol}")

    report = sv.events.refresh(symbols, max_age_hours=0 if force else ev.COMPANY_TTL_HOURS, progress=progress)
    if bar is not None:
        bar.empty()
    if report.failed:
        st.warning("Yahoo did not answer for " + ", ".join(report.failed) + " - showing what was cached; retried within 2 hours.")
    if economic:
        try:
            sv.events.refresh_economic(max_age_hours=0 if force else ev.ECONOMIC_TTL_HOURS)
        except Exception as exc:  # a format change upstream must not take the page down
            st.warning(f"Economic calendar unavailable right now ({exc}).")


# -- reusable blocks (also used by Stock Overview) ---------------------------------------------------
def news_list(items: list[dict], now: pd.Timestamp, *, show_symbol: bool = True) -> None:
    if not items:
        st.caption("No headlines found.")
        return
    for item in items:
        tag = f"`{item['symbol']}` · " if show_symbol and item.get("symbol") else ""
        url = item["url"].replace(" ", "%20").replace(")", "%29")
        title = f"[{md_escape(item['title'])}]({url})" if url else md_escape(item["title"])
        st.markdown(f"**{title}**  \n{tag}{md_escape(item['source'])} · {ago(item['published'], now)}")


def earnings_block(symbol: str) -> None:
    """Past reports with estimate, actual, surprise and the price move each one caused."""
    sv = services()
    close = ui_data.prices(symbol, "5y")["close"]
    hist = sv.events.earnings_history(symbol, close if not close.empty else None)
    past = hist[hist["eps_actual"].notna() | hist["reaction"].notna()]
    if past.empty:
        st.caption(f"No past earnings reports cached for {symbol}.")
        return
    surprises = past["surprise_pct"].dropna()
    moves = past["reaction"].dropna()
    k = st.columns(3)
    k[0].metric("Beat the EPS estimate", f"{int((surprises > 0).sum())} of {len(surprises)}" if len(surprises) else "-")
    k[1].metric("Average move on a report", pct(moves.abs().mean()) if len(moves) else "-",
                help="Mean absolute close-to-close move from the last close before a report to the first close after it.")
    k[2].metric("Last report's move", pct(moves.iloc[0], signed=True) if len(moves) else "-")
    view = pd.DataFrame({"Date": past["day"].dt.date, "EPS estimate": past["eps_estimate"], "EPS actual": past["eps_actual"],
                         "Surprise %": past["surprise_pct"], "Price move %": past["reaction"] * 100})
    show_table(heat(view.style.format({"EPS estimate": "{:,.2f}", "EPS actual": "{:,.2f}", "Surprise %": "{:+.1f}",
                                       "Price move %": "{:+.1f}"}, na_rep="-"),
                    subset=["Surprise %", "Price move %"], vmin=-10, vmax=10), hide_index=True)
    st.caption("Reports after midday exchange time are measured to the next session's close, earlier ones to the same day's. "
               "Past reactions describe how the stock has moved, not how it will.")


def dividends_block(symbol: str) -> None:
    divs = services().events.dividends(symbol)
    if divs.empty:
        st.caption(f"No dividends in the stored price history of {symbol}.")
        return
    by_year = divs.groupby(divs.index.year).sum().tail(10)
    by_year.index = by_year.index.astype(str)
    show_chart(charts.bar_figure(by_year, title="Dividends per share by ex-date year (this year so far)",
                                 horizontal=False, fmt=",.2f", height=280, sort=False))
    recent = divs.iloc[::-1].head(12)
    show_table(pd.DataFrame({"Ex-date": recent.index.date, "Amount": recent.to_numpy()}).style.format({"Amount": "{:,.4g}"}),
               hide_index=True)


# -- tabs --------------------------------------------------------------------------------------------
def _upcoming_tab(symbols: list[str], regions: list[str], key_only: bool, tz, today: pd.Timestamp) -> None:
    sv = services()
    c1, c2 = st.columns([2, 3])
    window = c1.radio("Window", list(WINDOWS), index=1, horizontal=True, key="cal_window")
    kinds = c2.multiselect("Company events", list(ev.COMPANY_KINDS), default=list(ev.COMPANY_KINDS), key="cal_kinds",
                           format_func=lambda k: f"{ev.KIND_ICONS[k]} {ev.KIND_LABELS[k]}")
    lo, hi = WINDOWS[window]
    start, end = today + pd.Timedelta(days=lo), today + pd.Timedelta(days=hi)

    company = sv.events.company_events(symbols, start, end, kinds=kinds)
    st.markdown(f"**Company events** · {start:%d %b} - {end:%d %b %Y}")
    if company.empty:
        st.caption("Nothing in this window for these symbols." if symbols else "No symbols selected.")
    else:
        show_table(company_table(company, today, tz), hide_index=True)

    econ = sv.events.economic_events(start, end, regions=regions, key_only=key_only, tz=tz)
    st.markdown(f"**Economic releases** · {', '.join(regions) or 'no region selected'} · times in your timezone")
    if econ.empty:
        st.caption("No releases in this window (Yahoo publishes them about five weeks ahead).")
    else:
        show_table(economic_table(econ, today, tz), hide_index=True,
                   column_config={"Actual": NUMBER, "Expected": NUMBER, "Previous": NUMBER})


def _month_tab(symbols: list[str], regions: list[str], key_only: bool, tz, today: pd.Timestamp) -> None:
    sv = services()
    months = [(today + pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(-3, 4)]
    picked = st.selectbox("Month", months, index=3, key="cal_month",
                          format_func=lambda m: pd.Timestamp(m + "-01").strftime("%B %Y"))
    year, month = map(int, picked.split("-"))
    weeks = calendar.Calendar(firstweekday=0).monthdatescalendar(year, month)
    events = sv.events.agenda(symbols, weeks[0][0], weeks[-1][-1], regions=regions, key_only=key_only, tz=tz)
    st.markdown(month_html(weeks, events, month, today, tz), unsafe_allow_html=True)
    st.caption("📊 earnings · 💰 ex-dividend (buy before this day to get the dividend) · 💵 payment · ✂️ split · "
               "🏛️ economic releases (hover for times). SET symbols are shown without .BK.")


def _earnings_tab(symbols: list[str], tz, today: pd.Timestamp) -> None:
    sv = services()
    stocks = [s for s in symbols if ev.has_company_events(s)]
    if not stocks:
        st.info("Pick stocks in the sidebar.")
        return
    nxt = sv.events.company_events(stocks, today, today + pd.Timedelta(days=120), kinds=(ev.EARNINGS,))
    st.markdown("**Next reports** (120 days)")
    if nxt.empty:
        st.caption("No scheduled reports cached for these symbols.")
    else:
        show_table(company_table(nxt, today, tz).drop(columns="Event"), hide_index=True)
    symbol = st.selectbox("Report history of", stocks, key="cal_earn_symbol")
    earnings_block(symbol)


def _dividends_tab(symbols: list[str]) -> None:
    sv = services()
    summary = sv.events.dividend_summary(symbols)
    if summary.empty or summary["last_ex_date"].isna().all():
        st.info("None of these symbols has dividends in the stored price history.")
        return
    payers = summary[summary["last_ex_date"].notna()].sort_values("ttm_yield", ascending=False)
    view = pd.DataFrame({
        "Name": [ui_data.symbol_names().get(s, "") for s in payers.index],
        "Last XD": payers["last_ex_date"].dt.date, "Amount": payers["last_amount"],
        "Last 12m": payers["ttm_dividends"], "Yield 12m %": payers["ttm_yield"] * 100,
        "Payments / yr": payers["payments_per_year"],
        "Next XD": payers["next_ex_date"].dt.date, "Next payment": payers["next_payment"].dt.date,
    }, index=payers.index)
    show_table(view.style.format({"Amount": "{:,.4g}", "Last 12m": "{:,.4g}", "Yield 12m %": "{:.2f}"}, na_rep="-"))
    st.caption("Yield = dividends with an ex-date in the last 12 months ÷ last close. Next dates appear once the company "
               "announces them. Buy before the XD date to receive the dividend.")
    symbol = st.selectbox("Dividend history of", list(payers.index), key="cal_div_symbol")
    dividends_block(symbol)


def _news_tab(symbols: list[str], now: pd.Timestamp) -> None:
    c1, c2 = st.columns([3, 2])
    picked = c1.multiselect("Headlines for", symbols, default=symbols[:5], key="cal_news_symbols")
    query = c2.text_input("Or search any topic", placeholder="e.g. Fed rate, ราคาทอง, SET index", key="cal_news_query").strip()
    if query:
        lang = "th" if THAI.search(query) else "en"
        try:
            items = ui_data.news_search(query, lang)
        except Exception as exc:
            st.warning(f"News search failed: {exc}")
            return
        news_list(items, now, show_symbol=False)
        return
    if not picked:
        st.caption("Pick symbols or type a topic.")
        return
    with st.spinner("Loading headlines ..."):
        items, errors = ui_data.headlines(tuple(picked))
    if errors:
        st.caption("No headlines for " + ", ".join(errors) + " right now.")
    news_list(items[:60], now)
    st.caption("US symbols: stories Yahoo tags with the ticker. SET symbols and topic searches: Google News (Thai for Thai text).")


def render() -> None:
    st.title("📅 Calendar & News")
    sv = services()
    tz = display_timezone()
    today = sv.events.today(tz)
    names = ui_data.symbol_names()

    tracked = tracked_symbols(sv.state)
    with st.sidebar:
        st.markdown("### Symbols")
        symbols = st.multiselect("Follow", list(dict.fromkeys([*tracked, *names])), default=tracked, key="cal_symbols",
                                 accept_new_options=True, format_func=lambda s: f"{s} - {names[s]}" if names.get(s) else s,
                                 help="Defaults to your watchlists and portfolio holdings. Type any Yahoo ticker, e.g. PTT.BK")
        st.markdown("### Economic releases")
        regions = st.multiselect("Regions", list(ev.REGIONS), default=list(ev.DEFAULT_REGIONS), key="cal_regions",
                                 format_func=lambda r: ev.REGIONS[r])
        key_only = st.toggle("Key releases only", value=True, key="cal_key_only",
                             help="Rate decisions, inflation, jobs, GDP, retail sales, PMIs, confidence, exports.")
        force = st.button("↻ Refresh from Yahoo", key="cal_refresh", width="stretch")
    symbols = [s.strip().upper() for s in symbols if s.strip()]

    load_events(symbols, force=force)
    fetched = sv.events.last_fetched(ev.ECONOMIC_SCOPE)
    st.caption(f"{len(symbols)} symbols · today {today:%a %d %b %Y} · times in your timezone (UTC{pd.Timestamp.now(tz=tz):%z})"
               + (f" · economic calendar fetched {fetched} UTC" if fetched else ""))
    if not symbols:
        st.info("Add symbols to a watchlist (or type them in the sidebar) to see their earnings and dividend dates. "
                "Economic releases are shown either way.")

    tab_up, tab_month, tab_earn, tab_div, tab_news = st.tabs(["Upcoming", "Month", "Earnings", "Dividends", "News"])
    with tab_up:
        _upcoming_tab(symbols, regions, key_only, tz, today)
    with tab_month:
        _month_tab(symbols, regions, key_only, tz, today)
    with tab_earn:
        _earnings_tab(symbols, tz, today)
    with tab_div:
        _dividends_tab(symbols)
    with tab_news:
        _news_tab(symbols, sv.events.now())
