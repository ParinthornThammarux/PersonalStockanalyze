"""Streamlit tab for the dividend deep-dive: the record, whether the company can afford it, and the XD day measured on the
stock's own history against every liquid dividend payer in its market."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from stockanalyzer.analysis import dividends as dv
from stockanalyzer.data.breadth import MARKETS
from stockanalyzer.domain.assets import currency_of
from stockanalyzer.ui import charts
from stockanalyzer.ui import data as ui_data
from stockanalyzer.ui.format import pct
from stockanalyzer.ui.widgets import show_chart, show_table

ICON = {"bad": "🔴", "warn": "🟠", "good": "🟢"}
MIN_SIZED = 3  # XDs with a dividend big enough to measure the drop on, before the drop is reported


def market_of(symbol: str) -> str | None:
    """The market a stock is compared with: Thai shares with the SET + mai, US-quoted ones with the US list."""
    ccy = currency_of(symbol)
    return "th" if ccy == "THB" else "us" if ccy == "USD" and "." not in symbol.replace("-", "") else None


def _num(value: float, fmt: str) -> str:
    return "-" if value is None or value != value else (">60" if np.isinf(value) else format(value, fmt))


def _record_section(divs: pd.Series, price: float, ccy: str, asof: pd.Timestamp, start: pd.Timestamp | None) -> None:
    rec = dv.record(divs, asof, start)
    years = dv.yearly(divs, asof)
    k = st.columns(4)
    k[0].metric("Dividends, last 12 months", f"{rec.ttm:,.2f} {ccy}", f"yield {pct(rec.ttm / price)}" if price else None,
                delta_color="off")
    k[1].metric("Years paid", f"{rec.years_paid}", f"since {rec.first_year}" if rec.first_year else None, delta_color="off",
                help="Complete calendar years with at least one ex-dividend date in the price history.")
    plus = "+" if rec.reaches_start and rec.no_cut_streak >= rec.years_paid - 1 else ""
    k[2].metric("Years without a cut", f"{rec.no_cut_streak}{plus}", f"{rec.raise_streak} raised in a row", delta_color="off",
                help=f"A year paying less than {dv.CUT_TOLERANCE:.0%} of the year before is a cut. '+' = the streak reaches the "
                     "start of the stored price history, so it may be longer.")
    k[3].metric("Growth a year (5 years)", pct(rec.cagr_5), f"3 years {pct(rec.cagr_3)}", delta_color="off")
    if not years.empty:
        full = years["dps"].where(~years["partial"])
        partial = years["dps"].where(years["partial"])
        shown = pd.DataFrame({"Complete year": full, "This year so far": partial})
        show_chart(charts.grouped_bar_figure(shown, title=f"Dividends per share by year of the ex-date ({ccy})", height=300,
                                             colors=[charts.PALETTE[0], "#bdbdbd"]))
    notes = []
    if rec.cuts:
        notes.append(f"Cut in {', '.join(map(str, rec.cuts))}.")
    notes.append("Years follow the ex-date: a Thai company's final dividend for one year goes ex the following April or May, so a "
                 "calendar year usually holds last year's final and this year's interim payment. Amounts are Yahoo's, adjusted for splits.")
    st.caption(" ".join(notes))


def _coverage_section(symbol: str, bank: bool) -> None:
    st.markdown("##### Can the company afford it?")
    cov = dv.coverage(ui_data.statements(symbol, False))
    notes = dv.coverage_notes(cov, bank=bank)
    for n in notes:
        st.markdown(f"{ICON[n.tone]} {n.text}")
    paid = cov[cov["dividends_paid"].fillna(0) > 0]
    if paid.empty:
        st.caption("The cash flow statements show no dividends paid.")
        return
    cols = ["payout_earnings"] if bank else ["payout_earnings", "payout_fcf", "shareholder_payout_fcf"]
    chart = (paid[cols] * 100).rename(columns={"payout_earnings": "Of net income", "payout_fcf": "Of free cash flow",
                                               "shareholder_payout_fcf": "Dividends + buybacks, of free cash flow"})
    chart.index = [f"{d:%Y}" for d in chart.index]
    fig = charts.grouped_bar_figure(chart, title="Dividends paid, % of ...", height=300)
    fig.add_hline(y=100, line=dict(color=charts.DOWN, width=1, dash="dot"))
    show_chart(fig)
    st.caption("Above 100% the company paid out more than it earned (or than its free cash flow) that year. Banks are judged on "
               "net income only: their cash flow follows deposits and loans." if bank else
               "Above 100% the company paid out more than it earned (or than its free cash flow) that year. A dividend that free "
               "cash flow does not cover is paid from cash on hand or borrowing, which cannot last.")


def _comparison(stock: dict, market: dict, peers: dict, label: str, peer_label: str) -> pd.DataFrame:
    rows = [
        ("XD days measured", "events", "{:,.0f}"),
        ("Dividend per XD (median yield)", "median_yield", "{:.2%}"),
        ("Share of the dividend off the price at the open", "median_drop_open", "{:.0%}"),
        ("... at the close, net of the market's move", "median_drop", "{:.0%}"),
        ("Trading days until the gap is filled (median)", "median_fill_days", "{:.0f}"),
        ("Gap filled within 5 days", "filled_5", "{:.0%}"),
        ("Gap filled within 20 days", "filled_20", "{:.0%}"),
        ("Gap filled within 60 days", "filled_60", "{:.0%}"),
        ("Return vs market, 10 days before the XD (avg)", "avg_pre_excess", "{:+.2%}"),
        ("Return vs market, 20 days after, dividend included (avg)", "avg_post_excess", "{:+.2%}"),
        ("Dividend capture after tax and costs (avg)", "avg_capture", "{:+.2%}"),
        ("Dividend capture made money", "capture_wins", "{:.0%}"),
    ]

    def cell(d: dict, key: str, fmt: str) -> str:
        v = d.get(key, np.nan)
        if v is None or v != v:
            return "-"
        return ">60" if np.isinf(v) else fmt.format(v)

    return pd.DataFrame({label: [cell(stock, k, f) for _, k, f in rows], peer_label: [cell(peers, k, f) for _, k, f in rows],
                         "All liquid payers": [cell(market, k, f) for _, k, f in rows]}, index=[r[0] for r in rows])


def _sentences(s: dict, peers: dict, sized: int, tax: float, cost: float, name: str) -> list[str]:
    out = []
    drop = s.get("median_drop", np.nan)
    if sized >= MIN_SIZED and drop == drop:
        expected = 1 - tax
        if abs(drop - expected) <= 0.15:
            how = f"about what the {tax:.0%} dividend tax predicts ({expected:.0%})"
        elif drop < expected:
            how = f"less than the {expected:.0%} that tax alone predicts - buyers keep bidding for it on the XD day"
        else:
            how = f"more than the {expected:.0%} that tax alone predicts"
        out.append(f"On the XD day {name}'s price typically gives up **{drop:.0%}** of the dividend by the close (after the market's own "
                   f"move): {how}.")
    fill, peer_fill = s.get("median_fill_days", np.nan), peers.get("median_fill_days", np.nan)
    if fill == fill:
        speed = "" if peer_fill != peer_fill else (
            f", faster than similar payers ({_num(peer_fill, '.0f')})" if fill < peer_fill else
            f", slower than similar payers ({_num(peer_fill, '.0f')})" if fill > peer_fill else ", like similar payers")
        unit = "trading day" if fill == 1 else "trading days"
        out.append(f"The price got back to its pre-XD close in a median **{_num(fill, '.0f')} {unit}**{speed}; within 20 days "
                   f"{_num(s.get('filled_20', np.nan), '.0%')} of the time.")
    pre = s.get("avg_pre_excess", np.nan)
    if pre == pre:
        out.append(f"In the 10 days before an XD the stock beat the market by **{pre:+.1%}** on average"
                   + (" - buyers arriving for the dividend." if pre > 0.005 else "." ))
    cap, wins = s.get("avg_capture", np.nan), s.get("capture_wins", np.nan)
    if cap == cap:
        out.append(f"Buying at the close before the XD and selling at the XD open to collect the dividend returned **{cap:+.2%}** on "
                   f"average after {tax:.0%} tax and {cost:.1%} costs, and made money {_num(wins, '.0%')} of the time.")
    return out


def _xd_section(symbol: str, price_ccy: str) -> None:
    st.markdown("##### What happens on the XD day")
    events = ui_data.xd_events(symbol)
    if events.empty:
        st.caption("No ex-dividend days in the stored price history.")
        return
    key = market_of(symbol)
    market = MARKETS.get(key) if key else None
    p = f"div_{symbol}_"
    st.session_state.setdefault(p + "tax", dv.DEFAULT_TAX.get(price_ccy, 0.15) * 100)
    st.session_state.setdefault(p + "cost", dv.DEFAULT_COST * 100)
    c1, c2 = st.columns(2)
    tax = c1.slider("Tax withheld on the dividend", 0.0, 30.0, step=1.0, format="%.0f%%", key=p + "tax",
                    help="Thai dividends: 10% withheld. US dividends for a Thai resident: 15% with a W-8BEN form, 30% without.") / 100
    cost = c2.slider("Trading costs, buy + sell", 0.0, 1.0, step=0.05, format="%.2f%%", key=p + "cost",
                     help="Commission, fees and VAT for the round trip; add something for the bid-ask spread on smaller stocks.") / 100

    bench = ui_data.benchmark_close(market.benchmark) if market else pd.Series(dtype="float64")
    ev = dv.event_metrics(events, bench if not bench.empty else None, tax=tax, cost=cost)
    stock = dv.summarize(ev)
    sized = int((ev["yield"] >= dv.DROP_MIN_YIELD).sum())
    market_stats, peer_stats, peer_label = {}, {}, "Similar payers"
    if market is not None:
        all_ev = ui_data.market_xd_events(key)
        floor = market.floors[market.default_floor][1]
        liquid = dv.liquid_events(dv.event_metrics(all_ev, bench if not bench.empty else None, tax=tax, cost=cost), floor)
        market_stats = dv.summarize(liquid)
        y = stock.get("median_yield", np.nan)
        if y == y and y > 0:
            similar = liquid[(liquid["yield"] >= 0.5 * y) & (liquid["yield"] <= 1.5 * y)]
            peer_stats = dv.summarize(similar)
            peer_label = f"Similar payers ({0.5 * y:.1%}-{1.5 * y:.1%} a time)"

    for line in _sentences(stock, peer_stats, sized, tax, cost, symbol.removesuffix(".BK")):
        st.markdown(f"- {line}")
    if sized < MIN_SIZED:
        st.caption(f"Only {sized} of the {len(ev)} dividends were at least {dv.DROP_MIN_YIELD:.1%} of the price: smaller ones vanish in "
                   "the day's normal price moves, so the drop ratio is not reported.")
    show_table(_comparison(stock, market_stats, peer_stats, symbol, peer_label))
    if market is not None:
        floor = market.floors[market.default_floor][1]
        sign = "฿" if market.currency == "THB" else "$"
        st.caption(f"Peers: every XD of {market.label} that traded at least {sign}{floor / 1e6:g}M a day over the 60 days before, "
                   "since the store begins. 'Similar payers' are those whose dividend was a similar share of the price - a big "
                   "dividend takes longer to fill than a small one, so that is the fair comparison.")

    observed = ev.dropna(subset=["prev_close"]).copy()
    if not observed.empty:
        days = observed["fill_days"].where(observed["fill_days"].notna(), np.where(observed["k_max"] >= 60, 61.0, np.nan))
        bars = pd.DataFrame({"Days to fill the gap": days.to_numpy()}, index=[f"{d:%Y-%m-%d}" for d in observed["ex_date"]])
        fig = charts.grouped_bar_figure(bars, title="Trading days until the price regained its pre-XD close (61 = not within 60)", height=280)
        show_chart(fig)
        table = observed.iloc[::-1][["ex_date", "amount", "yield", "drop_open", "drop_adjusted", "fill_days", "pre_excess", "post_excess",
                                     "capture"]].rename(columns={
            "ex_date": "XD", "amount": "Dividend", "yield": "Yield", "drop_open": "Off at open", "drop_adjusted": "Off at close (net of market)",
            "fill_days": "Days to fill", "pre_excess": "10 days before vs market", "post_excess": "20 days after vs market",
            "capture": "Capture trade"})
        table["XD"] = table["XD"].dt.date
        small = table["Yield"] < dv.DROP_MIN_YIELD
        table.loc[small, ["Off at open", "Off at close (net of market)"]] = np.nan
        show_table(table.style.format({"Dividend": "{:,.3f}", "Yield": "{:.2%}", "Off at open": "{:.0%}", "Off at close (net of market)": "{:.0%}",
                                       "Days to fill": "{:.0f}", "10 days before vs market": "{:+.1%}", "20 days after vs market": "{:+.1%}",
                                       "Capture trade": "{:+.2%}"}, na_rep="-"), hide_index=True)
    st.caption("'Off at open / close' = (close before the XD − price) ÷ dividend: 100% means the whole dividend came off the price. "
               "Days to fill counts trading days until a close at or above the close before the XD (0 = the same day). Every figure "
               "is from Yahoo's daily prices; one XD is an anecdote, the medians over many are the information. "
               "Only stocks listed today are in the store, so peers that were delisted are missing.")


def dividends_tab(symbol: str, info: dict, price: float, *, bank: bool = False) -> None:
    price_ccy = info.get("currency") or currency_of(symbol)
    prices = ui_data.prices(symbol, "max", "1d", False)  # puts the full history in the store if it was not there yet
    divs = ui_data.dividends(symbol)
    st.caption("The dividend record, whether the business earns enough to pay it, and what the share price does around the "
               "ex-dividend (XD) day - measured on this stock's own history and on every liquid dividend payer in its market.")
    if divs.dropna().empty:
        since = f" (prices since {prices.index[0]:%Y})" if not prices.empty else ""
        st.info(f"No dividends in the stored price history{since}.")
        return
    asof = prices.index[-1] if not prices.empty else pd.Timestamp.today()
    _record_section(divs, price, price_ccy, asof, prices.index[0] if not prices.empty else None)
    _coverage_section(symbol, bank)
    _xd_section(symbol, price_ccy)
