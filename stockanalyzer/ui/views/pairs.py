"""Pairs: correlation and cointegration inside one market, stocks with stocks or crypto with crypto.

Not a navigation page of its own: ``markets.render`` shows :func:`render_panel` as its "Pairs" tab.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from stockanalyzer.analysis import comovement as cm
from stockanalyzer.ui import charts
from stockanalyzer.ui import data as ui_data
from stockanalyzer.ui.context import services
from stockanalyzer.ui.widgets import show_chart

MARKETS = {"Stocks": "stock", "Crypto": "crypto"}
KINDS = {"stock": "a stock, ETF or stock index", "crypto": "a crypto pair such as BTC-USD"}
EXAMPLES = {"stock": "AAPL, MSFT, PTT.BK", "crypto": "SOL-USD, BNB-USD, XRP-USD"}
DEFAULT_PICKS = {"stock": ("SPY", "QQQ", "IWM", "EEM", "VNQ", "TLT", "GLD"), "crypto": ("BTC-USD", "ETH-USD")}
YEARS = {"1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5}
BARS = {"Daily": "D", "Weekly": "W"}
UNITS = {("D", "stock"): "trading days", ("D", "crypto"): "days", ("W", "stock"): "weeks", ("W", "crypto"): "weeks"}
ROLL = {"D": 60, "W": 26}
MAX_SYMBOLS = 40  # 780 pairs, ~20 s to scan
SECONDS_PER_PAIR = 0.025  # measured on 10 symbols / 2 years of daily bars
HISTORY_BARS = 5 * 366 + 10  # five years of calendar-day bars (crypto) plus a margin


# -- helpers --------------------------------------------------------------------------------------
def _parse_symbols(text: str) -> list[str]:
    return list(dict.fromkeys(s for s in text.upper().replace(";", ",").replace(" ", ",").split(",") if s))


def _window(closes: pd.DataFrame, years: int) -> pd.DataFrame:
    return closes.loc[closes.index.max() - pd.DateOffset(years=years):]


def _prime(key: str, options, default: list[str]) -> None:
    """Make the session value of a multiselect valid for ``options`` (first run: ``default``)."""
    current = st.session_state.get(key)
    allowed = set(options)
    st.session_state[key] = list(default) if current is None else [s for s in current if s in allowed]


def _num(value: float, fmt: str) -> str:
    return "-" if value is None or not np.isfinite(value) else fmt.format(value)


@st.cache_data(ttl=600, show_spinner=False, max_entries=6)
def _corr_matrix(prices: pd.DataFrame, frequency: str, method: str) -> pd.DataFrame:
    return cm.correlation_matrix(prices, method=method, frequency=frequency)


@st.cache_data(ttl=600, show_spinner=False, max_entries=6)
def _scan(prices: pd.DataFrame, frequency: str, method: str, min_corr: float) -> pd.DataFrame:
    return cm.scan_pairs(prices, frequency=frequency, method=method, min_corr=min_corr or None)


# -- choosing symbols -------------------------------------------------------------------------------
def _add_symbols(market: str, key: str) -> None:
    """Download what was typed in the box (only symbols of this market) and select it."""
    sv = services()
    c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
    typed = c1.text_input("Add symbols that are not in the list", key=f"pr_add_{market}", placeholder=EXAMPLES[market],
                          help="Comma separated, Yahoo format. Missing ones are downloaded into the local store.")
    if not c2.button("Add", key=f"pr_add_btn_{market}", width="stretch"):
        return
    wanted = _parse_symbols(typed)
    wrong = [s for s in wanted if cm.market_of(s) != market]
    if wrong:
        st.warning(f"{', '.join(wrong)} {'is' if len(wrong) == 1 else 'are'} not {KINDS[market]}: stocks are compared "
                   f"with stocks and crypto with crypto only.")
    good = [s for s in wanted if s not in wrong]
    missing = [s for s in good if not sv.md.store.exists(s)]
    if missing:
        with st.spinner(f"Downloading {', '.join(missing)} ..."):
            sv.md.update(missing)
    added = [s for s in good if sv.md.store.exists(s)]
    if len(added) < len(good):
        st.warning(f"No price history found for {', '.join(s for s in good if s not in added)}.")
    if added:
        ui_data.stored_symbols.clear()
        ui_data.stored_closes.clear()
        st.session_state[key] = list(dict.fromkeys([*st.session_state.get(key, []), *added]))


def _pick_symbols(market: str) -> list[str] | None:
    """The symbols chosen, or ``None`` when the store has none of this market (the page already said so)."""
    key = f"pr_picks_{market}"
    _add_symbols(market, key)
    stored = [s for s in ui_data.stored_symbols() if cm.market_of(s) == market]
    if not stored:
        st.info(f"No {market} prices in the local store yet. Add symbols above to download them.")
        return None
    names = ui_data.symbol_names()
    default = [s for s in DEFAULT_PICKS[market] if s in set(stored)] or stored[:5]
    _prime(key, stored, default)
    return st.multiselect("Symbols", stored, key=key, max_selections=MAX_SYMBOLS,
                          format_func=lambda s: f"{s} - {names[s]}" if names.get(s) else s,
                          help=f"Up to {MAX_SYMBOLS}. Every pair is tested, so a long list takes a while.")


# -- tabs -----------------------------------------------------------------------------------------------
def _correlation_tab(prices: pd.DataFrame, frequency: str, method: str, window: str, market: str) -> None:
    bars = "weekly" if frequency == "W" else "daily"
    mat = _corr_matrix(prices, frequency, method)
    show_chart(charts.heatmap_figure(mat, title=f"{method.capitalize()} correlation of {bars} log returns ({window})"))
    note = ("Measured on the dates both symbols traded, so a holiday on one side is not turned into a 0 % return. "
            "Blank = fewer than 30 shared returns.")
    if market == "stock":
        note += (" US and Thai stocks close hours apart, which understates daily correlation between the two; "
                 "Weekly bars remove most of that.")
    st.caption(note)


def _scan_view(table: pd.DataFrame, unit: str) -> pd.DataFrame:
    y, x = table["y"].fillna(table["a"]), table["x"].fillna(table["b"])
    return pd.DataFrame({
        "Pair": y + " ~ " + x, "Correlation": table["corr"], "Hedge ratio": table["hedge_ratio"],
        "p-value": table["p_value"], "p (adjusted)": table["p_adj"],
        f"Half-life ({unit})": table["half_life"].replace(np.inf, np.nan), "Z-score": table["zscore"],
        "Bars": table["n_obs"], "Cointegrated": table["cointegrated"], "Valid": table["both_integrated"],
    })


def _scan_table(table: pd.DataFrame, unit: str, only_hits: bool) -> None:
    shown = table[table["cointegrated"]] if only_hits else table
    if table["p_value"].isna().all():
        st.info(f"No pair had {cm.MIN_COINT_OBS} shared bars, the least the cointegration test needs. "
                f"Use a longer history or daily bars. Correlations are still in the table.")
    if shown.empty:
        st.info("No cointegrated pair in this scan." if only_hits else "No pairs to show.")
        return
    cfg = {"Correlation": st.column_config.NumberColumn(format="%.2f"),
           "Hedge ratio": st.column_config.NumberColumn(format="%.2f", help="Units of log(first) per unit of log(second) "
                                                                            "in the 'first ~ second' pair."),
           "p-value": st.column_config.NumberColumn(format="%.3f"),
           "p (adjusted)": st.column_config.NumberColumn(format="%.3f", help="Corrected for the number of pairs tested "
                                                                              "(Benjamini-Hochberg). Judge by this one."),
           f"Half-life ({unit})": st.column_config.NumberColumn(format="%.0f"),
           "Z-score": st.column_config.NumberColumn(format="%+.1f", help="Latest spread, in standard deviations from its mean."),
           "Bars": st.column_config.NumberColumn(format="%d"),
           "Cointegrated": st.column_config.CheckboxColumn(help="Adjusted p-value below 0.05 and both prices random-walk-like."),
           "Valid": st.column_config.CheckboxColumn(help="Both prices wander like random walks, which the test assumes. "
                                                         "Unticked pairs get no adjusted p-value.")}
    event = st.dataframe(_scan_view(shown, unit), hide_index=True, width="stretch", column_config=cfg,
                         on_select="rerun", selection_mode="single-row", key="pr_table")
    rows = event.selection.rows
    if rows:
        row = shown.iloc[rows[0]]
        pair = (row["y"] if pd.notna(row["y"]) else row["a"], row["x"] if pd.notna(row["x"]) else row["b"])
        if st.session_state.get("pr_picked") != pair:  # a new click, not the same selection surviving a rerun
            st.session_state["pr_picked"] = pair
            st.session_state["pr_a"], st.session_state["pr_b"] = pair
    st.caption("Click a row to inspect that pair below. 'first ~ second' is the direction with the stronger test.")


def _verdict(eg: cm.CointegrationResult) -> tuple[str, str]:
    if not np.isfinite(eg.p_value):
        return "info", (f"Only {eg.n_obs} shared bars; the cointegration test needs at least {cm.MIN_COINT_OBS}. "
                        f"Use a longer history or daily bars.")
    flat = [s for s, wanders in ((eg.y, eg.y_integrated), (eg.x, eg.x_integrated)) if not wanders]
    if flat:
        return "warning", (f"{' and '.join(flat)} does not wander like a price (it looks stationary by itself), which the "
                           f"test assumes, so the p-value of {eg.p_value:.3f} is not reliable.")
    if eg.p_value < 0.05:
        return "success", (f"Cointegrated (p = {eg.p_value:.3f}): {eg.y} and {eg.x} drift but the gap between them "
                           f"keeps returning to its mean. This p-value is for this one pair; when you picked it out of "
                           f"a scan, use the adjusted p-value there.")
    return "warning", f"No evidence of cointegration (p = {eg.p_value:.3f}): the gap between them can wander off for good."


def _inspector(prices: pd.DataFrame, frequency: str, method: str, market: str) -> None:
    st.markdown("#### Inspect a pair")
    options = list(prices.columns)
    for key, i in (("pr_a", 0), ("pr_b", 1)):
        if st.session_state.get(key) not in options:
            st.session_state[key] = options[min(i, len(options) - 1)]
    c1, c2 = st.columns(2)
    a = c1.selectbox("First", options, key="pr_a")
    b = c2.selectbox("Second", options, key="pr_b")
    if a == b:
        st.info("Pick two different symbols.")
        return
    pa, pb = prices[a], prices[b]
    corr = cm.correlation(pa, pb, method=method, frequency=frequency)
    eg = cm.cointegration(pa, pb, frequency=frequency)
    unit = UNITS[(frequency, market)]

    k = st.columns(5)
    k[0].metric("Correlation", _num(corr.corr, "{:+.2f}"),
                help=f"{method.capitalize()} correlation of returns over {corr.n_obs} shared bars; 95 % interval "
                     f"{_num(corr.ci_low, '{:+.2f}')} to {_num(corr.ci_high, '{:+.2f}')}.")
    k[1].metric("Cointegration p-value", _num(eg.p_value, "{:.3f}"), help="Engle-Granger, stronger of the two directions. "
                                                                         "Below 0.05 = cointegrated.")
    k[2].metric("Hedge ratio", _num(eg.hedge_ratio, "{:.2f}"),
                help=f"log({eg.y}) = {_num(eg.intercept, '{:.2f}')} + {_num(eg.hedge_ratio, '{:.2f}')} x log({eg.x}) + spread."
                     if np.isfinite(eg.hedge_ratio) else None)
    k[3].metric("Half-life", f"{eg.half_life:.0f} {unit}" if np.isfinite(eg.half_life) else
                ("never reverts" if np.isinf(eg.half_life) else "-"), help="Time for the spread to close half of its gap to the mean.")
    k[4].metric("Spread z-score", _num(eg.zscore, "{:+.1f}"), help="Latest spread in standard deviations from its mean.")
    kind, text = _verdict(eg)
    getattr(st, kind)(text)

    aligned = cm.align({a: pa, b: pb}, frequency)
    left, right = st.columns(2)
    with left:
        show_chart(charts.line_figure(aligned, title="Prices rebased to 100", rebase=True, height=320))
    with right:
        if len(eg.spread):
            show_chart(charts.spread_figure(eg.spread, title=f"Spread: log {eg.y} - {eg.hedge_ratio:.2f} x log {eg.x}"))
        else:
            st.empty()
    if len(eg.spread):
        st.caption(f"Above +2, {eg.y} is expensive against {eg.x}; below -2 it is cheap. That only means something while "
                   f"the pair stays cointegrated, which can end without warning.")

    roll = cm.rolling_correlation(pa, pb, window=ROLL[frequency], frequency=frequency)
    lag = cm.lead_lag(pa, pb, max_lag=3, method=method, frequency=frequency)
    lag.index = [f"{n:+d}" for n in lag.index]
    left, right = st.columns(2)
    with left:
        if len(roll):
            show_chart(charts.line_figure(roll.to_frame(f"{ROLL[frequency]}-bar correlation"), title="Rolling correlation", height=280))
    with right:
        show_chart(charts.bar_figure(lag, horizontal=False, fmt=".2f", sort=False, height=280,
                                     title=f"{a} at t against {b} at t + lag"))
    st.caption("A tall bar away from lag 0 means one moves first. Between US and Thai stocks that is mostly the "
               "closing-time gap, not a real lead.")


def _pairs_tab(prices: pd.DataFrame, picks: list[str], window: str, frequency: str, method: str, market: str) -> None:
    unit = UNITS[(frequency, market)]
    n_pairs = len(prices.columns) * (len(prices.columns) - 1) // 2
    c1, c2, c3 = st.columns([3, 2, 2], vertical_alignment="bottom")
    min_corr = c1.slider("Only test pairs with |correlation| of at least", 0.0, 0.9, 0.0, 0.05, key="pr_min_corr",
                         help="Cointegration does not need high correlation, but a floor saves time and tests fewer pairs.")
    only_hits = c2.checkbox("Cointegrated only", key="pr_only_hits")
    run = c3.button(f"Scan {n_pairs} pairs", type="primary", key="pr_scan_btn", width="stretch",
                    help=f"About {max(1, round(n_pairs * SECONDS_PER_PAIR))} s.")
    sig = (tuple(picks), window, frequency, method, min_corr)
    if run:
        with st.spinner(f"Testing {n_pairs} pairs ..."):
            st.session_state["pr_scan"] = {"sig": sig, "table": _scan(prices, frequency, method, min_corr)}
    scan = st.session_state.get("pr_scan")
    if scan:
        if scan["sig"] != sig:
            st.warning("The symbols or settings changed since this scan. Scan again to refresh it.")
        _scan_table(scan["table"], unit, only_hits)
    else:
        st.caption("Scan tests every pair for cointegration and ranks them. Or just inspect one pair below.")
    st.divider()
    _inspector(prices, frequency, method, market)


def _basket_tab(prices: pd.DataFrame, frequency: str, market: str) -> None:
    options = list(prices.columns)
    key = f"pr_basket_{market}"
    _prime(key, options, options[:4])
    chosen = st.multiselect("Basket", options, key=key, max_selections=cm.JOHANSEN_MAX_ASSETS,
                            help=f"2 to {cm.JOHANSEN_MAX_ASSETS} symbols tested together (Johansen).")
    if len(chosen) < 2:
        st.info("Pick at least two symbols.")
        return
    try:
        res = cm.johansen(prices[chosen], frequency=frequency)
    except np.linalg.LinAlgError:
        st.warning("These prices are too alike to tell apart (one is a near copy of another). Remove one.")
        return
    if not res.trace_stats:
        st.info(f"Only {res.n_obs} shared bars; the test needs at least {cm.MIN_COINT_OBS}.")
        return
    k = st.columns(2)
    k[0].metric("Cointegrating relationships", res.rank,
                help="Independent mixes of these prices that keep returning to a mean, at 95 % confidence.")
    k[1].metric("Shared bars", res.n_obs)
    trace = pd.DataFrame({"At most r relationships": range(len(res.trace_stats)), "Trace statistic": res.trace_stats,
                          "95 % critical value": res.trace_crit_95})
    trace["Rejected"] = trace["Trace statistic"] > trace["95 % critical value"]
    st.dataframe(trace, hide_index=True, width="stretch", column_config={
        "Trace statistic": st.column_config.NumberColumn(format="%.1f"),
        "95 % critical value": st.column_config.NumberColumn(format="%.1f")})
    if res.rank:
        st.markdown("**Weights of each relationship** (on log prices, first symbol = 1; a negative weight is a short)")
        st.dataframe(res.weights.style.format("{:+.2f}"), width="stretch")
    else:
        st.info("No combination of these prices keeps returning to a mean. Try a different basket or a longer history.")
    st.caption("Rank r means r independent stable mixes; the table reads top down, and the rank is the first row "
               "that is not rejected.")


# -- entry ------------------------------------------------------------------------------------------
def render_panel() -> None:
    market = MARKETS[st.radio("Compare", list(MARKETS), horizontal=True, key="pr_market",
                              help="Stocks are only compared with stocks and crypto with crypto.")]
    picks = _pick_symbols(market)
    if picks is None:
        return
    if len(picks) < 2:
        st.info("Pick at least two symbols.")
        return
    c1, c2, c3 = st.columns(3)
    window = c1.select_slider("History", list(YEARS), value="2Y", key="pr_window")
    frequency = BARS[c2.radio("Bars", list(BARS), horizontal=True, key="pr_bars",
                              help="Weekly (Friday to Friday) removes the closing-time gap between markets but has "
                                   "a quarter of the data; the cointegration test needs at least 2 years of it.")]
    method = c3.radio("Correlation", cm.METHODS[:2], horizontal=True, key="pr_method",
                      help="Spearman (ranks) is steadier when a few huge days dominate, as they do in crypto.")
    raw = ui_data.stored_closes(tuple(picks), HISTORY_BARS, False)
    prices = _window(raw, YEARS[window]).dropna(axis=1, how="all") if not raw.empty else raw
    prices = prices[[s for s in picks if s in prices.columns]]  # the order they were picked in, not alphabetical
    if prices.shape[1] < 2:
        st.info("Not enough stored prices for these symbols. Update them in Data Manager.")
        return
    st.caption(f"{prices.shape[1]} symbols, {prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}. "
               f"Returns are compared only on the dates both symbols traded.")
    tab_corr, tab_pairs, tab_basket = st.tabs(["Correlation", "Pairs & cointegration", "Basket (Johansen)"])
    with tab_corr:
        _correlation_tab(prices, frequency, method, window, market)
    with tab_pairs:
        _pairs_tab(prices, picks, window, frequency, method, market)
    with tab_basket:
        _basket_tab(prices, frequency, market)
