"""Fundamentals & Valuation: statements, ratios, quality scores, DCF with sensitivity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from stockanalyzer.analysis.fundamental import dcf_inputs, ratios, scores, valuation
from stockanalyzer.domain.assets import AssetClass, infer_asset_class
from stockanalyzer.ui import charts
from stockanalyzer.ui import data as ui_data
from stockanalyzer.ui.format import money, pct, statement_table
from stockanalyzer.ui.widgets import show_chart, show_table, symbol_picker

RATIO_GROUPS = {
    "Profitability": ["gross_margin", "operating_margin", "net_margin", "roe", "roa", "roic"],
    "Safety": ["current_ratio", "quick_ratio", "debt_to_equity", "net_debt_to_ebitda", "interest_coverage"],
    "Cash flow & efficiency": ["fcf_margin", "fcf_conversion", "capex_to_revenue", "asset_turnover", "payout_ratio"],
    "Growth": ["revenue_growth", "net_income_growth", "eps_growth", "fcf_growth", "share_count_change"],
}
FRACTION_COLUMNS = {"gross_margin", "operating_margin", "net_margin", "roe", "roa", "roic", "fcf_margin", "fcf_conversion",
                    "capex_to_revenue", "payout_ratio", "revenue_growth", "net_income_growth", "eps_growth",
                    "fcf_growth", "share_count_change"}


def _ratio_table(r: pd.DataFrame, cols: list[str]):
    t = r[cols].iloc[::-1].copy()  # newest period first
    t.index = [i.strftime("%Y-%m-%d") for i in t.index]
    return t.style.format({c: "{:.1%}" if c in FRACTION_COLUMNS else "{:,.2f}" for c in cols}, na_rep="-")


def _amount_scale(exact: float) -> tuple[float, str, int, float]:
    """(scale, unit letter, decimals, value shown in that unit) for a money/count box."""
    scale, unit = (1e9, "B") if abs(exact) >= 1e9 else (1e6, "M")
    decimals = 3 if scale == 1e9 else 2
    return scale, unit, decimals, round(exact / scale, decimals)


def _amount_input(label: str, exact: float, key: str, *, help: str | None = None, min_value: float | None = None) -> float:
    """Number box in billions/millions instead of 11-digit raw numbers.

    The exact suggested amount is returned until the user edits the shown number, so the display
    rounding never leaks into the calculation.
    """
    scale, unit, decimals, shown = _amount_scale(exact)
    step = max(float(f"{abs(shown) * 0.05:.1g}"), 10.0**-decimals)
    value = st.number_input(f"{label} ({unit})", key=key, step=step, format=f"%.{decimals}f", help=help, min_value=min_value)
    return exact if value == shown else value * scale


def _dcf_tab(symbol: str, stmts, snap: dict, info: dict) -> None:
    price_ccy = info.get("currency") or "USD"
    fin_ccy = info.get("financialCurrency") or price_ccy
    fx = 1.0
    if fin_ccy != price_ccy:
        fx = ui_data.fx_rate(fin_ccy, price_ccy)
        if fx is None:
            st.error(f"{symbol} reports in {fin_ccy} but its price is in {price_ccy}, and no exchange rate could be found. "
                     "A per-share DCF would mix two currencies, so it is switched off.")
            return
    price = info.get("currentPrice") or info.get("regularMarketPrice") or float(ui_data.prices(symbol, "5d")["close"].iloc[-1])
    sg = dcf_inputs.suggest_dcf_inputs(stmts, info, price, market_cap=snap.get("market_cap"), fx=fx)

    st.caption("A DCF estimates what the business is worth today from the free cash flow it should produce. "
               "Everything is pre-filled from the company's reported numbers - change what you disagree with; "
               "the result and the bear / bull range update immediately.")
    if fx != 1.0:
        st.info(f"{symbol} reports in {fin_ccy} but trades in {price_ccy}. Cash flows and debt were converted at "
                f"1 {fin_ccy} = {fx:.4g} {price_ccy} (latest Yahoo rate) so the value is comparable with the share price.")
    for note in sg.notes:
        st.caption("ℹ️ " + note)

    # -- widget state: one key per input, per symbol and statement frequency ----------------------
    p = f"dcf_{symbol}_{'q' if stmts.quarterly else 'a'}_"
    options = {o.key: o for o in sg.base_options}
    choices = [*options, "manual"]
    labels = {k: f"{o.label} · {money(o.value)} {price_ccy}" for k, o in options.items()} | {"manual": "Enter my own estimate"}
    net_exact, shares_exact, manual_exact = sg.net_debt, sg.shares or 0.0, sg.base_fcf
    defaults = {
        p + "base": sg.default_base or "manual", p + "growth": round(sg.growth * 100, 1), p + "years": sg.years,
        p + "term": round(sg.terminal_growth * 100, 1), p + "wacc": round(sg.wacc * 100, 1),
        p + "manual": _amount_scale(manual_exact)[3], p + "net_debt": _amount_scale(net_exact)[3],
        p + "shares": _amount_scale(shares_exact)[3],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if st.session_state[p + "base"] not in choices:  # the option set can change when Yahoo data refreshes
        st.session_state[p + "base"] = defaults[p + "base"]

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.button("↺ Reset to suggested values", key=p + "reset", width="stretch",
                  on_click=lambda: st.session_state.update(defaults))

        st.markdown("##### 1 · Starting free cash flow")
        choice = st.radio("Base the forecast on", choices, format_func=labels.get, key=p + "base",
                          help="Free cash flow = cash from operations minus capital spending: what the business could pay out. "
                               "The 3-period average smooths lumpy years.")
        base_fcf = options[choice].value if choice in options else _amount_input(
            f"Free cash flow per year, {price_ccy}", manual_exact, p + "manual")

        st.markdown("##### 2 · Growth")
        growth_pct = st.slider("Growth in year 1 (per year)", -10.0, 40.0, step=0.5, format="%.1f%%", key=p + "growth",
                               help="How fast free cash flow grows next year. It then fades in a straight line to the "
                                    "long-run rate below. Suggested from the median revenue growth of the last 3 periods (kept within 0-25%).")
        years = st.slider("Years to project", 3, 15, key=p + "years",
                          help="Cash flows are forecast year by year for this long; after that a single terminal value takes over.")
        term_pct = st.slider("Long-run growth after that", 0.0, 5.0, step=0.1, format="%.1f%%", key=p + "term",
                             help="Growth forever after the projection. Keep it at or below long-run economic growth (about 2-3%).")

        st.markdown("##### 3 · Required return (WACC)")
        wacc_pct = st.slider("Discount rate", 5.0, 16.0, step=0.1, format="%.1f%%", key=p + "wacc",
                             help="The yearly return investors demand for the risk. Higher means a lower value today.")
        d = sg.wacc_detail
        st.caption(f"Suggested {sg.wacc:.1%} = cost of equity {d['cost_equity']:.1%} (4.3% risk-free + beta {d['beta']:.2f} × 5% "
                   f"equity premium) blended with after-tax debt cost by market weights.")

        with st.expander("4 · Debt and share count"):
            net_debt = _amount_input(f"Net debt, {price_ccy} (negative = net cash)", net_exact, p + "net_debt")
            shares = _amount_input("Shares outstanding", shares_exact, p + "shares", min_value=0.0,
                                   help="Yahoo's count matches the quoted price (ADR-equivalent for ADRs).")

    growth, term, wacc = round(growth_pct / 100, 6), round(term_pct / 100, 6), round(wacc_pct / 100, 6)
    with right:
        if base_fcf <= 0:
            st.warning(f"The starting free cash flow is {money(base_fcf)} {price_ccy}: a DCF cannot value a business that is "
                       "not generating cash yet. Pick another base on the left (last year, average) or enter your own estimate.")
            return
        if not shares or shares <= 0:
            st.warning("Enter the number of shares (under '4 · Debt and share count') to get a value per share.")
            return
        if wacc <= term:
            st.error("The discount rate must be above long-run growth, otherwise the terminal value is undefined.")
            return

        res = valuation.dcf(base_fcf, growth, years=years, terminal_growth=term, wacc=wacc, net_debt=net_debt,
                            shares=shares, price=price)
        scen = valuation.dcf_scenarios(base_fcf, growth, years=years, terminal_growth=term, wacc=wacc,
                                       net_debt=net_debt, shares=shares, price=price)
        lo, hi = float(scen["value_per_share"].min()), float(scen["value_per_share"].max())
        where = ("below the whole bear-to-bull range" if price < lo else
                 "above the whole bear-to-bull range" if price > hi else "inside the bear-to-bull range")
        st.info(f"With these inputs the business is worth **{res.intrinsic_per_share:,.2f} {price_ccy}** per share, "
                f"against a price of {price:,.2f} ({pct(res.upside, signed=True)}). "
                f"The price sits **{where}** ({lo:,.2f} - {hi:,.2f}).")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Intrinsic value / share", f"{res.intrinsic_per_share:,.2f}")
        m2.metric("Current price", f"{price:,.2f}")
        m3.metric("Upside / (downside)", pct(res.upside, signed=True))
        m4.metric("Terminal value share of EV", pct(res.terminal_share, 0),
                  help="Share of today's value that comes from everything after the projection years.")
        if res.terminal_share > 0.75:
            st.warning("More than 75% of the value comes from the terminal value - the result is very sensitive to the discount rate and long-run growth.")

        st.markdown("##### Bear · Base · Bull")
        show_chart(charts.scenario_figure(scen, price))
        shown = scen.rename(columns={"scenario": "Scenario", "growth": "Year-1 growth", "wacc": "Discount rate",
                                     "terminal_growth": "Long-run growth", "value_per_share": "Value / share", "upside": "vs price"})
        show_table(shown.style.format({"Year-1 growth": "{:.1%}", "Discount rate": "{:.1%}", "Long-run growth": "{:.1%}",
                                       "Value / share": "{:,.2f}", "vs price": "{:+.1%}"}, na_rep="-"), hide_index=True)
        st.caption("Bear / Bull move year-1 growth ∓5 points, the discount rate ±1 point and long-run growth ∓0.5 point around your inputs.")
        by_base = [(o.label, valuation.dcf(o.value, growth, years=years, terminal_growth=term, wacc=wacc, net_debt=net_debt,
                                           shares=shares).intrinsic_per_share) for o in sg.base_options if o.value > 0]
        if len(by_base) > 1:
            st.caption("Same growth and discount rate, different starting cash flow → "
                       + " · ".join(f"{label}: **{value:,.2f}**" for label, value in by_base))

        st.markdown("##### How sensitive is the value?")
        grid = valuation.dcf_sensitivity(base_fcf, growth, waccs=[round(wacc + x, 4) for x in (-0.02, -0.01, 0, 0.01, 0.02)],
                                         terminal_growths=[round(term + x, 4) for x in (-0.01, -0.005, 0, 0.005, 0.01)],
                                         years=years, net_debt=net_debt, shares=shares)
        if grid.notna().any().any():
            show_chart(charts.sensitivity_figure(grid, price))
            st.caption("Rows: discount rate. Columns: long-run growth. The centre cell is your inputs.")

        implied = valuation.implied_growth(price, base_fcf, years=years, terminal_growth=term, wacc=wacc,
                                           net_debt=net_debt, shares=shares)
        if np.isnan(implied):
            st.caption("Reverse DCF: no year-1 growth between -50% and +100% reproduces the current price with these inputs.")
        else:
            st.info(f"**What is the market assuming?** At a {wacc:.1%} discount rate the current price implies year-1 cash-flow growth of "
                    f"**{implied:.1%}** a year, fading to {term:.1%}. Compare that with the growth you actually expect.")

        with st.expander("Year-by-year projection"):
            show_table(res.projection.style.format({"growth": "{:.1%}", "fcf": "{:,.0f}", "discount_factor": "{:.3f}", "pv": "{:,.0f}"}),
                       hide_index=True)
        with st.expander("How the number is calculated"):
            st.markdown(
                "- Free cash flow grows at the year-1 rate, fading in a straight line to the long-run rate in the last projected year.\n"
                "- Each year's cash flow is discounted back at the discount rate (year-end convention).\n"
                "- Terminal value = last cash flow × (1 + long-run growth) ÷ (discount rate − long-run growth), also discounted back.\n"
                "- Enterprise value = sum of both; equity value = enterprise value − net debt; value per share = equity value ÷ shares.\n"
                "- The suggested discount rate uses fixed market assumptions (4.3% risk-free rate, 5% equity premium); move the slider if you disagree.")


def render() -> None:
    st.title("📑 Fundamentals & Valuation")
    symbol = symbol_picker()
    quarterly = st.sidebar.toggle("Quarterly statements", value=False)
    if not symbol:
        st.info("Enter a symbol in the sidebar.")
        return
    if infer_asset_class(symbol) is not AssetClass.EQUITY:
        st.warning("Financial statements exist for companies only (not indices, futures, FX or crypto).")
        return

    with st.spinner(f"Loading statements for {symbol} ..."):
        stmts = ui_data.statements(symbol, quarterly)
        info = ui_data.info(symbol)
    if stmts.is_empty:
        st.error(f"No financial statements available for {symbol} (ETF, recent IPO or delisted?).")
        return

    snap = valuation.snapshot_from_info(info)
    st.subheader(f"{snap.get('name') or symbol}" + (f" · {snap['sector']} · {snap['industry']}" if snap.get("sector") else ""))
    c = st.columns(6)
    c[0].metric("Market cap", money(snap.get("market_cap")))
    c[1].metric("P/E", f"{snap['pe']:.1f}" if snap.get("pe") else "-")
    c[2].metric("Fwd P/E", f"{snap['forward_pe']:.1f}" if snap.get("forward_pe") else "-")
    c[3].metric("PEG", f"{snap['peg']:.2f}" if snap.get("peg") else "-", help="Yahoo's published PEG, else P/E ÷ earnings growth. None when growth ≤ 0.")
    c[4].metric("EV/EBITDA", f"{snap['ev_ebitda']:.1f}" if snap.get("ev_ebitda") else "-")
    c[5].metric("FCF yield", pct(snap.get("fcf_yield")))

    r = ratios.compute_ratios(stmts)
    t_stat, t_ratio, t_score, t_dcf = st.tabs(["Statements", "Ratios", "Quality scores", "DCF valuation"])

    with t_stat:
        which = st.radio("Statement", ["Income", "Balance sheet", "Cash flow"], horizontal=True)
        frame = {"Income": stmts.income, "Balance sheet": stmts.balance, "Cash flow": stmts.cashflow}[which]
        show_table(statement_table(frame))

    with t_ratio:
        group = st.selectbox("Group", list(RATIO_GROUPS))
        cols = [c for c in RATIO_GROUPS[group] if c in r.columns]
        chart_df = r[cols].dropna(how="all")
        if not chart_df.empty:
            plot = chart_df * 100 if all(c in FRACTION_COLUMNS for c in cols) else chart_df
            show_chart(charts.line_figure(plot, title=group + (" (%)" if plot is not chart_df else ""), height=320))
        show_table(_ratio_table(r, cols))

    with t_score:
        pio = scores.piotroski_f_score(stmts)
        st.markdown(f"**Piotroski F-score: {pio.score} / {pio.available}** · {pio.verdict}")
        show_table(pd.DataFrame([{"signal": c.name, "result": {True: "✅ pass", False: "❌ fail", None: "– n/a"}[c.passed],
                                  "current": c.current, "previous": c.previous} for c in pio.criteria]),
                   hide_index=True)
        manufacturer = st.checkbox("Treat as manufacturer (original Altman Z)", value=False,
                                   help="Unchecked uses Z'' for non-manufacturers/services.")
        z = scores.altman_z(stmts, snap.get("market_cap"), manufacturer=manufacturer)
        if z is None:
            st.info("Altman Z could not be computed from the available statements.")
        else:
            st.markdown(f"**Altman {z.variant}: {z.z:.2f}** · zone: **{z.zone}**" + (f" · _{z.note}_" if z.note else ""))
            show_table(pd.DataFrame([z.components], index=["value"]).style.format("{:.3f}"))
            if snap.get("industry") and any(k in snap["industry"] for k in ("Banks", "Insurance", "Capital Markets")):
                st.warning("Altman Z is not meaningful for banks and insurers.")

    with t_dcf:
        _dcf_tab(symbol, stmts, snap, info)
