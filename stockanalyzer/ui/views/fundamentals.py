"""Fundamentals & Valuation: statements, ratios, quality scores, valuation bands, DCF with sensitivity, Graham value."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from stockanalyzer.analysis.fundamental import bands, dcf_inputs, ratios, scores, valuation
from stockanalyzer.analysis.fundamental import bank as bank_mod
from stockanalyzer.analysis.fundamental import estimates as est_mod
from stockanalyzer.domain.assets import AssetClass, infer_asset_class
from stockanalyzer.ui import charts
from stockanalyzer.ui import data as ui_data
from stockanalyzer.ui import dividend_panel, graham_panel, quality_panel
from stockanalyzer.ui import growth_panels as gp
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


BANK_GROUPS = {
    "Returns and capital": ["roe", "roa", "equity_to_assets", "payout_ratio"],
    "Income": ["nii_to_assets", "nii_share", "cost_provision_share"],
    "Loans": ["loans_to_assets", "loan_growth"],
    "Growth": ["net_income_growth", "eps_growth", "nii_growth"],
}
BANK_FRACTIONS = {c for cols in BANK_GROUPS.values() for c in cols}
BANK_LABELS = {
    "roe": "ROE", "roa": "ROA", "equity_to_assets": "Equity / assets", "payout_ratio": "Dividend payout",
    "nii_to_assets": "Net interest income / assets", "nii_share": "Net interest income / revenue",
    "cost_provision_share": "Costs + provisions / revenue", "loans_to_assets": "Net loans / assets",
    "loan_growth": "Loan growth", "net_income_growth": "Net income growth", "eps_growth": "EPS growth",
    "nii_growth": "Net interest income growth",
}


def _ratio_table(r: pd.DataFrame, cols: list[str], fractions=FRACTION_COLUMNS):
    t = r[cols].iloc[::-1].copy()  # newest period first
    t.index = [i.strftime("%Y-%m-%d") for i in t.index]
    return t.style.format({c: "{:.1%}" if c in fractions else "{:,.2f}" for c in cols}, na_rep="-")


def _current_price(symbol: str, info: dict) -> float:
    return info.get("currentPrice") or info.get("regularMarketPrice") or float(ui_data.prices(symbol, "5d")["close"].iloc[-1])


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
    price = _current_price(symbol, info)
    sg = dcf_inputs.suggest_dcf_inputs(stmts, info, price, market_cap=snap.get("market_cap"), fx=fx, currency=price_ccy)
    rs = dcf_inputs.suggest_revenue_inputs(stmts, info, fx=fx, currency=price_ccy, years=sg.years)
    cons = est_mod.consensus_growth(ui_data.estimates(symbol))

    st.caption("A DCF estimates what the business is worth today from the cash it should produce. "
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
    manual_exact = sg.base_fcf
    defaults = {p + "base": sg.default_base or "manual", p + "manual": gp.amount_scale(manual_exact)[3],
                **gp.shared_defaults(p, sg), **gp.revenue_defaults(p, rs)}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault(p + "method", gp.REVENUE_METHOD if gp.prefers_revenue(sg, rs) else gp.FCF_METHOD)
    if st.session_state[p + "base"] not in choices:  # the option set can change when Yahoo data refreshes
        st.session_state[p + "base"] = defaults[p + "base"]

    method = st.radio("Valuation method", [gp.FCF_METHOD, gp.REVENUE_METHOD], horizontal=True, key=p + "method",
                      help="Free cash flow starts from the cash the business makes today. Revenue × margin starts from sales and the margin "
                           "you expect it to earn once mature: the right choice when free cash flow is small or negative because the company "
                           "is still investing in growth.")
    if method == gp.REVENUE_METHOD:
        gp.revenue_panel(sg=sg, rs=rs, price=price, price_ccy=price_ccy, p=p, defaults=defaults,
                         consensus=(cons.revenue, f"revenue, next {cons.revenue_years} yr") if cons.revenue is not None else None)
        return
    if sg.base_fcf <= 0 or (rs.current_margin < 0.05 and sg.growth >= 0.15):
        st.caption("This company reinvests most of its cash in growth, so a DCF from today's free cash flow will look low. "
                   "Switch to 'Revenue × margin' above for a value based on where the margin can settle.")

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.button("↺ Reset to suggested values", key=p + "reset", width="stretch",
                  on_click=lambda: st.session_state.update(defaults))

        st.markdown("##### 1 · Starting free cash flow")
        choice = st.radio("Base the forecast on", choices, format_func=labels.get, key=p + "base",
                          help="Free cash flow = cash from operations minus capital spending: what the business could pay out. "
                               "The 3-period average smooths lumpy years.")
        base_fcf = options[choice].value if choice in options else gp.amount_input(
            f"Free cash flow per year, {price_ccy}", manual_exact, p + "manual")

        st.markdown("##### 2 · Growth")
        growth, hold, years, term = gp.growth_inputs(
            p, consensus=(cons.eps, f"EPS, next {cons.eps_years} yr") if cons.eps is not None else None)

        st.markdown("##### 3 · Required return (WACC)")
        wacc = gp.wacc_input(p, sg)
        net_debt, shares = gp.debt_inputs(p, sg, price_ccy, label="4 · Debt and share count")

    with right:
        if base_fcf <= 0:
            st.warning(f"The starting free cash flow is {money(base_fcf)} {price_ccy}: a DCF cannot value a business that is "
                       "not generating cash yet. Pick another base on the left (last year, average), enter your own estimate, "
                       "or switch to 'Revenue × margin' above.")
            return
        if not shares or shares <= 0:
            st.warning("Enter the number of shares (under '4 · Debt and share count') to get a value per share.")
            return
        if wacc <= term:
            st.error("The discount rate must be above long-run growth, otherwise the terminal value is undefined.")
            return

        res = valuation.dcf(base_fcf, growth, years=years, terminal_growth=term, wacc=wacc, net_debt=net_debt,
                            shares=shares, price=price, hold=hold)
        scen = valuation.dcf_scenarios(base_fcf, growth, years=years, terminal_growth=term, wacc=wacc,
                                       net_debt=net_debt, shares=shares, price=price, hold=hold)
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
                                           shares=shares, hold=hold).intrinsic_per_share) for o in sg.base_options if o.value > 0]
        if len(by_base) > 1:
            st.caption("Same growth and discount rate, different starting cash flow → "
                       + " · ".join(f"{label}: **{value:,.2f}**" for label, value in by_base))

        st.markdown("##### How sensitive is the value?")
        grid = valuation.dcf_sensitivity(base_fcf, growth, waccs=[round(wacc + x, 4) for x in (-0.02, -0.01, 0, 0.01, 0.02)],
                                         terminal_growths=[round(term + x, 4) for x in (-0.01, -0.005, 0, 0.005, 0.01)],
                                         years=years, net_debt=net_debt, shares=shares, hold=hold)
        if grid.notna().any().any():
            show_chart(charts.sensitivity_figure(grid, price))
            st.caption("Rows: discount rate. Columns: long-run growth. The centre cell is your inputs.")

        implied = valuation.implied_growth(price, base_fcf, years=years, terminal_growth=term, wacc=wacc,
                                           net_debt=net_debt, shares=shares, hold=hold)
        if np.isnan(implied):
            st.caption("Reverse DCF: no year-1 growth between -50% and +100% reproduces the current price with these inputs.")
        else:
            held = f"held {hold} years, then fading" if hold > 1 else "fading"
            st.info(f"**What is the market assuming?** At a {wacc:.1%} discount rate the current price implies year-1 cash-flow growth of "
                    f"**{implied:.1%}** a year ({held} to {term:.1%}). Compare that with the growth you actually expect."
                    + gp.consensus_sentence(implied, cons.eps, f"EPS, next {cons.eps_years} yr"))

        with st.expander("Year-by-year projection"):
            show_table(res.projection.style.format({"growth": "{:.1%}", "fcf": "{:,.0f}", "discount_factor": "{:.3f}", "pv": "{:,.0f}"}),
                       hide_index=True)
        with st.expander("How the number is calculated"):
            st.markdown(
                "- Free cash flow grows at the year-1 rate for the years at full speed, then fades in a straight line to the long-run rate in the last projected year.\n"
                "- Each year's cash flow is discounted back at the discount rate (year-end convention).\n"
                "- Terminal value = last cash flow × (1 + long-run growth) ÷ (discount rate − long-run growth), also discounted back.\n"
                "- Enterprise value = sum of both; equity value = enterprise value − net debt; value per share = equity value ÷ shares.\n"
                "- The suggested discount rate uses fixed market assumptions (4.3% risk-free rate, 5% equity premium); move the slider if you disagree.")


WINDOWS = {"3 years": 3, "5 years": 5, "10 years": 10, "All": None}
STANCE_TEXT = {"cheap": "on the **cheap** side of its own history", "fair": "**in line** with its own history",
               "expensive": "on the **expensive** side of its own history"}


def _bands_tab(symbol: str, snap: dict, info: dict, *, bank: bool = False) -> None:
    price_ccy = info.get("currency") or "USD"
    fin_ccy = info.get("financialCurrency") or price_ccy
    fx = 1.0
    if fin_ccy != price_ccy:
        fx = ui_data.fx_rate(fin_ccy, price_ccy)
        if fx is None:
            st.error(f"{symbol} reports in {fin_ccy} but its price is in {price_ccy}, and no exchange rate could be found, "
                     "so its multiples would mix two currencies.")
            return
        st.info(f"Statements are in {fin_ccy}, the price in {price_ccy}: earnings and book value were converted at today's "
                f"rate (1 {fin_ccy} = {fx:.4g} {price_ccy}) for the whole history, so past currency moves are ignored.")

    hist = ui_data.valuation_history(symbol, fx)
    if hist.empty:
        st.info("No price history for this symbol yet.")
        return
    yld = bands.dividend_yield_history(hist["close"], ui_data.dividends(symbol))
    # name -> (series, per-share figure behind the price bands, lower is cheaper, tick format)
    metrics = {
        "P/E": (hist["pe"], hist["eps_ttm"], True, ".1f"),
        "P/B": (hist["pb"], hist["bvps"], True, ".2f"),
        "EV/EBITDA": (hist["ev_ebitda"], None, True, ".1f"),
        "P/S": (hist["ps"], hist["sps"], True, ".1f"),
        "EV/Sales": (hist["ev_sales"], None, True, ".1f"),
        "Dividend yield": (yld["dividend_yield"], yld["dps_ttm"], False, ".1%"),
    }
    if bank:  # a bank's debt is its raw material and its "revenue" includes interest: enterprise value, EBITDA and sales multiples mean nothing
        for name in ("EV/EBITDA", "P/S", "EV/Sales"):
            metrics.pop(name)
    window = st.radio("Look-back", list(WINDOWS), index=len(WINDOWS) - 1, horizontal=True, key=f"band_window_{symbol}")
    years = WINDOWS[window]
    ready = {name: m for name, m in metrics.items() if bands.band_stats(m[0], years=years) is not None}
    if not ready:
        first = hist["eps_ttm"].first_valid_index()
        st.info("Not enough history to draw bands: a multiple needs at least 120 trading days with positive earnings / book "
                "value" + (f" (the first figures here are from {first:%b %Y})." if first is not None else ".")
                + " Each time you open a company its statements are archived, so the history grows with every report.")
        return
    name = st.radio("Multiple", list(ready), horizontal=True, key=f"band_metric_{symbol}")
    series, per_share, lower_cheaper, tick = ready[name]
    stats = bands.band_stats(series, years=years)
    known = series.dropna()
    start = known.index[-1] - pd.DateOffset(years=years) if years else known.index[0]
    fmt =(lambda v: f"{v:.1%}") if name == "Dividend yield" else (lambda v: f"{v:.2f}")

    where = bands.stance(stats, higher_is_cheaper=not lower_cheaper)
    st.info(f"{name} is **{fmt(stats.current)}**, {STANCE_TEXT[where]}: higher than {stats.percentile:.0%} of trading days over the "
            f"{stats.years:.1f} years measured (average {fmt(stats.mean)}, ±1 SD {fmt(max(stats.level(-1), 0))} - {fmt(stats.level(1))}).")
    m = st.columns(4)
    m[0].metric(f"{name} now", fmt(stats.current))
    m[1].metric("Average", fmt(stats.mean), help="Mean of the daily multiple over the window, outliers removed.")
    m[2].metric("Percentile", f"{stats.percentile:.0%}", help="Share of days in the window with a multiple at or below today's.")
    m[3].metric("Distance from average", f"{stats.zscore:+.1f} SD" if stats.zscore == stats.zscore else "-")

    if per_share is not None:
        implied = bands.implied_prices(per_share, stats, inverse=not lower_cheaper)
        show_chart(charts.band_figure(hist["close"].loc[start:], implied.loc[start:], title=f"Price vs {name} bands ({price_ccy})"))
        st.caption("Each line is the price the company's own average multiple (or ±1 / ±2 standard deviations) would imply "
                   "given the earnings, book value or dividends known on that day.")
    show_chart(charts.multiple_figure(series.loc[start:], stats, name=name, tickformat=tick))

    notes = [f"{stats.days:,} trading days measured" + (f"; {stats.excluded} days with near-zero earnings ignored" if stats.excluded else "")]
    first = hist["eps_ttm"].first_valid_index()
    if first is not None:
        notes.append(f"statements start {first:%b %Y} (Yahoo gives about 4-5 years; older periods exist only once the app has archived them)")
    st.caption(" · ".join(notes))
    mine, yahoo = hist["pe"].dropna(), snap.get("pe")
    if name == "P/E" and len(mine) and yahoo and abs(mine.iloc[-1] / yahoo - 1) > 0.25:
        st.warning(f"The P/E here ({mine.iloc[-1]:.1f}) differs from Yahoo's trailing P/E ({yahoo:.1f}). Per-share figures in the statements "
                   "can be on another share basis than the quoted price (ADRs) or the last report may be stale, so read this band as indicative.")


def _bank_tab(symbol: str, snap: dict, info: dict) -> None:
    """Justified P/B: what a bank's return on equity is worth against the return shareholders require."""
    price_ccy = info.get("currency") or "USD"
    fin_ccy = info.get("financialCurrency") or price_ccy
    fx = 1.0
    if fin_ccy != price_ccy:
        fx = ui_data.fx_rate(fin_ccy, price_ccy)
        if fx is None:
            st.error(f"{symbol} reports in {fin_ccy} but its price is in {price_ccy}, and no exchange rate could be found.")
            return
    annual = ui_data.statements(symbol, False)  # ROE must be yearly, whatever the page's quarterly toggle says
    m = bank_mod.metrics(annual)
    price = info.get("currentPrice") or info.get("regularMarketPrice") or float(ui_data.prices(symbol, "5d")["close"].iloc[-1])
    bv = info.get("bookValue") or (float(m["bvps"].dropna().iloc[-1]) if m["bvps"].notna().any() else None)
    if not bv or bv <= 0:
        st.warning("No book value per share is available for this bank, so a price-to-book valuation cannot be made.")
        return
    bv = float(bv) * fx
    yearly_roe = m["roe"].dropna()
    refs = {"Average of the last 3 years": float(yearly_roe.tail(3).mean()) if len(yearly_roe) else None,
            "Latest year": float(yearly_roe.iloc[-1]) if len(yearly_roe) else None,
            "Yahoo trailing": info.get("returnOnEquity")}
    start_roe = next((v for v in refs.values() if v is not None and v == v), 0.10)
    coe = bank_mod.cost_of_equity(info.get("beta"), price_ccy)
    start_g = bank_mod.LONG_RUN_GROWTH.get(price_ccy, 0.025)

    st.caption("Banks live on borrowed money, so free cash flow, EBITDA and working capital say nothing about them. What matters is "
               "what shareholders earn on their equity (ROE) against what they require (cost of equity): "
               "a fair price-to-book is (ROE − growth) ÷ (cost of equity − growth). Pre-filled from the bank's own numbers - "
               "change what you disagree with.")
    p = f"bank_{symbol}_"
    defaults = {p + "roe": round(start_roe * 100, 1), p + "coe": round(coe.value * 100, 1), p + "g": round(start_g * 100, 1)}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.button("↺ Reset to suggested values", key=p + "reset", width="stretch", on_click=lambda: st.session_state.update(defaults))
        st.markdown("##### 1 · Return on equity")
        roe = st.slider("Sustainable ROE", 0.0, 30.0, step=0.5, format="%.1f%%", key=p + "roe",
                        help="What the bank can keep earning on its equity. Averages smooth a good or bad year.") / 100
        st.caption(" · ".join(f"{k}: **{v:.1%}**" for k, v in refs.items() if v is not None and v == v) or "No ROE history available.")
        st.markdown("##### 2 · Required return (cost of equity)")
        cost = st.slider("Cost of equity", 4.0, 18.0, step=0.1, format="%.1f%%", key=p + "coe",
                         help="The yearly return shareholders demand for the risk. A bank with weak loans or thin capital deserves more.") / 100
        beta_note = (f"Yahoo's beta {coe.raw_beta:.2f} was raised to {coe.beta:.2f}: " if coe.clipped and coe.raw_beta is not None
                     else "" if coe.raw_beta is not None else "No beta available, so ")
        st.caption(f"Suggested {coe.value:.1%} = risk-free {coe.risk_free:.1%} + beta {coe.beta:.2f} × premium {coe.premium:.1%}. "
                   f"{beta_note}{'a bank is a leveraged, cyclical business and is not assumed less risky than the market. ' if coe.clipped else ''}"
                   f"The rates are rough starting points for {price_ccy} - set the discount rate to the current 10-year government yield plus the premium you require.")
        st.markdown("##### 3 · Growth")
        g = st.slider("Long-run growth", 0.0, 5.0, step=0.1, format="%.1f%%", key=p + "g",
                      help="Growth of earnings and book value forever. Keep it at or below long-run economic growth.") / 100
        book = st.number_input(f"Book value per share ({price_ccy})", min_value=0.01, value=round(bv, 2), step=max(round(bv * 0.01, 2), 0.01),
                               key=p + f"bv{round(bv, 2)}", help="Yahoo's latest book value per share; changes after each report.")

    with right:
        if cost <= g:
            st.error("The cost of equity must be above long-run growth, otherwise the value is undefined.")
            return
        pb_now = price / book
        fair_pb = bank_mod.justified_pb(roe, cost, g)
        value = fair_pb * book
        upside = value / price - 1
        verdict = ("**cheaper** than" if upside > 0.10 else "**more expensive** than" if upside < -0.10 else "**close to**")
        st.info(f"At a {roe:.1%} return on equity against a {cost:.1%} requirement, the bank deserves **{fair_pb:.2f}× book value**, "
                f"worth **{value:,.2f} {price_ccy}** per share. The price is {price:,.2f} ({pb_now:.2f}× book) - {verdict} that value.")
        k = st.columns(4)
        k[0].metric("Justified P/B", f"{fair_pb:.2f}×", help="(ROE − growth) ÷ (cost of equity − growth)")
        k[1].metric("Value / share", f"{value:,.2f}", pct(upside, signed=True) + " vs price")
        k[2].metric("P/B now", f"{pb_now:.2f}×")
        k[3].metric("ROE − cost of equity", f"{(roe - cost) * 100:+.1f} pts",
                    help="Positive: the bank creates value above its book. Negative: it destroys it, and deserves less than 1× book.")
        implied = bank_mod.implied_roe(pb_now, cost, g)
        st.caption(f"What is the market assuming? At a {cost:.1%} cost of equity, {pb_now:.2f}× book implies the bank keeps earning "
                   f"**{implied:.1%}** on equity forever - compare it with the {roe:.1%} you expect.")

        scen = bank_mod.scenarios(book, roe, cost, g, price)
        st.markdown("##### Bear · Base · Bull")
        show_chart(charts.scenario_figure(scen, price))
        shown = scen.rename(columns={"scenario": "Scenario", "roe": "ROE", "coe": "Cost of equity", "justified_pb": "Justified P/B",
                                     "value_per_share": "Value / share", "upside": "vs price"})
        show_table(shown.style.format({"ROE": "{:.1%}", "Cost of equity": "{:.1%}", "Justified P/B": "{:.2f}×", "Value / share": "{:,.2f}",
                                       "vs price": "{:+.1%}"}, na_rep="-"), hide_index=True)
        st.caption("Bear takes 2 points off the ROE and adds 1 point to the cost of equity; Bull does the opposite.")

        st.markdown("##### How sensitive is the value?")
        roes = [round(roe + x, 4) for x in (-0.04, -0.02, 0, 0.02, 0.04)]
        coes = [round(cost + x, 4) for x in (-0.02, -0.01, 0, 0.01, 0.02)]
        grid = bank_mod.sensitivity(book, roes, coes, g)
        if grid.notna().any().any():
            fig = charts.heatmap_figure(grid.rename(index=lambda v: f"{v:.1%}", columns=lambda v: f"{v:.1%}"), fmt=".1f",
                                        zmin=float(np.nanmin(grid.to_numpy())), zmax=float(np.nanmax(grid.to_numpy())), colorscale="Viridis",
                                        title=f"Value per share (price {price:,.2f})")
            fig.update_xaxes(title_text="ROE")
            fig.update_yaxes(title_text="Cost of equity")
            show_chart(fig)
            st.caption("Rows: cost of equity. Columns: ROE. The centre cell is your inputs - one point of cost of equity moves the value more than most analysts' views differ.")

        paid = ui_data.dividends(symbol)
        dps = info.get("dividendRate") or (float(paid.tail(4).sum()) if not paid.empty else None)
        ddm = bank_mod.ddm_value((dps or 0.0) * fx, cost, g)
        if ddm == ddm:
            st.caption(f"Cross-check, dividends only: {(dps or 0.0) * fx:,.2f} {price_ccy} a year growing {g:.1%} and discounted at {cost:.1%} "
                       f"is worth **{ddm:,.2f}** per share. It is lower than the P/B value when the bank keeps more profit than it pays out.")
        with st.expander("What Yahoo cannot tell you"):
            st.markdown("- **Asset quality** - the share of bad loans (NPL) and how well they are provided for.\n"
                        "- **Capital** - the regulatory capital ratio (CAR / CET1) and whether the dividend is safe.\n"
                        "- Read both in the bank's own report; weak numbers justify a higher cost of equity (raise the slider).\n"
                        "- Loan figures from Yahoo are patchy, so treat the loan ratios as a rough guide.")


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
    is_bank = bank_mod.is_bank(snap.get("industry"))
    st.subheader(f"{snap.get('name') or symbol}" + (f" · {snap['sector']} · {snap['industry']}" if snap.get("sector") else ""))
    c = st.columns(6)
    c[0].metric("Market cap", money(snap.get("market_cap")))
    c[1].metric("P/E", f"{snap['pe']:.1f}" if snap.get("pe") else "-")
    if is_bank:
        bank_ratios = bank_mod.metrics(stmts)
        capital = bank_ratios["equity_to_assets"].dropna()
        c[2].metric("P/B", f"{snap['pb']:.2f}" if snap.get("pb") else "-", help="Price ÷ book value per share: the anchor for valuing a bank.")
        c[3].metric("ROE", pct(snap.get("roe")), help="Yahoo's trailing return on equity.")
        c[4].metric("Dividend yield", pct(snap.get("dividend_yield")))
        c[5].metric("Equity / assets", pct(float(capital.iloc[-1])) if len(capital) else "-", help="Capital cushion: how much of the balance sheet is shareholders' money.")
        st.caption("🏦 Bank mode: free cash flow, EBITDA, Altman Z and working-capital ratios do not describe a bank, so they are replaced "
                   "by return on equity, capital, income mix and a justified price-to-book valuation.")
    else:
        c[2].metric("Fwd P/E", f"{snap['forward_pe']:.1f}" if snap.get("forward_pe") else "-")
        c[3].metric("PEG", f"{snap['peg']:.2f}" if snap.get("peg") else "-", help="Yahoo's published PEG, else P/E ÷ earnings growth. None when growth ≤ 0.")
        c[4].metric("EV/EBITDA", f"{snap['ev_ebitda']:.1f}" if snap.get("ev_ebitda") else "-")
        c[5].metric("FCF yield", pct(snap.get("fcf_yield")))

    r = bank_mod.metrics(stmts) if is_bank else ratios.compute_ratios(stmts)
    groups, fractions, label_of = (BANK_GROUPS, BANK_FRACTIONS, BANK_LABELS) if is_bank else (RATIO_GROUPS, FRACTION_COLUMNS, {})
    tab_names = ["Statements", "Ratios", "Quality scores", "Earnings quality", "Dividends", "Valuation bands",
                 "Bank valuation" if is_bank else "DCF valuation", "Graham value"]
    if not is_bank:
        tab_names.append("Growth profile")
    t_stat, t_ratio, t_score, t_quality, t_div, t_bands, t_dcf, t_graham, *t_growth = st.tabs(tab_names)

    with t_stat:
        which = st.radio("Statement", ["Income", "Balance sheet", "Cash flow"], horizontal=True)
        frame = {"Income": stmts.income, "Balance sheet": stmts.balance, "Cash flow": stmts.cashflow}[which]
        show_table(statement_table(frame))

    with t_ratio:
        group = st.selectbox("Group", list(groups))
        cols = [c for c in groups[group] if c in r.columns]
        chart_df = r[cols].dropna(how="all")
        if not chart_df.empty:
            plot = chart_df * 100 if all(c in fractions for c in cols) else chart_df
            show_chart(charts.line_figure(plot.rename(columns=label_of), title=group + (" (%)" if plot is not chart_df else ""), height=320))
        show_table(_ratio_table(r, cols, fractions).relabel_index([label_of.get(c, c) for c in cols], axis=1))
        if is_bank:
            st.caption("Net interest income / assets approximates the net interest margin (Yahoo has no earning-asset figure). "
                       "Costs + provisions / revenue is one minus pre-tax profit ÷ revenue. Yahoo's loan data is patchy.")

    with t_score:
        if is_bank:
            checks = bank_mod.checks(stmts)
            passed = sum(c.passed for c in checks if c.passed is not None)
            known = sum(c.passed is not None for c in checks)
            st.markdown(f"**Bank health checks: {passed} / {known}** on the two latest {'quarters' if quarterly else 'years'}")
            show_table(pd.DataFrame([{"signal": c.name, "result": {True: "✅ pass", False: "❌ fail", None: "– n/a"}[c.passed],
                                      "current": c.current, "previous": c.previous} for c in checks]), hide_index=True)
            st.caption("The Piotroski F-score and Altman Z lean on cash flow, gross margin and working capital, which a bank does not have, "
                       "so these six checks replace them. They say which way the bank is moving, not how safe its loans are.")
        else:
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
                if snap.get("industry") and any(k in snap["industry"] for k in ("Insurance", "Capital Markets")):
                    st.warning("Altman Z is not meaningful for insurers and brokers.")

    with t_quality:
        quality_panel.quality_tab(symbol, snap, info, _current_price(symbol, info), bank=is_bank)

    with t_div:
        dividend_panel.dividends_tab(symbol, info, _current_price(symbol, info), bank=is_bank)

    with t_bands:
        _bands_tab(symbol, snap, info, bank=is_bank)

    with t_dcf:
        if is_bank:
            _bank_tab(symbol, snap, info)
        else:
            _dcf_tab(symbol, stmts, snap, info)

    with t_graham:
        graham_panel.graham_tab(symbol, snap, info, _current_price(symbol, info), bank=is_bank)

    if t_growth:
        with t_growth[0]:
            gp.growth_profile_tab(symbol, stmts, snap, info, _current_price(symbol, info), info.get("currency") or "USD")
