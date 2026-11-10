"""Streamlit tab for Benjamin Graham's valuation: Graham Number, growth formula, net current asset value and the defensive checklist."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from stockanalyzer.analysis.fundamental import estimates as est_mod
from stockanalyzer.analysis.fundamental import graham
from stockanalyzer.ui import charts
from stockanalyzer.ui import data as ui_data
from stockanalyzer.ui.format import pct
from stockanalyzer.ui.widgets import show_chart, show_table

RESULT = {True: "✅ pass", False: "❌ fail", None: "– n/a"}
MARGIN_DEFAULT = 25.0  # percent; a starting point for the margin of safety, Graham left the size to the investor


def _num(v: float) -> str:
    return f"{v:,.2f}" if v == v else "-"


def _margin_cell(value: float, price: float) -> str:
    """Margin of safety as text; a price far above the value reads better as a multiple than as -849%."""
    mos = graham.margin_of_safety(value, price)
    if mos != mos:
        return "-"
    return f"{mos:+.0%}" if mos >= -1 else f"price {price / value:.1f}× value"


def _margin_text(value: float, price: float) -> str | None:
    cell = _margin_cell(value, price)
    return None if cell == "-" else cell if cell.startswith("price") else f"{cell} margin of safety"


def _size_floor(price_ccy: str) -> float | None:
    """Graham's sales floor in the price currency (``None`` when there is no USD rate for it)."""
    if price_ccy == "USD":
        return graham.SIZE_FLOOR_USD
    rate = ui_data.fx_rate("USD", price_ccy)
    return graham.SIZE_FLOOR_USD * rate if rate else None


def graham_tab(symbol: str, snap: dict, info: dict, price: float, *, bank: bool = False) -> None:
    price_ccy = info.get("currency") or "USD"
    fin_ccy = info.get("financialCurrency") or price_ccy
    fx = 1.0
    if fin_ccy != price_ccy:
        fx = ui_data.fx_rate(fin_ccy, price_ccy)
        if fx is None:
            st.error(f"{symbol} reports in {fin_ccy} but its price is in {price_ccy}, and no exchange rate could be found. "
                     "Graham's per-share values would mix two currencies, so they are switched off.")
            return
        st.info(f"{symbol} reports in {fin_ccy} but trades in {price_ccy}. Earnings, book value and assets were converted at "
                f"1 {fin_ccy} = {fx:.4g} {price_ccy} (latest Yahoo rate).")

    annual = ui_data.statements(symbol, False)  # Graham's tests are about years, whatever the page's quarterly toggle says
    quarterly = ui_data.statements(symbol, True)
    hist = ui_data.valuation_history(symbol, fx)
    ttm_eps = float(hist["eps_ttm"].dropna().iloc[-1]) if not hist.empty and hist["eps_ttm"].notna().any() else None
    bvps = float(hist["bvps"].dropna().iloc[-1]) if not hist.empty and hist["bvps"].notna().any() else None
    if not bvps and info.get("bookValue"):
        bvps = float(info["bookValue"]) * fx
    cons = est_mod.consensus_growth(ui_data.estimates(symbol))

    options, default_eps = graham.eps_options(annual, ttm_eps, fx)
    growth_s = graham.suggest_growth(annual, cons.eps, fx)
    aaa = graham.default_bond_yield(price_ccy)

    st.caption("Benjamin Graham valued a company from what it earns and owns today, with a wide margin of safety, never from a long "
               "forecast. Three yardsticks: the **Graham Number** (the most a defensive investor should pay), his **growth formula** "
               "(earnings × (8.5 + 2 × growth), adjusted for bond yields) and **net current asset value** (what the company would be "
               "worth if it only sold its short-term assets and paid every debt). Pre-filled from the reported numbers - change what you disagree with.")

    p = f"graham_{symbol}_"
    by_key = {o.key: o for o in options}
    first = by_key[default_eps].value if default_eps else 0.0
    defaults = {p + "eps": default_eps or "manual", p + "manual": round(first, 2), p + "growth": growth_s.value * 100,
                p + "yield": round(aaa * 100, 1), p + "margin": MARGIN_DEFAULT}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    choices = [*by_key, "manual"]
    labels = {k: f"{o.label} · {o.value:,.2f} {price_ccy}" for k, o in by_key.items()} | {"manual": "Enter my own estimate"}
    if st.session_state[p + "eps"] not in choices:  # the option set can change when Yahoo data refreshes
        st.session_state[p + "eps"] = defaults[p + "eps"]

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.button("↺ Reset to suggested values", key=p + "reset", width="stretch", on_click=lambda: st.session_state.update(defaults))

        st.markdown("##### 1 · Earnings per share")
        choice = st.radio("Value the company on", choices, format_func=labels.get, key=p + "eps",
                          help="Graham looked at normal earnings, not one good or bad year, so the default is the average of the last three years.")
        eps = by_key[choice].value if choice in by_key else st.number_input(
            f"Earnings per share, {price_ccy}", step=0.01, format="%.2f", key=p + "manual")

        st.markdown("##### 2 · Expected growth")
        growth = st.slider("Growth per year, next 7-10 years", 0.0, 20.0, step=0.5, format="%.1f%%", key=p + "growth",
                           help="Graham's own warning: few companies grow faster than 10-15% for that long. Be conservative.") / 100
        raw = growth_s.raw
        note = ("." if raw is None or abs(raw - growth_s.value) < 0.0051 else
                f" ({raw:.1%}), held at {graham.GROWTH_CAP:.0%} because growth that fast rarely lasts for a decade." if raw > growth_s.value else
                f" ({raw:.1%}), so it starts at 0%: Graham's formula is built for earnings that at least hold up.")
        st.caption(f"Suggested {growth_s.value:.1%} from {growth_s.source}" + note)

        st.markdown("##### 3 · AAA corporate bond yield")
        bond = st.slider("Yield today", 1.5, 10.0, step=0.1, format="%.1f%%", key=p + "yield",
                         help="The formula pays less when safe bonds pay more. Graham calibrated it to 4.4%, the AAA yield of 1962.") / 100
        st.caption(f"Suggested {aaa:.1%}: the {price_ccy} risk-free rate plus {graham.AAA_SPREAD:.1%} - a rough guess, not market data. "
                   "Set it to the current yield of AAA / top-rated corporate bonds in this currency."
                   + ("" if price_ccy == "USD" else f" The 4.4% base is a 1962 US figure; with low {price_ccy} yields the formula gives high values, "
                                                    "so lean on the Graham Number and the original formula below as cross-checks."))

        st.markdown("##### 4 · Book value per share")
        book = st.number_input(f"Book value per share ({price_ccy})", min_value=0.0, value=round(bvps or 0.0, 2),
                               step=max(round((bvps or 1.0) * 0.01, 2), 0.01), key=p + f"bv{round(bvps or 0.0, 2)}",
                               help="Equity ÷ shares from the latest report.")

        st.markdown("##### 5 · Margin of safety you want")
        wanted = st.slider("Buy only this far below value", 0.0, 50.0, step=5.0, format="%.0f%%", key=p + "margin",
                           help="Graham insisted on paying clearly less than the estimated value, because the estimate can be wrong.") / 100

    with right:
        if eps <= 0:
            st.warning("Graham's Number and growth formula need positive earnings, and the earnings chosen are not. Pick another starting "
                       "point on the left (for example the 3-year average) or enter your own. Net current asset value and the checklist still apply.")
        number = graham.graham_number(eps, book)
        formula = graham.growth_value(eps, growth, bond)
        original = graham.growth_value(eps, growth)
        ncav = None if bank else graham.latest_ncav(annual, quarterly, fx)
        ncav_value = ncav[0] if ncav else float("nan")

        found = [v for v in (number, formula) if v == v]
        if found:
            lo, hi = min(found), max(found)
            if len(found) == 2:
                where = "below both" if price < lo else "above both" if price > hi else "between the two"
            else:
                where = "below it" if price < lo else "above it"
            parts = ([f"the Graham Number is **{number:,.2f}**"] if number == number else []) + \
                    ([f"his growth formula ({growth:.1%} growth, {bond:.1%} AAA yield) gives **{formula:,.2f}**"] if formula == formula else [])
            st.info(f"On {eps:,.2f} earnings and {book:,.2f} book value per share, " + " and ".join(parts)
                    + f". The price is {price:,.2f} {price_ccy}, **{where}**.")

        k = st.columns(4)
        k[0].metric("Graham Number", _num(number), _margin_text(number, price),
                    help="√(22.5 × EPS × book value per share): the price at which P/E is 15 and P/B is 1.5 together. A ceiling, not a target.")
        k[1].metric("Graham growth formula", _num(formula), _margin_text(formula, price),
                    help="EPS × (8.5 + 2 × growth) × 4.4 ÷ AAA yield. 8.5 is the P/E of a company that does not grow.")
        k[2].metric("Net current asset value / share", _num(ncav_value) if ncav else "-", _margin_text(ncav_value, price),
                    help="(Current assets − all liabilities) ÷ shares. Graham bought 'net-nets' at two thirds of this or less. Rare today, and not defined for banks.")
        k[3].metric("Price now", f"{price:,.2f}", f"P/E {price / eps:.1f} · P/B {price / book:.2f}" if eps > 0 and book > 0 else None, delta_color="off",
                    help="P/E here is the price over the earnings chosen on the left (the 3-year average by default), so it can differ from the trailing P/E above.")

        rows = [("Graham Number", number, wanted), ("Growth formula (with bond yield)", formula, wanted),
                ("Growth formula, original 1962 (no yield term)", original, wanted),
                ("Net current asset value", ncav_value, 1 - graham.NET_NET_FRACTION)]
        table = pd.DataFrame([{"Method": name, "Value / share": value, "Buy below": value * (1 - need) if value > 0 else float("nan"),
                               "Margin of safety": _margin_cell(value, price), "Where the price is": graham.reading(value, price, need)}
                              for name, value, need in rows])
        positive = {n: v for n, v, _ in rows if v == v and v > 0}
        if positive:
            show_chart(charts.method_figure(positive, price))
        show_table(table.style.format({"Value / share": "{:,.2f}", "Buy below": "{:,.2f}"}, na_rep="-"), hide_index=True)
        st.caption(f"'Buy below' takes your {wanted:.0%} margin of safety off each value; for net current asset value it is Graham's own "
                   f"two-thirds rule. Net current asset value is {'not defined for a bank' if bank else 'negative for most companies, which then fall outside this test'}"
                   + (f" (balance sheet of {ncav[1]:%b %Y})." if ncav else "."))

        implied = graham.implied_growth(price, eps, bond)
        if implied == implied:
            if implied < 0:
                st.info(f"**What is the market assuming?** At a {bond:.1%} AAA yield the price is below what Graham's formula gives a company "
                        f"that does not grow at all ({_num(graham.growth_value(eps, 0.0, bond))}).")
            else:
                st.info(f"**What is the market assuming?** At a {bond:.1%} AAA yield the price equals Graham's formula if earnings grow "
                        f"**{implied:.1%}** a year for the next 7-10 years. Compare that with the {growth:.1%} you expect"
                        + (f" and the {cons.eps:.1%} the analysts expect." if cons.eps is not None else "."))

        if eps > 0:
            growths = [round(growth + d, 4) for d in (-0.04, -0.02, 0.0, 0.02, 0.04)]
            yields = [round(bond + d, 4) for d in (-0.01, -0.005, 0.0, 0.005, 0.01)]
            grid = graham.sensitivity(eps, growths, yields)
            st.markdown("##### How sensitive is the value?")
            fig = charts.heatmap_figure(grid.rename(index=lambda v: f"{v:.1%}", columns=lambda v: f"{v:.1%}"), fmt=".1f",
                                        zmin=float(np.nanmin(grid.to_numpy())), zmax=float(np.nanmax(grid.to_numpy())), colorscale="Viridis",
                                        title=f"Growth-formula value per share (price {price:,.2f})")
            fig.update_xaxes(title_text="AAA bond yield")
            fig.update_yaxes(title_text="Expected growth")
            show_chart(fig)
            st.caption("Rows: expected growth. Columns: bond yield. The centre cell is your inputs - a point of yield or two points of growth "
                       "moves the value more than most analysts' views differ.")

    st.markdown("##### The defensive investor's checklist")
    tests = graham.defensive_checklist(annual, price=price, bvps=book or None, dividends=ui_data.dividends(symbol), fx=fx,
                                       size_floor=_size_floor(price_ccy), bank=bank, currency=price_ccy,
                                       history_start=hist.index[0] if not hist.empty else None)
    passed, known = sum(t.passed is True for t in tests), sum(t.passed is not None for t in tests)
    st.markdown(f"**{passed} / {known} tests passed**" + (f" · {len(tests) - known} could not be judged" if known < len(tests) else ""))
    show_table(pd.DataFrame([{"Test": t.name, "Graham's rule": t.rule, "Result": RESULT[t.passed],
                              "Detail": t.detail + (" · shorter record than Graham asked for" if t.partial else "")} for t in tests]),
               hide_index=True)
    st.caption("These are Graham's minimum standards for an investor who wants safety first. The earnings tests use the reported yearly figures, "
               "not the earnings chosen above. Yahoo gives only about 4-5 years of statements and the app archives every report you open, "
               "so the record grows over time; Graham asked for 10 years of earnings and 20 of dividends.")

    with st.expander("How the numbers are calculated"):
        st.markdown(
            "- **Graham Number** = √(22.5 × EPS × book value per share). 22.5 = 15 (the highest P/E he accepted) × 1.5 (the highest P/B).\n"
            "- **Growth formula** = EPS × (8.5 + 2 × g) × 4.4 ÷ Y, with g the expected yearly growth in points and Y the AAA corporate bond "
            "yield in points. The original 1962 version had no 4.4 ÷ Y term; it was added when interest rates began to move.\n"
            "- **Net current asset value** = current assets − total liabilities, per share. Graham counted every liability and gave no credit "
            "to buildings, machines or brands.\n"
            "- **Margin of safety** = 1 − price ÷ value: how far below the estimated value the price is.\n"
            "- The rules were written for US industrial companies in the 1970s. They suit mature, profitable businesses and undervalue companies "
            "whose worth is in the future (technology, growth). For those, compare the DCF tab.")
