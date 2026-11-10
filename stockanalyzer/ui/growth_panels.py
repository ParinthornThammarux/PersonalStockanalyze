"""Streamlit pieces for valuing growth companies: the DCF inputs both methods share, the revenue x margin model and the
growth-profile tab (Rule of 40, sales multiples and analyst consensus)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from stockanalyzer.analysis.fundamental import dcf_inputs, ratios, valuation
from stockanalyzer.analysis.fundamental import estimates as est_mod
from stockanalyzer.domain.estimates import PERIOD_NAMES, PERIODS, Estimates
from stockanalyzer.ui import charts
from stockanalyzer.ui import data as ui_data
from stockanalyzer.ui.format import money, pct
from stockanalyzer.ui.widgets import amount_input, amount_scale, show_chart, show_table

GROWTH_SLIDER = (-10.0, 80.0)  # percent; wide enough for a company growing 60% a year
MARGIN_SLIDER = (-60.0, 60.0)
FCF_METHOD = "Free cash flow"
REVENUE_METHOD = "Revenue × margin (growth companies)"


# -- inputs both methods share -----------------------------------------------------------------------
def growth_inputs(p: str, *, consensus: tuple[float, str] | None = None) -> tuple[float, int, int, float]:
    """Year-1 growth, years at full speed, years to project and long-run growth, as fractions and counts.

    ``consensus`` is ``(growth, label)`` of what analysts expect; it adds a button that copies it into the growth slider.
    """
    growth_pct = st.slider("Growth in year 1 (per year)", *GROWTH_SLIDER, step=0.5, format="%.1f%%", key=p + "growth",
                           help="How fast the business grows next year. It keeps this pace for the 'years at full speed' below, then fades in "
                                "a straight line to the long-run rate. Suggested from the median revenue growth of the last 3 periods (kept within 0-40%).")
    if consensus:
        value, label = consensus
        shown = round(min(max(value * 100, GROWTH_SLIDER[0]), GROWTH_SLIDER[1]) * 2) / 2
        capped = f", capped at {shown:g}%" if abs(shown - value * 100) >= 0.5 and shown in GROWTH_SLIDER else ""  # the slider has limits; say so
        st.button(f"Use analysts' {value:.1%} ({label}{capped})", key=p + "use_consensus",
                  on_click=lambda: st.session_state.update({p + "growth": shown}),
                  help="Copies the consensus growth into the slider above. Analysts look only a year or two ahead.")
    years = st.slider("Years to project", 3, 15, key=p + "years",
                      help="Cash flows are forecast year by year for this long; after that a single terminal value takes over.")
    st.session_state[p + "hold"] = int(min(max(st.session_state.get(p + "hold", 1), 1), years - 1))  # keep it valid when years shrinks
    hold = st.slider("Years at full speed", 1, years - 1, key=p + "hold",
                     help="How many years growth stays at the year-1 rate before it starts to fade. 1 fades from the start. "
                          "A company with a long runway (a big market, a lead over rivals) deserves more; the value moves a lot with this.")
    term_pct = st.slider("Long-run growth after that", 0.0, 5.0, step=0.1, format="%.1f%%", key=p + "term",
                         help="Growth forever after the projection. Keep it at or below long-run economic growth (about 2-3%).")
    return round(growth_pct / 100, 6), int(hold), int(years), round(term_pct / 100, 6)


def wacc_input(p: str, sg: dcf_inputs.DCFSuggestion) -> float:
    wacc_pct = st.slider("Discount rate", 5.0, 16.0, step=0.1, format="%.1f%%", key=p + "wacc",
                         help="The yearly return investors demand for the risk. Higher means a lower value today.")
    d = sg.wacc_detail
    beta_text = f"beta {d['beta']:.2f}" + (f" (Yahoo: {d['raw_beta']:.2f})" if d["raw_beta"] is not None and abs(d["raw_beta"] - d["beta"]) > 1e-9 else "")
    st.caption(f"Suggested {sg.wacc:.1%} = cost of equity {d['cost_equity']:.1%} ({d['risk_free']:.1%} risk-free + {beta_text} × "
               f"{d['equity_risk_premium']:.0%} equity premium) blended with after-tax debt cost ({d['tax_rate']:.0%} tax) by market weights.")
    return round(wacc_pct / 100, 6)


def debt_inputs(p: str, sg: dcf_inputs.DCFSuggestion, price_ccy: str, *, label: str) -> tuple[float, float]:
    with st.expander(label):
        net_debt = amount_input(f"Net debt, {price_ccy} (negative = net cash)", sg.net_debt, p + "net_debt")
        shares = amount_input("Shares outstanding", sg.shares or 0.0, p + "shares", min_value=0.0,
                              help="Yahoo's count matches the quoted price (ADR-equivalent for ADRs).")
    return net_debt, shares


def shared_defaults(p: str, sg: dcf_inputs.DCFSuggestion) -> dict:
    """Session-state defaults of the inputs the free-cash-flow and the revenue methods use alike."""
    return {p + "growth": round(sg.growth * 100, 1), p + "years": sg.years, p + "hold": sg.hold_years,
            p + "term": round(sg.terminal_growth * 100, 1), p + "wacc": round(sg.wacc * 100, 1),
            p + "net_debt": amount_scale(sg.net_debt)[3], p + "shares": amount_scale(sg.shares or 0.0)[3]}


def revenue_defaults(p: str, rs: dcf_inputs.RevenueSuggestion) -> dict:
    return {p + "rbase": rs.default_base or "manual", p + "rmanual": amount_scale(rs.base_revenue)[3],
            p + "cur_margin": round(rs.current_margin * 100, 1), p + "target_margin": round(rs.target_margin * 100, 1),
            p + "margin_years": rs.margin_years, p + "sbc_cost": False}


def prefers_revenue(sg: dcf_inputs.DCFSuggestion, rs: dcf_inputs.RevenueSuggestion) -> bool:
    """Start on the revenue method when free cash flow cannot carry a DCF: none, or thin next to a company that is growing fast."""
    return sg.base_fcf <= 0 or (rs.current_margin < 0.05 and sg.growth >= 0.15)


def consensus_sentence(implied: float | None, consensus: float | None, label: str) -> str:
    """'Analysts expect 18.0% (EPS, next 2 years): the price asks for more growth than they forecast (+9.0 points).'"""
    gap = est_mod.expectation_gap(implied, consensus)
    if gap is None:
        return ""
    diff, verdict = gap
    return f" Analysts expect **{consensus:.1%}** ({label}): the price {verdict} ({diff * 100:+.1f} points)."


# -- revenue x margin DCF ----------------------------------------------------------------------------
def revenue_panel(*, sg: dcf_inputs.DCFSuggestion, rs: dcf_inputs.RevenueSuggestion, price: float, price_ccy: str, p: str,
                  defaults: dict, consensus: tuple[float, str] | None) -> None:
    for note in rs.notes:
        st.caption("ℹ️ " + note)
    options = {o.key: o for o in rs.base_options}
    choices = [*options, "manual"]
    labels = {k: f"{o.label} · {money(o.value)} {price_ccy}" for k, o in options.items()} | {"manual": "Enter my own estimate"}
    if st.session_state.get(p + "rbase") not in choices:
        st.session_state[p + "rbase"] = defaults[p + "rbase"]

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.button("↺ Reset to suggested values", key=p + "rreset", width="stretch", on_click=lambda: st.session_state.update(defaults))
        st.markdown("##### 1 · Starting revenue")
        choice = st.radio("Base the forecast on", choices, format_func=labels.get, key=p + "rbase",
                          help="Revenue over the last twelve months, or the last reported period.")
        base_revenue = options[choice].value if choice in options else amount_input(
            f"Revenue per year, {price_ccy}", rs.base_revenue, p + "rmanual")

        st.markdown("##### 2 · Growth")
        growth, hold, years, term = growth_inputs(p, consensus=consensus)

        st.markdown("##### 3 · Free-cash-flow margin")
        current = st.slider("Margin today", *MARGIN_SLIDER, step=0.5, format="%.1f%%", key=p + "cur_margin",
                            help="Free cash flow ÷ revenue now. Negative while the company spends more than it earns.") / 100
        target = st.slider("Lasting margin", 0.0, MARGIN_SLIDER[1], step=0.5, format="%.1f%%", key=p + "target_margin",
                           help="The share of revenue left as free cash flow once the business has matured. This is the most important "
                                "assumption: look at what the best mature companies in the same line of business keep.") / 100
        st.session_state[p + "margin_years"] = int(min(max(st.session_state.get(p + "margin_years", 1), 1), years))
        ramp = st.slider("Years to reach it", 1, years, key=p + "margin_years",
                         help="The margin moves from today's level to the lasting one in equal steps over this many years.")
        adj = 0.0
        if rs.sbc_margin:
            if st.checkbox("Treat stock pay as a cost", key=p + "sbc_cost",
                           help=f"Stock-based pay is {rs.sbc_margin:.1%} of revenue. Free cash flow adds it back although it dilutes "
                                "shareholders; ticking this subtracts it from both margins."):
                adj = rs.sbc_margin
                st.caption(f"Margins used: {current - adj:.1%} today, {target - adj:.1%} lasting.")

        st.markdown("##### 4 · Required return (WACC)")
        wacc = wacc_input(p, sg)
        net_debt, shares = debt_inputs(p, sg, price_ccy, label="5 · Debt and share count")

    cur_eff, target_eff = current - adj, target - adj
    with right:
        if base_revenue <= 0:
            st.warning("There is no revenue figure to start from. Pick another base on the left or enter your own estimate.")
            return
        if not shares or shares <= 0:
            st.warning("Enter the number of shares (under '5 · Debt and share count') to get a value per share.")
            return
        if wacc <= term:
            st.error("The discount rate must be above long-run growth, otherwise the terminal value is undefined.")
            return
        if target_eff <= 0:
            st.warning("The lasting margin is zero or negative, so the business never earns a cash flow to value. Raise it above zero.")
            return

        kw = dict(current_margin=cur_eff, target_margin=target_eff, margin_years=ramp, years=years, terminal_growth=term, wacc=wacc,
                  net_debt=net_debt, shares=shares, hold=hold)
        res = valuation.revenue_dcf(base_revenue, growth, price=price, **kw)
        scen = valuation.revenue_dcf_scenarios(base_revenue, growth, price=price, **kw)
        lo, hi = float(scen["value_per_share"].min()), float(scen["value_per_share"].max())
        where = ("below the whole bear-to-bull range" if price < lo else
                 "above the whole bear-to-bull range" if price > hi else "inside the bear-to-bull range")
        last = res.projection.iloc[-1]
        worthless = res.intrinsic_per_share <= 0
        if worthless:
            reason = ("the cash it burns before it matures is worth more than what it earns afterwards" if res.enterprise_value <= 0
                      else "its net debt is larger than the value of its cash flows")
            st.warning(f"At these inputs the business is worth nothing to shareholders: {reason}. Raise the growth or the lasting margin, "
                       f"or shorten the years to reach it. The price of {price:,.2f} {price_ccy} is only justified by the reverse DCF below.")
        else:
            st.info(f"A business growing {growth:.1%} for {hold} year{'s' if hold > 1 else ''}, then fading, whose margin reaches {target_eff:.1%}, "
                    f"is worth **{res.intrinsic_per_share:,.2f} {price_ccy}** per share against a price of {price:,.2f} "
                    f"({pct(res.upside, signed=True)}). The price sits **{where}** ({lo:,.2f} - {hi:,.2f}). "
                    f"By year {years} revenue is {last['revenue'] / base_revenue:.1f}× today's.")

        m = st.columns(4)
        m[0].metric("Intrinsic value / share", "≤ 0" if worthless else f"{res.intrinsic_per_share:,.2f}")
        m[1].metric("Current price", f"{price:,.2f}")
        m[2].metric("Upside / (downside)", "-" if worthless else pct(res.upside, signed=True))
        m[3].metric("Terminal value share of EV", "-" if res.enterprise_value <= 0 else pct(res.terminal_share, 0),
                    help="Share of today's value that comes from everything after the projection years.")
        if res.enterprise_value > 0 and res.terminal_share > 0.75:
            st.warning("More than 75% of the value comes from the terminal value - for a growth company that is normal, "
                       "but it means the result leans on the lasting margin and the discount rate. Read the range, not the point.")

        st.markdown("##### Bear · Base · Bull")
        show_chart(charts.scenario_figure(scen, price))
        shown = scen.rename(columns={"scenario": "Scenario", "growth": "Year-1 growth", "target_margin": "Lasting margin", "wacc": "Discount rate",
                                     "terminal_growth": "Long-run growth", "value_per_share": "Value / share", "upside": "vs price"})
        show_table(shown.style.format({"Year-1 growth": "{:.1%}", "Lasting margin": "{:.1%}", "Discount rate": "{:.1%}",
                                       "Long-run growth": "{:.1%}", "Value / share": "{:,.2f}", "vs price": "{:+.1%}"}, na_rep="-"),
                   hide_index=True)
        st.caption(f"Bear / Bull move year-1 growth ∓{max(5.0, 20 * abs(growth)):.0f} points (a fifth of it, at least 5), the lasting margin ∓3 points, "
                   "the discount rate ±1 point and long-run growth ∓0.5 point.")

        st.markdown("##### How sensitive is the value?")
        common = {k: v for k, v in kw.items() if k != "target_margin"}  # everything the solvers and the grid take besides the margin they vary
        margins = sorted({round(max(target_eff + x, 0.0), 4) for x in (-0.10, -0.05, 0, 0.05, 0.10)})
        growths = [round(growth + x, 4) for x in (-0.10, -0.05, 0, 0.05, 0.10)]
        grid = valuation.revenue_sensitivity(base_revenue, growths=growths, margins=margins, **common)
        fig = charts.heatmap_figure(grid.rename(index=lambda v: f"{v:.1%}", columns=lambda v: f"{v:.1%}"), fmt=".1f",
                                    zmin=float(np.nanmin(grid.to_numpy())), zmax=float(np.nanmax(grid.to_numpy())), colorscale="Viridis",
                                    title=f"Value per share (price {price:,.2f})")
        fig.update_xaxes(title_text="Year-1 growth")
        fig.update_yaxes(title_text="Lasting margin")
        show_chart(fig)
        st.caption("Rows: lasting margin. Columns: year-1 growth. The centre cell is your inputs.")

        implied_g = valuation.implied_revenue_growth(price, base_revenue, target_margin=target_eff, **common)
        implied_m = valuation.implied_margin(price, base_revenue, growth, **common)
        st.markdown("##### What is the market assuming?")
        if np.isnan(implied_g):
            st.caption(f"Reverse DCF: no year-1 growth between -50% and +150% reproduces the price with a {target_eff:.1%} lasting margin.")
        else:
            st.info(f"If the margin settles at {target_eff:.1%}, the price needs year-1 revenue growth of **{implied_g:.1%}** "
                    f"(held {hold} year{'s' if hold > 1 else ''}, then fading to {term:.1%})." + (consensus_sentence(implied_g, *consensus) if consensus else ""))
        if np.isnan(implied_m):
            st.caption(f"Reverse DCF: no lasting margin between 0% and 90% reproduces the price with {growth:.1%} growth.")
        else:
            st.info(f"If revenue grows {growth:.1%} as assumed, the price needs a lasting free-cash-flow margin of **{implied_m:.1%}**. "
                    "Compare it with what mature companies in this business really keep.")

        with st.expander("Year-by-year projection"):
            show_table(res.projection.style.format({"growth": "{:.1%}", "revenue": "{:,.0f}", "margin": "{:.1%}", "fcf": "{:,.0f}",
                                                    "discount_factor": "{:.3f}", "pv": "{:,.0f}"}), hide_index=True)
        with st.expander("How the number is calculated"):
            st.markdown(
                "- Revenue grows at the year-1 rate for the years at full speed, then fades in a straight line to the long-run rate.\n"
                "- The free-cash-flow margin moves in equal steps from today's level to the lasting one, then stays there.\n"
                "- Free cash flow = revenue × margin. Each year's cash flow is discounted at the discount rate (year-end convention).\n"
                "- Terminal value = last cash flow × (1 + long-run growth) ÷ (discount rate − long-run growth), also discounted back.\n"
                "- Enterprise value = sum of both; equity value = enterprise value − net debt; value per share = equity value ÷ shares.\n"
                "- Share dilution is not modelled: tick 'Treat stock pay as a cost' to charge it against the margin instead.")


# -- growth profile tab ------------------------------------------------------------------------------
def _consensus_table(est: Estimates) -> pd.DataFrame:
    rows = []
    for period in PERIODS:
        eps, rev = est.eps.get(period), est.revenue.get(period)
        if not eps and not rev:
            continue
        rows.append({
            "Period": PERIOD_NAMES[period],
            "EPS (average)": eps.get("avg") if eps else None, "EPS low": eps.get("low") if eps else None,
            "EPS high": eps.get("high") if eps else None, "EPS growth": eps.get("growth") if eps else None,
            "Revenue (average)": rev.get("avg") if rev else None, "Revenue growth": rev.get("growth") if rev else None,
            "Analysts": (eps or rev or {}).get("analysts"),
        })
    return pd.DataFrame(rows)


def _level(value: float) -> str:
    return f"{value:,.2f}" if abs(value) < 100 else f"{value:,.0f}"


def growth_profile_tab(symbol: str, stmts, snap: dict, info: dict, price: float, price_ccy: str) -> None:
    st.caption("How fast the company grows, whether that growth pays for itself, what the market charges for it and what analysts expect.")
    m = st.columns(5)
    m[0].metric("Revenue growth", pct(snap.get("revenue_growth")), help="Yahoo's latest quarter against the same quarter a year earlier.")
    m[1].metric("FCF margin", pct(snap.get("fcf_margin")), help="Trailing free cash flow ÷ trailing revenue.")
    r40 = snap.get("rule_of_40")
    m[2].metric("Rule of 40", pct(r40), help="Revenue growth + free-cash-flow margin. 40% or more means growth is not being bought "
                                             "with unsustainable losses.")
    m[3].metric("EV/Sales", f"{snap['ev_sales']:.1f}" if snap.get("ev_sales") else "-", help="Enterprise value ÷ trailing revenue.")
    m[4].metric("EV/Sales per point of growth", f"{snap['ev_sales_growth']:.2f}" if snap.get("ev_sales_growth") else "-",
                help="EV/Sales ÷ revenue growth in points: what each point of growth costs. Under about 0.5 is cheap for a grower; "
                     "above 1 needs the growth to last. Only defined while growth is positive.")
    if r40 is not None:
        st.caption(("✅ Above the 40% bar: growth and cash generation together are healthy." if r40 >= 0.40 else
                    "Below the 40% bar: growth is not yet paying for itself, or is slow.") + " Rule of 40 is a screen, not a valuation.")

    r = ratios.compute_ratios(stmts)
    parts = r[["revenue_growth", "fcf_margin", "rule_of_40"]].dropna(how="all")
    if not parts.empty:
        show_chart(charts.line_figure((parts * 100).rename(columns={"revenue_growth": "Revenue growth", "fcf_margin": "FCF margin",
                                                                    "rule_of_40": "Rule of 40"}), title="Growth, margin and Rule of 40 (%)", height=320))
    sbc = r["sbc_to_revenue"].dropna()
    if len(sbc) and sbc.iloc[-1] >= dcf_inputs.SBC_NOTE_SHARE:
        st.caption(f"Stock-based pay is {sbc.iloc[-1]:.1%} of revenue and is added back in free cash flow, so the margin and Rule of 40 above are flattered by it.")

    st.markdown("##### What analysts expect")
    est = ui_data.estimates(symbol)
    if est.is_empty:
        st.info("Yahoo has no analyst estimates for this stock (or could not be reached just now).")
        return
    fin_ccy = est.currency or price_ccy
    fx = 1.0
    if fin_ccy != price_ccy:
        fx = ui_data.fx_rate(fin_ccy, price_ccy) or 0.0
        if fx:
            st.info(f"Estimates are in {fin_ccy} and the price in {price_ccy}: amounts were converted at 1 {fin_ccy} = {fx:.4g} {price_ccy}. "
                    "For a company whose shares trade as ADRs, check that the estimate and the price are for the same share.")
        else:
            st.warning(f"Estimates are in {fin_ccy} but no {fin_ccy}/{price_ccy} rate was found, so forward P/E and EPS amounts are not shown.")
    shown_est = est_mod.convert(est, fx) if fx else est
    cons = est_mod.consensus_growth(est)

    g = st.columns(4)
    g[0].metric("Revenue growth expected", pct(cons.revenue), help=f"Yearly growth from the last reported fiscal year through the "
                                                                  f"{'next two' if cons.revenue_years == 2 else 'current'} fiscal year(s), by {cons.revenue_analysts} analysts.")
    g[1].metric("EPS growth expected", pct(cons.eps), help="The same for earnings per share. Empty when the base year was a loss.")
    pes = est_mod.forward_pe(price, shown_est) if fx else {}
    g[2].metric(f"P/E on {PERIOD_NAMES['0y'].lower()}", f"{pes['0y']:.1f}" if "0y" in pes else "-")
    g[3].metric(f"P/E on {PERIOD_NAMES['+1y'].lower()}", f"{pes['+1y']:.1f}" if "+1y" in pes else "-")
    peg = est_mod.forward_peg(price, shown_est) if fx else None
    if peg is not None:
        st.caption(f"Forward PEG {peg:.2f}: the P/E on the furthest estimate divided by the {cons.eps:.0%} a year EPS growth analysts expect. "
                   "Below 1 is the usual 'cheap for its growth' mark, but analysts' growth is the weakest input in it.")

    view = est_mod.target_view(price, est)
    if view is not None:
        t = st.columns(4)
        t[0].metric("Mean price target", f"{view.mean:,.2f}" if view.mean else "-", pct(view.upside_mean, signed=True) if view.upside_mean is not None else None,
                    help="Average of the analysts' 12-month targets, against the current price.")
        t[1].metric("Target range", f"{_level(view.low)} - {_level(view.high)}" if view.low and view.high else "-")
        t[2].metric("Analysts", f"{view.analysts}" if view.analysts else "-")
        word = est_mod.recommendation_word(view.recommendation_mean) if view.recommendation_mean else None
        t[3].metric("Rating (1 buy - 5 sell)", f"{view.recommendation_mean:.1f} · {word}" if word else "-")
        st.caption("Targets sit above prices most of the time - analysts rarely publish a sell. Read the spread between high and low, and the revisions below.")

    rev = est_mod.revision(est, "0y")
    if rev.change_30d is not None or rev.up_30d is not None:
        bits = []
        if rev.change_30d is not None:
            bits.append(f"{rev.change_30d:+.1%} in 30 days")
        if rev.change_90d is not None:
            bits.append(f"{rev.change_90d:+.1%} in 90 days")
        if rev.up_30d is not None:
            bits.append(f"{rev.up_30d} analysts raised and {rev.down_30d} cut it this month")
        icon = {"rising": "📈", "falling": "📉"}.get(rev.direction, "➖")
        st.markdown(f"{icon} **EPS estimates for the current fiscal year are {rev.direction}:** " + " · ".join(bits) + ". "
                    "Estimates that keep rising are a better sign than a high absolute growth number.")

    table = _consensus_table(shown_est if fx else est)
    if not table.empty:
        if fx:
            show_table(table.style.format({"EPS (average)": "{:,.2f}", "EPS low": "{:,.2f}", "EPS high": "{:,.2f}", "EPS growth": "{:+.1%}",
                                           "Revenue (average)": "{:,.0f}", "Revenue growth": "{:+.1%}", "Analysts": "{:.0f}"}, na_rep="-"),
                       hide_index=True)
            st.caption(f"Amounts in {price_ccy}. 'Current fiscal year' is the one still in progress; growth is against the year before.")
    history = est_mod.history_frame(ui_data.estimates_history(symbol))
    if len(history) >= 2:
        with st.expander(f"Consensus history ({len(history)} daily snapshots)"):
            targets = history[["target_mean"]].dropna()
            if len(targets) >= 2:
                show_chart(charts.line_figure(targets.rename(columns={"target_mean": "Mean price target"}), title="Mean analyst price target", height=260))
            eps = history[["eps_0y", "eps_1y"]].dropna(how="all")
            if len(eps) >= 2:
                show_chart(charts.line_figure(eps.rename(columns={"eps_0y": "EPS, current year", "eps_1y": "EPS, next year"}),
                                              title="EPS estimates over time", height=260))
    else:
        st.caption("Consensus is saved once a day each time this page is opened; a history of targets and estimates builds up from here.")
