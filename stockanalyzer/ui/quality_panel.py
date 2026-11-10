"""Streamlit tab for earnings quality and value creation: cash backing, working capital, Beneish M-score, ROIC vs the cost
of capital, DuPont, stock pay and dilution."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from stockanalyzer.analysis.fundamental import dcf_inputs
from stockanalyzer.analysis.fundamental import earnings_quality as eq
from stockanalyzer.ui import charts
from stockanalyzer.ui import data as ui_data
from stockanalyzer.ui.format import money, pct
from stockanalyzer.ui.widgets import show_chart, show_table

ICON = {"bad": "🔴", "warn": "🟠", "info": "ℹ️", "good": "🟢"}


def _years(index) -> list[str]:
    return [f"{d:%Y}" for d in index]


def _suggested_wacc(symbol: str, annual, snap: dict, info: dict, price: float) -> float:
    price_ccy = info.get("currency") or "USD"
    fin_ccy = info.get("financialCurrency") or price_ccy
    fx = ui_data.fx_rate(fin_ccy, price_ccy) if fin_ccy != price_ccy else 1.0
    try:
        return dcf_inputs.suggest_dcf_inputs(annual, info, price, market_cap=snap.get("market_cap"), fx=fx or 1.0,
                                             currency=price_ccy).wacc
    except Exception:  # a cost of capital is a convenience here, never a reason to hide the tab
        return 0.09


def _flags(found: list[eq.Flag]) -> None:
    counts = {t: sum(f.tone == t for f in found) for t in ("bad", "warn", "good")}
    st.markdown(f"**{counts['bad']} red flags · {counts['warn']} worth a look · {counts['good']} reassuring**")
    for f in found:
        st.markdown(f"{ICON[f.tone]} **{f.title}** - {f.detail}")
    if not found:
        st.caption("Nothing to judge: the statements lack the lines these checks need.")


def _cash_section(q: pd.DataFrame, ccy: str) -> None:
    st.markdown("##### Is the profit backed by cash?")
    money_cols = q[["net_income", "cfo", "fcf"]].dropna(how="all")
    if not money_cols.empty:
        shown = money_cols.rename(columns={"net_income": "Net income (whole group)", "cfo": "Operating cash flow", "fcf": "Free cash flow"})
        shown.index = _years(shown.index)
        show_chart(charts.grouped_bar_figure(shown, title=f"Profit vs cash ({ccy})", colors=["#9e9e9e", charts.PALETTE[0], charts.UP]))
    table = q[["cash_conversion", "fcf_conversion", "accruals_ratio"]].iloc[::-1]
    table.index = _years(table.index)
    show_table(table.rename(columns={"cash_conversion": "Operating cash flow ÷ net income", "fcf_conversion": "Free cash flow ÷ net income",
                                     "accruals_ratio": "Accruals ÷ average assets"})
               .style.format("{:.0%}", na_rep="-").format("{:+.1%}", subset=["Accruals ÷ average assets"], na_rep="-"))
    st.caption(f"Profit is an accounting opinion, cash is a fact. Operating cash flow should match or beat net income over a few years "
               f"(ratio ≥ {eq.CASH_CONVERSION_OK:.0%}). Accruals = net income − operating cash flow, over average assets: above "
               f"{eq.ACCRUALS_WARN:.0%} the gap is large, above {eq.ACCRUALS_BAD:.0%} it is in the zone where Sloan (1996) found "
               "earnings tend to fall back. Ratios are blank when net income is not positive. Net income here includes the part "
               "that belongs to minority shareholders of subsidiaries, because operating cash flow covers the whole group too.")


def _working_capital_section(q: pd.DataFrame) -> None:
    st.markdown("##### Working capital")
    days = q[["dso", "dio", "dpo", "ccc"]].dropna(how="all")
    if days.empty:
        st.caption("No receivables, inventory or payables in the statements.")
        return
    named = days.rename(columns={"dso": "Days of sales in receivables", "dio": "Days of inventory", "dpo": "Days of payables",
                                 "ccc": "Cash conversion cycle"})
    show_chart(charts.line_figure(named, height=300))
    table = q[["dso", "dio", "dpo", "ccc", "receivables_growth", "revenue_growth", "inventory_growth"]].iloc[::-1]
    table.index = _years(table.index)
    show_table(table.rename(columns={"dso": "Receivable days", "dio": "Inventory days", "dpo": "Payable days", "ccc": "Cycle (days)",
                                     "receivables_growth": "Receivables growth", "revenue_growth": "Sales growth",
                                     "inventory_growth": "Inventory growth"})
               .style.format("{:.0f}", na_rep="-")
               .format("{:+.0%}", subset=["Receivables growth", "Sales growth", "Inventory growth"], na_rep="-"))
    st.caption("Cash conversion cycle = receivable days + inventory days − payable days: how long cash is tied up between paying "
               "suppliers and being paid. Receivables or inventory growing much faster than sales (more than "
               f"{eq.GAP_WARN * 100:.0f} points) is the classic early sign of weaker demand or aggressive revenue booking; it is only "
               f"flagged when the item is at least {eq.MATERIAL:.0%} of assets.")


def _beneish_section(stmts) -> None:
    st.markdown("##### Beneish M-score: signs of earnings manipulation")
    scores = eq.beneish(stmts)
    if not scores:
        st.caption("Needs two consecutive years with revenue, receivables, total assets, net income and operating cash flow.")
        return
    last = scores[-1]
    k = st.columns(3)
    k[0].metric(f"M-score FY {last.period:%Y}", f"{last.m:.2f}", last.zone, delta_color="off",
                help=f"Above {eq.BENEISH_FLAG}: resembles companies later caught manipulating; below {eq.BENEISH_GREY}: does not.")
    k[1].metric("Years scored", len(scores))
    k[2].metric("Biggest push", last.drivers().index[0], help="The variable that lifts the score most above a typical honest company.")
    years = pd.DataFrame({f"FY {s.period:%Y}": s.indices for s in scores})
    years["Honest firms (avg)"] = [eq.BENEISH_MEANS[k][0] for k in years.index]
    years["Manipulators (avg)"] = [eq.BENEISH_MEANS[k][1] for k in years.index]
    years.index = [f"{k} · {eq.BENEISH_NAMES[k]}" for k in years.index]
    years.loc["M-score"] = [s.m for s in scores] + [None, None]
    show_table(years.style.format("{:.3f}", na_rep="-"))
    st.caption(f"Beneish's 1999 model weighs eight ratios that moved differently at companies later caught by the SEC. Each index "
               f"compares a year with the one before (1.0 = no change; TATA is a level). Above **{eq.BENEISH_FLAG}** is his cut-off, "
               f"**{eq.BENEISH_GREY}** a stricter one many use. It was fitted on US manufacturers of the 1980s-90s and misses most "
               "frauds while flagging honest fast growers - read it as a reason to look closer, never as an accusation. "
               + (f"Missing data set to neutral in the latest year: {', '.join(last.neutral)}." if last.neutral else ""))


def _value_section(stmts, wacc: float, ccy: str) -> None:
    st.markdown("##### Does the business create value?")
    vc = eq.value_creation(stmts, wacc)
    t = vc.table.dropna(subset=["roic"])
    if t.empty:
        st.caption("No operating profit or invested capital in the statements.")
        return
    latest = t.iloc[-1]
    k = st.columns(4)
    k[0].metric(f"ROIC FY {t.index[-1]:%Y}", pct(latest["roic"]), f"{(latest['roic'] - wacc) * 100:+.1f} pts vs cost of capital",
                help="Operating profit after tax ÷ invested capital (debt + equity).")
    k[1].metric("Incremental ROIC", pct(vc.incremental_roic) if vc.incremental_roic == vc.incremental_roic else "n/m",
                help="Extra operating profit per unit of extra capital, first to last year. 'n/m' when capital grew less than "
                     f"{eq.MIN_CAPITAL_GROWTH:.0%} (buybacks, paid-down debt): dividing by a tiny change gives a meaningless number.")
    k[2].metric("Reinvestment rate", pct(vc.reinvestment_rate) if vc.reinvestment_rate == vc.reinvestment_rate else "-",
                help="Capital added ÷ operating profit earned over the same years: how much of its profit the business ploughs back.")
    k[3].metric("Growth it can fund", pct(vc.funded_growth) if vc.funded_growth == vc.funded_growth else "-",
                help="Reinvestment rate × average ROIC: the profit growth the business can pay for from its own earnings.")
    show_chart(charts.line_figure(pd.DataFrame({"ROIC": t["roic"] * 100, "Cost of capital": wacc * 100}, index=t.index),
                                  title="Return on invested capital vs cost of capital (%)", height=300))
    ep = t["economic_profit"].dropna()
    if not ep.empty:
        st.caption(f"Economic profit (ROIC − cost of capital) × invested capital, {ccy}: "
                   + " · ".join(f"{d:%Y} **{money(v)}**" for d, v in ep.items()))
    st.caption("A company creates value only when it earns more on its capital than that capital costs; growth below the cost of "
               "capital makes owners poorer the faster it goes. ROIC uses the same definition as the Ratios tab.")


def _dupont_section(stmts) -> None:
    st.markdown("##### What drives return on equity (DuPont)")
    dp = eq.dupont(stmts).dropna(how="all")
    if dp.dropna().empty:
        st.caption("The statements lack pre-tax income, operating profit or equity for a full split.")
        return
    table = dp.iloc[::-1].rename(columns={**eq.DRIVERS, "tax_burden": "Tax (net ÷ pre-tax)", "interest_burden": "Interest (pre-tax ÷ EBIT)",
                                          "ebit_margin": "Operating margin", "asset_turnover": "Asset turnover",
                                          "equity_multiplier": "Leverage (assets ÷ equity)", "roe": "ROE"})
    table.index = _years(table.index)
    show_table(table.style.format("{:.2f}", na_rep="-").format("{:.1%}", subset=["Operating margin", "ROE"], na_rep="-"))
    drivers = eq.roe_drivers(dp)
    if drivers is not None:
        start, end, roe0, roe1, contrib = drivers
        show_chart(charts.bar_figure(contrib * 100, title=f"ROE {roe0:.1%} → {roe1:.1%} ({start:%Y} → {end:%Y}): points from each driver",
                                     fmt="+.1f"))
    st.caption("ROE = tax burden × interest burden × operating margin × asset turnover × leverage. ROE that rises with margins and "
               "turnover is earned; ROE that rises with leverage is borrowed.")


def _reinvestment_section(q: pd.DataFrame) -> None:
    st.markdown("##### Reinvestment, stock pay, dilution and one-offs")
    table = q[["capex_to_da", "sbc_to_fcf", "sbc_to_revenue", "share_change", "unusual_share"]].iloc[::-1]
    table.index = _years(table.index)
    show_table(table.rename(columns={"capex_to_da": "Capital spending ÷ depreciation", "sbc_to_fcf": "Stock pay ÷ free cash flow",
                                     "sbc_to_revenue": "Stock pay ÷ revenue", "share_change": "Share count change",
                                     "unusual_share": "One-off items ÷ pre-tax profit"})
               .style.format("{:.0%}", na_rep="-").format("{:+.1%}", subset=["Share count change", "One-off items ÷ pre-tax profit"], na_rep="-"))
    st.caption("Capital spending persistently below depreciation can flatter free cash flow (but see the lease note below). Stock pay "
               "is a cost settled in shares, and a rising share count is how owners pay it. One-off items: positive = gains that "
               "lifted profit, negative = charges that held it down.")


def quality_tab(symbol: str, snap: dict, info: dict, price: float, *, bank: bool = False) -> None:
    annual = ui_data.statements(symbol, False)  # quality is judged on years, whatever the page's quarterly toggle says
    fin_ccy = info.get("financialCurrency") or info.get("currency") or "USD"
    st.caption("Two questions the headline numbers do not answer: is the reported profit real cash, and does the business earn "
               "more on its capital than that capital costs? Judged on the yearly statements (Yahoo gives 4-5 years).")
    p = f"quality_{symbol}_"
    wacc = None
    if not bank:
        suggested = _suggested_wacc(symbol, annual, snap, info, price)
        st.session_state.setdefault(p + "wacc", round(suggested * 100, 1))
        wacc = st.slider("Cost of capital (WACC)", 4.0, 16.0, step=0.1, format="%.1f%%", key=p + "wacc",
                         help="What investors and lenders require. Pre-filled with the DCF tab's suggestion for this market.") / 100
        st.caption(f"Suggested {suggested:.1%}, the same starting point as the DCF tab.")
    _flags(eq.flags(annual, wacc=wacc, bank=bank))

    q = eq.quality_table(annual)
    if bank:
        st.info("🏦 A bank's cash flow follows deposits and loans, and it has no inventory, payables cycle or capital spending to "
                "speak of, so the cash-backing, working-capital, Beneish and ROIC checks do not apply. What remains: DuPont "
                "(return on assets × leverage), dilution and one-off items. Bank health checks are under Quality scores.")
        _dupont_section(annual)
        _reinvestment_section(q[["capex_to_da", "sbc_to_fcf", "sbc_to_revenue", "share_change", "unusual_share"]])
        return
    _cash_section(q, fin_ccy)
    _working_capital_section(q)
    _beneish_section(annual)
    _value_section(annual, wacc, fin_ccy)
    _dupont_section(annual)
    _reinvestment_section(q)
    with st.expander("How the numbers are calculated, and their limits"):
        st.markdown(
            "- **Accruals ratio** = (net income − operating cash flow) ÷ average total assets (Sloan 1996, cash-flow form).\n"
            "- **Receivable / inventory / payable days** = year-end balance ÷ sales (receivables) or cost of sales × 365.\n"
            "- **Beneish M-score** = −4.84 + 0.920·DSRI + 0.528·GMI + 0.404·AQI + 0.892·SGI + 0.115·DEPI − 0.172·SGAI + 4.679·TATA "
            "− 0.327·LVGI. Depreciation here is Yahoo's depreciation and amortisation; a missing variable counts as 1 (no change).\n"
            "- **ROIC** = EBIT × (1 − effective tax rate) ÷ invested capital (Yahoo's debt + equity). Incremental ROIC divides the "
            "change in that profit by the change in capital one year earlier.\n"
            "- **IFRS / TFRS 16 (leases)**: Thai and other non-US companies book leased stores and planes as assets and their payments "
            "partly as financing, so operating cash flow looks stronger and capital spending ÷ depreciation weaker than for a US "
            "company with the same business. Compare companies under the same rules.\n"
            "- The thresholds are rules of thumb from published research, not laws; a flag is a question to ask the annual report.")
