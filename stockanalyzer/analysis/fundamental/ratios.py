"""Financial ratios per period from statements (profitability, safety, efficiency, growth)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockanalyzer.analysis.fundamental.statements import line
from stockanalyzer.domain.financials import FinancialStatements


def _div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.where(b != 0)


def ebit(stmts: FinancialStatements) -> pd.Series:
    return line(stmts, "ebit").fillna(line(stmts, "operating_income"))


def nopat(stmts: FinancialStatements) -> pd.Series:
    """Operating profit after tax: EBIT × (1 − effective tax rate, kept within 0-40%, 21% when unknown)."""
    tax_rate = _div(line(stmts, "tax"), line(stmts, "pretax_income")).clip(0, 0.4).fillna(0.21)
    return ebit(stmts) * (1 - tax_rate)


def invested_capital(stmts: FinancialStatements) -> pd.Series:
    """Yahoo's invested capital (debt + equity), else equity + debt − cash."""
    debt = line(stmts, "total_debt").fillna(line(stmts, "long_term_debt"))
    return line(stmts, "invested_capital").fillna(line(stmts, "equity") + debt - line(stmts, "cash"))


def compute_ratios(stmts: FinancialStatements) -> pd.DataFrame:
    """Rows are periods (oldest first); growth columns need one prior year (four quarters)."""
    L = lambda k: line(stmts, k)  # noqa: E731
    revenue, ni = L("revenue"), L("net_income")
    op_profit = ebit(stmts)
    assets, equity = L("total_assets"), L("equity")
    debt = L("total_debt").fillna(L("long_term_debt"))
    cash = L("cash")
    fcf = L("fcf")
    lag = 4 if stmts.quarterly else 1

    avg_equity = (equity + equity.shift(1)) / 2
    avg_assets = (assets + assets.shift(1)) / 2

    out = pd.DataFrame(
        {
            "revenue": revenue,
            "net_income": ni,
            "ebitda": L("ebitda"),
            "fcf": fcf,
            # profitability
            "gross_margin": _div(L("gross_profit"), revenue),
            "operating_margin": _div(L("operating_income").fillna(op_profit), revenue),
            "net_margin": _div(ni, revenue),
            "roe": _div(ni, avg_equity.fillna(equity)),
            "roa": _div(ni, avg_assets.fillna(assets)),
            "roic": _div(nopat(stmts), invested_capital(stmts)),
            # safety
            "current_ratio": _div(L("current_assets"), L("current_liabilities")),
            "quick_ratio": _div(L("current_assets") - L("inventory").fillna(0), L("current_liabilities")),
            "debt_to_equity": _div(debt, equity),
            "net_debt_to_ebitda": _div(debt - cash, L("ebitda")),
            "interest_coverage": _div(op_profit, L("interest_expense").abs()),
            # efficiency & cash
            "asset_turnover": _div(revenue, avg_assets.fillna(assets)),
            "fcf_margin": _div(fcf, revenue),
            "fcf_conversion": _div(fcf, ni),
            "capex_to_revenue": _div(-L("capex"), revenue),
            "payout_ratio": _div(-L("dividends"), ni),
            "sbc_to_revenue": _div(L("sbc"), revenue),
            # growth
            "revenue_growth": revenue.pct_change(lag, fill_method=None),
            "net_income_growth": ni.pct_change(lag, fill_method=None),
            "eps_growth": L("eps").pct_change(lag, fill_method=None),
            "fcf_growth": fcf.pct_change(lag, fill_method=None),
            "share_count_change": L("shares").pct_change(lag, fill_method=None),
        }
    )
    out["rule_of_40"] = out["revenue_growth"] + out["fcf_margin"]  # growth + free-cash-flow margin: above 40% is a healthy grower
    # growth from a negative base is meaningless (-100% -> +50% would read as "growth")
    for growth, base in [("net_income_growth", ni), ("eps_growth", L("eps")), ("fcf_growth", fcf)]:
        out.loc[base.shift(lag) <= 0, growth] = np.nan
    return out.sort_index().replace([np.inf, -np.inf], np.nan)


def cagr(series: pd.Series, years: float | None = None) -> float:
    """Compound annual growth between first and last positive observation."""
    s = series.dropna()
    if len(s) < 2 or s.iloc[0] <= 0 or s.iloc[-1] <= 0:
        return float("nan")
    if years is None:
        years = (s.index[-1] - s.index[0]).days / 365.25
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1) if years > 0 else float("nan")
