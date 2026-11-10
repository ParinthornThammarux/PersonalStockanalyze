"""Banks: what to look at instead of free cash flow, EBITDA and working capital.

A bank's debt and deposits are its raw material, so the usual non-financial toolkit (FCF-based DCF, EV/EBITDA, Altman Z,
current ratio) says nothing about it. What matters is what shareholders earn on their equity against what they require.
The anchor is the *justified price-to-book*: with a steady return on equity (ROE), cost of equity (COE) and growth (g),

    P/B = (ROE - g) / (COE - g)

so a bank earning more than its COE deserves more than book value, and one earning less deserves less.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockanalyzer.analysis.fundamental.scores import Criterion
from stockanalyzer.analysis.fundamental.statements import line
from stockanalyzer.domain.financials import FinancialStatements

# currency -> (risk-free rate, equity risk premium): starting points to be overridden, not market data
MARKET_ASSUMPTIONS = {"USD": (0.043, 0.05), "THB": (0.025, 0.06)}
FALLBACK_ASSUMPTION = (0.04, 0.06)
LONG_RUN_GROWTH = {"USD": 0.025, "THB": 0.03}
BETA_RANGE = (1.0, 1.6)  # banks are leveraged and cyclical: a measured beta well below 1 (KBANK 0.15) is an artefact of thin trading or the wrong index


def is_bank(industry: str | None) -> bool:
    """Yahoo files banks under 'Banks - Regional' / 'Banks - Diversified'."""
    return bool(industry) and industry.strip().lower().startswith("banks")


@dataclass(frozen=True)
class CostOfEquity:
    value: float
    risk_free: float
    premium: float
    beta: float  # the beta actually used (clipped)
    raw_beta: float | None
    clipped: bool


def cost_of_equity(beta: float | None, currency: str) -> CostOfEquity:
    """CAPM cost of equity with a market-appropriate risk-free rate and a beta kept inside a sensible range."""
    rf, erp = MARKET_ASSUMPTIONS.get(currency, FALLBACK_ASSUMPTION)
    raw = float(beta) if beta is not None and beta == beta else None
    used = min(max(raw if raw is not None else 1.0, BETA_RANGE[0]), BETA_RANGE[1])
    return CostOfEquity(rf + used * erp, rf, erp, used, raw, raw is None or used != raw)


def justified_pb(roe: float, coe: float, growth: float) -> float:
    """P/B a steady ROE deserves; NaN when the required return does not exceed growth."""
    return (roe - growth) / (coe - growth) if coe > growth else float("nan")


def implied_roe(pb: float, coe: float, growth: float) -> float:
    """The steady ROE the current P/B assumes (the reverse of ``justified_pb``)."""
    return pb * (coe - growth) + growth


def ddm_value(dps: float, coe: float, growth: float) -> float:
    """Gordon growth value of a dividend stream: next year's dividend / (COE - g)."""
    return dps * (1 + growth) / (coe - growth) if coe > growth and dps > 0 else float("nan")


def scenarios(bvps: float, roe: float, coe: float, growth: float, price: float | None = None,
              shifts: dict[str, tuple[float, float]] | None = None) -> pd.DataFrame:
    """Bear / base / bull value per share: ROE and COE moved around the inputs (default -2 pts / +1 pt for the bear case)."""
    shifts = shifts or {"Bear": (-0.02, 0.01), "Base": (0.0, 0.0), "Bull": (0.02, -0.01)}
    rows = []
    for name, (d_roe, d_coe) in shifts.items():
        r, c = roe + d_roe, coe + d_coe
        pb = justified_pb(r, c, growth)
        value = pb * bvps
        rows.append({"scenario": name, "roe": r, "coe": c, "justified_pb": pb, "value_per_share": value,
                     "upside": value / price - 1 if price and value == value else float("nan")})
    return pd.DataFrame(rows)


def sensitivity(bvps: float, roes, coes, growth: float) -> pd.DataFrame:
    """Value per share: rows = cost of equity, columns = ROE."""
    return pd.DataFrame([[justified_pb(r, c, growth) * bvps for r in roes] for c in coes], index=list(coes), columns=list(roes))


def metrics(stmts: FinancialStatements) -> pd.DataFrame:
    """Bank ratios per period (oldest first; growth needs one prior year, four quarters for quarterly statements)."""
    L = lambda k: line(stmts, k)  # noqa: E731
    revenue, ni, pretax = L("revenue"), L("net_income"), L("pretax_income")
    equity, assets, nii = L("equity"), L("total_assets"), L("net_interest_income")
    loans, shares = L("net_loans"), L("shares")
    lag = 4 if stmts.quarterly else 1
    avg_equity = ((equity + equity.shift(1)) / 2).fillna(equity)
    avg_assets = ((assets + assets.shift(1)) / 2).fillna(assets)

    def ratio(a: pd.Series, b: pd.Series) -> pd.Series:
        return a / b.where(b != 0)

    out = pd.DataFrame({
        "net_income": ni, "revenue": revenue, "net_interest_income": nii, "equity": equity, "total_assets": assets,
        "roe": ratio(ni, avg_equity), "roa": ratio(ni, avg_assets),
        "equity_to_assets": ratio(equity, assets),
        "nii_to_assets": ratio(nii, avg_assets),
        "nii_share": ratio(nii, revenue),
        "cost_provision_share": 1 - ratio(pretax, revenue),
        "loans_to_assets": ratio(loans, assets),
        "bvps": ratio(equity, shares),
        "payout_ratio": ratio(-L("dividends"), ni),
        "net_income_growth": ni.pct_change(lag, fill_method=None),
        "eps_growth": L("eps").pct_change(lag, fill_method=None),
        "nii_growth": nii.pct_change(lag, fill_method=None),
        "loan_growth": loans.pct_change(lag, fill_method=None),
    })
    for growth, base in [("net_income_growth", ni), ("eps_growth", L("eps")), ("nii_growth", nii)]:
        out.loc[base.shift(lag) <= 0, growth] = np.nan  # growth from a loss is meaningless
    return out.sort_index().replace([np.inf, -np.inf], np.nan)


def checks(stmts: FinancialStatements) -> list[Criterion]:
    """Six pass / fail signals on the two latest periods, the bank counterpart of the Piotroski score."""
    m = metrics(stmts)

    def two(column: str) -> tuple[float | None, float | None]:
        s = m[column].dropna()
        return (float(s.iloc[-1]) if len(s) else None), (float(s.iloc[-2]) if len(s) > 1 else None)

    def compare(name: str, column: str, ok) -> Criterion:
        cur, prev = two(column)
        return Criterion(name, None if cur is None or prev is None else bool(ok(cur, prev)), cur, prev)

    cur, _ = two("net_income")
    shares = line(stmts, "shares").dropna()
    dilution = Criterion("No share dilution", None if len(shares) < 2 else bool(shares.iloc[-1] <= shares.iloc[-2] * 1.005),
                         float(shares.iloc[-1]) if len(shares) else None, float(shares.iloc[-2]) if len(shares) > 1 else None)
    return [
        Criterion("Profitable", None if cur is None else cur > 0, cur),
        compare("ROE improved", "roe", lambda a, b: a > b),
        compare("Net interest income grew", "net_interest_income", lambda a, b: a > b),
        compare("Capital cushion not thinner (equity / assets)", "equity_to_assets", lambda a, b: a >= b - 1e-4),
        compare("Costs and provisions took no bigger share of revenue", "cost_provision_share", lambda a, b: a <= b + 1e-4),
        dilution,
    ]
