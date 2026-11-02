"""Look up standard line items in provider statements whatever their exact label is."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockanalyzer.domain.financials import FinancialStatements

# canonical key -> candidate labels (Yahoo naming), first match wins
ALIASES: dict[str, dict[str, list[str]]] = {
    "income": {
        "revenue": ["Total Revenue", "Operating Revenue", "Revenue"],
        "cogs": ["Cost Of Revenue", "Reconciled Cost Of Revenue"],
        "gross_profit": ["Gross Profit"],
        "operating_income": ["Operating Income", "Total Operating Income As Reported"],
        "ebit": ["EBIT"],
        "ebitda": ["EBITDA", "Normalized EBITDA"],
        "net_income": ["Net Income", "Net Income Common Stockholders",
                       "Net Income From Continuing Operation Net Minority Interest"],
        "net_income_common": ["Net Income Common Stockholders", "Diluted NI Availto Com Stockholders", "Net Income"],
        "interest_expense": ["Interest Expense", "Interest Expense Non Operating"],
        "pretax_income": ["Pretax Income"],
        "tax": ["Tax Provision"],
        "eps": ["Diluted EPS", "Basic EPS"],
        "shares_avg": ["Diluted Average Shares", "Basic Average Shares"],
    },
    "balance": {
        "total_assets": ["Total Assets"],
        "current_assets": ["Current Assets"],
        "current_liabilities": ["Current Liabilities"],
        "total_liabilities": ["Total Liabilities Net Minority Interest"],
        "equity": ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"],
        "retained_earnings": ["Retained Earnings"],
        "long_term_debt": ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
        "total_debt": ["Total Debt"],
        "cash": ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"],
        "inventory": ["Inventory"],
        "shares": ["Ordinary Shares Number", "Share Issued"],
        "invested_capital": ["Invested Capital"],
        "minority": ["Minority Interest"],
    },
    "cashflow": {
        "cfo": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
        "capex": ["Capital Expenditure"],  # reported negative
        "fcf": ["Free Cash Flow"],
        "dividends": ["Cash Dividends Paid", "Common Stock Dividend Paid"],
        "buybacks": ["Repurchase Of Capital Stock"],
        "sbc": ["Stock Based Compensation"],
        "da": ["Depreciation And Amortization", "Depreciation Amortization Depletion"],
    },
}


def _find(df: pd.DataFrame, labels: list[str]) -> pd.Series | None:
    if df is None or df.empty:
        return None
    lookup = {str(i).lower(): i for i in df.index}
    for label in labels:
        key = lookup.get(label.lower())
        if key is not None:
            row = df.loc[key]
            if isinstance(row, pd.DataFrame):  # duplicated label
                row = row.iloc[0]
            return pd.to_numeric(row, errors="coerce")
    return None


def line(stmts: FinancialStatements, key: str) -> pd.Series:
    """Series on the common period index of all statements (NaN where missing or unsupported).

    Every series shares one index, so ratios can combine items from different statements
    even when the provider returns a different number of periods for each.
    """
    for statement, items in ALIASES.items():
        if key in items:
            found = _find(getattr(stmts, statement), items[key])
            if found is None and key == "fcf":  # derive when absent
                found = line(stmts, "cfo") + line(stmts, "capex")
            if found is None:
                found = pd.Series(np.nan, index=stmts.periods, dtype="float64")
            return found.reindex(stmts.periods).astype("float64")
    raise KeyError(key)


def table(stmts: FinancialStatements, keys: list[str]) -> pd.DataFrame:
    """Selected items as a periods x keys frame (oldest first)."""
    return pd.DataFrame({k: line(stmts, k) for k in keys}).sort_index()
