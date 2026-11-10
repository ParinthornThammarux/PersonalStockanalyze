"""Number formatting and ``st.dataframe`` column configs."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from stockanalyzer.screener.features import FEATURE_INFO

# fundamentals columns of the screen table (fractions are shown as %), keyed like the snapshot
FUNDAMENTAL_INFO = {
    "pe": ("P/E", "num"), "forward_pe": ("Forward P/E", "num"), "pb": ("P/B", "num"), "ps": ("P/S", "num"),
    "ev_ebitda": ("EV/EBITDA", "num"), "peg": ("PEG", "num"), "eps": ("EPS", "num"), "beta": ("Beta", "num"),
    "roe": ("ROE", "pct"), "roa": ("ROA", "pct"), "profit_margin": ("Net margin", "pct"),
    "operating_margin": ("Operating margin", "pct"), "gross_margin": ("Gross margin", "pct"),
    "revenue_growth": ("Revenue growth", "pct"), "earnings_growth": ("Earnings growth", "pct"),
    "dividend_yield": ("Dividend yield", "pct"), "fcf_yield": ("FCF yield", "pct"),
    "debt_to_equity": ("Debt / equity", "num"), "current_ratio": ("Current ratio", "num"),
    "ev_sales": ("EV/Sales", "num"), "ev_sales_growth": ("EV/Sales per point of growth", "num"),
    "fcf_margin": ("FCF margin", "pct"), "rule_of_40": ("Rule of 40", "pct"), "forward_eps": ("Forward EPS", "num"),
    "target_upside": ("Upside to analyst target", "pct"), "analyst_count": ("Analysts covering", "int"),
    "recommendation_mean": ("Analyst rating (1 buy - 5 sell)", "num"),
}

# screen-table columns that do not come from the feature snapshot
EXTRA_INFO = {**FUNDAMENTAL_INFO,
              "market_cap": ("Market cap ($)", "usd"), "currency": ("Currency", "text"),
              "score": ("Score", "scorebar"), "score_momentum": ("Momentum", "score"), "score_trend": ("Trend", "score"),
              "score_risk": ("Risk", "score"), "score_value": ("Value", "score"), "score_quality": ("Quality", "score"),
              "score_liquidity": ("Liquidity", "score")}

OP_WORDS = {">": "above", ">=": "at least", "<": "below", "<=": "at most", "==": "is", "!=": "is not",
            "between": "between", "in": "is one of", "not in": "is none of"}


def label_of(field: str) -> str:
    return (FEATURE_INFO.get(field) or EXTRA_INFO.get(field) or (field.replace("_", " "), ""))[0]


def format_feature(field: str, value, *, compact: bool = False) -> str:
    """One value the way the screener table shows it: fractions as %, dollars compact, flags as yes / no.

    ``compact`` drops needless decimals from plain numbers (a threshold of 35, not 35.00).
    """
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return "-"
    kind = (FEATURE_INFO.get(field) or EXTRA_INFO.get(field) or ("", "num"))[1]
    try:
        if kind == "pct":
            return pct(float(value))
        if kind == "usd":
            return "$" + money(float(value))
        if kind in ("int", "score", "scorebar"):
            return f"{float(value):.0f}"
        if kind == "bool":
            return "yes" if bool(value) else "no"
        if kind == "date":
            return f"{value:%Y-%m-%d}"
        if kind == "text":
            return str(value)
        return f"{float(value):g}" if compact else f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def describe_rule(rule) -> str:
    """'RSI(14) below 35', 'From 52w high between -3.0% and 0.0%' - a rule as a phrase."""
    if rule.op == "between":
        target = f"{format_feature(rule.field, rule.value[0], compact=True)} and {format_feature(rule.field, rule.value[1], compact=True)}"
    elif rule.op in ("in", "not in"):
        target = ", ".join(format_feature(rule.field, v, compact=True) for v in rule.value)
    else:
        target = format_feature(rule.field, rule.value, compact=True)
    return f"{label_of(rule.field)} {OP_WORDS[rule.op]} {target}"


def money(value: float | None, digits: int = 2) -> str:
    """12_345_678 -> '12.35 M'; small numbers keep their decimals."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    a = abs(value)
    for limit, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if a >= limit:
            return f"{value / limit:,.{digits}f} {suffix}"
    return f"{value:,.{digits}f}" if a < 1000 else f"{value:,.0f}"


def pct(value: float | None, digits: int = 1, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value * 100:+.{digits}f}%" if signed else f"{value * 100:.{digits}f}%"


def _heat_css(value, vmin: float, vmax: float) -> str:
    """Red <- 0 -> green background with opacity growing toward the range ends (theme-neutral)."""
    if value is None or value != value:
        return ""
    span = (vmax if value >= 0 else -vmin) or 1.0
    strength = min(abs(value) / span, 1.0)
    r, g, b = (38, 166, 154) if value >= 0 else (239, 83, 80)
    return f"background-color: rgba({r},{g},{b},{0.08 + 0.42 * strength:.2f})"


def heat(styler, subset=None, vmin: float = -1.0, vmax: float = 1.0):
    """Colour cells like ``Styler.background_gradient`` without needing matplotlib."""
    return styler.map(lambda v: _heat_css(v, vmin, vmax), subset=subset)


def statement_table(df: pd.DataFrame) -> pd.DataFrame:
    """Statement with human-readable amounts; EPS-like rows keep their decimals."""
    if df.empty:
        return df
    out = df.copy()
    out.columns = [c.strftime("%Y-%m-%d") if hasattr(c, "strftime") else str(c) for c in out.columns]
    out = out.apply(lambda row: row.map(money))  # EPS-like rows (|x| < 1000) keep their decimals
    return out.iloc[:, ::-1]  # newest first


def feature_view(df: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Display copy of a feature/screen table: fractions x100 with ``%`` formats, labels from FEATURE_INFO."""
    cols = [c for c in (columns or df.columns) if c in df.columns]
    out = df[cols].copy()
    config: dict = {}
    for c in cols:
        label, kind = FEATURE_INFO.get(c) or EXTRA_INFO.get(c) or (c.replace("_", " ").title(), "num")
        if kind in ("pct", "usd", "num", "int", "score", "scorebar") and not pd.api.types.is_bool_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")  # a column Yahoo left empty arrives as None, not NaN
        if kind == "scorebar":
            config[c] = st.column_config.ProgressColumn(label, min_value=0, max_value=100, format="%.0f")
        elif kind == "pct":
            out[c] = out[c] * 100
            config[c] = st.column_config.NumberColumn(label, format="%.1f%%")
        elif kind == "usd":
            out[c] = out[c] / 1e6
            config[c] = st.column_config.NumberColumn(label.replace("$", "$M"), format="%.1f")
        elif kind == "int":
            config[c] = st.column_config.NumberColumn(label, format="%d")
        elif kind == "score":
            config[c] = st.column_config.NumberColumn(label, format="%.0f", help="Percentile rank 0-100")
        elif kind == "bool":
            config[c] = st.column_config.CheckboxColumn(label)
        elif kind == "date":
            config[c] = st.column_config.DateColumn(label)
        elif kind == "text":
            config[c] = st.column_config.TextColumn(label)
        elif pd.api.types.is_numeric_dtype(out[c]):
            config[c] = st.column_config.NumberColumn(label, format="%.2f")
    return out, config
