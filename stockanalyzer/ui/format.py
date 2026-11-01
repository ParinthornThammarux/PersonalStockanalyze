"""Number formatting and ``st.dataframe`` column configs."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from stockanalyzer.screener.features import FEATURE_INFO


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
        label, kind = FEATURE_INFO.get(c, (c.replace("_", " ").title(), "num"))
        if kind == "pct":
            out[c] = out[c] * 100
            config[c] = st.column_config.NumberColumn(label, format="%.1f%%")
        elif kind == "usd":
            out[c] = out[c] / 1e6
            config[c] = st.column_config.NumberColumn(label.replace("$", "$M"), format="%.1f")
        elif kind == "int":
            config[c] = st.column_config.NumberColumn(label, format="%d")
        elif kind == "bool":
            config[c] = st.column_config.CheckboxColumn(label)
        elif kind == "date":
            config[c] = st.column_config.DateColumn(label)
        elif kind == "text":
            config[c] = st.column_config.TextColumn(label)
        elif pd.api.types.is_numeric_dtype(out[c]):
            config[c] = st.column_config.NumberColumn(label, format="%.2f")
    return out, config
