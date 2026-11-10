"""Reusable input widgets."""

from __future__ import annotations

import streamlit as st

from stockanalyzer.analysis.technical.registry import Param
from stockanalyzer.data import universe as uni
from stockanalyzer.ui import data as ui_data

PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]
INTERVALS = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo", "Hourly (last 2y)": "1h"}
INTRADAY_PERIODS = ["5d", "1mo", "3mo", "6mo", "1y", "2y"]


def symbol_picker(key: str = "symbol", default: str = "AAPL", *, label: str = "Symbol", container=None) -> str:
    """Ticker box shared across pages. If the text is not a known ticker, offer matches by name."""
    box = container or st.sidebar
    current = st.session_state.get(key, default)
    typed = box.text_input(label, value=current, key=f"_{key}_input",
                           help="Yahoo format: AAPL, BRK-B, PTT.BK, ^GSPC, GC=F, BTC-USD, THB=X").strip()
    value = typed.upper()
    table = ui_data.universe_table()
    if typed and value not in set(table["symbol"]):
        hits = uni.search(table, typed, limit=8)
        if not hits.empty:
            names = dict(zip(hits["symbol"], hits["name"]))
            value = box.selectbox("Did you mean", [value, *hits["symbol"]], key=f"_{key}_match",
                                  format_func=lambda s: s if s not in names else f"{s} - {names[s]}")
    st.session_state[key] = value
    return value


def period_and_interval(container=None, *, default_period: str = "1y", allow_intraday: bool = True) -> tuple[str, str]:
    box = container or st.sidebar
    options = list(INTERVALS) if allow_intraday else [k for k, v in INTERVALS.items() if v != "1h"]
    interval = INTERVALS[box.selectbox("Bar size", options, key="_interval")]
    if interval == "1h":
        period = box.selectbox("Period", INTRADAY_PERIODS, index=2, key="_period_intraday")
    else:
        period = box.selectbox("Period", PERIODS, index=PERIODS.index(default_period), key="_period")
    return period, interval


def param_inputs(params: tuple[Param, ...], key: str, container=None) -> dict:
    """One number input per indicator/strategy parameter, honouring ranges and int/float."""
    box = container or st
    values = {}
    for p in params:
        if p.is_int:
            values[p.name] = int(box.number_input(p.name, int(p.min), int(p.max), int(p.default), int(p.step),
                                                  key=f"{key}_{p.name}"))
        else:
            values[p.name] = float(box.number_input(p.name, float(p.min), float(p.max), float(p.default),
                                                    float(p.step), key=f"{key}_{p.name}"))
    return values


def show_table(df, **kwargs) -> None:
    st.dataframe(df, width="stretch", **kwargs)


def show_chart(fig, **kwargs) -> None:
    st.plotly_chart(fig, width="stretch", **kwargs)


def amount_scale(exact: float) -> tuple[float, str, int, float]:
    """(scale, unit letter, decimals, value shown in that unit) for a money/count box."""
    scale, unit = (1e9, "B") if abs(exact) >= 1e9 else (1e6, "M")
    decimals = 3 if scale == 1e9 else 2
    return scale, unit, decimals, round(exact / scale, decimals)


def amount_input(label: str, exact: float, key: str, *, help: str | None = None, min_value: float | None = None) -> float:
    """Number box in billions/millions instead of 11-digit raw numbers.

    The exact suggested amount is returned until the user edits the shown number, so the display
    rounding never leaks into the calculation.
    """
    scale, unit, decimals, shown = amount_scale(exact)
    step = max(float(f"{abs(shown) * 0.05:.1g}"), 10.0**-decimals)
    value = st.number_input(f"{label} ({unit})", key=key, step=step, format=f"%.{decimals}f", help=help, min_value=min_value)
    return exact if value == shown else value * scale
