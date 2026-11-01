"""Plotly figure builders (no Streamlit dependency, so they are unit-testable)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

UP, DOWN = "#26a69a", "#ef5350"
PALETTE = ["#2962ff", "#ff6d00", "#7b1fa2", "#00897b", "#c2185b", "#6d4c41", "#546e7a", "#afb42b"]


@dataclass
class Pane:
    """A separate panel under the price chart (RSI, MACD ...)."""
    title: str
    series: list[tuple[str, pd.Series]] = field(default_factory=list)
    levels: Sequence[float] = ()
    bars: Sequence[str] = ()  # series names drawn as bars (e.g. MACD histogram)


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def price_figure(
    df: pd.DataFrame,
    *,
    title: str = "",
    overlays: Sequence[tuple[str, pd.Series]] = (),
    panes: Sequence[Pane] = (),
    volume: bool = True,
    zones: Sequence = (),
    fib: dict[float, float] | None = None,
    markers: dict[str, pd.Series] | None = None,
    skip_weekends: bool = True,
    height: int | None = None,
    zone_colors: tuple[str, str, str] = ("#3B82F6", "#EF4444", "#A3A3A3"),
    zone_alpha: float = 0.2,
) -> go.Figure:
    """Candlesticks + overlays on top, optional volume, then one panel per ``Pane``."""
    rows = 1 + int(volume) + len(panes)
    pane_h = 0.16
    heights = [0.55 if rows > 1 else 1.0]
    if volume:
        heights.append(0.12)
    heights += [pane_h] * len(panes)
    scale = sum(heights)
    heights = [h / scale for h in heights]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.025, row_heights=heights)
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                 name="Price", increasing_line_color=UP, decreasing_line_color=DOWN), row=1, col=1)
    for i, (name, s) in enumerate(overlays):
        fig.add_trace(go.Scatter(x=s.index, y=s, name=name, mode="lines", line=dict(width=1.3, color=PALETTE[i % len(PALETTE)])),
                      row=1, col=1)

    for name, s in (markers or {}).items():
        hits = s[s != 0].dropna()
        if len(hits):
            ys = df["low"].reindex(hits.index) * 0.995 if (hits > 0).all() else df["high"].reindex(hits.index) * 1.005
            fig.add_trace(go.Scatter(x=hits.index, y=ys, mode="markers", name=name,
                                     marker=dict(symbol="triangle-up" if (hits > 0).all() else "triangle-down", size=9,
                                                 color=UP if (hits > 0).all() else DOWN)), row=1, col=1)

    row = 2
    if volume:
        colors = np.where(df["close"] >= df["open"], UP, DOWN)
        fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color=colors, showlegend=False),
                      row=row, col=1)
        fig.update_yaxes(title_text="Vol", row=row, col=1)
        row += 1

    for pane in panes:
        for i, (name, s) in enumerate(pane.series):
            if name in pane.bars:
                fig.add_trace(go.Bar(x=s.index, y=s, name=name, marker_color=_rgba(PALETTE[i % len(PALETTE)], 0.6)),
                              row=row, col=1)
            else:
                fig.add_trace(go.Scatter(x=s.index, y=s, name=name, mode="lines",
                                         line=dict(width=1.3, color=PALETTE[i % len(PALETTE)])), row=row, col=1)
        for level in pane.levels:
            fig.add_hline(y=level, line_dash="dot", line_color="gray", line_width=1, row=row, col=1)
        fig.update_yaxes(title_text=pane.title, row=row, col=1)
        row += 1

    if len(df):
        last_close = float(df["close"].iloc[-1])
        x_end = df.index[-1]
        for z in zones:
            low, high = float(z.low), float(z.high)
            role = z.role(last_close)
            base = zone_colors[0] if role == "support" else zone_colors[1] if role == "resistance" else zone_colors[2]
            x0 = df.index[min(int(z.first_idx), len(df) - 1)]
            fig.add_shape(type="rect", xref="x", yref="y", x0=x0, x1=x_end, y0=low, y1=high,
                          fillcolor=_rgba(base, zone_alpha), line=dict(color=_rgba(base, min(0.95, zone_alpha + 0.35)), width=1),
                          layer="below")
            fig.add_trace(go.Scatter(x=[x0, x_end], y=[(low + high) / 2] * 2, mode="lines", line=dict(width=0, color="rgba(0,0,0,0)"),
                                     name=f"{role.title()} zone ({z.touches} pivots)", showlegend=False,
                                     hovertemplate=f"{role.title()} zone<br>{low:,.2f} - {high:,.2f}<br>{z.touches} pivots<extra></extra>"),
                          row=1, col=1)
        for ratio, price in (fib or {}).items():
            fig.add_hline(y=price, line_dash="dot", line_color="#9e9e9e", line_width=1, row=1, col=1,
                          annotation_text=f"Fib {ratio:.1%}", annotation_position="right", annotation_font_size=10)

    fig.update_layout(title=title, height=height or 460 + 150 * (rows - 1), hovermode="x unified",
                      xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=50 if title else 20, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0))
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1)
    if skip_weekends:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


def equity_figure(strategy: pd.Series, benchmark: pd.Series, drawdown: pd.Series | None = None,
                  strategy_name: str = "Strategy") -> go.Figure:
    rows = 2 if drawdown is not None else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.7, 0.3] if rows == 2 else [1.0])
    fig.add_trace(go.Scatter(x=strategy.index, y=strategy, name=strategy_name, line=dict(width=2, color=PALETTE[0])), row=1, col=1)
    fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark, name="Buy & hold", line=dict(width=1.5, color="#9e9e9e")), row=1, col=1)
    if drawdown is not None:
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown * 100, name="Drawdown %", fill="tozeroy",
                                 line=dict(width=1, color=DOWN)), row=2, col=1)
        fig.update_yaxes(title_text="DD %", row=2, col=1)
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_layout(height=520, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0))
    return fig


def line_figure(df: pd.DataFrame, *, title: str = "", rebase: bool = False, height: int = 420) -> go.Figure:
    data = df.dropna(how="all")
    if rebase:
        data = data.apply(lambda s: 100 * s / s.dropna().iloc[0] if s.notna().any() else s)
    fig = go.Figure()
    for i, col in enumerate(data.columns):
        fig.add_trace(go.Scatter(x=data.index, y=data[col], name=str(col), mode="lines",
                                 line=dict(width=1.5, color=PALETTE[i % len(PALETTE)])))
    fig.update_layout(title=title, height=height, hovermode="x unified", margin=dict(l=10, r=10, t=50 if title else 20, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0))
    return fig


def heatmap_figure(matrix: pd.DataFrame, *, title: str = "", zmin: float = -1.0, zmax: float = 1.0,
                   colorscale: str = "RdBu", fmt: str = ".2f", height: int | None = None) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=matrix.to_numpy(), x=[str(c) for c in matrix.columns], y=[str(i) for i in matrix.index],
                               zmin=zmin, zmax=zmax, colorscale=colorscale, reversescale=colorscale == "RdBu",
                               text=matrix.map(lambda v: format(v, fmt) if pd.notna(v) else "").to_numpy(),
                               texttemplate="%{text}", hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=title, height=height or max(320, 40 * len(matrix) + 120), margin=dict(l=10, r=10, t=50 if title else 20, b=10))
    return fig


def bar_figure(series: pd.Series, *, title: str = "", horizontal: bool = True, fmt: str = ".1%", height: int | None = None) -> go.Figure:
    s = series.sort_values()
    fig = go.Figure(go.Bar(x=s.values if horizontal else s.index, y=s.index if horizontal else s.values,
                           orientation="h" if horizontal else "v", marker_color=PALETTE[0],
                           text=[format(v, fmt) for v in s.values], textposition="auto"))
    fig.update_layout(title=title, height=height or max(240, 28 * len(s) + 100), margin=dict(l=10, r=10, t=50 if title else 20, b=10))
    return fig


def scenario_figure(scen: pd.DataFrame, price: float | None = None) -> go.Figure:
    """Bear / base / bull value per share as bars, with the current price as a dashed line."""
    colors = {"Bear": DOWN, "Base": PALETTE[0], "Bull": UP}
    fig = go.Figure(go.Bar(x=scen["scenario"], y=scen["value_per_share"],
                           marker_color=[colors.get(s, PALETTE[2]) for s in scen["scenario"]],
                           text=[f"{v:,.2f}" for v in scen["value_per_share"]], textposition="outside"))
    if price:
        fig.add_hline(y=price, line_dash="dash", line_color="#9e9e9e", annotation_text=f"Price {price:,.2f}",
                      annotation_position="top left")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Value per share", showlegend=False)
    return fig


def sensitivity_figure(grid: pd.DataFrame, price: float | None = None) -> go.Figure:
    fig = heatmap_figure(grid.rename(index=lambda v: f"{v:.1%}", columns=lambda v: f"{v:.1%}"), fmt=".1f",
                         zmin=float(np.nanmin(grid.to_numpy())), zmax=float(np.nanmax(grid.to_numpy())), colorscale="Viridis")
    fig.update_xaxes(title_text="Terminal growth")
    fig.update_yaxes(title_text="WACC")
    if price:
        fig.update_layout(title=f"Intrinsic value per share (current price {price:,.2f})")
    return fig
