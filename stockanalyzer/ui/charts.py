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
                  strategy_name: str = "Strategy", benchmark_name: str = "Buy & hold") -> go.Figure:
    rows = 2 if drawdown is not None else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.7, 0.3] if rows == 2 else [1.0])
    fig.add_trace(go.Scatter(x=strategy.index, y=strategy, name=strategy_name, line=dict(width=2, color=PALETTE[0])), row=1, col=1)
    fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark, name=benchmark_name, line=dict(width=1.5, color="#9e9e9e")), row=1, col=1)
    if drawdown is not None:
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown * 100, name="Drawdown %", fill="tozeroy",
                                 line=dict(width=1, color=DOWN)), row=2, col=1)
        fig.update_yaxes(title_text="DD %", row=2, col=1)
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_layout(height=520, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0))
    return fig


def weights_figure(weights: pd.DataFrame, *, top: int = 12, height: int = 340) -> go.Figure:
    """Share of the equity held in each asset over time, stacked; what is missing up to 100% is cash.

    With more than ``top`` assets the ones held least on average are summed into "Other (n)".
    """
    order = weights.mean().sort_values(ascending=False).index
    if len(order) > top:
        rest = order[top:]
        weights = pd.concat([weights[order[:top]], weights[rest].sum(axis=1).rename(f"Other ({len(rest)})")], axis=1)
    fig = go.Figure()
    for i, col in enumerate(weights.columns):
        color = "#9e9e9e" if str(col).startswith("Other (") else PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(x=weights.index, y=weights[col] * 100, name=str(col), mode="lines", stackgroup="w",
                                 line=dict(width=0.5, color=color), fillcolor=_rgba(color, 0.75)))
    fig.update_yaxes(title_text="% of equity", range=[0, 100])
    fig.update_layout(height=height, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
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


def spread_figure(spread: pd.Series, *, title: str = "", band: float = 2.0, height: int = 320) -> go.Figure:
    """A spread as a z-score (standard deviations from its own mean) with the mean and +/- ``band`` lines.

    Above the upper line the first leg is expensive against the second, below the lower line it is cheap.
    """
    sd = spread.std(ddof=1)
    z = (spread - spread.mean()) / sd if sd and sd > 0 else spread * 0.0
    fig = go.Figure(go.Scatter(x=z.index, y=z, name="z-score", mode="lines", line=dict(width=1.5, color=PALETTE[0])))
    fig.add_hline(y=0, line=dict(color="#888888", width=1))
    fig.add_hline(y=band, line=dict(color=DOWN, width=1, dash="dot"), annotation_text=f"+{band:g}", annotation_position="top left")
    fig.add_hline(y=-band, line=dict(color=UP, width=1, dash="dot"), annotation_text=f"-{band:g}", annotation_position="bottom left")
    fig.update_layout(title=title, height=height, hovermode="x unified", showlegend=False,
                      margin=dict(l=10, r=10, t=50 if title else 20, b=10), yaxis_title="std devs")
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


def bar_figure(series: pd.Series, *, title: str = "", horizontal: bool = True, fmt: str = ".1%", height: int | None = None,
               sort: bool = True) -> go.Figure:
    s = series.sort_values() if sort else series
    fig = go.Figure(go.Bar(x=s.values if horizontal else s.index, y=s.index if horizontal else s.values,
                           orientation="h" if horizontal else "v", marker_color=PALETTE[0],
                           text=[format(v, fmt) for v in s.values], textposition="auto"))
    fig.update_layout(title=title, height=height or max(240, 28 * len(s) + 100), margin=dict(l=10, r=10, t=50 if title else 20, b=10))
    return fig


QUADRANT_COLORS = {"Leading": UP, "Weakening": "#ffa726", "Lagging": DOWN, "Improving": PALETTE[0]}


def quadrant_of(ratio: float, momentum: float) -> str:
    if ratio >= 100:
        return "Leading" if momentum >= 100 else "Weakening"
    return "Improving" if momentum >= 100 else "Lagging"


def rrg_figure(ratio: pd.DataFrame, momentum: pd.DataFrame, *, trail: int = 8, height: int = 600) -> go.Figure:
    """Relative rotation graph: each sector's path over the last ``trail`` bars, coloured by its current quadrant.

    A sector normally travels clockwise: Improving -> Leading -> Weakening -> Lagging.
    """
    ratio = ratio.tail(trail + 1)
    momentum = momentum.reindex(ratio.index)
    fig = go.Figure()
    xs, ys = [100.0], [100.0]
    for name in ratio.columns:
        keep = ratio[name].notna() & momentum[name].notna()
        x, y = ratio.loc[keep, name], momentum.loc[keep, name]
        if x.empty:
            continue
        color = QUADRANT_COLORS[quadrant_of(x.iloc[-1], y.iloc[-1])]
        xs += list(x)
        ys += list(y)
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers+text", name=name, showlegend=False,
            line=dict(width=1.5, color=_rgba(color, 0.55)),
            marker=dict(size=[5] * (len(x) - 1) + [12], color=color), text=[""] * (len(x) - 1) + [name],
            textposition="top center", customdata=[d.strftime("%Y-%m-%d") for d in x.index],
            hovertemplate=f"<b>{name}</b> %{{customdata}}<br>RS-Ratio %{{x:.1f}}<br>RS-Momentum %{{y:.1f}}<extra></extra>"))
    half_x, half_y = (max(abs(v - 100) for v in vs) * 1.15 + 0.5 for vs in (xs, ys))
    x0, x1, y0, y1 = 100 - half_x, 100 + half_x, 100 - half_y, 100 + half_y
    corners = {"Leading": (100, x1, 100, y1, "right", "top"), "Weakening": (100, x1, y0, 100, "right", "bottom"),
               "Lagging": (x0, 100, y0, 100, "left", "bottom"), "Improving": (x0, 100, 100, y1, "left", "top")}
    for quad, (xa, xb, ya, yb, xanchor, yanchor) in corners.items():
        fig.add_shape(type="rect", x0=xa, x1=xb, y0=ya, y1=yb, fillcolor=_rgba(QUADRANT_COLORS[quad], 0.07), line_width=0, layer="below")
        fig.add_annotation(x=xb if xanchor == "right" else xa, y=yb if yanchor == "top" else ya, text=f"<b>{quad}</b>", showarrow=False,
                           xanchor=xanchor, yanchor=yanchor, font=dict(color=QUADRANT_COLORS[quad], size=13))
    fig.update_xaxes(range=[x0, x1], title_text="RS-Ratio  (relative strength vs its own trend)", zeroline=False)
    fig.update_yaxes(range=[y0, y1], title_text="RS-Momentum  (is it improving?)", zeroline=False)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=20, b=10), hovermode="closest")
    return fig


def bump_figure(ranks: pd.DataFrame, *, highlight: Sequence[str] = (), height: int | None = None) -> go.Figure:
    """Rank of every sector in every period (1 at the top); ``highlight`` sectors are drawn bold, the rest faded."""
    labels = [d.strftime("%b %y") for d in ranks.index]
    fig = go.Figure()
    for i, name in enumerate(ranks.columns):
        r = ranks[name]
        on = not highlight or name in highlight
        color = PALETTE[i % len(PALETTE)] if on else "#bdbdbd"
        last = r.last_valid_index()
        text = [name if d == last else "" for d in ranks.index]
        fig.add_trace(go.Scatter(x=labels, y=r, name=name, mode="lines+markers+text", text=text, textposition="middle right",
                                 cliponaxis=False, line=dict(width=3 if on and highlight else 1.5, color=color), marker=dict(size=7 if on else 4),
                                 opacity=1.0 if on else 0.55, hovertemplate=f"<b>{name}</b> %{{x}}: rank %{{y:.0f}}<extra></extra>"))
    fig.update_yaxes(range=[len(ranks.columns) + 0.5, 0.5], dtick=1, title_text="Rank (1 = best)")  # reversed: rank 1 on top
    fig.update_xaxes(type="category")
    fig.update_layout(height=height or max(360, 34 * len(ranks.columns) + 120), showlegend=False, margin=dict(l=10, r=190, t=20, b=10),
                      hovermode="closest")
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


def method_figure(values: dict[str, float], price: float | None = None) -> go.Figure:
    """One bar per valuation method, teal where the value is above the price (cheap) and red where it is below, with the price as a dashed line."""
    names = [k for k, v in values.items() if v == v]
    fig = go.Figure(go.Bar(x=names, y=[values[k] for k in names], text=[f"{values[k]:,.2f}" for k in names], textposition="outside",
                           marker_color=[UP if not price or values[k] >= price else DOWN for k in names]))
    if price:
        fig.add_hline(y=price, line_dash="dash", line_color="#9e9e9e", annotation_text=f"Price {price:,.2f}", annotation_position="top left")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Value per share", showlegend=False)
    return fig


BAND_COLORS = {-2: "#00897b", -1: "#80cbc4", 0: "#9e9e9e", 1: "#ef9a9a", 2: "#c62828"}


def _sd_label(k: float) -> str:
    return "mean" if k == 0 else f"{k:+g} SD"


def band_figure(price: pd.Series, implied: pd.DataFrame, *, title: str = "", height: int = 420) -> go.Figure:
    """Price over the prices that the company's own average multiple, +-1 and +-2 SD would imply that day.

    ``implied`` has one column per band level (see ``bands.implied_prices``); low levels are teal (cheap), high red.
    """
    fig = go.Figure()
    for k in implied.columns:
        s = implied[k].dropna()
        fig.add_trace(go.Scatter(x=s.index, y=s, name=_sd_label(k), mode="lines",
                                 line=dict(width=1.1, color=BAND_COLORS.get(k, "#9e9e9e"), dash="solid" if k == 0 else "dash")))
    fig.add_trace(go.Scatter(x=price.index, y=price, name="Price", mode="lines", line=dict(width=1.8, color=PALETTE[0])))
    fig.update_layout(title=title, height=height, hovermode="x unified", margin=dict(l=10, r=10, t=50 if title else 20, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0))
    return fig


def multiple_figure(series: pd.Series, stats, *, name: str, tickformat: str = ".1f", height: int = 320) -> go.Figure:
    """A multiple through time with its average and +-1 / +-2 SD lines (``stats``: ``bands.BandStats``)."""
    s = series.dropna()
    fig = go.Figure()
    fig.add_hrect(y0=stats.level(-1), y1=stats.level(1), fillcolor="#9e9e9e", opacity=0.12, line_width=0)
    for k in (-2, -1, 0, 1, 2):
        fig.add_hline(y=stats.level(k), line_width=1, line_dash="solid" if k == 0 else "dot",
                      line_color=BAND_COLORS[k], annotation_text=_sd_label(k), annotation_position="right")
    fig.add_trace(go.Scatter(x=s.index, y=s, name=name, mode="lines", line=dict(width=1.6, color=PALETTE[0])))
    fig.update_layout(height=height, hovermode="x unified", showlegend=False, margin=dict(l=10, r=60, t=20, b=10),
                      yaxis=dict(tickformat=tickformat, title=name))
    return fig


def sensitivity_figure(grid: pd.DataFrame, price: float | None = None) -> go.Figure:
    fig = heatmap_figure(grid.rename(index=lambda v: f"{v:.1%}", columns=lambda v: f"{v:.1%}"), fmt=".1f",
                         zmin=float(np.nanmin(grid.to_numpy())), zmax=float(np.nanmax(grid.to_numpy())), colorscale="Viridis")
    fig.update_xaxes(title_text="Terminal growth")
    fig.update_yaxes(title_text="WACC")
    if price:
        fig.update_layout(title=f"Intrinsic value per share (current price {price:,.2f})")
    return fig


def grouped_bar_figure(df: pd.DataFrame, *, title: str = "", height: int = 340, tickformat: str | None = None,
                       colors: Sequence[str] | None = None) -> go.Figure:
    """One group of bars per row (e.g. a year), one bar per column."""
    fig = go.Figure()
    for i, col in enumerate(df.columns):
        fig.add_trace(go.Bar(x=[str(x) for x in df.index], y=df[col], name=str(col),
                             marker_color=(colors[i] if colors and i < len(colors) else PALETTE[i % len(PALETTE)])))
    fig.add_hline(y=0, line=dict(color="#888888", width=1))
    fig.update_layout(title=title, barmode="group", height=height, hovermode="x unified", yaxis_tickformat=tickformat,
                      margin=dict(l=10, r=10, t=50 if title else 20, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0))
    return fig


# -- market breadth -----------------------------------------------------------------------------------
REGIME_COLORS = {"Healthy uptrend": UP, "Narrow uptrend": "#ffa726", "Improving underneath": PALETTE[0], "Downtrend": DOWN,
                 "Calm": UP, "Stressed": "#ffa726", "Crisis": DOWN}


def stacked_figure(top: pd.DataFrame, bottom: pd.DataFrame, *, top_title: str = "", bottom_title: str = "",
                   bottom_levels: Sequence[float] = (), bottom_bars: Sequence[str] = (), bottom_format: str | None = None,
                   log_top: bool = False, height: int = 560) -> go.Figure:
    """Two panels on one time axis: e.g. the index on top and a breadth reading under it."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.55, 0.45])
    for i, col in enumerate(top.columns):
        fig.add_trace(go.Scatter(x=top.index, y=top[col], name=str(col), mode="lines",
                                 line=dict(width=1.6 if i == 0 else 1.1, color=PALETTE[i] if i == 0 else "#9e9e9e",
                                           dash="solid" if i == 0 else "dot")), row=1, col=1)
    for i, col in enumerate(bottom.columns):
        color = PALETTE[(i + 1) % len(PALETTE)]
        if col in bottom_bars:
            s = bottom[col]
            fig.add_trace(go.Bar(x=s.index, y=s, name=str(col), marker_color=[UP if v >= 0 else DOWN for v in s.fillna(0)],
                                 marker_line_width=0), row=2, col=1)
        else:
            fig.add_trace(go.Scatter(x=bottom.index, y=bottom[col], name=str(col), mode="lines", line=dict(width=1.4, color=color)),
                          row=2, col=1)
    for level in bottom_levels:
        fig.add_hline(y=level, line=dict(color="#888888", width=1, dash="dot"), row=2, col=1)
    fig.update_yaxes(title_text=top_title, type="log" if log_top else "linear", row=1, col=1)
    fig.update_yaxes(title_text=bottom_title, tickformat=bottom_format, row=2, col=1)
    fig.update_layout(height=height, hovermode="x unified", bargap=0, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0))
    return fig


def highs_lows_figure(new_highs: pd.Series, new_lows: pd.Series, net_10: pd.Series, *, height: int = 380) -> go.Figure:
    """New 52-week highs as bars above zero, new lows below, with the 10-day average of highs minus lows."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=new_highs.index, y=new_highs, name="New highs", marker_color=UP, marker_line_width=0))
    fig.add_trace(go.Bar(x=new_lows.index, y=-new_lows, name="New lows", marker_color=DOWN, marker_line_width=0))
    fig.add_trace(go.Scatter(x=net_10.index, y=net_10, name="Highs − lows, 10-day average", mode="lines",
                             line=dict(width=1.6, color=PALETTE[0])))
    fig.add_hline(y=0, line=dict(color="#888888", width=1))
    fig.update_layout(height=height, barmode="relative", bargap=0, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title="stocks", legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0))
    return fig


def regime_figure(level: pd.Series, regime: pd.Series, *, name: str = "Index", log: bool = True, height: int = 420) -> go.Figure:
    """The index with its background shaded by breadth regime."""
    fig = go.Figure(go.Scatter(x=level.index, y=level, name=name, mode="lines", line=dict(width=1.6, color="#212121")))
    s = regime.reindex(level.index)
    run = (s != s.shift()).cumsum()
    seen: set[str] = set()
    for _, g in s.groupby(run):
        label = g.iloc[0]
        if not isinstance(label, str):
            continue
        fig.add_vrect(x0=g.index[0], x1=g.index[-1], fillcolor=REGIME_COLORS.get(label, "#bdbdbd"), opacity=0.18, line_width=0)
        if label not in seen:  # a legend entry per regime
            seen.add(label)
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label,
                                     marker=dict(size=10, symbol="square", color=REGIME_COLORS.get(label, "#bdbdbd"))))
    fig.update_yaxes(type="log" if log else "linear")
    fig.update_layout(height=height, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0))
    return fig


def fan_figure(bands: pd.DataFrame, invested: pd.Series, *, samples: pd.DataFrame | None = None,
               compare: pd.Series | None = None, compare_name: str = "Base case median", currency: str = "",
               log: bool = False, height: int = 460) -> go.Figure:
    """Simulated portfolio value: 5-95 % and 25-75 % bands, the median, the money put in and a few single paths."""
    fig = go.Figure()
    x = bands.index
    if samples is not None:
        for i, col in enumerate(samples.columns):
            fig.add_trace(go.Scatter(x=x, y=samples[col], mode="lines", line=dict(width=0.7, color="rgba(120,120,120,0.35)"),
                                     name="Single paths", legendgroup="paths", showlegend=i == 0, hoverinfo="skip"))
    for lo, hi, alpha, label in (("p5", "p95", 0.15, "5-95 % of paths"), ("p25", "p75", 0.3, "25-75 % of paths")):
        fig.add_trace(go.Scatter(x=x, y=bands[hi], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=bands[lo], mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor=_rgba(PALETTE[0], alpha), name=label, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=bands["p50"], mode="lines", line=dict(width=2.2, color=PALETTE[0]), name="Median"))
    if compare is not None:
        fig.add_trace(go.Scatter(x=compare.index, y=compare, mode="lines", line=dict(width=1.8, color=PALETTE[1], dash="dot"),
                                 name=compare_name))
    fig.add_trace(go.Scatter(x=invested.index, y=invested, mode="lines", line=dict(width=1.6, color="#424242", dash="dash"),
                             name="Money put in"))
    fig.update_yaxes(title=currency or None, type="log" if log else "linear", tickformat=",.0f")
    fig.update_layout(height=height, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0))
    return fig
