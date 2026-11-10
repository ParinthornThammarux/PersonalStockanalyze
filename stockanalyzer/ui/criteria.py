"""What the Screener's criteria builder needs to know about a column: its group, kind, units, operators and a sensible start.

Pure functions (no Streamlit) so they can be tested. A *kind* decides which input the page shows:
``pct`` (a fraction shown and typed as %), ``usd`` (dollars shown as US$ millions), ``num``, ``int``, ``bool`` and ``text``.
"""

from __future__ import annotations

import math

import pandas as pd

from stockanalyzer.ui.format import EXTRA_INFO, FEATURE_INFO, format_feature, label_of

# group -> fields, in the order the picker lists them
GROUPS: dict[str, tuple[str, ...]] = {
    "Price & momentum": ("ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_126d", "ret_252d", "mom_12_1", "rs_126",
                         "pct_from_52w_high", "pct_from_52w_low", "new_52w_high", "drawdown_now", "drawdown_1y_max", "close"),
    "Trend": ("trend_label", "trend_score", "above_sma50", "above_sma200", "golden_cross", "sma20_dist", "sma50_dist",
              "sma200_dist", "adx14", "trend_r2", "trend_slope_annual"),
    "Oscillators": ("rsi14", "stoch_k", "mfi14", "macd_hist_pct", "roc_12", "bb_pctb", "bb_bandwidth"),
    "Volatility & risk": ("atr_pct", "hvol_20", "hvol_60", "beta_1y"),
    "Volume & size": ("volume_ratio", "dollar_volume_20", "market_cap"),
    "Support & resistance": ("support_dist_atr", "support_dist_pct", "resistance_dist_atr", "resistance_dist_pct", "n_zones"),
    "Valuation": ("pe", "forward_pe", "pb", "ps", "ev_sales", "ev_ebitda", "peg", "ev_sales_growth", "dividend_yield", "fcf_yield",
                  "eps", "forward_eps"),
    "Quality & growth": ("roe", "roa", "profit_margin", "operating_margin", "gross_margin", "fcf_margin", "revenue_growth",
                         "earnings_growth", "rule_of_40", "debt_to_equity", "current_ratio"),
    "Analysts": ("target_upside", "recommendation_mean", "analyst_count"),
}
OTHER = "Other"
GROUP_OF = {field: group for group, fields in GROUPS.items() for field in fields}
TEXT_FIELDS = ("trend_label", "currency")  # the only text columns worth a rule; names and industries have their own filters
NEVER_OFFERED = {"name", "industry", "asset_class", "sector", "fundamentals", "fundamentals_at", "last_date",
                 "revenue_ttm", "target_mean"}  # raw amounts in each stock's own currency: not comparable across the pool

NUMERIC_OPS = [(">", "is above"), (">=", "is at least"), ("<", "is below"), ("<=", "is at most"), ("between", "is between")]
OPS_BY_KIND = {
    "bool": [("==", "is")],
    "text": [("==", "is"), ("!=", "is not"), ("in", "is one of"), ("not in", "is none of")],
}
EXTRA_NUMERIC_OPS = [("==", "equals"), ("!=", "differs from")]  # offered only when a loaded rule already uses them
MAX_TEXT_OPTIONS = 60


def kind_of(field: str, series: pd.Series | None = None) -> str:
    info = FEATURE_INFO.get(field) or EXTRA_INFO.get(field)
    kind = info[1] if info else None
    if kind in ("pct", "usd", "bool", "text", "int"):
        return kind
    if series is not None and kind is None:
        if pd.api.types.is_bool_dtype(series):
            return "bool"
        if not pd.api.types.is_numeric_dtype(series):
            return "text"
    return "num"


def offered_fields(pool: pd.DataFrame) -> list[str]:
    """Columns a rule can use, grouped and in a stable order."""
    fields = []
    for column in pool.columns:
        info = FEATURE_INFO.get(column) or EXTRA_INFO.get(column)
        kind = info[1] if info else None
        if column in NEVER_OFFERED or kind in ("date", "score", "scorebar"):
            continue
        if kind == "text" or (kind is None and not pd.api.types.is_numeric_dtype(pool[column]) and not pd.api.types.is_bool_dtype(pool[column])):
            if column not in TEXT_FIELDS:
                continue
        fields.append(column)
    groups = [*GROUPS, OTHER]

    def order(field: str):
        group = GROUP_OF.get(field, OTHER)
        position = GROUPS[group].index(field) if group in GROUPS else 0
        return groups.index(group), position, label_of(field)

    return sorted(fields, key=order)


def unit_of(kind: str) -> str:
    return {"pct": " (%)", "usd": " (US$ M)"}.get(kind, "")


def option_label(field: str, series: pd.Series | None = None) -> str:
    """'Oscillators · RSI(14)', 'Quality & growth · ROE (%)': what the metric picker shows."""
    return f"{GROUP_OF.get(field, OTHER)} · {label_of(field)}{unit_of(kind_of(field, series))}"


def ops_for(kind: str, current: str | None = None) -> list[tuple[str, str]]:
    """The operators to offer for a column, plus ``current`` when a loaded rule already uses one that is not on the list."""
    ops = list(OPS_BY_KIND.get(kind, NUMERIC_OPS))  # a copy: this list is appended to below
    if current and current not in dict(ops):
        words = dict(EXTRA_NUMERIC_OPS if kind not in OPS_BY_KIND else OPS_BY_KIND["text"])
        ops.append((current, words.get(current, current)))
    return ops


# -- values: what the rule holds versus what is typed ------------------------------------------------------
def to_ui(value, kind: str):
    """Rule value -> the number shown (fractions as %, dollars as millions)."""
    if isinstance(value, (list, tuple)) and kind not in ("text",):
        return tuple(to_ui(v, kind) for v in value)
    if kind == "pct":
        return float(value) * 100
    if kind == "usd":
        return float(value) / 1e6
    if kind == "int":
        return int(round(float(value)))
    if kind == "num":
        return float(value)
    return value


def from_ui(value, kind: str):
    """The reverse of :func:`to_ui`."""
    if isinstance(value, (list, tuple)) and kind not in ("text",):
        return tuple(from_ui(v, kind) for v in value)
    if kind == "pct":
        return float(value) / 100
    if kind == "usd":
        return float(value) * 1e6
    if kind == "int":
        return int(round(float(value)))
    if kind == "num":
        return float(value)
    return value


def _numbers(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return pd.Series(dtype="float64")
    return pd.to_numeric(series, errors="coerce").dropna()


def _round_sig(value: float, digits: int = 2) -> float:
    return float(f"{value:.{digits}g}") if value else 0.0


def text_options(series: pd.Series) -> list[str]:
    counts = series.dropna().astype(str).value_counts()
    return list(counts.index[:MAX_TEXT_OPTIONS])


def default_value(kind: str, op: str, series: pd.Series | None):
    """A value in the *displayed* units that splits the pool sensibly: the median, or the middle half for 'between'."""
    if kind == "bool":
        return True
    if kind == "text":
        options = text_options(series) if series is not None else []
        if op in ("in", "not in"):
            return options[:1]
        return options[0] if options else ""
    nums = _numbers(series) if series is not None else pd.Series(dtype="float64")
    if nums.empty:
        return (0.0, 0.0) if op == "between" else (0 if kind == "int" else 0.0)
    scale = to_ui(1.0, "pct") if kind == "pct" else (1 / 1e6 if kind == "usd" else 1.0)
    lo, mid, hi = (_round_sig(float(nums.quantile(q)) * scale) for q in (0.25, 0.5, 0.75))
    if op == "between":
        pair = (lo, hi)
        return tuple(int(round(v)) for v in pair) if kind == "int" else pair
    return int(round(mid)) if kind == "int" else mid


def step_for(series: pd.Series | None, kind: str) -> float | int:
    """A number-input step of about 1% of the usual spread, as a power of ten (1 for percentages and counts)."""
    if kind == "int":
        return 1
    if kind == "pct":
        return 1.0
    if kind == "usd":
        return 10.0
    nums = _numbers(series) if series is not None else pd.Series(dtype="float64")
    spread = float(nums.quantile(0.9) - nums.quantile(0.1)) if len(nums) > 1 else 0.0
    if not spread > 0:
        return 1.0
    return float(min(max(10 ** math.floor(math.log10(spread / 100)), 0.01), 100.0))


def number_format(step: float | int, kind: str) -> str:
    if kind == "int":
        return "%d"
    return "%.1f" if kind == "pct" else ("%.0f" if step >= 1 else "%.2f")


def has_data(series: pd.Series) -> bool:
    return bool(series.notna().any())


def _short(field: str, kind: str, value: float) -> str:
    """A threshold-sized figure: three significant digits (36.6, not 36.5828), % and $ as everywhere else."""
    if kind in ("pct", "usd"):
        return format_feature(field, value, compact=True)
    return f"{value:.3g}"


def typical(series: pd.Series, field: str) -> str:
    """'typical 38 to 60 (median 48)': where most of the pool sits, to pick a threshold against."""
    kind = kind_of(field, series)
    if kind in ("bool", "text"):
        return ""
    nums = _numbers(series)
    if len(nums) < 10:
        return ""
    lo, mid, hi = (_short(field, kind, float(nums.quantile(q))) for q in (0.25, 0.5, 0.75))
    return f"typical {lo} to {hi} (median {mid})"
