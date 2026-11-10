"""Declarative screening rules and ready-made presets.

A :class:`Rule` is ``field op value`` over the screen table (features, optionally joined
with fundamentals). Missing values never match, and a rule on a column that does not exist
matches nothing - so a preset that needs fundamentals returns an empty list instead of
silently ignoring that condition.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

_SCALAR_OPS: dict[str, Callable] = {
    ">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le,
    "==": operator.eq, "!=": operator.ne,
}
OPERATORS = [*_SCALAR_OPS, "between", "in", "not in"]


@dataclass(frozen=True)
class Rule:
    field: str
    op: str
    value: Any

    def __post_init__(self):
        if self.op not in OPERATORS:
            raise ValueError(f"unknown operator {self.op!r}; use one of {OPERATORS}")
        if self.op == "between" and (not isinstance(self.value, (tuple, list)) or len(self.value) != 2):
            raise ValueError("'between' needs a (low, high) pair")

    def mask(self, df: pd.DataFrame) -> pd.Series:
        if self.field not in df.columns:
            return pd.Series(False, index=df.index)
        col = df[self.field]
        if self.op in _SCALAR_OPS:
            out = _SCALAR_OPS[self.op](col, self.value)
        elif self.op == "between":
            out = col.between(self.value[0], self.value[1])
        elif self.op == "in":
            out = col.isin(list(self.value))
        else:
            out = ~col.isin(list(self.value))
        return out.fillna(False).astype(bool) & col.notna()

    def __str__(self) -> str:
        return f"{self.field} {self.op} {self.value}"


@dataclass
class Screen:
    name: str
    rules: list[Rule] = field(default_factory=list)
    sort_by: str | None = None
    ascending: bool = False
    limit: int | None = None
    description: str = ""

    def mask(self, df: pd.DataFrame) -> pd.Series:
        m = pd.Series(True, index=df.index)
        for rule in self.rules:
            m &= rule.mask(df)
        return m

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[self.mask(df)]
        if self.sort_by and self.sort_by in out.columns:
            out = out.sort_values(self.sort_by, ascending=self.ascending, na_position="last")
        return out.head(self.limit) if self.limit else out

    def explain(self, df: pd.DataFrame) -> pd.DataFrame:
        """How many symbols each rule alone keeps, and the cumulative funnel."""
        rows, running = [], pd.Series(True, index=df.index)
        for rule in self.rules:
            m = rule.mask(df)
            running &= m
            rows.append({"rule": str(rule), "passes": int(m.sum()), "cumulative": int(running.sum())})
        return pd.DataFrame(rows, columns=["rule", "passes", "cumulative"])


def explain_row(rules: list[Rule], row: pd.DataFrame) -> pd.DataFrame:
    """For one stock (a one-row frame), each rule with the stock's value for it and whether it passes."""
    if len(row) != 1:
        raise ValueError("explain_row takes a frame with exactly one row")
    out = []
    for rule in rules:
        value = row[rule.field].iloc[0] if rule.field in row.columns else float("nan")  # a missing figure is empty, and never passes
        out.append({"rule": rule, "value": value, "passed": bool(rule.mask(row).iloc[0])})
    return pd.DataFrame(out, columns=["rule", "value", "passed"])


# No preset carries its own liquidity floor: how much trading is "enough" depends on the market (US$5M a day is the whole
# top of the Thai market), so the screener page and the CLI apply one floor per market before any preset runs.

PRESETS: dict[str, Screen] = {
    "Oversold in an uptrend": Screen(
        "Oversold in an uptrend",
        [Rule("rsi14", "<", 35), Rule("above_sma200", "==", True), Rule("trend_score", ">=", 1)],
        sort_by="rsi14", ascending=True,
        description="Pullbacks inside a longer-term uptrend: RSI < 35 while above the 200-day average.",
    ),
    "Breakout near 52-week high": Screen(
        "Breakout near 52-week high",
        [Rule("pct_from_52w_high", ">=", -0.03), Rule("volume_ratio", ">", 1.5), Rule("adx14", ">", 20)],
        sort_by="volume_ratio",
        description="Within 3% of the 52-week high on 1.5x volume with a trending ADX.",
    ),
    "Golden cross + momentum": Screen(
        "Golden cross + momentum",
        [Rule("golden_cross", "==", True), Rule("above_sma50", "==", True), Rule("ret_63d", ">", 0.10)],
        sort_by="ret_63d",
        description="50-day above 200-day average, price above the 50-day, 3-month return above 10%.",
    ),
    "Near support": Screen(
        "Near support",
        [Rule("support_dist_atr", "between", (0, 1.0)), Rule("rsi14", "<", 50)],
        sort_by="support_dist_atr", ascending=True,
        description="Price within 1 ATR above a confirmed support zone.",
    ),
    "Strong relative strength": Screen(
        "Strong relative strength",
        [Rule("rs_126", ">", 0.10), Rule("above_sma200", "==", True), Rule("hvol_60", "<", 0.60)],
        sort_by="rs_126",
        description="Beating the benchmark by 10+ points over 6 months, above the 200-day, not wildly volatile.",
    ),
    "Value + positive momentum": Screen(
        "Value + positive momentum",
        [Rule("pe", "between", (0, 15)), Rule("fcf_yield", ">", 0.05), Rule("ret_126d", ">", 0)],
        sort_by="fcf_yield",
        description="Needs fundamentals: P/E under 15, FCF yield over 5%, rising over 6 months.",
    ),
    "Cheap with a dividend": Screen(
        "Cheap with a dividend",
        [Rule("pe", "between", (0, 15)), Rule("pb", "<", 2.0), Rule("dividend_yield", ">", 0.03)],
        sort_by="dividend_yield",
        description="Needs P/E, P/B and yield only - the market-wide snapshot is enough: P/E under 15, P/B under 2, yield over 3%.",
    ),
    "Quality compounders": Screen(
        "Quality compounders",
        [Rule("roe", ">", 0.15), Rule("debt_to_equity", "<", 1.0), Rule("operating_margin", ">", 0.15),
         Rule("above_sma200", "==", True)],
        sort_by="roe",
        description="Needs fundamentals: ROE > 15%, D/E < 1, operating margin > 15%, long-term uptrend.",
    ),
    "Growth that pays: Rule of 40": Screen(
        "Growth that pays: Rule of 40",
        [Rule("revenue_growth", ">", 0.15), Rule("rule_of_40", ">=", 0.40), Rule("gross_margin", ">", 0.40),
         Rule("above_sma200", "==", True)],
        sort_by="rule_of_40",
        description="Needs fundamentals: revenue growing over 15% while growth plus free-cash-flow margin is at least 40% "
                    "(the usual bar for a healthy grower), gross margin above 40%, in a long-term uptrend.",
    ),
    "Growth at a reasonable price": Screen(
        "Growth at a reasonable price",
        [Rule("revenue_growth", ">", 0.15), Rule("ev_sales_growth", "between", (0, 0.5)), Rule("rule_of_40", ">=", 0.25)],
        sort_by="ev_sales_growth", ascending=True,
        description="Needs fundamentals: revenue growing over 15% but EV/Sales under half a point per point of growth "
                    "(a 20x sales multiple needs 40%+ growth) and a Rule of 40 of at least 25%. A rule of thumb, not a valuation.",
    ),
}
