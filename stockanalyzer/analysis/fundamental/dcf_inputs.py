"""Suggested DCF inputs from statements + Yahoo info (the page's pre-filled values).

Kept out of the UI so the defaults are testable. Every amount is returned in the *price*
currency: when a company reports in another currency (ADRs such as TSM report in TWD while
the ADR trades in USD) pass ``fx`` = price-currency units per 1 reporting-currency unit.
Share counts and market cap come from Yahoo already in price-currency / ADR-equivalent terms
and are not converted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stockanalyzer.analysis.fundamental import ratios, valuation
from stockanalyzer.analysis.fundamental.statements import line
from stockanalyzer.domain.financials import FinancialStatements

GROWTH_RANGE = (0.0, 0.25)  # suggested initial growth is clipped to this band
WACC_RANGE = (0.05, 0.16)
BASE_DIVERGENCE = 2.0  # warn when trailing FCF is more than this many times above/below its multi-year average


def _compact(value: float) -> str:
    return f"{value / 1e9:,.1f}B" if abs(value) >= 1e9 else f"{value / 1e6:,.1f}M" if abs(value) >= 1e6 else f"{value:,.0f}"
FALLBACK_GROWTH = 0.05  # used when there is not enough history to measure revenue growth


@dataclass(frozen=True)
class BaseOption:
    key: str  # "ttm" | "last_fy" | "avg3"
    label: str
    value: float


@dataclass
class DCFSuggestion:
    price: float
    shares: float | None
    net_debt: float
    base_options: list[BaseOption]
    default_base: str | None  # key of the option used by default (None = nothing usable)
    growth: float
    wacc: float
    wacc_detail: dict
    terminal_growth: float
    years: int
    notes: list[str] = field(default_factory=list)

    @property
    def base_fcf(self) -> float:
        """Free cash flow the page starts from (0.0 when no option is available)."""
        for o in self.base_options:
            if o.key == self.default_base:
                return o.value
        return 0.0


def suggest_dcf_inputs(
    stmts: FinancialStatements,
    info: dict,
    price: float,
    *,
    market_cap: float | None = None,
    fx: float = 1.0,
    terminal_growth: float = 0.025,
    years: int = 10,
) -> DCFSuggestion:
    notes: list[str] = []
    fcf = (line(stmts, "fcf").dropna() * fx)

    options: list[BaseOption] = []
    ttm = info.get("freeCashflow")
    if ttm:
        options.append(BaseOption("ttm", "Trailing 12 months", float(ttm) * fx))
    if len(fcf):
        options.append(BaseOption("last_fy", f"Last reported {'quarter' if stmts.quarterly else 'year'} "
                                              f"({fcf.index[-1]:%Y-%m})", float(fcf.iloc[-1])))
    if len(fcf) >= 2:
        n = min(3, len(fcf))
        options.append(BaseOption("avg3", f"{n}-period average", float(fcf.tail(n).mean())))
    default = "ttm" if ttm else "last_fy" if len(fcf) else None
    by_key = {o.key: o for o in options}
    if "ttm" in by_key and "avg3" in by_key and by_key["avg3"].value > 0:
        ratio = by_key["ttm"].value / by_key["avg3"].value
        if ratio < 1 / BASE_DIVERGENCE or ratio > BASE_DIVERGENCE:
            notes.append(
                f"Trailing-12-month free cash flow ({_compact(by_key['ttm'].value)}) is far "
                f"{'below' if ratio < 1 else 'above'} the {by_key['avg3'].label.split('-')[0]}-period average "
                f"({_compact(by_key['avg3'].value)}) - often a temporary swing in spending or working capital. "
                "Compare the other starting points before trusting the headline value.")

    shares = info.get("sharesOutstanding")
    if not shares:
        s = line(stmts, "shares").dropna()
        shares = float(s.iloc[-1]) if len(s) else None
        if shares:
            notes.append("Share count taken from the balance sheet (ordinary shares) - check it against what the price quotes.")
    net_debt = (float(info.get("totalDebt") or 0.0) - float(info.get("totalCash") or 0.0)) * fx

    growth_hist = ratios.compute_ratios(stmts)["revenue_growth"].tail(3).dropna()  # as the original page did
    if growth_hist.empty:
        raw_growth = FALLBACK_GROWTH
        notes.append("Not enough history to measure revenue growth - starting from a neutral 5%.")
    else:
        raw_growth = float(growth_hist.median())
    growth = round(float(np.clip(raw_growth, *GROWTH_RANGE)), 2)

    mcap = market_cap or (price * shares if shares else 0)
    interest = line(stmts, "interest_expense").dropna()
    detail = valuation.estimate_wacc(
        beta=float(info.get("beta") or 1.0),
        market_cap=float(mcap or 1.0),
        total_debt=float(info.get("totalDebt") or 0.0) * fx,
        interest_expense=float(interest.iloc[-1]) * fx if len(interest) else 0.0,
    )
    wacc = float(np.clip(round(detail["wacc"], 3), *WACC_RANGE))
    detail = {**detail, "beta": float(info.get("beta") or 1.0)}

    return DCFSuggestion(price=float(price), shares=float(shares) if shares else None, net_debt=net_debt,
                         base_options=options, default_base=default, growth=growth, wacc=wacc,
                         wacc_detail=detail, terminal_growth=terminal_growth, years=years, notes=notes)
