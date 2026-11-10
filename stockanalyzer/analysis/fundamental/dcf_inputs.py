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
from stockanalyzer.analysis.fundamental.bank import FALLBACK_ASSUMPTION, MARKET_ASSUMPTIONS
from stockanalyzer.analysis.fundamental.statements import line
from stockanalyzer.domain.financials import FinancialStatements

GROWTH_RANGE = (0.0, 0.40)  # suggested initial growth is clipped to this band (it was 0-25%, which cut a company growing 40% in half)
HOLD_YEARS = ((0.30, 3), (0.15, 2))  # (growth at least, years at full speed): a fast grower is assumed to keep its pace a while
MARGIN_RANGE = (0.02, 0.40)  # suggested lasting free-cash-flow margin of a company that is not there yet
MARGIN_YEARS = 7  # years a margin takes to move from today's level to its target
SBC_NOTE_SHARE = 0.03  # stock pay above this share of revenue is worth a warning
WACC_RANGE = (0.05, 0.16)
TAX_RATES = {"USD": 0.21, "THB": 0.20}  # statutory corporate tax per price currency (editable starting points)
LOCAL_BETA_RANGE = (0.8, 1.6)  # outside the US a measured beta is not trusted below 0.8: Yahoo gives ADVANC 0.04 and AOT 0.10
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
    hold_years: int = 1  # years at the initial growth before it starts to fade
    raw_growth: float | None = None  # measured revenue growth before it was kept within GROWTH_RANGE

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
    currency: str | None = None,
) -> DCFSuggestion:
    """Pre-filled DCF inputs.

    ``currency`` is the price currency. ``None`` or ``"USD"`` keeps the US assumptions (4.3% risk-free, 5% premium, 21% tax,
    Yahoo's beta as it is). Any other currency uses that market's own risk-free rate, premium and tax rate and keeps beta
    within ``LOCAL_BETA_RANGE``, because a beta measured against a thin local index says little about risk.
    """
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
    if raw_growth > GROWTH_RANGE[1] + 0.005:
        notes.append(f"Revenue has been growing {raw_growth:.0%} a year; the suggestion is kept at {GROWTH_RANGE[1]:.0%} because growth that "
                     "fast rarely lasts. Raise it if you believe it will, and check how long it can hold under 'Years at full speed'.")

    mcap = market_cap or (price * shares if shares else 0)
    interest = line(stmts, "interest_expense").dropna()
    raw_beta = info.get("beta")
    beta = float(raw_beta or 1.0)
    market = {}
    if currency and currency != "USD":
        rf, erp = MARKET_ASSUMPTIONS.get(currency, FALLBACK_ASSUMPTION)
        beta = float(np.clip(beta, *LOCAL_BETA_RANGE))
        market = {"risk_free": rf, "equity_risk_premium": erp, "tax_rate": TAX_RATES.get(currency, TAX_RATES["USD"])}
        if raw_beta is None or abs(beta - float(raw_beta)) > 1e-9:
            shown = f"{float(raw_beta):.2f}" if raw_beta is not None else "missing"
            notes.append(f"Yahoo's beta ({shown}) was moved to {beta:.2f}: a beta measured on a small local market understates "
                         f"risk. The discount rate uses {currency} assumptions ({rf:.1%} risk-free, {erp:.0%} equity premium, "
                         f"{market['tax_rate']:.0%} tax) - change the rate below if you disagree.")
    detail = valuation.estimate_wacc(
        beta=beta,
        market_cap=float(mcap or 1.0),
        total_debt=float(info.get("totalDebt") or 0.0) * fx,
        interest_expense=float(interest.iloc[-1]) * fx if len(interest) else 0.0,
        **market,
    )
    wacc = float(np.clip(round(detail["wacc"], 3), *WACC_RANGE))
    detail = {**detail, "beta": beta, "raw_beta": None if raw_beta is None else float(raw_beta),
              "risk_free": market.get("risk_free", 0.043), "equity_risk_premium": market.get("equity_risk_premium", 0.05),
              "tax_rate": market.get("tax_rate", 0.21)}

    return DCFSuggestion(price=float(price), shares=float(shares) if shares else None, net_debt=net_debt,
                         base_options=options, default_base=default, growth=growth, wacc=wacc,
                         wacc_detail=detail, terminal_growth=terminal_growth, years=years, notes=notes,
                         hold_years=suggest_hold_years(growth, years), raw_growth=None if growth_hist.empty else raw_growth)


def suggest_hold_years(growth: float, years: int = 10) -> int:
    """Years a company growing ``growth`` a year is assumed to keep that pace before it fades (1 = fade straight away).

    A rule of thumb to start from, not a forecast: 3 years from 30% growth, 2 from 15%, otherwise 1 - which is what
    the model did before this setting existed. Never more than half the projection.
    """
    for floor, hold in HOLD_YEARS:
        if growth >= floor:
            return min(hold, max(1, years // 2))
    return 1


# -- revenue x margin DCF ---------------------------------------------------------------------
@dataclass
class RevenueSuggestion:
    base_options: list[BaseOption]  # revenue per year, price currency
    default_base: str | None
    current_margin: float  # free cash flow / revenue today (negative while the company burns cash)
    target_margin: float  # the margin it is assumed to settle at
    margin_years: int
    sbc_margin: float | None  # stock-based pay as a share of revenue (None when the statements do not show it)
    notes: list[str] = field(default_factory=list)

    @property
    def base_revenue(self) -> float:
        for o in self.base_options:
            if o.key == self.default_base:
                return o.value
        return 0.0


def suggest_revenue_inputs(
    stmts: FinancialStatements, info: dict, *, fx: float = 1.0, currency: str | None = None, years: int = 10,
) -> RevenueSuggestion:
    """Pre-filled inputs for the revenue x margin DCF.

    The current margin is trailing free cash flow over trailing revenue. The target margin is the best of that and the
    operating margin after tax (what the business would keep once growth spending settles); a company with no profit yet
    gets a quarter of its gross margin, which is a placeholder, and the note says so.
    """
    notes: list[str] = []
    revenue = (line(stmts, "revenue").dropna() * fx)
    options: list[BaseOption] = []
    ttm = info.get("totalRevenue")
    if ttm:
        options.append(BaseOption("ttm", "Trailing 12 months", float(ttm) * fx))
    if len(revenue):
        options.append(BaseOption("last_fy", f"Last reported {'quarter' if stmts.quarterly else 'year'} "
                                              f"({revenue.index[-1]:%Y-%m})", float(revenue.iloc[-1])))
    default = "ttm" if ttm else "last_fy" if len(revenue) else None

    r = ratios.compute_ratios(stmts)
    fcf_ttm = info.get("freeCashflow")
    if ttm and fcf_ttm is not None:
        current = float(fcf_ttm) / float(ttm)
    else:
        margins = r["fcf_margin"].dropna()
        current = float(margins.iloc[-1]) if len(margins) else 0.0
        if not len(margins):
            notes.append("No free cash flow figure is available - the current margin is assumed to be 0%.")

    tax = TAX_RATES.get(currency or "USD", TAX_RATES["USD"])
    opm, gm = info.get("operatingMargins"), info.get("grossMargins")
    candidates = [current] + ([float(opm) * (1 - tax)] if isinstance(opm, (int, float)) and opm == opm and opm > 0 else [])
    best = max(candidates)
    if best > 0:
        target = max(float(np.clip(best, *MARGIN_RANGE)), current)
    elif isinstance(gm, (int, float)) and gm == gm and gm > 0:
        target = float(np.clip(gm * 0.25, 0.05, 0.25))
        notes.append(f"The company is not profitable yet, so the lasting margin is a placeholder: a quarter of its {gm:.0%} gross margin "
                     f"({target:.0%}). What a mature business of this kind keeps is the single most important number here - set it yourself.")
    else:
        target = 0.10
        notes.append("The company is not profitable and has no gross margin figure, so the lasting margin is a neutral 10% placeholder.")

    sbc = r["sbc_to_revenue"].dropna()
    sbc_margin = float(sbc.iloc[-1]) if len(sbc) and sbc.iloc[-1] > 0 else None
    if sbc_margin is not None and sbc_margin >= SBC_NOTE_SHARE:
        notes.append(f"Stock-based pay is {sbc_margin:.1%} of revenue and is added back in free cash flow, so the margins above overstate "
                     "what shareholders keep. Tick 'Treat stock pay as a cost' to take it out.")
    return RevenueSuggestion(base_options=options, default_base=default, current_margin=round(current, 3),
                             target_margin=round(target, 2), margin_years=min(MARGIN_YEARS, years), sbc_margin=sbc_margin, notes=notes)
