"""Valuation: DCF (free cash flow or revenue x margin, with sensitivity and reverse-DCF), WACC estimate, PEG and info snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd


# -- DCF -------------------------------------------------------------------------------------
@dataclass
class DCFResult:
    enterprise_value: float
    equity_value: float
    intrinsic_per_share: float
    terminal_value: float
    pv_terminal: float
    pv_cash_flows: float
    projection: pd.DataFrame  # year, growth, (revenue, margin,) fcf, discount_factor, pv
    upside: float | None = None  # vs the supplied price

    @property
    def terminal_share(self) -> float:
        """Share of enterprise value that comes from the terminal value (>75% = fragile)."""
        return self.pv_terminal / self.enterprise_value if self.enterprise_value else float("nan")


def growth_path(initial: float, terminal: float, years: int, fade: bool = True, hold: int = 1) -> list[float]:
    """Growth for each projection year: constant, or ``initial`` for ``hold`` years and then fading linearly to ``terminal``.

    ``hold=1`` (the default) fades from the second year on: the first year is ``initial`` and the last is ``terminal``.
    A longer ``hold`` models a company that keeps growing at full speed for a while before it matures. A ``hold`` that
    covers every year leaves growth at ``initial`` throughout.
    """
    hold = max(1, int(hold))
    if not fade or years == 1 or hold >= years:
        return [initial] * years
    return [initial if t < hold else initial + (terminal - initial) * (t - hold + 1) / (years - hold) for t in range(years)]


def _present_value(
    fcf: np.ndarray, rates: Sequence[float], *, terminal_growth: float, wacc: float, net_debt: float, shares: float,
    price: float | None, extra: dict[str, Sequence[float]] | None = None,
) -> DCFResult:
    """Discount a free-cash-flow path (year-end convention), add the Gordon terminal value and bridge to a per-share value."""
    years = len(fcf)
    t = np.arange(1, years + 1)
    factor = 1 / (1 + wacc) ** t
    pv = fcf * factor
    terminal = fcf[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal * factor[-1]
    ev = float(pv.sum() + pv_terminal)
    equity = ev - net_debt
    per_share = equity / shares
    columns = {"year": t, "growth": list(rates), **(extra or {}), "fcf": fcf, "discount_factor": factor, "pv": pv}
    return DCFResult(
        enterprise_value=ev, equity_value=float(equity), intrinsic_per_share=float(per_share),
        terminal_value=float(terminal), pv_terminal=float(pv_terminal), pv_cash_flows=float(pv.sum()),
        projection=pd.DataFrame(columns),
        upside=None if price is None or price <= 0 else float(per_share / price - 1),
    )


def _check(wacc: float, terminal_growth: float, shares: float) -> None:
    if wacc <= terminal_growth:
        raise ValueError("WACC must exceed terminal growth or the terminal value is undefined")
    if shares <= 0:
        raise ValueError("shares must be positive")


def _rates(growth: float | Sequence[float], years: int, terminal_growth: float, fade: bool, hold: int) -> list[float]:
    rates = list(growth) if isinstance(growth, (list, tuple, np.ndarray, pd.Series)) \
        else growth_path(float(growth), terminal_growth, years, fade, hold)
    if len(rates) != years:
        raise ValueError(f"need {years} growth rates, got {len(rates)}")
    return rates


def dcf(
    base_fcf: float,
    growth: float | Sequence[float],
    *,
    years: int = 10,
    terminal_growth: float = 0.025,
    wacc: float = 0.09,
    net_debt: float = 0.0,
    shares: float = 1.0,
    price: float | None = None,
    fade: bool = True,
    hold: int = 1,
) -> DCFResult:
    """Two-stage DCF. ``growth`` is an initial rate (held ``hold`` years, then fading to terminal) or one rate per year.

    Cash flows are discounted at year end. ``base_fcf`` is the last twelve months' free cash flow.
    """
    _check(wacc, terminal_growth, shares)
    rates = _rates(growth, years, terminal_growth, fade, hold)
    fcf = base_fcf * np.cumprod(1 + np.asarray(rates, dtype=float))
    return _present_value(fcf, rates, terminal_growth=terminal_growth, wacc=wacc, net_debt=net_debt, shares=shares, price=price)


def dcf_sensitivity(
    base_fcf: float, growth: float, *, waccs: Sequence[float], terminal_growths: Sequence[float],
    **kwargs,
) -> pd.DataFrame:
    """Intrinsic value per share over a WACC x terminal-growth grid (invalid pairs are NaN)."""
    grid = pd.DataFrame(index=pd.Index(waccs, name="wacc"), columns=pd.Index(terminal_growths, name="terminal_g"),
                        dtype="float64")
    for w in waccs:
        for g in terminal_growths:
            if w > g:
                grid.loc[w, g] = dcf(base_fcf, growth, wacc=w, terminal_growth=g, **kwargs).intrinsic_per_share
    return grid


# Bear / base / bull = shifts (absolute, in rate points) to initial growth, WACC and terminal growth.
SCENARIO_SHIFTS: dict[str, tuple[float, float, float]] = {
    "Bear": (-0.05, +0.01, -0.005),
    "Base": (0.0, 0.0, 0.0),
    "Bull": (+0.05, -0.01, +0.005),
}


def dcf_scenarios(
    base_fcf: float,
    growth: float,
    *,
    years: int = 10,
    terminal_growth: float = 0.025,
    wacc: float = 0.09,
    net_debt: float = 0.0,
    shares: float = 1.0,
    price: float | None = None,
    shifts: dict[str, tuple[float, float, float]] | None = None,
    hold: int = 1,
) -> pd.DataFrame:
    """Value per share under bear / base / bull variants of the same model.

    The base row is exactly ``dcf(...)`` with the given inputs. A shifted terminal growth is kept
    at least 0.5 point below its (shifted) WACC so the terminal value stays defined; the rates
    actually used are returned so nothing is adjusted silently.
    """
    rows = []
    for name, (d_growth, d_wacc, d_term) in (shifts or SCENARIO_SHIFTS).items():
        g, w, t = growth + d_growth, wacc + d_wacc, terminal_growth + d_term
        if (d_growth, d_wacc, d_term) != (0.0, 0.0, 0.0):
            t = min(max(t, 0.0), w - 0.005)
        res = dcf(base_fcf, g, years=years, terminal_growth=t, wacc=w, net_debt=net_debt, shares=shares, price=price, hold=hold)
        rows.append({"scenario": name, "growth": g, "wacc": w, "terminal_growth": t,
                     "value_per_share": res.intrinsic_per_share,
                     "upside": res.upside if res.upside is not None else float("nan")})
    return pd.DataFrame(rows)


def _bisect(gap: Callable[[float], float], lo: float, hi: float) -> float:
    """Root of ``gap`` between ``lo`` and ``hi`` (NaN when the two ends do not straddle zero)."""
    f_lo, f_hi = gap(lo), gap(hi)
    if f_lo * f_hi > 0:
        return float("nan")  # price is outside what any value in the bracket can explain
    for _ in range(80):
        mid = (lo + hi) / 2
        f_mid = gap(mid)
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def implied_growth(
    price: float, base_fcf: float, *, years: int = 10, terminal_growth: float = 0.025,
    wacc: float = 0.09, net_debt: float = 0.0, shares: float = 1.0, bounds=(-0.5, 1.0), hold: int = 1,
) -> float:
    """Reverse DCF: the initial growth rate (held ``hold`` years, then fading to terminal) the market price implies."""
    def gap(g: float) -> float:
        return dcf(base_fcf, g, years=years, terminal_growth=terminal_growth, wacc=wacc,
                   net_debt=net_debt, shares=shares, hold=hold).intrinsic_per_share - price

    return _bisect(gap, *bounds)


# -- revenue x margin DCF ---------------------------------------------------------------------
# For companies whose free cash flow is small or negative because they are still investing in growth, a DCF that starts
# from today's cash flow says nothing. This model forecasts revenue and the margin it will earn, and values the product.
def margin_path(current: float, target: float, years: int, ramp: int) -> list[float]:
    """Free-cash-flow margin per year: ``current`` moving in equal steps to ``target`` by year ``ramp``, then flat.

    Year 1 is already one step along (``ramp=1`` reaches the target at once).
    """
    ramp = min(max(1, int(ramp)), years)
    return [current + (target - current) * min(t + 1, ramp) / ramp for t in range(years)]


def revenue_dcf(
    base_revenue: float,
    growth: float | Sequence[float],
    *,
    current_margin: float,
    target_margin: float,
    margin_years: int = 7,
    years: int = 10,
    terminal_growth: float = 0.025,
    wacc: float = 0.09,
    net_debt: float = 0.0,
    shares: float = 1.0,
    price: float | None = None,
    hold: int = 1,
) -> DCFResult:
    """DCF of revenue x free-cash-flow margin.

    Revenue grows like ``dcf``'s cash flow does (``growth`` held ``hold`` years, then fading to ``terminal_growth``), and
    the margin moves from ``current_margin`` to ``target_margin`` over ``margin_years``. The projection also reports
    ``revenue`` and ``margin`` for each year. A target margin at or below zero gives a terminal value that is not positive.
    """
    _check(wacc, terminal_growth, shares)
    if base_revenue <= 0:
        raise ValueError("starting revenue must be positive")
    rates = _rates(growth, years, terminal_growth, True, hold)
    revenue = base_revenue * np.cumprod(1 + np.asarray(rates, dtype=float))
    margin = np.asarray(margin_path(current_margin, target_margin, years, margin_years), dtype=float)
    return _present_value(revenue * margin, rates, terminal_growth=terminal_growth, wacc=wacc, net_debt=net_debt,
                          shares=shares, price=price, extra={"revenue": revenue, "margin": margin})


def revenue_dcf_scenarios(
    base_revenue: float, growth: float, *, current_margin: float, target_margin: float, margin_years: int = 7,
    years: int = 10, terminal_growth: float = 0.025, wacc: float = 0.09, net_debt: float = 0.0, shares: float = 1.0,
    price: float | None = None, hold: int = 1,
) -> pd.DataFrame:
    """Bear / base / bull value per share of the revenue model.

    Bear and Bull move year-1 growth by 5 points or a fifth of it, whichever is larger (five points is nothing for a company
    growing 60%), the target margin by 3 points, the discount rate by 1 point and long-run growth by half a point. The base
    row is exactly ``revenue_dcf(...)``; a shifted terminal growth stays 0.5 point below its WACC. The rates used are returned.
    """
    spread = max(0.05, 0.2 * abs(growth))
    steps = {"Bear": -1, "Base": 0, "Bull": 1}
    rows = []
    for name, k in steps.items():
        g, m, w, t = growth + k * spread, target_margin + k * 0.03, wacc - k * 0.01, terminal_growth + k * 0.005
        if k:
            m, t = max(m, 0.0), min(max(t, 0.0), w - 0.005)
        res = revenue_dcf(base_revenue, g, current_margin=current_margin, target_margin=m, margin_years=margin_years, years=years,
                          terminal_growth=t, wacc=w, net_debt=net_debt, shares=shares, price=price, hold=hold)
        rows.append({"scenario": name, "growth": g, "target_margin": m, "wacc": w, "terminal_growth": t,
                     "value_per_share": res.intrinsic_per_share,
                     "upside": res.upside if res.upside is not None else float("nan")})
    return pd.DataFrame(rows)


def revenue_sensitivity(base_revenue: float, *, growths: Sequence[float], margins: Sequence[float], **kwargs) -> pd.DataFrame:
    """Value per share over a target-margin (rows) x year-1-growth (columns) grid; ``kwargs`` go to ``revenue_dcf``."""
    grid = pd.DataFrame(index=pd.Index(margins, name="target_margin"), columns=pd.Index(growths, name="growth"), dtype="float64")
    for m in margins:
        for g in growths:
            grid.loc[m, g] = revenue_dcf(base_revenue, g, target_margin=m, **kwargs).intrinsic_per_share
    return grid


def implied_revenue_growth(
    price: float, base_revenue: float, *, current_margin: float, target_margin: float, margin_years: int = 7, years: int = 10,
    terminal_growth: float = 0.025, wacc: float = 0.09, net_debt: float = 0.0, shares: float = 1.0, hold: int = 1,
    bounds=(-0.5, 1.5),
) -> float:
    """Reverse DCF of the revenue model: the year-1 revenue growth the price needs, given the margin you expect."""
    def gap(g: float) -> float:
        return revenue_dcf(base_revenue, g, current_margin=current_margin, target_margin=target_margin, margin_years=margin_years,
                           years=years, terminal_growth=terminal_growth, wacc=wacc, net_debt=net_debt, shares=shares,
                           hold=hold).intrinsic_per_share - price

    return _bisect(gap, *bounds)


def implied_margin(
    price: float, base_revenue: float, growth: float, *, current_margin: float, margin_years: int = 7, years: int = 10,
    terminal_growth: float = 0.025, wacc: float = 0.09, net_debt: float = 0.0, shares: float = 1.0, hold: int = 1,
    bounds=(0.0, 0.9),
) -> float:
    """Reverse DCF the other way round: the lasting free-cash-flow margin the price needs, given the growth you expect."""
    def gap(m: float) -> float:
        return revenue_dcf(base_revenue, growth, current_margin=current_margin, target_margin=m, margin_years=margin_years,
                           years=years, terminal_growth=terminal_growth, wacc=wacc, net_debt=net_debt, shares=shares,
                           hold=hold).intrinsic_per_share - price

    return _bisect(gap, *bounds)


def estimate_wacc(
    *, beta: float, market_cap: float, total_debt: float = 0.0, interest_expense: float = 0.0,
    tax_rate: float = 0.21, risk_free: float = 0.043, equity_risk_premium: float = 0.05,
) -> dict[str, float]:
    """CAPM cost of equity blended with after-tax cost of debt (all inputs are assumptions)."""
    cost_equity = risk_free + beta * equity_risk_premium
    cost_debt = abs(interest_expense) / total_debt if total_debt > 0 else 0.0
    cost_debt = min(cost_debt, 0.15)
    total = market_cap + total_debt
    w_e = market_cap / total if total else 1.0
    wacc = w_e * cost_equity + (1 - w_e) * cost_debt * (1 - tax_rate)
    return {"wacc": wacc, "cost_equity": cost_equity, "cost_debt_after_tax": cost_debt * (1 - tax_rate),
            "equity_weight": w_e}


# -- PEG & info snapshot -----------------------------------------------------------------------
def peg_ratio(info: dict) -> float | None:
    """PEG from Yahoo data: published trailing/forward PEG first, else P/E / earnings growth %.

    Replaces the old HTML scraper (slow and brittle). Returns ``None`` when growth is not
    positive - a PEG with negative growth is meaningless.
    """
    for key in ("trailingPegRatio", "pegRatio"):
        v = info.get(key)
        if isinstance(v, (int, float)) and v == v and v > 0:
            return float(v)
    pe, growth = info.get("trailingPE"), info.get("earningsGrowth")
    if isinstance(pe, (int, float)) and isinstance(growth, (int, float)) and pe > 0 and growth > 0:
        return float(pe / (growth * 100))
    return None


_PLAIN = {
    "market_cap": "marketCap", "pe": "trailingPE", "forward_pe": "forwardPE", "pb": "priceToBook",
    "ps": "priceToSalesTrailing12Months", "ev_ebitda": "enterpriseToEbitda", "roe": "returnOnEquity",
    "roa": "returnOnAssets", "profit_margin": "profitMargins", "operating_margin": "operatingMargins",
    "gross_margin": "grossMargins", "revenue_growth": "revenueGrowth", "earnings_growth": "earningsGrowth",
    "current_ratio": "currentRatio", "beta": "beta", "fcf": "freeCashflow", "enterprise_value": "enterpriseValue",
    "short_pct_float": "shortPercentOfFloat", "sector": "sector", "industry": "industry",
    "name": "longName", "currency": "currency",
    "ev_sales": "enterpriseToRevenue", "revenue_ttm": "totalRevenue", "forward_eps": "forwardEps",
    "target_mean": "targetMeanPrice", "analyst_count": "numberOfAnalystOpinions", "recommendation_mean": "recommendationMean",
}
# columns computed from the plain ones; every snapshot carries all of them (None when they cannot be worked out)
DERIVED = ("debt_to_equity", "peg", "fcf_yield", "fcf_margin", "rule_of_40", "ev_sales_growth", "target_upside")


def snapshot_from_info(info: dict) -> dict:
    """Flat, unit-normalised fundamentals row for screening (ratios as fractions, not %)."""
    out: dict = {}
    for key, src in _PLAIN.items():
        v = info.get(src)
        out[key] = v if isinstance(v, str) or (isinstance(v, (int, float)) and v == v) else None
    de = info.get("debtToEquity")  # Yahoo reports percent (78.4 = 0.784x)
    out["debt_to_equity"] = de / 100 if isinstance(de, (int, float)) else None
    dy = info.get("trailingAnnualDividendYield")  # fraction; "dividendYield" is in percent
    out["dividend_yield"] = dy if isinstance(dy, (int, float)) else None
    out["peg"] = peg_ratio(info)
    cap, fcf = out.get("market_cap"), out.get("fcf")
    out["fcf_yield"] = fcf / cap if isinstance(cap, (int, float)) and cap and isinstance(fcf, (int, float)) else None
    out.update(growth_columns(out, info.get("currentPrice") or info.get("regularMarketPrice")))
    return out


def growth_columns(snap: dict, price) -> dict:
    """``fcf_margin``, ``rule_of_40``, ``ev_sales_growth`` and ``target_upside`` of a snapshot (None where an input is missing).

    Rule of 40 is revenue growth plus free-cash-flow margin, both as fractions: a company growing 30% with a 10% margin
    scores 0.40. ``ev_sales_growth`` is EV/Sales per point of revenue growth (only for positive growth and sales multiple).
    """
    def num(key: str):
        v = snap.get(key)
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v else None

    revenue, fcf, growth, ev_sales = num("revenue_ttm"), num("fcf"), num("revenue_growth"), num("ev_sales")
    margin = fcf / revenue if fcf is not None and revenue else None
    target = num("target_mean")
    return {
        "fcf_margin": margin,
        "rule_of_40": growth + margin if growth is not None and margin is not None else None,
        "ev_sales_growth": ev_sales / (growth * 100) if ev_sales is not None and ev_sales > 0 and growth is not None and growth > 0 else None,
        "target_upside": target / price - 1 if target and isinstance(price, (int, float)) and price > 0 else None,
    }


def snapshot_from_quote(quote: dict) -> dict:
    """The few fundamentals a market-wide quote carries, in the same shape as ``snapshot_from_info``.

    Marked ``_light``: ROE, margins, free cash flow and debt are missing until the full snapshot is fetched.
    """
    def num(key: str):
        v = quote.get(key)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v else None

    out = {key: None for key in (*_PLAIN, *DERIVED)}
    out.update(market_cap=num("marketCap"), pe=num("trailingPE"), forward_pe=num("forwardPE"), pb=num("priceToBook"),
               dividend_yield=num("trailingAnnualDividendYield"), currency=quote.get("currency"),
               name=quote.get("longName") or quote.get("shortName"))
    out["eps"] = num("epsTrailingTwelveMonths")
    out["_light"] = True
    return out
