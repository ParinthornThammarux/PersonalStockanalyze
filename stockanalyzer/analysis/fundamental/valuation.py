"""Valuation: DCF (with sensitivity and reverse-DCF), WACC estimate, PEG and info snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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
    projection: pd.DataFrame  # year, growth, fcf, discount_factor, pv
    upside: float | None = None  # vs the supplied price

    @property
    def terminal_share(self) -> float:
        """Share of enterprise value that comes from the terminal value (>75% = fragile)."""
        return self.pv_terminal / self.enterprise_value if self.enterprise_value else float("nan")


def growth_path(initial: float, terminal: float, years: int, fade: bool = True) -> list[float]:
    """Growth for each projection year: constant, or fading linearly towards ``terminal``."""
    if not fade or years == 1:
        return [initial] * years
    return [initial + (terminal - initial) * t / (years - 1) for t in range(years)]


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
) -> DCFResult:
    """Two-stage DCF. ``growth`` is an initial rate (fading to terminal) or one rate per year.

    Cash flows are discounted at year end. ``base_fcf`` is the last twelve months' free cash flow.
    """
    if wacc <= terminal_growth:
        raise ValueError("WACC must exceed terminal growth or the terminal value is undefined")
    if shares <= 0:
        raise ValueError("shares must be positive")
    rates = list(growth) if isinstance(growth, (list, tuple, np.ndarray, pd.Series)) \
        else growth_path(float(growth), terminal_growth, years, fade)
    if len(rates) != years:
        raise ValueError(f"need {years} growth rates, got {len(rates)}")

    fcf = base_fcf * np.cumprod(1 + np.asarray(rates, dtype=float))
    t = np.arange(1, years + 1)
    factor = 1 / (1 + wacc) ** t
    pv = fcf * factor
    terminal = fcf[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal * factor[-1]
    ev = float(pv.sum() + pv_terminal)
    equity = ev - net_debt
    per_share = equity / shares
    return DCFResult(
        enterprise_value=ev, equity_value=float(equity), intrinsic_per_share=float(per_share),
        terminal_value=float(terminal), pv_terminal=float(pv_terminal), pv_cash_flows=float(pv.sum()),
        projection=pd.DataFrame({"year": t, "growth": rates, "fcf": fcf, "discount_factor": factor, "pv": pv}),
        upside=None if price is None or price <= 0 else float(per_share / price - 1),
    )


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
        res = dcf(base_fcf, g, years=years, terminal_growth=t, wacc=w, net_debt=net_debt, shares=shares, price=price)
        rows.append({"scenario": name, "growth": g, "wacc": w, "terminal_growth": t,
                     "value_per_share": res.intrinsic_per_share,
                     "upside": res.upside if res.upside is not None else float("nan")})
    return pd.DataFrame(rows)


def implied_growth(
    price: float, base_fcf: float, *, years: int = 10, terminal_growth: float = 0.025,
    wacc: float = 0.09, net_debt: float = 0.0, shares: float = 1.0, bounds=(-0.5, 1.0),
) -> float:
    """Reverse DCF: the initial growth rate (fading to terminal) the market price implies."""
    def gap(g: float) -> float:
        return dcf(base_fcf, g, years=years, terminal_growth=terminal_growth, wacc=wacc,
                   net_debt=net_debt, shares=shares).intrinsic_per_share - price

    lo, hi = bounds
    f_lo, f_hi = gap(lo), gap(hi)
    if f_lo * f_hi > 0:
        return float("nan")  # price is outside what any growth in the bracket can explain
    for _ in range(80):
        mid = (lo + hi) / 2
        f_mid = gap(mid)
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


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
}


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
    return out
