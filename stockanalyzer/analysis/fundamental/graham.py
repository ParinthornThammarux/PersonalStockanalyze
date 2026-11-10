"""Benjamin Graham's value rules: the Graham Number, his growth formula, net current asset value and the defensive-investor tests.

Graham priced a stock from what it earns and owns today, never from a long forecast:

* **Graham Number** - the most a defensive investor should pay: the price at which P/E is 15 and P/B is 1.5 together
  (15 x 1.5 = 22.5), so ``sqrt(22.5 x EPS x book value per share)``.
* **Growth formula** (*The Intelligent Investor*) - ``V = EPS x (8.5 + 2g) x 4.4 / Y``: 8.5 is the P/E of a company that does
  not grow, ``g`` the yearly growth expected over the next 7-10 years (in points), 4.4 the AAA corporate bond yield of 1962 and
  ``Y`` today's. The first version, without the bond-yield term, is ``V = EPS x (8.5 + 2g)``.
* **Net current asset value** (NCAV, "net-net") - current assets minus *all* liabilities, per share. Graham bought at two
  thirds of it or less.
* **Defensive-investor tests** - seven rules on size, balance sheet, earnings record, dividend record, growth and price.

Amounts are per share and in one currency; rates are fractions (0.05 = 5%).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockanalyzer.analysis.fundamental import ratios
from stockanalyzer.analysis.fundamental.bands import eps_by_period, stock_points
from stockanalyzer.analysis.fundamental.bank import FALLBACK_ASSUMPTION, MARKET_ASSUMPTIONS
from stockanalyzer.analysis.fundamental.statements import line
from stockanalyzer.domain.financials import FinancialStatements

NAN = float("nan")
NO_GROWTH_PE = 8.5  # P/E of a company with no growth
GROWTH_MULTIPLE = 2.0  # extra P/E per point of expected yearly growth
BASE_YIELD = 0.044  # AAA corporate bond yield of 1962, the level the formula is calibrated to
MAX_PE, MAX_PB, PE_PB_CAP = 15.0, 1.5, 22.5  # moderate P/E, moderate P/B, or their product
NET_NET_FRACTION = 2 / 3  # buy at two thirds of net current asset value or less
AAA_SPREAD = 0.008  # AAA corporate yield over the government 10-year rate: a rough starting point, not market data
GROWTH_CAP = 0.15  # suggested growth is held here: few companies keep growing faster for 7-10 years
MIN_CURRENT_RATIO = 2.0
STABLE_YEARS, DIVIDEND_YEARS = 10, 20  # years of profits / uninterrupted dividends Graham asked for
GROWTH_PERIOD = 7  # years between the 3-year averages at the start and end of a 10-year record
GROWTH_PER_YEAR = (4 / 3) ** (1 / GROWTH_PERIOD) - 1  # "at least one third" over that period, per year (~4.2%)
SIZE_FLOOR_USD = 700e6  # Graham's $100M of sales in the early 1970s, in today's money (about 7x)


def _pos(x) -> bool:
    return isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, bool) and x == x and x > 0


def _short(value: float, currency: str = "") -> str:
    v = abs(value)
    text = f"{value / 1e9:,.1f}B" if v >= 1e9 else f"{value / 1e6:,.1f}M" if v >= 1e6 else f"{value:,.0f}"
    return f"{text} {currency}".strip()


# -- the three values ------------------------------------------------------------------------
def graham_number(eps: float, bvps: float) -> float:
    """``sqrt(22.5 x EPS x BVPS)``; NaN unless both earnings and book value are positive."""
    return math.sqrt(PE_PB_CAP * eps * bvps) if _pos(eps) and _pos(bvps) else NAN


def growth_value(eps: float, growth: float, bond_yield: float | None = None) -> float:
    """``EPS x (8.5 + 2g)``, scaled by ``4.4 / Y`` when a bond yield is given (``None`` = Graham's original 1962 form).

    ``growth`` is a fraction (0.07 = 7%); a growth below -4.25% would make the multiple negative and is held at zero.
    """
    if not _pos(eps):
        return NAN
    base = eps * max(NO_GROWTH_PE + GROWTH_MULTIPLE * growth * 100, 0.0)
    if bond_yield is None:
        return base
    return base * BASE_YIELD / bond_yield if _pos(bond_yield) else NAN


def implied_growth(price: float, eps: float, bond_yield: float) -> float:
    """The yearly growth that makes the formula equal today's price (the reverse of ``growth_value``)."""
    if not (_pos(price) and _pos(eps) and _pos(bond_yield)):
        return NAN
    pe_at_base_yield = price / eps * bond_yield / BASE_YIELD
    return (pe_at_base_yield - NO_GROWTH_PE) / GROWTH_MULTIPLE / 100


def sensitivity(eps: float, growths, yields) -> pd.DataFrame:
    """Growth-formula value per share: rows = growth, columns = bond yield."""
    return pd.DataFrame([[growth_value(eps, g, y) for y in yields] for g in growths], index=list(growths), columns=list(yields))


def margin_of_safety(value: float, price: float) -> float:
    """Share of the value that the price is below (positive = a discount, negative = the price is above the value)."""
    return 1 - price / value if _pos(value) and _pos(price) else NAN


def reading(value: float, price: float, wanted: float) -> str:
    """One phrase on where the price sits against a value, given the margin of safety the investor asks for."""
    mos = margin_of_safety(value, price)
    if mos != mos:
        return "n/a"
    if mos >= wanted:
        return "Inside your buy zone"
    return "Below value, but margin thinner than you want" if mos >= 0 else "Price above value"


# -- net current asset value ----------------------------------------------------------------
def ncav_per_share(stmts: FinancialStatements) -> pd.Series:
    """(Current assets - total liabilities) / shares for each period; NaN where any of them is missing (banks have no current assets)."""
    shares = line(stmts, "shares").fillna(line(stmts, "shares_avg"))
    liabilities = line(stmts, "total_liabilities").fillna(
        line(stmts, "total_assets") - line(stmts, "equity") - line(stmts, "minority").fillna(0.0))
    out = (line(stmts, "current_assets") - liabilities) / shares.where(shares > 0)
    return out.replace([np.inf, -np.inf], np.nan)


def latest_ncav(annual: FinancialStatements, quarterly: FinancialStatements | None = None, fx: float = 1.0) -> tuple[float, pd.Timestamp] | None:
    """Newest net current asset value per share (the latest quarter when there is one) and its period end."""
    points = stock_points(ncav_per_share(annual), ncav_per_share(quarterly) if quarterly is not None else pd.Series(dtype="float64"))
    points = points.dropna()
    return None if points.empty else (float(points.iloc[-1]) * fx, points.index[-1])


# -- suggested inputs -------------------------------------------------------------------------
@dataclass(frozen=True)
class EpsOption:
    key: str  # "ttm" | "last_fy" | "avg3"
    label: str
    value: float


def annual_eps(annual: FinancialStatements, fx: float = 1.0) -> pd.Series:
    """Diluted EPS of each reported year (oldest first) in the price currency."""
    return (eps_by_period(annual).dropna() * fx).sort_index()


def eps_options(annual: FinancialStatements, ttm_eps: float | None = None, fx: float = 1.0) -> tuple[list[EpsOption], str | None]:
    """Earnings the valuation can start from and the key of the default.

    Graham asked for *normal* earnings, so the default is the average of the last three years when it is positive; trailing
    earnings are the next choice (a recovering company's average still carries its bad years).
    """
    eps = annual_eps(annual, fx)
    options: list[EpsOption] = []
    if isinstance(ttm_eps, (int, float, np.floating)) and ttm_eps == ttm_eps:
        options.append(EpsOption("ttm", "Trailing 12 months", float(ttm_eps)))
    if len(eps):
        options.append(EpsOption("last_fy", f"Last fiscal year ({eps.index[-1]:%Y})", float(eps.iloc[-1])))
    if len(eps) >= 2:
        n = min(3, len(eps))
        options.append(EpsOption("avg3", f"{n}-year average (Graham's normal earnings)", float(eps.tail(n).mean())))
    by_key = {o.key: o for o in options}
    order = ["avg3", "ttm", "last_fy"] if len(eps) >= 3 else ["ttm", "last_fy", "avg3"]
    default = next((k for k in order if k in by_key and by_key[k].value > 0), next(iter(by_key), None))
    return options, default


@dataclass(frozen=True)
class GrowthSuggestion:
    value: float  # the starting slider value, a fraction held between 0 and GROWTH_CAP
    raw: float | None  # what the source said before the cap
    source: str


def suggest_growth(annual: FinancialStatements, consensus_eps: float | None = None, fx: float = 1.0) -> GrowthSuggestion:
    """Yearly growth to start from: the analysts' EPS growth, else the company's own EPS growth over its record, else 3%."""
    raw: float | None
    if consensus_eps is not None and consensus_eps == consensus_eps:
        raw, source = float(consensus_eps), "analyst consensus EPS growth"
    else:
        eps = annual_eps(annual, fx)
        raw = ratios.cagr(eps) if len(eps) >= 3 else NAN
        if raw == raw:
            source = f"EPS growth over the last {len(eps) - 1} years"
        else:
            raw, source = None, "no usable growth data - a neutral 3%"
    base = 0.03 if raw is None else raw
    return GrowthSuggestion(round(min(max(base, 0.0), GROWTH_CAP) * 200) / 200, raw, source)


def default_bond_yield(currency: str) -> float:
    """AAA corporate bond yield to start from: the market's risk-free rate plus a typical spread (an assumption to overwrite)."""
    rf, _ = MARKET_ASSUMPTIONS.get(currency, FALLBACK_ASSUMPTION)
    return round(rf + AAA_SPREAD, 3)


# -- defensive-investor tests -----------------------------------------------------------------
@dataclass(frozen=True)
class GrahamTest:
    name: str
    rule: str  # Graham's requirement in plain words
    passed: bool | None  # None = cannot be judged from the data
    detail: str
    partial: bool = False  # judged on a shorter record than Graham asked for


def normal_eps(annual: FinancialStatements, fx: float = 1.0, years: int = 3) -> tuple[float, int]:
    """Average EPS of the last ``years`` reported years and how many years it really covers (NaN, 0 without any)."""
    eps = annual_eps(annual, fx).tail(years)
    return (float(eps.mean()), len(eps)) if len(eps) else (NAN, 0)


def _streak(dividends: pd.Series | None, today: pd.Timestamp | None) -> tuple[int, int]:
    """(calendar years in a row with a dividend, the latest of them); (0, 0) when none."""
    if dividends is None or len(dividends) == 0:
        return 0, 0
    paid = dividends[dividends > 0]
    years = set(pd.DatetimeIndex(paid.index).year)
    year = (today or pd.Timestamp.today()).year
    if year not in years:
        year -= 1  # this year's dividend may simply not have been declared yet
    latest, streak = year, 0
    while year in years:
        streak += 1
        year -= 1
    return streak, latest if streak else 0


def dividend_streak(dividends: pd.Series, today: pd.Timestamp | None = None) -> int:
    """Calendar years in a row, up to the latest one, in which at least one dividend was paid."""
    return _streak(dividends, today)[0]


def _size_test(annual, fx, floor, currency) -> GrahamTest:
    rule = f"Sales of at least {_short(floor, currency) if floor else 'about USD 700M'} (Graham's $100M in the 1970s, adjusted for inflation)"
    revenue = line(annual, "revenue").dropna()
    if floor is None or revenue.empty:
        return GrahamTest("Adequate size", rule, None, "No sales figure or no exchange rate to compare with.")
    sales = float(revenue.iloc[-1]) * fx
    return GrahamTest("Adequate size", rule, sales >= floor, f"Sales {_short(sales, currency)}")


def _condition_test(annual, fx, bank, currency) -> GrahamTest:
    rule = "Current assets at least twice current liabilities, and long-term debt no more than net current assets"
    if bank:
        return GrahamTest("Strong financial condition", rule, None, "Not meaningful for a bank: it has no current assets or current liabilities.")
    bal = pd.DataFrame({"ca": line(annual, "current_assets"), "cl": line(annual, "current_liabilities"),
                        "ltd": line(annual, "long_term_debt").fillna(line(annual, "total_debt"))}).dropna(subset=["ca", "cl"])
    if bal.empty or float(bal["cl"].iloc[-1]) <= 0:
        return GrahamTest("Strong financial condition", rule, None, "No current assets / liabilities reported.")
    last = bal.iloc[-1]
    ca, cl = float(last["ca"]) * fx, float(last["cl"]) * fx
    debt = 0.0 if last["ltd"] != last["ltd"] else float(last["ltd"]) * fx  # Yahoo omits the debt row of a company that has none
    ratio = ca / cl
    ok = ratio >= MIN_CURRENT_RATIO and debt <= ca - cl
    return GrahamTest("Strong financial condition", rule, ok,
                      f"Current ratio {ratio:.2f}; long-term debt {_short(debt, currency)} vs net current assets {_short(ca - cl, currency)}")


def _stability_test(eps: pd.Series) -> GrahamTest:
    rule = f"Positive earnings in each of the last {STABLE_YEARS} years"
    if len(eps) < 3:
        return GrahamTest("Earnings stability", rule, None, f"Only {len(eps)} years of earnings available.", True)
    good = int((eps > 0).sum())
    return GrahamTest("Earnings stability", rule, good == len(eps), f"Profitable in {good} of the last {len(eps)} years", len(eps) < STABLE_YEARS)


def _dividend_test(dividends, today, history_start) -> GrahamTest:
    rule = f"Dividends paid every year for at least {DIVIDEND_YEARS} years"
    streak, latest = _streak(dividends, today)
    if not streak:
        return GrahamTest("Dividend record", rule, False, "No dividend in the record")
    detail = f"Paid in {streak} straight year{'s' if streak != 1 else ''}"
    if streak >= DIVIDEND_YEARS:
        return GrahamTest("Dividend record", rule, True, detail)
    if history_start is not None and latest - streak <= pd.Timestamp(history_start).year:
        # the run reaches the first year on file: the company may well have paid for longer, the record just does not go back that far
        return GrahamTest("Dividend record", rule, None, f"{detail} - every year of the price history on file; earlier years cannot be checked", True)
    return GrahamTest("Dividend record", rule, False, detail)


def _growth_test(eps: pd.Series) -> GrahamTest:
    rule = f"EPS up at least one third over ten years, comparing three-year averages (about {GROWTH_PER_YEAR:.1%} a year)"
    n = len(eps)
    if n < 4:
        return GrahamTest("Earnings growth", rule, None, f"Needs at least 4 years of earnings; {n} available.", True)
    k = 3 if n >= 6 else 2  # a short record is compared on two-year averages
    start, end = float(eps.iloc[:k].mean()), float(eps.iloc[-k:].mean())
    if start <= 0:
        return GrahamTest("Earnings growth", rule, None, "The starting years were loss-making, so a growth rate is undefined.", True)
    years = n - k  # distance between the middles of the first and last windows
    rate = (end / start) ** (1 / years) - 1 if end > 0 else -1.0
    return GrahamTest("Earnings growth", rule, rate >= GROWTH_PER_YEAR,
                      f"{k}-year average EPS grew {rate:+.1%} a year over {years} years", n < STABLE_YEARS or k < 3)


def _price_tests(price, avg_eps, n_eps, bvps) -> list[GrahamTest]:
    pe_rule = f"Price no more than {MAX_PE:.0f}x average earnings of the last three years"
    pb_rule = f"Price no more than {MAX_PB}x book value, or P/E x P/B no more than {PE_PB_CAP:g}"
    if not _pos(price):
        return [GrahamTest("Moderate P/E", pe_rule, None, "No price."), GrahamTest("Moderate P/B", pb_rule, None, "No price.")]
    partial = 0 < n_eps < 3
    if n_eps == 0:
        pe_test = GrahamTest("Moderate P/E", pe_rule, None, "No earnings history.")
    elif not _pos(avg_eps):
        pe_test = GrahamTest("Moderate P/E", pe_rule, False, "Average earnings are not positive.", partial)
    else:
        pe_test = GrahamTest("Moderate P/E", pe_rule, price / avg_eps <= MAX_PE, f"P/E on average earnings {price / avg_eps:.1f}", partial)
    if not _pos(bvps):
        return [pe_test, GrahamTest("Moderate P/B", pb_rule, None if bvps is None or bvps != bvps else False, "No positive book value.")]
    pb = price / bvps
    product = price / avg_eps * pb if _pos(avg_eps) else NAN
    ok = pb <= MAX_PB or (product == product and product <= PE_PB_CAP)
    detail = f"P/B {pb:.2f}" + (f"; P/E x P/B {product:.1f}" if product == product else "")
    return [pe_test, GrahamTest("Moderate P/B", pb_rule, ok, detail, partial and pb > MAX_PB)]


def defensive_checklist(
    annual: FinancialStatements,
    *,
    price: float,
    bvps: float | None,
    dividends: pd.Series | None = None,
    fx: float = 1.0,
    size_floor: float | None = None,
    bank: bool = False,
    currency: str = "",
    today: pd.Timestamp | None = None,
    history_start: pd.Timestamp | None = None,
) -> list[GrahamTest]:
    """Graham's seven tests for the defensive investor, each pass / fail / not judgeable.

    ``annual`` are yearly statements (the tests are about years); ``fx`` converts them to the price currency, ``size_floor``
    is the sales floor in that currency (``None`` skips the size test) and ``bvps`` the latest book value per share.
    ``history_start`` is the first day of price data on file: dividends are read from it, so a payment run that reaches back to
    that year says nothing about earlier ones and the dividend test is left undecided rather than failed.
    """
    eps = annual_eps(annual, fx)
    avg, n = normal_eps(annual, fx)
    return [
        _size_test(annual, fx, size_floor, currency),
        _condition_test(annual, fx, bank, currency),
        _stability_test(eps),
        _dividend_test(dividends, today, history_start),
        _growth_test(eps),
        *_price_tests(price, avg, n, bvps),
    ]
