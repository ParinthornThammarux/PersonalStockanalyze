"""Valuation bands: where today's multiple sits in the company's own history.

Each day's P/E, P/B and EV/EBITDA uses only figures the market already had that day: a period's
numbers count from its announcement date (the first cached earnings date after the period ends,
else period end + 45 days for a quarter / 60 days for a year-end, the usual filing deadlines) and
never before. Without that, a quarter's EPS would colour prices from weeks earlier.

Prices must be split-adjusted but not dividend-adjusted (the store's raw ``close``): Yahoo restates
EPS and share counts for splits, so ``close / EPS`` stays consistent across them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from stockanalyzer.analysis.fundamental.statements import line
from stockanalyzer.domain.financials import FinancialStatements

QUARTER_LAG, ANNUAL_LAG = 45, 60  # days from period end to the filing when no report date is cached
REPORT_WINDOW = 90  # a cached report date counts for a period only if it falls this soon after its end (the longest
#                     filing deadline); any later one is the next period's, so a gap in the cache cannot shift a report
OUTLIER_X = 5.0  # multiples above this many medians come from earnings near zero, not from valuation
MIN_DAYS = 120
LEVELS = (-2, -1, 0, 1, 2)


def _naive(s: pd.Series | pd.DataFrame):
    out = s.copy()
    out.index = pd.DatetimeIndex(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    return out.sort_index()


def announce_dates(period_ends: pd.DatetimeIndex, report_days: Iterable = (), lags: Iterable[int] | None = None) -> pd.DatetimeIndex:
    """When each period's numbers became public: the first report date after the period end within
    ``REPORT_WINDOW`` days, else the period end plus its filing lag."""
    reports = pd.DatetimeIndex(sorted(pd.to_datetime(list(report_days)))) if report_days is not None else pd.DatetimeIndex([])
    lags = [QUARTER_LAG] * len(period_ends) if lags is None else list(lags)
    out = []
    for end, lag in zip(period_ends, lags):
        i = reports.searchsorted(end, side="right")
        if i < len(reports) and (reports[i] - end).days <= REPORT_WINDOW:
            out.append(reports[i])
        else:
            out.append(end + pd.Timedelta(days=int(lag)))
    return pd.DatetimeIndex(out)


def ttm(quarterly: pd.Series) -> pd.Series:
    """Sum of four quarters, only where those four really are consecutive (a missing quarter breaks it)."""
    s = quarterly.dropna().sort_index()
    if len(s) < 4:
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
    span = pd.Series(s.index, index=s.index).diff(3).dt.days
    total = s.rolling(4).sum()
    return total[span.between(250, 295)]


def eps_by_period(stmts: FinancialStatements) -> pd.Series:
    """Reported diluted EPS; where Yahoo leaves a period blank, net income to common over the average share count.

    Yahoo often blanks the latest quarter's EPS and share count while keeping its net income, which would break
    every trailing-four-quarter sum and freeze P/E on an old year. The share count is borrowed from the nearest
    periods (two at most), as it moves slowly.
    """
    shares = line(stmts, "shares_avg").fillna(line(stmts, "shares")).ffill(limit=2).bfill(limit=2)
    derived = (line(stmts, "net_income_common") / shares).replace([float("inf"), float("-inf")], float("nan"))
    return line(stmts, "eps").fillna(derived)


def flow_points(annual: pd.Series, quarterly: pd.Series) -> pd.Series:
    """Trailing-12-month value by period end: reported years, plus every consecutive four-quarter sum."""
    return annual.dropna().combine_first(ttm(quarterly)).sort_index()


def stock_points(annual: pd.Series, quarterly: pd.Series) -> pd.Series:
    """Balance-sheet value by period end from both frequencies."""
    return annual.dropna().combine_first(quarterly.dropna()).sort_index()


def _asof(points: pd.Series, annual_periods: pd.DatetimeIndex, report_days: Iterable, index: pd.DatetimeIndex) -> pd.Series:
    """Step series on ``index``: each value from its announcement date until the next period's."""
    if points.empty:
        return pd.Series(float("nan"), index=index)
    lags = [ANNUAL_LAG if p in annual_periods else QUARTER_LAG for p in points.index]
    known = pd.Series(points.to_numpy(), index=announce_dates(points.index, report_days, lags)).sort_index()
    known = known[~known.index.duplicated(keep="last")]
    return known.reindex(known.index.union(index)).ffill().reindex(index)


def valuation_history(
    close: pd.Series,
    annual: FinancialStatements,
    quarterly: FinancialStatements | None = None,
    report_days: Iterable = (),
    fx: float = 1.0,
) -> pd.DataFrame:
    """Daily ``close, eps_ttm, bvps, sps, pe, pb, ps, ev_ebitda, ev_sales`` (NaN where the company had no usable figure).

    A multiple is NaN while its denominator is not positive: a P/E on a loss is not a cheap stock.
    ``fx`` converts statement currency into the price currency (one rate for the whole history).
    """
    quarterly = quarterly if quarterly is not None else FinancialStatements(annual.symbol, quarterly=True)
    px = _naive(close.dropna())
    px = px[px > 0]
    idx = px.index
    days = list(report_days)
    years = pd.DatetimeIndex(annual.periods)

    def both(key: str, fn):
        return fn(line(annual, key), line(quarterly, key))

    def share_count(s: FinancialStatements) -> pd.Series:
        return line(s, "shares").fillna(line(s, "shares_avg"))

    def book_per_share(s: FinancialStatements) -> pd.Series:
        return (line(s, "equity") / share_count(s)).replace([float("inf"), float("-inf")], float("nan"))

    def ev_bridge(s: FinancialStatements) -> pd.Series:
        """Market cap to enterprise value: net debt plus minority interest (consolidated EBITDA counts subsidiaries in full)."""
        return line(s, "total_debt") - line(s, "cash") + line(s, "minority").fillna(0.0)

    eps = _asof(flow_points(eps_by_period(annual), eps_by_period(quarterly)), years, days, idx) * fx
    ebitda = _asof(both("ebitda", flow_points), years, days, idx) * fx
    sales = _asof(both("revenue", flow_points), years, days, idx) * fx
    bvps = _asof(stock_points(book_per_share(annual), book_per_share(quarterly)), years, days, idx) * fx
    shares = _asof(stock_points(share_count(annual), share_count(quarterly)), years, days, idx)
    bridge = _asof(stock_points(ev_bridge(annual), ev_bridge(quarterly)), years, days, idx) * fx
    ev = px * shares + bridge
    sps = (sales / shares).replace([float("inf"), float("-inf")], float("nan"))  # sales per share
    out = pd.DataFrame({
        "close": px, "eps_ttm": eps, "bvps": bvps, "sps": sps,
        "pe": px / eps.where(eps > 0), "pb": px / bvps.where(bvps > 0), "ps": px / sps.where(sps > 0),
        "ev_ebitda": ev / ebitda.where(ebitda > 0), "ev_sales": ev / sales.where(sales > 0),
    })
    return out


def dividend_yield_history(close: pd.Series, dividends: pd.Series) -> pd.DataFrame:
    """Daily ``dps_ttm, dividend_yield``: dividends with an ex-date in the last 365 days over the close.

    The first year of price history is left NaN because the dividends before it are unknown.
    """
    px = _naive(close.dropna())
    px = px[px > 0]
    if px.empty:
        return pd.DataFrame(columns=["dps_ttm", "dividend_yield"])
    if dividends.dropna().empty:
        return pd.DataFrame({"dps_ttm": float("nan"), "dividend_yield": float("nan")}, index=px.index)
    paid = _naive(dividends.dropna()).groupby(level=0).sum()
    cum = paid.cumsum()

    def at(index: pd.DatetimeIndex) -> pd.Series:
        return cum.reindex(cum.index.union(index)).ffill().reindex(index).fillna(0.0)

    dps = at(px.index) - at(px.index - pd.Timedelta(days=365)).set_axis(px.index)
    warm = px.index >= px.index[0] + pd.Timedelta(days=365)
    dps = dps.where(warm)
    if not (dps > 0).any():
        dps = pd.Series(float("nan"), index=px.index)
    return pd.DataFrame({"dps_ttm": dps, "dividend_yield": dps / px})


@dataclass(frozen=True)
class BandStats:
    current: float
    mean: float
    std: float
    median: float
    percentile: float  # share of the history at or below the current value, 0-1
    zscore: float
    days: int
    years: float
    excluded: int  # days dropped as outliers before mean / std / percentile

    def level(self, k: float) -> float:
        return self.mean + k * self.std


def band_stats(series: pd.Series, *, years: float | None = None) -> BandStats | None:
    """Mean, spread and rank of the latest value; ``None`` with fewer than ``MIN_DAYS`` usable days."""
    s = series.dropna()
    s = s[s > 0]
    if years is not None and not s.empty:
        s = s[s.index >= s.index[-1] - pd.DateOffset(years=years)]
    if len(s) < MIN_DAYS:
        return None
    keep = s[s <= OUTLIER_X * s.median()]
    current = float(s.iloc[-1])
    std = float(keep.std())
    return BandStats(
        current=current, mean=float(keep.mean()), std=std, median=float(keep.median()),
        percentile=float((keep <= current).mean()),
        zscore=(current - float(keep.mean())) / std if std > 0 else float("nan"),
        days=len(keep), years=(s.index[-1] - s.index[0]).days / 365.25, excluded=len(s) - len(keep),
    )


def implied_prices(per_share: pd.Series, stats: BandStats, levels: Iterable[float] = LEVELS, *, inverse: bool = False) -> pd.DataFrame:
    """The price each band level would mean given the per-share figure of the day (``inverse`` for yields)."""
    cols = {}
    for k in levels:
        level = stats.level(k)
        if level > 0:
            cols[k] = per_share / level if inverse else per_share * level
    return pd.DataFrame(cols)


def stance(stats: BandStats, *, higher_is_cheaper: bool = False) -> str:
    """'cheap' / 'fair' / 'expensive' against the company's own history (top / bottom fifth of days)."""
    rank = 1 - stats.percentile if higher_is_cheaper else stats.percentile
    return "expensive" if rank >= 0.8 else "cheap" if rank <= 0.2 else "fair"
