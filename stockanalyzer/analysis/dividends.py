"""Dividend deep-dive: the payment record, whether the company can afford it, and what the share price does around the
ex-dividend (XD) day.

The XD study answers questions Thai investors ask all the time - how much of the dividend comes off the price, how long the
gap takes to fill ("เติมปันผล"), whether the price runs up before the XD and whether buying the day before to collect the
dividend pays after tax and costs. Each figure is a median or average over the stock's own XDs, shown next to the same
figure for every liquid dividend payer in its market.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockanalyzer.analysis.fundamental.statements import line
from stockanalyzer.domain.financials import FinancialStatements

DROP_MIN_YIELD = 0.005  # below this the day's normal price noise swamps the dividend, so the drop ratio means nothing
FILL_WINDOWS = (5, 20, 60)
CUT_TOLERANCE = 0.95  # a year paying less than this share of the year before counts as a cut
DEFAULT_TAX = {"THB": 0.10, "USD": 0.15}  # Thai dividend withholding; the US rate for a Thai resident with a W-8BEN form
DEFAULT_COST = 0.003  # round trip (buy + sell), commission and fees
PAYOUT_WARN, PAYOUT_BAD = 0.8, 1.0


# -- the record ------------------------------------------------------------------------------------------
def yearly(dividends: pd.Series, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """Dividends per share by calendar year of the ex-date: total, number of payments, change on the year before.
    The current year is marked ``partial`` (more payments may come)."""
    d = dividends.dropna()
    d = d[d > 0]
    if d.empty:
        return pd.DataFrame(columns=["dps", "payments", "growth", "partial"])
    asof = pd.Timestamp(asof) if asof is not None else d.index[-1]
    g = d.groupby(d.index.year)
    out = pd.DataFrame({"dps": g.sum(), "payments": g.size()})
    out = out.reindex(range(out.index.min(), max(out.index.max(), asof.year) + 1), fill_value=0)
    out.index.name = "year"
    out["growth"] = out["dps"].pct_change(fill_method=None).where(out["dps"].shift(1) > 0)
    out["partial"] = out.index == asof.year
    return out


@dataclass
class DividendRecord:
    first_year: int | None
    years_paid: int
    no_cut_streak: int  # consecutive complete years without a cut, counting back from the last complete year
    raise_streak: int  # consecutive complete years with a raise
    cagr_3: float
    cagr_5: float
    cuts: list[int]
    ttm: float  # dividends with an ex-date in the last 365 days
    reaches_start: bool  # the record starts with the price history, so the streaks may be longer in reality


def record(dividends: pd.Series, asof: pd.Timestamp, history_start: pd.Timestamp | None = None) -> DividendRecord:
    y = yearly(dividends, asof)
    full = y[~y["partial"]]
    paid = full[full["dps"] > 0]
    if paid.empty:
        return DividendRecord(None, 0, 0, 0, np.nan, np.nan, [], 0.0, False)
    cuts = [int(yr) for yr in full.index[1:] if full.loc[yr, "dps"] < CUT_TOLERANCE * full.loc[yr - 1, "dps"] and full.loc[yr - 1, "dps"] > 0]
    no_cut = raise_ = 0
    for yr in reversed(full.index[1:]):
        prev, cur = full.loc[yr - 1, "dps"], full.loc[yr, "dps"]
        if prev <= 0 or cur < CUT_TOLERANCE * prev:
            break
        no_cut += 1
    for yr in reversed(full.index[1:]):
        if not full.loc[yr, "dps"] > full.loc[yr - 1, "dps"] > 0:
            break
        raise_ += 1

    def cagr(n: int) -> float:
        if len(full) <= n:
            return np.nan
        a, b = full["dps"].iloc[-1 - n], full["dps"].iloc[-1]
        return float((b / a) ** (1 / n) - 1) if a > 0 and b > 0 else np.nan

    d = dividends.dropna()
    ttm = float(d[d.index > pd.Timestamp(asof) - pd.Timedelta(days=365)].sum())
    first = int(paid.index[0])
    reaches = history_start is not None and first <= pd.Timestamp(history_start).year + 1
    return DividendRecord(first, int(len(paid)), no_cut, raise_, cagr(3), cagr(5), cuts, ttm, reaches)


# -- can the company afford it ----------------------------------------------------------------------------
def coverage(stmts: FinancialStatements) -> pd.DataFrame:
    """Per year: dividends paid, buybacks, and both against net income and free cash flow; net debt and its change."""
    L = lambda k: line(stmts, k)  # noqa: E731
    paid, buybacks = -L("dividends"), -L("buybacks")
    ni, fcf = L("net_income"), L("fcf")
    debt = L("total_debt").fillna(L("long_term_debt"))
    net_debt = debt - L("cash")
    out = pd.DataFrame({"dividends_paid": paid, "buybacks": buybacks, "net_income": ni, "fcf": fcf}, index=stmts.periods)
    out["payout_earnings"] = paid / ni.where(ni > 0)
    out["payout_fcf"] = paid / fcf.where(fcf > 0)
    out["shareholder_payout_fcf"] = (paid + buybacks.fillna(0)) / fcf.where(fcf > 0)
    out["net_debt"] = net_debt
    out["net_debt_change"] = net_debt.diff()
    return out.sort_index().replace([np.inf, -np.inf], np.nan)


@dataclass
class Note:
    tone: str  # "bad" | "warn" | "good"
    text: str


def coverage_notes(cov: pd.DataFrame, *, bank: bool = False) -> list[Note]:
    out: list[Note] = []
    c = cov.dropna(subset=["dividends_paid"])
    c = c[c["dividends_paid"] > 0]
    if c.empty:
        return out
    pe = c["payout_earnings"].dropna()
    if len(pe):
        last = float(pe.iloc[-1])
        tone = "bad" if last > PAYOUT_BAD else "warn" if last > PAYOUT_WARN else "good"
        out.append(Note(tone, f"Paid out {last:.0%} of net income in {pe.index[-1]:%Y}"
                              + (" - more than it earned." if last > PAYOUT_BAD else " - little room if profit dips." if last > PAYOUT_WARN
                                 else " - room to keep paying through a weaker year.")))
    if bank:
        return out
    fcf_years = c.dropna(subset=["fcf"])
    if len(fcf_years):
        short = fcf_years[fcf_years["dividends_paid"] > fcf_years["fcf"]]
        if len(short):
            funded = short["net_debt_change"].dropna()
            borrowed = int((funded > 0).sum())
            out.append(Note("bad" if len(short) >= 2 else "warn",
                            f"Dividends exceeded free cash flow in {len(short)} of {len(fcf_years)} years ({', '.join(f'{d:%Y}' for d in short.index)})"
                            + (f"; net debt rose in {borrowed} of them, so part of the dividend was borrowed." if borrowed else ".")))
        else:
            out.append(Note("good", f"Free cash flow covered the dividend in each of the {len(fcf_years)} years on record."))
    sp = c["shareholder_payout_fcf"].dropna()
    if len(sp) and float(sp.iloc[-1]) > 1.0 and c["buybacks"].fillna(0).iloc[-1] > 0:
        out.append(Note("warn", f"Dividends plus buybacks were {float(sp.iloc[-1]):.0%} of free cash flow in {sp.index[-1]:%Y}."))
    return out


# -- the XD day --------------------------------------------------------------------------------------------
def _asof(series: pd.Series, when: pd.Series) -> np.ndarray:
    """Value of ``series`` on (or the last day before) each date in ``when``."""
    s = series.dropna().sort_index()
    if s.empty:
        return np.full(len(when), np.nan)
    pos = s.index.searchsorted(pd.DatetimeIndex(when), side="right") - 1
    vals = s.to_numpy()[np.clip(pos, 0, None)]
    return np.where((pos >= 0) & pd.notna(when).to_numpy(), vals, np.nan)


def event_metrics(events: pd.DataFrame, benchmark: pd.Series | None = None, *, tax: float = 0.10,
                  cost: float = DEFAULT_COST) -> pd.DataFrame:
    """Derived figures per XD from :func:`data.dividend_events.xd_events`:

    - ``yield``: dividend ÷ the close before the XD;
    - ``drop_open`` / ``drop_close``: how much of the dividend came off the price at the open / by the close (1.0 = all of it);
    - ``drop_adjusted``: the same at the close after taking out the market's move that day (needs ``benchmark``);
    - ``pre_excess``: return over the 10 days before the XD minus the market's; ``post_excess``: return with the dividend
      over the 20 days after minus the market's;
    - ``capture``: buy at the close before, sell at the XD open, keep the dividend after ``tax``, pay ``cost`` - the
      dividend-capture trade;
    - ``fill_days`` and ``filled_<n>``: whether the close regained the pre-XD close within n trading days (NaN while the
      window is not over yet).
    """
    ev = events.copy()
    if ev.empty:
        return ev
    prev = ev["prev_close"].astype("float64")
    amount = ev["amount"].astype("float64")
    ev["yield"] = amount / prev
    ev["drop_open"] = (prev - ev["open0"]) / amount
    ev["drop_close"] = (prev - ev["close0"]) / amount
    ev["pre_return"] = prev / ev["close_m11"] - 1
    ev["post_return"] = ev["adj_p20"] / ev["adj_m1"] - 1
    if benchmark is not None and not benchmark.dropna().empty:
        b = benchmark.copy()
        b.index = pd.DatetimeIndex(b.index).tz_localize(None) if getattr(b.index, "tz", None) else pd.DatetimeIndex(b.index)
        b_m1, b_0 = _asof(b, ev["ts_m1"]), _asof(b, ev["ex_date"])
        b_m11, b_p20 = _asof(b, ev["ts_m11"]), _asof(b, ev["ts_p20"])
        market_day = b_0 / b_m1 - 1
        ev["drop_adjusted"] = (prev * (1 + market_day) - ev["close0"]) / amount
        ev["pre_excess"] = ev["pre_return"] - (b_m1 / b_m11 - 1)
        ev["post_excess"] = ev["post_return"] - (b_p20 / b_m1 - 1)
    else:
        ev["drop_adjusted"], ev["pre_excess"], ev["post_excess"] = ev["drop_close"], np.nan, np.nan
    ev["capture"] = (ev["open0"] - prev + amount * (1 - tax)) / prev - cost
    for n in FILL_WINDOWS:
        filled = ev["fill_days"] <= n
        ev[f"filled_{n}"] = np.where(filled, 1.0, np.where(ev["k_max"] >= n, 0.0, np.nan))
    return ev.replace([np.inf, -np.inf], np.nan)


def summarize(ev: pd.DataFrame) -> dict[str, float]:
    """Medians / averages over a set of XDs (one stock's, or a whole market's)."""
    if ev.empty:
        return {"events": 0}
    sized = ev[ev["yield"] >= DROP_MIN_YIELD]
    observed = ev[ev["k_max"] >= FILL_WINDOWS[-1]]
    fill = observed["fill_days"].fillna(np.inf)
    out = {
        "events": int(len(ev)),
        "median_yield": float(ev["yield"].median()),
        "median_drop_open": float(sized["drop_open"].median()) if len(sized) else np.nan,
        "median_drop": float(sized["drop_adjusted"].median()) if len(sized) else np.nan,
        "median_fill_days": float(np.median(fill)) if len(fill) else np.nan,
        "avg_pre_excess": float(ev["pre_excess"].mean()),
        "avg_post_excess": float(ev["post_excess"].mean()),
        "avg_capture": float(ev["capture"].mean()),
        "capture_wins": float((ev["capture"].dropna() > 0).mean()) if ev["capture"].notna().any() else np.nan,
    }
    for n in FILL_WINDOWS:
        col = ev[f"filled_{n}"].dropna()
        out[f"filled_{n}"] = float(col.mean()) if len(col) else np.nan
    return out


def liquid_events(ev: pd.DataFrame, min_traded: float, *, min_days: int = 40) -> pd.DataFrame:
    """XDs of stocks that traded at least ``min_traded`` a day over the 60 days before (judged at the time)."""
    return ev[(ev["tv60"] >= min_traded) & (ev["n_before"] >= min_days)]
