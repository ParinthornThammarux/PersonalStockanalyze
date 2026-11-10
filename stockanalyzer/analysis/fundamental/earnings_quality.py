"""Earnings quality and value creation: is the reported profit backed by cash, and does the business earn more on its
capital than that capital costs?

- cash backing: operating cash flow against net income, Sloan's accruals ratio;
- working capital: days of receivables, inventory and payables, and receivables / inventory growing faster than sales;
- Beneish's M-score (1999, 8 variables) for signs of earnings manipulation;
- reinvestment: capital spending against depreciation, stock pay against free cash flow, share dilution, one-off items;
- value creation: ROIC against the cost of capital, economic profit, incremental ROIC, the growth reinvestment can fund;
- DuPont: which of margin, asset turnover and leverage moved the return on equity.

Thresholds are rules of thumb from the accounting literature and are labelled as such on the page. Yahoo gives 4-5 years
of statements (the oldest is often half empty), so trends rest on three or four years. None of this describes a bank:
its cash flow follows deposits and loans, so ``flags(bank=True)`` keeps only what still means something.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stockanalyzer.analysis.fundamental import ratios
from stockanalyzer.analysis.fundamental.statements import line
from stockanalyzer.domain.financials import FinancialStatements

# Beneish (1999) 8-variable probit model; the averages are his sample's non-manipulators vs manipulators
BENEISH_CONSTANT = -4.84
BENEISH_WEIGHTS = {"DSRI": 0.920, "GMI": 0.528, "AQI": 0.404, "SGI": 0.892, "DEPI": 0.115, "SGAI": -0.172, "TATA": 4.679,
                   "LVGI": -0.327}
BENEISH_MEANS = {"DSRI": (1.031, 1.465), "GMI": (1.014, 1.193), "AQI": (1.039, 1.254), "SGI": (1.134, 1.607),
                 "DEPI": (1.001, 1.077), "SGAI": (1.054, 1.041), "TATA": (0.018, 0.031), "LVGI": (1.037, 1.111)}
BENEISH_NAMES = {
    "DSRI": "Receivables vs sales (days' sales in receivables index)",
    "GMI": "Gross margin shrinking (gross margin index)",
    "AQI": "Soft assets growing (asset quality index)",
    "SGI": "Sales growth index",
    "DEPI": "Depreciation slowing (depreciation index)",
    "SGAI": "Overheads vs sales (SG&A index)",
    "TATA": "Accruals: profit not backed by cash (total accruals ÷ assets)",
    "LVGI": "Leverage rising (leverage index)",
}
BENEISH_REQUIRED = ("DSRI", "SGI", "TATA")
BENEISH_FLAG, BENEISH_GREY = -1.78, -2.22  # above -1.78: Beneish's cut-off for a likely manipulator; -2.22 is a stricter, popular one

CASH_CONVERSION_OK, CASH_CONVERSION_LOW = 1.0, 0.8
ACCRUALS_WARN, ACCRUALS_BAD = 0.05, 0.10  # (net income − operating cash flow) ÷ average assets
GAP_WARN, GAP_BAD = 0.10, 0.25  # receivables / inventory growth minus sales / cost growth
MATERIAL = 0.05  # receivables or inventory below this share of total assets are too small for their growth to matter
CAPEX_DA_LOW = 0.8
SBC_FCF_WARN, SBC_FCF_BAD = 0.10, 0.30
DILUTION_WARN, DILUTION_BAD = 0.02, 0.05  # share count growth a year
UNUSUAL_WARN = 0.20  # one-off items as a share of pre-tax profit
SPREAD_OK = 0.02  # ROIC this far above the cost of capital counts as clearly creating value
MIN_CAPITAL_GROWTH = 0.05  # incremental ROIC needs capital to have grown at least this much


def _div(a, b):
    return a / b.where(b != 0) if isinstance(b, pd.Series) else (a / b if b else np.nan)


def _days(stmts: FinancialStatements) -> float:
    return 91.25 if stmts.quarterly else 365.0


def _avg(series: pd.Series) -> pd.Series:
    return ((series + series.shift(1)) / 2).fillna(series)


def cost_of_revenue(stmts: FinancialStatements) -> pd.Series:
    return line(stmts, "cogs").fillna(line(stmts, "revenue") - line(stmts, "gross_profit"))


def quality_table(stmts: FinancialStatements) -> pd.DataFrame:
    """Per period (oldest first): cash backing, working capital days and gaps, reinvestment, stock pay, dilution, one-offs."""
    L = lambda k: line(stmts, k)  # noqa: E731
    # cash flow is consolidated, so compare it with the group's whole profit (minority shareholders' part included)
    ni = L("net_income_consolidated").fillna(L("net_income"))
    cfo, fcf, revenue = L("cfo"), L("fcf"), L("revenue")
    cogs = cost_of_revenue(stmts)
    rec, inv, pay = L("receivables"), L("inventory"), L("payables")
    days = _days(stmts)
    lag = 4 if stmts.quarterly else 1
    pretax = L("pretax_income")
    positive_ni = ni.where(ni > 0)

    out = pd.DataFrame(index=stmts.periods)
    out["net_income"], out["cfo"], out["fcf"] = ni, cfo, fcf
    out["cash_conversion"] = _div(cfo, positive_ni)
    out["fcf_conversion"] = _div(fcf, positive_ni)
    out["accruals_ratio"] = _div(ni - cfo, _avg(L("total_assets")))
    out["dso"] = _div(rec, revenue) * days
    out["dio"] = _div(inv, cogs) * days
    out["dpo"] = _div(pay, cogs) * days
    out["ccc"] = out["dso"] + out["dio"].fillna(0) - out["dpo"]
    growth = lambda s: s.pct_change(lag, fill_method=None).where(s.shift(lag) > 0)  # noqa: E731
    out["revenue_growth"] = growth(revenue)
    out["receivables_growth"] = growth(rec)
    out["receivables_gap"] = out["receivables_growth"] - out["revenue_growth"]
    out["inventory_growth"] = growth(inv)
    out["inventory_gap"] = out["inventory_growth"] - growth(cogs)
    out["receivables_share"] = _div(rec, L("total_assets"))
    out["inventory_share"] = _div(inv, L("total_assets"))
    out["capex_to_da"] = _div(-L("capex"), L("da").where(L("da") > 0))
    out["sbc_to_fcf"] = _div(L("sbc"), fcf.where(fcf > 0))
    out["sbc_to_revenue"] = _div(L("sbc"), revenue)
    out["share_change"] = L("shares").pct_change(lag, fill_method=None)
    out["unusual_share"] = _div(L("unusual_items"), pretax.where(pretax.abs() > 0))
    return out.sort_index().replace([np.inf, -np.inf], np.nan)


def dupont(stmts: FinancialStatements) -> pd.DataFrame:
    """ROE split into tax burden × interest burden × EBIT margin × asset turnover × equity multiplier (averages of the
    balance sheet items, so the product is net income ÷ average equity)."""
    ni, pretax, revenue = line(stmts, "net_income"), line(stmts, "pretax_income"), line(stmts, "revenue")
    op = ratios.ebit(stmts)
    assets, equity = _avg(line(stmts, "total_assets")), _avg(line(stmts, "equity"))
    out = pd.DataFrame({
        "tax_burden": _div(ni, pretax),
        "interest_burden": _div(pretax, op),
        "ebit_margin": _div(op, revenue),
        "asset_turnover": _div(revenue, assets),
        "equity_multiplier": _div(assets, equity),
    }, index=stmts.periods)
    out["roe"] = out.prod(axis=1, min_count=5)
    return out.sort_index().replace([np.inf, -np.inf], np.nan)


DRIVERS = {"tax_burden": "Tax", "interest_burden": "Interest", "ebit_margin": "Operating margin",
           "asset_turnover": "Asset turnover", "equity_multiplier": "Leverage"}


def roe_drivers(dp: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, float, float, pd.Series] | None:
    """How much each DuPont factor moved ROE between the first and last complete period, in ROE points (the log change
    of each factor, shared out over the total change). ``None`` without two complete periods with every factor positive."""
    ok = dp.dropna()
    ok = ok[(ok[list(DRIVERS)] > 0).all(axis=1)]
    if len(ok) < 2:
        return None
    first, last = ok.iloc[0], ok.iloc[-1]
    logs = np.log(last[list(DRIVERS)] / first[list(DRIVERS)])
    total = float(np.log(last["roe"] / first["roe"]))
    change = float(last["roe"] - first["roe"])
    share = logs / total if abs(total) > 1e-9 else logs * 0
    return ok.index[0], ok.index[-1], float(first["roe"]), float(last["roe"]), (share * change).rename(index=DRIVERS)


@dataclass
class ValueCreation:
    table: pd.DataFrame  # per period: nopat, invested_capital, roic, spread, economic_profit
    wacc: float
    incremental_roic: float  # extra operating profit per unit of extra capital, first to last period
    reinvestment_rate: float  # capital added ÷ operating profit earned over the same years
    funded_growth: float  # reinvestment rate × average ROIC: the growth the business can pay for itself


def value_creation(stmts: FinancialStatements, wacc: float) -> ValueCreation:
    nopat, ic = ratios.nopat(stmts), ratios.invested_capital(stmts)
    table = pd.DataFrame({"nopat": nopat, "invested_capital": ic}, index=stmts.periods).sort_index()
    table["roic"] = _div(table["nopat"], table["invested_capital"].where(table["invested_capital"] > 0))
    table["spread"] = table["roic"] - wacc
    table["economic_profit"] = table["spread"] * table["invested_capital"]
    ok = table.dropna(subset=["nopat", "invested_capital"])
    inc = reinvest = funded = np.nan
    if len(ok) >= 3:
        # profit of year t comes from capital at the end of year t-1; a capital change under 5% makes the ratio explode
        d_nopat = ok["nopat"].iloc[-1] - ok["nopat"].iloc[1]
        d_ic = ok["invested_capital"].iloc[-2] - ok["invested_capital"].iloc[0]
        inc = float(d_nopat / d_ic) if d_ic > MIN_CAPITAL_GROWTH * abs(ok["invested_capital"].iloc[0]) else np.nan
        earned = ok["nopat"].iloc[1:].sum()
        added = ok["invested_capital"].iloc[-1] - ok["invested_capital"].iloc[0]
        reinvest = float(added / earned) if earned > 0 else np.nan
        avg_roic = ok["roic"].iloc[1:].mean()
        funded = float(reinvest * avg_roic) if reinvest == reinvest and avg_roic == avg_roic and reinvest > 0 else np.nan
    return ValueCreation(table.replace([np.inf, -np.inf], np.nan), wacc, inc, reinvest, funded)


@dataclass
class BeneishResult:
    period: pd.Timestamp
    m: float
    indices: dict[str, float]
    neutral: list[str] = field(default_factory=list)  # variables with missing data, set to 1 (no signal) / 0 for TATA

    @property
    def zone(self) -> str:
        return "red flag" if self.m > BENEISH_FLAG else "grey zone" if self.m > BENEISH_GREY else "no sign"

    def drivers(self) -> pd.Series:
        """Each variable's push on the score against a typical honest company (weight × distance from the
        non-manipulators' average), largest first."""
        push = {k: BENEISH_WEIGHTS[k] * (v - BENEISH_MEANS[k][0]) for k, v in self.indices.items()}
        return pd.Series(push, dtype="float64").sort_values(ascending=False)


def beneish(stmts: FinancialStatements) -> list[BeneishResult]:
    """M-score for every period that has the one before it (yearly statements; quarterly ones are seasonal)."""
    L = lambda k: line(stmts, k)  # noqa: E731
    sales, rec, ta = L("revenue"), L("receivables"), L("total_assets")
    gm = _div(sales - cost_of_revenue(stmts), sales)
    ca, ppe, dep, sga = L("current_assets"), L("net_ppe"), L("da"), L("sga")
    debt = L("long_term_debt").fillna(0)
    lev = _div(L("current_liabilities") + debt, ta)
    soft = 1 - _div(ca + ppe, ta)
    dep_rate = _div(dep, dep + ppe)
    income = L("income_continuing").fillna(L("net_income"))
    tata = _div(income - L("cfo"), ta)
    p = lambda s: s.shift(1)  # noqa: E731
    indices = pd.DataFrame({
        "DSRI": _div(_div(rec, sales), p(_div(rec, sales))),
        "GMI": _div(p(gm), gm.where(gm > 0)),
        "AQI": _div(soft, p(soft)),
        "SGI": _div(sales, p(sales)),
        "DEPI": _div(p(dep_rate), dep_rate),
        "SGAI": _div(_div(sga, sales), p(_div(sga, sales))),
        "TATA": tata,
        "LVGI": _div(lev, p(lev)),
    }, index=stmts.periods).sort_index().replace([np.inf, -np.inf], np.nan)
    out = []
    for period, row in indices.iloc[1:].iterrows():
        if row[list(BENEISH_REQUIRED)].isna().any():
            continue
        neutral = [k for k in BENEISH_WEIGHTS if pd.isna(row[k])]
        values = {k: (float(row[k]) if pd.notna(row[k]) else (0.0 if k == "TATA" else 1.0)) for k in BENEISH_WEIGHTS}
        m = BENEISH_CONSTANT + sum(BENEISH_WEIGHTS[k] * v for k, v in values.items())
        out.append(BeneishResult(pd.Timestamp(period), float(m), values, neutral))
    return out


@dataclass
class Flag:
    tone: str  # "bad" | "warn" | "good" | "info"
    title: str
    detail: str


def _last(series: pd.Series) -> tuple[float, pd.Timestamp | None]:
    s = series.dropna()
    return (float(s.iloc[-1]), s.index[-1]) if len(s) else (np.nan, None)


def _year(ts: pd.Timestamp | None, stmts: FinancialStatements) -> str:
    if ts is None:
        return ""
    return f"Q{ts.quarter} {ts.year}" if stmts.quarterly else f"FY {ts:%Y}"


def flags(stmts: FinancialStatements, *, wacc: float | None = None, bank: bool = False) -> list[Flag]:
    """What stands out, worst first. A bank only gets the checks that do not lean on cash flow or working capital."""
    q = quality_table(stmts)
    out: list[Flag] = []
    if not bank:
        recent = q[["net_income", "cfo"]].dropna().tail(3)
        if len(recent) and recent["net_income"].sum() > 0:
            conv = recent["cfo"].sum() / recent["net_income"].sum()
            span = f"over the last {len(recent)} {'periods' if len(recent) > 1 else 'period'}"
            if conv < CASH_CONVERSION_LOW:
                out.append(Flag("bad", "Profit not backed by cash", f"Operating cash flow was {conv:.0%} of net income {span}: "
                                "earnings are running ahead of the cash coming in."))
            elif conv < CASH_CONVERSION_OK:
                out.append(Flag("warn", "Cash a little behind profit", f"Operating cash flow was {conv:.0%} of net income {span}."))
            else:
                out.append(Flag("good", "Profit backed by cash", f"Operating cash flow was {conv:.0%} of net income {span}."))

        acc, when = _last(q["accruals_ratio"])
        if acc == acc and acc > ACCRUALS_WARN:
            out.append(Flag("bad" if acc > ACCRUALS_BAD else "warn", "High accruals",
                            f"Net income exceeded operating cash flow by {acc:.1%} of assets in {_year(when, stmts)}. In Sloan's "
                            "research, companies with high accruals saw their earnings and share prices fall back later."))

        for col, what, against in (("receivables_gap", "Receivables", "sales"), ("inventory_gap", "Inventory", "cost of sales")):
            gap, when = _last(q[col])
            size = q.loc[when, col.replace("gap", "share")] if when is not None else np.nan
            if gap == gap and gap > GAP_WARN and size == size and size >= MATERIAL:
                grew = q.loc[when, col.replace("gap", "growth")]
                out.append(Flag("bad" if gap > GAP_BAD else "warn", f"{what} growing faster than {against}",
                                f"{what} ({size:.0%} of assets) grew {grew:+.0%} in {_year(when, stmts)}, {gap * 100:.0f} points "
                                f"faster than {against}: " + ("customers may be paying later, or sales were pulled forward."
                                                             if what == "Receivables" else "unsold goods are piling up.")))

        dso = q["dso"].dropna()
        rec_size, _ = _last(q["receivables_share"])
        if len(dso) >= 4 and (dso.diff().iloc[-3:] > 0).all() and dso.iloc[-1] > 1.2 * dso.iloc[-4] and rec_size >= MATERIAL:
            out.append(Flag("warn", "Customers paying ever later",
                            f"Days of sales in receivables rose three periods in a row, {dso.iloc[-4]:.0f} → {dso.iloc[-1]:.0f} days."))

        capex = q["capex_to_da"].dropna().tail(3)
        if len(capex) >= 2 and capex.mean() < CAPEX_DA_LOW:
            out.append(Flag("warn", "Spending less than depreciation",
                            f"Capital spending averaged {capex.mean():.0%} of depreciation over {len(capex)} periods: free cash flow "
                            "looks better than it can stay if the assets are wearing out. Under IFRS / TFRS 16 depreciation also "
                            "covers leased stores, planes and offices, whose payments are not capital spending, so lease-heavy "
                            "companies (retail, airlines) read low here without under-investing."))

        sbc, when = _last(q["sbc_to_fcf"])
        sbc_rev, _ = _last(q["sbc_to_revenue"])
        fcf_last, _ = _last(q["fcf"])
        if sbc == sbc and sbc > SBC_FCF_WARN:
            out.append(Flag("bad" if sbc > SBC_FCF_BAD else "warn",
                            "Stock pay eats free cash flow" if sbc > SBC_FCF_BAD else "Stock pay is a real cost",
                            f"Stock-based pay was {sbc:.0%} of free cash flow in {_year(when, stmts)}: it is paid in shares instead "
                            "of cash, so free cash flow overstates what owners keep."))
        elif fcf_last == fcf_last and fcf_last <= 0 and sbc_rev == sbc_rev and sbc_rev > 0.05:
            out.append(Flag("bad", "Stock pay with no free cash flow",
                            f"Stock-based pay is {sbc_rev:.0%} of revenue while free cash flow is negative."))

    shares = line(stmts, "shares").dropna()
    shares = shares[shares > 0]
    if len(shares) >= 2:
        years = (shares.index[-1] - shares.index[0]).days / 365.25
        rate = (shares.iloc[-1] / shares.iloc[0]) ** (1 / years) - 1 if years > 0.5 else np.nan
        if rate == rate and rate > DILUTION_WARN:
            out.append(Flag("bad" if rate > DILUTION_BAD else "warn", "Shareholders being diluted",
                            f"The share count grew {rate:.1%} a year ({shares.index[0]:%Y} → {shares.index[-1]:%Y}): each share "
                            "owns a shrinking slice of the profit."))
        elif rate == rate and rate < -0.01:
            out.append(Flag("good", "Share count shrinking", f"The share count fell {abs(rate):.1%} a year ({shares.index[0]:%Y} → "
                                                             f"{shares.index[-1]:%Y}): buybacks raise each share's slice."))

    one_off, when = _last(q["unusual_share"])
    if one_off == one_off and abs(one_off) > UNUSUAL_WARN:
        if one_off > 0:
            out.append(Flag("warn", "One-off gains flatter profit", f"Unusual items added {one_off:.0%} of pre-tax profit in "
                                                                    f"{_year(when, stmts)}; underlying profit is lower."))
        else:
            out.append(Flag("info", "One-off charges depress profit", f"Unusual items cost {abs(one_off):.0%} of pre-tax profit in "
                                                                       f"{_year(when, stmts)}; underlying profit is higher."))

    if not bank and not stmts.quarterly:
        scores = beneish(stmts)
        if scores:
            last = scores[-1]
            tone = {"red flag": "bad", "grey zone": "warn", "no sign": "good"}[last.zone]
            top = last.drivers().index[0]
            growth_note = ""
            if last.zone != "no sign" and top == "SGI":
                tone = "warn"
                growth_note = (f" Most of the score comes from sales growth (SGI {last.indices['SGI']:.2f}): the model reads fast growth "
                               "as pressure to manage earnings, so fast growers often score high with nothing wrong. Look at the "
                               "other variables before reading anything into it.")
            elif last.zone != "no sign":
                growth_note = f" The biggest push comes from {BENEISH_NAMES[top].split(' (')[0].lower()} ({top} {last.indices[top]:.2f})."
            out.append(Flag(tone, f"Beneish M-score {last.m:.2f}: {last.zone}",
                            f"FY {last.period:%Y}. Above {BENEISH_FLAG} the pattern resembles companies later caught "
                            f"manipulating earnings; below {BENEISH_GREY} it does not." + growth_note
                            + (f" Missing data set to neutral: {', '.join(last.neutral)}." if last.neutral else "")))

    if not bank and wacc is not None:
        vc = value_creation(stmts, wacc)
        roic = vc.table["roic"].dropna().tail(3)
        if len(roic):
            avg = float(roic.mean())
            span = f"averaged {avg:.1%} over {len(roic)} {'periods' if len(roic) > 1 else 'period'}"
            if avg - wacc > SPREAD_OK:
                out.append(Flag("good", "Creates value", f"Return on invested capital {span}, above the {wacc:.1%} cost of capital."))
            elif avg < wacc:
                out.append(Flag("bad", "Earns less than its capital costs", f"Return on invested capital {span}, below the "
                                                                              f"{wacc:.1%} cost of capital: growth destroys value."))
            else:
                out.append(Flag("warn", "Barely covers its cost of capital", f"Return on invested capital {span}, against a "
                                                                              f"{wacc:.1%} cost of capital."))

    drivers = roe_drivers(dupont(stmts))
    if drivers is not None:
        start, end, roe0, roe1, contrib = drivers
        if roe1 - roe0 > 0.02 and contrib.get("Leverage", 0) > 0.5 * (roe1 - roe0):
            out.append(Flag("warn", "Return on equity rose mainly on debt",
                            f"ROE went from {roe0:.1%} to {roe1:.1%} ({start:%Y} → {end:%Y}); more than half of the rise came from "
                            "a thinner equity base rather than better margins or turnover."))

    order = {"bad": 0, "warn": 1, "info": 2, "good": 3}
    return sorted(out, key=lambda f: order[f.tone])
