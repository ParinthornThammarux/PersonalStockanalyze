"""Quality / distress scores: Piotroski F-score and Altman Z."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from stockanalyzer.analysis.fundamental.statements import line
from stockanalyzer.domain.financials import FinancialStatements


@dataclass
class Criterion:
    name: str
    passed: bool | None  # None = not computable from the data we have
    current: float | None = None
    previous: float | None = None


@dataclass
class PiotroskiResult:
    score: int
    available: int
    criteria: list[Criterion] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.available < 7:
            return "insufficient data"
        pct = self.score / self.available
        return "strong" if pct >= 8 / 9 else "weak" if pct <= 2 / 9 else "average"


def _two(series: pd.Series) -> tuple[float | None, float | None]:
    s = series.dropna()
    if len(s) < 2:
        return (float(s.iloc[-1]) if len(s) else None), None
    return float(s.iloc[-1]), float(s.iloc[-2])


def piotroski_f_score(stmts: FinancialStatements) -> PiotroskiResult:
    """Nine binary signals on the two most recent periods (profitability, leverage, efficiency).

    Simplification vs the paper: ROA uses end-of-period assets, and "leverage" is long-term
    debt / total assets where *not increasing* earns the point (a debt-free company would
    otherwise score 0 forever).
    """
    ni, cfo, assets = line(stmts, "net_income"), line(stmts, "cfo"), line(stmts, "total_assets")
    ltd = line(stmts, "long_term_debt").fillna(0)
    ca, cl = line(stmts, "current_assets"), line(stmts, "current_liabilities")
    shares, revenue, gp = line(stmts, "shares"), line(stmts, "revenue"), line(stmts, "gross_profit")

    roa = ni / assets
    leverage = ltd / assets
    current_ratio = ca / cl
    gross_margin = gp / revenue
    turnover = revenue / assets

    def cmp(name, series, op):
        cur, prev = _two(series)
        if cur is None or prev is None:
            return Criterion(name, None, cur, prev)
        return Criterion(name, bool(op(cur, prev)), cur, prev)

    def positive(name, series):
        cur, _ = _two(series)
        return Criterion(name, None if cur is None else cur > 0, cur)

    cfo_cur, _ = _two(cfo)
    ni_cur, _ = _two(ni)
    accrual = Criterion(
        "CFO > net income (earnings quality)",
        None if cfo_cur is None or ni_cur is None else cfo_cur > ni_cur,
        cfo_cur, ni_cur,
    )
    criteria = [
        positive("ROA > 0", roa),
        positive("Operating cash flow > 0", cfo),
        cmp("ROA improved", roa, lambda a, b: a > b),
        accrual,
        cmp("Leverage did not increase", leverage, lambda a, b: a <= b + 1e-12),
        cmp("Current ratio improved", current_ratio, lambda a, b: a > b),
        cmp("No share dilution", shares, lambda a, b: a <= b * 1.005),
        cmp("Gross margin improved", gross_margin, lambda a, b: a > b),
        cmp("Asset turnover improved", turnover, lambda a, b: a > b),
    ]
    scored = [c for c in criteria if c.passed is not None]
    return PiotroskiResult(score=sum(c.passed for c in scored), available=len(scored), criteria=criteria)


@dataclass
class AltmanResult:
    z: float
    zone: str  # "safe" | "grey" | "distress"
    variant: str
    components: dict[str, float]
    note: str = ""


def altman_z(
    stmts: FinancialStatements, market_cap: float | None = None, *, manufacturer: bool = True
) -> AltmanResult | None:
    """Altman Z (public manufacturers) or Z'' (everything else, no sales term).

    Without ``market_cap`` book equity is used for the equity term (flagged in ``note``).
    Not meaningful for banks/insurers. Returns ``None`` if the statements lack inputs.
    """
    def last(key: str) -> float | None:
        s = line(stmts, key).dropna()
        return float(s.iloc[-1]) if len(s) else None

    ta, tl = last("total_assets"), last("total_liabilities")
    ca, cl = last("current_assets"), last("current_liabilities")
    re_, ebit = last("retained_earnings"), last("ebit")
    if ebit is None:
        ebit = last("operating_income")
    sales, book_equity = last("revenue"), last("equity")
    if None in (ta, tl, ca, cl, re_, ebit) or not ta or not tl:
        return None

    note = ""
    equity_value = market_cap
    if equity_value is None:
        equity_value, note = book_equity, "book equity used instead of market cap"
    if equity_value is None:
        return None

    x1, x2, x3 = (ca - cl) / ta, re_ / ta, ebit / ta
    if manufacturer:
        if sales is None:
            return None
        x4, x5 = equity_value / tl, sales / ta
        z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
        safe, distress, variant = 2.99, 1.81, "Z (manufacturers)"
        comps = {"WC/TA": x1, "RE/TA": x2, "EBIT/TA": x3, "Equity/TL": x4, "Sales/TA": x5}
    else:
        x4 = (book_equity if book_equity is not None else equity_value) / tl
        z = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
        safe, distress, variant = 2.6, 1.1, "Z'' (non-manufacturers)"
        comps = {"WC/TA": x1, "RE/TA": x2, "EBIT/TA": x3, "BookEquity/TL": x4}
        note = ""
    zone = "safe" if z > safe else "distress" if z < distress else "grey"
    return AltmanResult(float(z), zone, variant, {k: float(v) for k, v in comps.items()}, note)
