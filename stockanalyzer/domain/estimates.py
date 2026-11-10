"""Analyst consensus for one company, in a shape that survives a round trip through JSON (the daily snapshot table).

Yahoo's estimate tables are keyed by period: ``0q`` (the quarter being reported), ``+1q``, ``0y`` (the fiscal year in
progress) and ``+1y``. ``0y`` is *not* the last completed year: its ``year_ago`` figure is.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

PERIODS = ("0q", "+1q", "0y", "+1y")
PERIOD_NAMES = {"0q": "Current quarter", "+1q": "Next quarter", "0y": "Current fiscal year", "+1y": "Next fiscal year"}


@dataclass
class Estimates:
    symbol: str
    currency: str | None = None
    # period -> {avg, low, high, year_ago, analysts, growth}; a period nobody covers is left out
    eps: dict[str, dict[str, float]] = field(default_factory=dict)
    revenue: dict[str, dict[str, float]] = field(default_factory=dict)
    # period -> {current, d7, d30, d60, d90}: the average EPS estimate now and that many days ago
    eps_trend: dict[str, dict[str, float]] = field(default_factory=dict)
    # period -> {up7, up30, down7, down30}: analysts who raised / cut their EPS estimate in the last 7 / 30 days
    eps_revisions: dict[str, dict[str, float]] = field(default_factory=dict)
    target: dict[str, float] = field(default_factory=dict)  # mean, median, high, low, analysts
    recommendation: dict[str, object] = field(default_factory=dict)  # mean (1 strong buy .. 5 sell), key, analysts
    as_of: str | None = None  # ISO time the figures were fetched

    @property
    def is_empty(self) -> bool:
        return not (self.eps or self.revenue or self.target or self.recommendation.get("mean"))

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict) -> "Estimates":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
