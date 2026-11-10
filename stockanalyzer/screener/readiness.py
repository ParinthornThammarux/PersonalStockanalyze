"""Can a screen be judged with the fundamentals loaded so far, and which stocks would settle it?

A market-wide (light) snapshot has P/E, P/B, market cap and yield; ROE, margins, cash flow and debt come only from the
full, per-stock fetch. A rule on one of those columns cannot match a stock that has only a light snapshot, so a screen
built on them returns nothing until some are fetched - and fetching only for the *matches*, of which there are none,
goes nowhere. ``readiness`` names the stocks worth fetching instead: those that already pass every rule that *can* be judged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from stockanalyzer.data.fundamentals import FULL_ONLY_FIELDS
from stockanalyzer.screener.rules import Rule, Screen

SECONDS_PER_STOCK = 0.05  # measured on 2026-11-02: 400 stocks in 15 s with 4 requests in parallel, no throttling


@dataclass
class Readiness:
    needs_full: list[Rule] = field(default_factory=list)  # rules that only a full snapshot can answer
    covered: int = 0  # stocks in the pool that have a full snapshot
    total: int = 0
    candidates: list[str] = field(default_factory=list)  # to fetch, largest market cap first

    @property
    def blocked(self) -> bool:
        """True when the screen needs financials and no stock in the pool has them yet."""
        return bool(self.needs_full) and self.covered == 0

    @property
    def partial(self) -> bool:
        return bool(self.needs_full) and 0 < self.covered < self.total


def readiness(rules: list[Rule], pool: pd.DataFrame) -> Readiness:
    needs = [r for r in rules if r.field in FULL_ONLY_FIELDS]
    full = pool["fundamentals"].eq("full") if "fundamentals" in pool.columns else pd.Series(False, index=pool.index)
    out = Readiness(needs_full=needs, covered=int(full.sum()), total=len(pool))
    if not needs:
        return out
    judgeable = Screen("judgeable", [r for r in rules if r.field not in FULL_ONLY_FIELDS]).mask(pool)
    waiting = pool[judgeable & ~full]
    if "market_cap" in waiting.columns:
        waiting = waiting.sort_values("market_cap", ascending=False, na_position="last")
    out.candidates = list(waiting.index)
    return out


def fetch_seconds(count: int) -> int:
    """A rounded-up wall-clock estimate for fetching ``count`` full snapshots (5 s steps, at least 5)."""
    return max(5, int(-(-count * SECONDS_PER_STOCK // 5) * 5))
