"""Financial-statement container shared by providers and the fundamental analysis code."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


def _empty() -> pd.DataFrame:
    return pd.DataFrame()


@dataclass
class FinancialStatements:
    """Rows are line items (Yahoo labels), columns are period-end dates, oldest first."""

    symbol: str
    income: pd.DataFrame = field(default_factory=_empty)
    balance: pd.DataFrame = field(default_factory=_empty)
    cashflow: pd.DataFrame = field(default_factory=_empty)
    quarterly: bool = False

    @property
    def is_empty(self) -> bool:
        return self.income.empty and self.balance.empty and self.cashflow.empty

    @property
    def periods(self) -> pd.DatetimeIndex:
        """Union of the periods of all three statements (Yahoo often reports different spans)."""
        frames = [df for df in (self.income, self.balance, self.cashflow) if not df.empty]
        if not frames:
            return pd.DatetimeIndex([])
        return pd.DatetimeIndex(sorted(set().union(*(df.columns for df in frames))))
