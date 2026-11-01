"""Sanity checks and light repairs for OHLCV data before it enters the store."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from stockanalyzer.data.providers.base import is_intraday


@dataclass
class QualityReport:
    rows_in: int = 0
    rows_out: int = 0
    dropped_invalid: int = 0  # NaN / non-positive prices
    repaired_ohlc: int = 0  # high/low widened to contain open & close
    gaps: list[tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=list)
    big_moves: list[tuple[pd.Timestamp, float]] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (self.dropped_invalid or self.repaired_ohlc or self.gaps or self.big_moves)

    def summary(self) -> str:
        parts = [f"{self.rows_in}->{self.rows_out} rows"]
        if self.dropped_invalid:
            parts.append(f"{self.dropped_invalid} invalid dropped")
        if self.repaired_ohlc:
            parts.append(f"{self.repaired_ohlc} OHLC repaired")
        if self.gaps:
            parts.append(f"{len(self.gaps)} gaps")
        if self.big_moves:
            parts.append(f"{len(self.big_moves)} moves >threshold")
        return ", ".join(parts)


def clean_ohlcv(
    df: pd.DataFrame,
    *,
    interval: str = "1d",
    max_move: float = 0.5,
    max_gap_days: int = 7,
    require_positive: bool = True,
) -> tuple[pd.DataFrame, QualityReport]:
    """Drop impossible rows, repair inconsistent high/low, flag gaps and extreme moves.

    Extreme moves and gaps are only *reported*: they are often real (earnings gaps,
    holidays, halts) and silently altering them would corrupt the series.
    ``require_positive=False`` is for macro/rate series, which can legitimately be <= 0
    (spreads, policy rates); percentage moves are meaningless there so they are not flagged.
    """
    report = QualityReport(rows_in=len(df))
    if df.empty:
        return df, report

    out = df.copy()
    price_cols = ["open", "high", "low", "close"]
    valid = out[price_cols].notna().all(axis=1)
    if require_positive:
        valid &= out[price_cols].gt(0).all(axis=1)
    report.dropped_invalid = int((~valid).sum())
    out = out[valid]

    hi = out[["open", "high", "close"]].max(axis=1)
    lo = out[["open", "low", "close"]].min(axis=1)
    bad = (hi > out["high"]) | (lo < out["low"])
    report.repaired_ohlc = int(bad.sum())
    out = out.assign(high=hi, low=lo)

    if len(out) > 1:
        if require_positive:
            moves = out["adj_close"].pct_change()
            report.big_moves = [(ts, float(m)) for ts, m in moves[moves.abs() > max_move].items()]
        if not is_intraday(interval):
            gap = out.index.to_series().diff()
            for end, delta in gap[gap > pd.Timedelta(days=max_gap_days)].items():
                report.gaps.append((end - delta, end))

    report.rows_out = len(out)
    return out, report
