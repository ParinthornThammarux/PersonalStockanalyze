"""Every ex-dividend (XD) day in the price store with the prices around it, extracted inside DuckDB so a whole market's
events (tens of thousands) come out in seconds: only the bars from 61 days before to 60 days after each XD are touched.

Prices are Yahoo's raw closes (split-adjusted, not dividend-adjusted), which is what the XD gap is measured on; the
dividend-adjusted close is used for returns that should include the dividend.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from stockanalyzer.data.store import PriceStore

BEFORE, AFTER = 61, 60  # trading days of context kept around each XD day
COLUMNS = ["symbol", "ex_date", "amount", "prev_close", "close_m11", "open0", "close0", "adj_m1", "adj_p20", "ts_m1", "ts_m11",
           "ts_p20", "tv60", "n_before", "k_max", "fill_days"]

_SQL = f"""
WITH p AS (
  SELECT symbol, ts, open, close, adj_close, close * volume AS tv, dividends,
         row_number() OVER (PARTITION BY symbol ORDER BY ts) AS rn
  FROM read_parquet(?, union_by_name=true)
  WHERE close > 0
),
e AS (SELECT symbol, rn AS ern, ts AS ex_date, dividends AS amount FROM p WHERE dividends > 0 AND ts >= ?),
w AS (
  SELECT e.symbol, e.ex_date, e.amount, p.rn - e.ern AS k, p.ts, p.open, p.close, p.adj_close, p.tv
  FROM e JOIN p ON p.symbol = e.symbol AND p.rn BETWEEN e.ern - {BEFORE} AND e.ern + {AFTER}
),
base AS (
  SELECT symbol, ex_date, any_value(amount) AS amount,
    max(close) FILTER (WHERE k = -1) AS prev_close,
    max(close) FILTER (WHERE k = -11) AS close_m11,
    max(open) FILTER (WHERE k = 0) AS open0,
    max(close) FILTER (WHERE k = 0) AS close0,
    max(adj_close) FILTER (WHERE k = -1) AS adj_m1,
    max(adj_close) FILTER (WHERE k = 20) AS adj_p20,
    max(ts) FILTER (WHERE k = -1) AS ts_m1,
    max(ts) FILTER (WHERE k = -11) AS ts_m11,
    max(ts) FILTER (WHERE k = 20) AS ts_p20,
    avg(tv) FILTER (WHERE k BETWEEN -60 AND -1) AS tv60,
    count(*) FILTER (WHERE k BETWEEN -60 AND -1) AS n_before,
    max(k) AS k_max
  FROM w GROUP BY symbol, ex_date
),
fill AS (
  SELECT w.symbol, w.ex_date, min(w.k) AS fill_days
  FROM w JOIN base ON w.symbol = base.symbol AND w.ex_date = base.ex_date
  WHERE w.k >= 0 AND w.close >= base.prev_close
  GROUP BY w.symbol, w.ex_date
)
SELECT base.*, fill.fill_days
FROM base LEFT JOIN fill ON base.symbol = fill.symbol AND base.ex_date = fill.ex_date
"""


def xd_events(store: PriceStore, symbols: Iterable[str], *, start="1990-01-01") -> pd.DataFrame:
    """One row per ex-dividend day of ``symbols``: the dividend, the closes and opens around it, the traded value over the
    60 days before, and ``fill_days`` - trading days until the close first got back to the close before the XD
    (0 = the same day, NaN = not within the days available, at most 60; ``k_max`` says how many there were)."""
    files = store._files(list(symbols), "1d")
    if not files:
        return pd.DataFrame(columns=COLUMNS)
    parts = store._query_files(_SQL, files, [pd.Timestamp(start).to_pydatetime()])
    out = pd.concat(parts, ignore_index=True)
    if out.empty:
        return pd.DataFrame(columns=COLUMNS)
    dates = ("ex_date", "ts_m1", "ts_m11", "ts_p20")
    for col in COLUMNS:
        if col in dates:
            out[col] = pd.to_datetime(out[col]).astype("datetime64[ns]")
        elif col != "symbol":  # DuckDB hands counts back as nullable integers; plain floats keep NaN arithmetic simple
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out[COLUMNS].sort_values(["symbol", "ex_date"], ignore_index=True)
