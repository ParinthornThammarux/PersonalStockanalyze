"""Market breadth straight from the price store: per day, how many stocks are above their averages, at new highs or lows,
up or down. The counting runs inside DuckDB (window functions per symbol), so the whole US market takes about 2 seconds
and the Thai market half a second - no cache file is needed.

Only today's listed stocks are in the store (delisted ones are missing), so past breadth leans a little positive:
the stocks that fell out of the market are not there to drag the counts down.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd

from stockanalyzer.analysis import breadth as calc
from stockanalyzer.data import universe as uni
from stockanalyzer.data.sectors import THAI_SECTORS
from stockanalyzer.data.store import PriceStore

LOOKBACK_MONTH = 21  # trading days in "a month ago"
STALE_DAYS = 7  # a symbol whose last bar is older than this (calendar days) before the market date is not counted


@dataclass(frozen=True)
class BreadthMarket:
    key: str
    label: str
    currency: str
    benchmark: str
    benchmark_name: str
    floors: tuple[tuple[str, float], ...]  # (label, minimum average traded value a day in the local currency)
    default_floor: int = 1


US_MARKET = BreadthMarket(
    "us", "US stocks", "USD", "SPY", "S&P 500 (SPY)",
    (("Every stock", 0.0), ("Trades at least $1M a day", 1e6), ("Trades at least $5M a day (the Screener's floor)", 5e6)),
)
TH_MARKET = BreadthMarket(
    "th", "Thai stocks (SET and mai)", "THB", "TDEX.BK", "SET50 ETF (TDEX)",
    (("Every stock", 0.0), ("Trades at least ฿2M a day", 2e6), ("Trades at least ฿10M a day (the Screener's floor)", 1e7)),
)
MARKETS: dict[str, BreadthMarket] = {m.key: m for m in (US_MARKET, TH_MARKET)}


def market_universe(key: str) -> pd.DataFrame:
    """The market's stock list (SPACs left out: they sit still at their trust value and would only add 'unchanged' days)."""
    table = uni.load_th_stocks() if key == "th" else uni.load_us_stocks()
    return table[table["industry"] != uni.SPAC_INDUSTRY].reset_index(drop=True)


_FEATURES = """
WITH b AS (
  SELECT symbol, ts, close, adj_close,
    avg(close) OVER w50 AS sma50, count(close) OVER w50 AS n50,
    avg(close) OVER w200 AS sma200, count(close) OVER w200 AS n200,
    max(close) OVER w252 AS hi252, min(close) OVER w252 AS lo252, count(close) OVER w252 AS n252,
    lag(close) OVER ws AS prev, lag(adj_close) OVER ws AS prev_adj,
    adj_close / lag(adj_close, 63) OVER ws - 1 AS ret63,
    avg(close * volume) OVER w20 AS tv20
  FROM read_parquet(?, union_by_name=true)
  WHERE close > 0
  WINDOW ws AS (PARTITION BY symbol ORDER BY ts),
         w20 AS (PARTITION BY symbol ORDER BY ts ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
         w50 AS (PARTITION BY symbol ORDER BY ts ROWS BETWEEN 49 PRECEDING AND CURRENT ROW),
         w200 AS (PARTITION BY symbol ORDER BY ts ROWS BETWEEN 199 PRECEDING AND CURRENT ROW),
         w252 AS (PARTITION BY symbol ORDER BY ts ROWS BETWEEN 251 PRECEDING AND CURRENT ROW)
)"""

_BENCH_JOIN = """LEFT JOIN (
  SELECT ts AS bts, adj_close / lag(adj_close, 63) OVER (ORDER BY ts) - 1 AS bret63 FROM read_parquet(?)
) m ON m.bts = b.ts"""

_DAILY = _FEATURES + """, j AS (SELECT b.*, {bench_col} AS bret63 FROM b {bench_join})
SELECT ts,
  count(*) AS n,
  count(*) FILTER (WHERE n50 = 50) AS n50,
  count(*) FILTER (WHERE n50 = 50 AND close > sma50) AS above50,
  count(*) FILTER (WHERE n200 = 200) AS n200,
  count(*) FILTER (WHERE n200 = 200 AND close > sma200) AS above200,
  count(*) FILTER (WHERE n252 = 252) AS n252,
  count(*) FILTER (WHERE n252 = 252 AND close >= hi252) AS new_highs,
  count(*) FILTER (WHERE n252 = 252 AND close <= lo252) AS new_lows,
  count(*) FILTER (WHERE close > prev) AS adv,
  count(*) FILTER (WHERE close < prev) AS dec,
  count(*) FILTER (WHERE close = prev) AS unch,
  sum(least(greatest(adj_close / prev_adj - 1, ?), ?)) FILTER (WHERE prev_adj > 0) AS ret_sum,
  count(*) FILTER (WHERE prev_adj > 0) AS ret_n,
  sum(least(greatest(ret63, ?), ?)) FILTER (WHERE ret63 IS NOT NULL) AS ret63_sum,
  count(ret63) AS ret63_n,
  count(*) FILTER (WHERE ret63 IS NOT NULL AND bret63 IS NOT NULL) AS n63,
  count(*) FILTER (WHERE ret63 > bret63) AS beat63
FROM j
WHERE tv20 >= ?
GROUP BY ts
"""

_STATES = _FEATURES + """, s AS (
  SELECT symbol, ts, tv20, n200,
    CASE WHEN n50 = 50 THEN close > sma50 END AS above50,
    CASE WHEN n200 = 200 THEN close > sma200 END AS above200,
    CASE WHEN n252 = 252 THEN close >= hi252 END AS new_high,
    CASE WHEN n252 = 252 THEN close <= lo252 END AS new_low,
    adj_close / lag(adj_close, {lag}) OVER ws - 1 AS ret_month,
    lag(CASE WHEN n200 = 200 THEN close > sma200 END, {lag}) OVER ws AS above200_before
  FROM b
  WINDOW ws AS (PARTITION BY symbol ORDER BY ts)
)
SELECT symbol, ts, tv20, above50, above200, new_high, new_low, ret_month, above200_before
FROM s
WHERE ts <= ?
QUALIFY row_number() OVER (PARTITION BY symbol ORDER BY ts DESC) = 1
"""


def daily_counts(store: PriceStore, symbols: Iterable[str], *, min_traded: float = 0.0, benchmark: str | None = None) -> pd.DataFrame:
    """Breadth counts per trading day for ``symbols`` that traded at least ``min_traded`` a day (20-day average, local
    currency, judged each day with what was known then). Thin early history and half-updated last days are dropped.
    With a ``benchmark`` in the store, ``beat63`` counts the stocks that beat it over the last 63 trading days."""
    files = store._files(list(symbols), "1d")
    if not files:
        return pd.DataFrame(columns=calc.COUNT_COLUMNS)
    low, high = calc.RETURN_CLIP
    low3, high3 = calc.RETURN_CLIP_3M
    bench_path = store.path_for(benchmark) if benchmark else None
    if bench_path is not None and bench_path.exists():
        sql = _DAILY.format(bench_col="m.bret63", bench_join=_BENCH_JOIN)
        params = [str(bench_path), low, high, low3, high3, float(min_traded)]
    else:
        sql = _DAILY.format(bench_col="CAST(NULL AS DOUBLE)", bench_join="")
        params = [low, high, low3, high3, float(min_traded)]
    parts = store._query_files(sql, files, params)
    counts = pd.concat(parts, ignore_index=True).groupby("ts").sum(numeric_only=True).sort_index()
    counts.index = pd.DatetimeIndex(counts.index.astype("datetime64[ns]"), name="ts")
    return calc.complete_days(counts)


def symbol_states(store: PriceStore, symbols: Iterable[str], asof, *, min_traded: float = 0.0) -> pd.DataFrame:
    """Each stock's position on ``asof`` (its last bar at or before it): above its 50 / 200-day average, at a 52-week
    high or low, its return over the last month and whether it was above the 200-day average a month earlier."""
    files = store._files(list(symbols), "1d")
    cols = ["ts", "tv20", "above50", "above200", "new_high", "new_low", "ret_month", "above200_before"]
    if not files:
        return pd.DataFrame(columns=cols)
    asof = pd.Timestamp(asof)
    parts = store._query_files(_STATES.format(lag=LOOKBACK_MONTH), files, [asof.to_pydatetime()])
    states = pd.concat(parts, ignore_index=True).set_index("symbol")
    states["ts"] = states["ts"].astype("datetime64[ns]")
    fresh = states["ts"] >= asof - pd.Timedelta(days=STALE_DAYS)
    return states.loc[fresh & (states["tv20"] >= min_traded), cols]


def benchmark_close(store: PriceStore, symbol: str) -> pd.Series:
    """Dividend-adjusted close of the market's benchmark (empty when it is not in the store)."""
    df = store.read(symbol)
    return df["adj_close"].dropna() if not df.empty else pd.Series(dtype="float64")


def groups(key: str, table: pd.DataFrame, fundamentals: pd.DataFrame | None = None) -> dict[str, str]:
    """Symbol -> group the industry table rolls up to: Yahoo's sector, or for Thai stocks the app's sector baskets first
    (only a few dozen Thai shares carry an industry until their full fundamentals are fetched)."""
    industry = dict(zip(table["symbol"], table["industry"]))
    if fundamentals is not None and not fundamentals.empty and "industry" in fundamentals.columns:
        for symbol, value in fundamentals["industry"].dropna().items():
            if not industry.get(symbol):
                industry[symbol] = value
    out = {s: g for s, i in industry.items() if (g := uni.sector_of(i))}
    if key == "th":
        out.update({m: name for name, members in THAI_SECTORS.sectors.items() for m in members})
    return out


def industries(table: pd.DataFrame, fundamentals: pd.DataFrame | None = None) -> dict[str, str]:
    """Symbol -> Yahoo industry (the finer level under the sector)."""
    out = {s: i for s, i in zip(table["symbol"], table["industry"]) if uni.sector_of(i)}
    if fundamentals is not None and not fundamentals.empty and "industry" in fundamentals.columns:
        for symbol, value in fundamentals["industry"].dropna().items():
            if symbol not in out and uni.sector_of(value):
                out[symbol] = value
    return out


def load(store: PriceStore, key: str, *, min_traded: float = 0.0) -> calc.Breadth | None:
    """Everything the breadth page shows for one market, or ``None`` when the store holds no prices for it."""
    market = MARKETS[key]
    symbols = market_universe(key)["symbol"].tolist()
    counts = daily_counts(store, symbols, min_traded=min_traded, benchmark=market.benchmark)
    if counts.empty or counts["n200"].max() == 0:
        return None
    bench = benchmark_close(store, market.benchmark)
    return calc.Breadth(calc.indicators(counts, bench if not bench.empty else None), has_benchmark=not bench.empty)


def group_table(store: PriceStore, key: str, asof, labels: Mapping[str, str], *, min_traded: float = 0.0,
                min_stocks: int = 3) -> tuple[pd.DataFrame, int, int]:
    """Participation per group on ``asof``, the number of counted stocks that have a group, and the number counted."""
    states = symbol_states(store, market_universe(key)["symbol"].tolist(), asof, min_traded=min_traded)
    placed = int(sum(1 for s in states.index if labels.get(s)))
    return calc.group_breadth(states, labels, min_stocks=min_stocks), placed, len(states)
