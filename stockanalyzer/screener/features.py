"""Per-symbol feature snapshot (latest value of ~50 indicators/statistics) for screening.

``compute_features`` is a pure function of one OHLCV frame. ``compute_universe_features``
reads the whole store with a single DuckDB query and fans the symbols out over processes.

Units: returns, distances, volatilities and margins are *fractions* (0.05 = 5%);
oscillators (RSI, ADX, stochastic, MFI) keep their natural 0-100 scale.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable

import numpy as np
import pandas as pd

from stockanalyzer.analysis.technical import indicators as ind
from stockanalyzer.analysis.technical import signals, sr_zones
from stockanalyzer.data.service import adjust_ohlc
from stockanalyzer.data.store import PriceStore
from stockanalyzer.domain.assets import DEFAULT_BENCHMARK, AssetClass, benchmark_for, infer_asset_class

log = logging.getLogger(__name__)

MIN_BARS = 60
NAN = float("nan")
READ_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]

# column -> (label, format) ; formats: pct, num, int, usd, bool, date, text
FEATURE_INFO: dict[str, tuple[str, str]] = {
    "last_date": ("Last bar", "date"), "bars": ("Bars", "int"), "close": ("Close", "num"),
    "dollar_volume_20": ("Avg $ volume (20d)", "usd"), "volume_ratio": ("Volume vs 20d avg", "num"),
    "ret_1d": ("1d return", "pct"), "ret_5d": ("1w return", "pct"), "ret_21d": ("1m return", "pct"),
    "ret_63d": ("3m return", "pct"), "ret_126d": ("6m return", "pct"), "ret_252d": ("12m return", "pct"),
    "mom_12_1": ("Momentum 12-1", "pct"),
    "sma20_dist": ("vs SMA20", "pct"), "sma50_dist": ("vs SMA50", "pct"), "sma200_dist": ("vs SMA200", "pct"),
    "above_sma50": ("Above SMA50", "bool"), "above_sma200": ("Above SMA200", "bool"),
    "golden_cross": ("SMA50 > SMA200", "bool"), "trend_score": ("Trend score (-5..5)", "int"),
    "trend_label": ("Trend", "text"), "adx14": ("ADX(14)", "num"), "trend_r2": ("Trend R² (60d)", "num"),
    "trend_slope_annual": ("Trend slope (annualised)", "pct"),
    "rsi14": ("RSI(14)", "num"), "stoch_k": ("Stochastic %K", "num"), "mfi14": ("MFI(14)", "num"),
    "macd_hist_pct": ("MACD hist / price", "pct"), "roc_12": ("ROC(12)", "pct"),
    "atr_pct": ("ATR(14) / price", "pct"), "hvol_20": ("Volatility 20d", "pct"),
    "hvol_60": ("Volatility 60d", "pct"), "bb_pctb": ("Bollinger %B", "num"),
    "bb_bandwidth": ("Bollinger width", "pct"),
    "pct_from_52w_high": ("From 52w high", "pct"), "pct_from_52w_low": ("From 52w low", "pct"),
    "new_52w_high": ("New 52w high", "bool"), "drawdown_1y_max": ("Max drawdown 1y", "pct"),
    "drawdown_now": ("Drawdown now", "pct"), "beta_1y": ("Beta 1y", "num"), "rs_126": ("Rel. strength 6m", "pct"),
    "support_dist_atr": ("Support distance (ATR)", "num"), "resistance_dist_atr": ("Resistance distance (ATR)", "num"),
    "support_dist_pct": ("Support distance", "pct"), "resistance_dist_pct": ("Resistance distance", "pct"),
    "n_zones": ("S/R zones", "int"),
}


def _last(s: pd.Series) -> float:
    v = s.iloc[-1] if len(s) else NAN
    return float(v) if pd.notna(v) else NAN


def compute_features(
    df: pd.DataFrame, benchmark: pd.Series | None = None, *, include_zones: bool = True
) -> dict | None:
    """Latest-bar feature dict, or ``None`` if there is not enough history."""
    if df is None or len(df) < MIN_BARS:
        return None
    d = adjust_ohlc(df)
    close, high, low = d["close"], d["high"], d["low"]
    volume = df["volume"]
    n, last = len(d), float(close.iloc[-1])

    def ret(k: int) -> float:
        return last / float(close.iloc[-1 - k]) - 1 if n > k else NAN

    sma = {k: ind.sma(close, k) for k in (20, 50, 200)}
    macd = ind.macd(close)
    stoch = ind.stochastic(high, low, close)
    bb = ind.bollinger(close)
    adx = ind.adx(high, low, close)
    atr = _last(ind.atr(high, low, close))
    trend = signals.trend_state(d)
    year = d.tail(252)
    hi_52, lo_52 = float(year["high"].max()), float(year["low"].min())
    prior_high = float(high.iloc[-253:-1].max()) if n > 1 else NAN
    equity = (close.tail(253) / close.tail(253).iloc[0])
    dd = equity / equity.cummax() - 1

    def dist(k: int) -> float:
        v = _last(sma[k])
        return last / v - 1 if v == v and v else NAN

    def above(k: int) -> float | bool:
        """Price over its SMA(k); unknown (NaN) while the stock has fewer than k bars, never a made-up False."""
        d = dist(k)
        return NAN if d != d else bool(d > 0)

    fast, slow = _last(sma[50]), _last(sma[200])
    votes = max(int(trend["available"]), 1)

    daily = close.pct_change(fill_method=None)
    beta = rs = NAN
    if benchmark is not None:
        bench = benchmark.reindex(d.index).ffill()
        pair = pd.concat([daily, bench.pct_change(fill_method=None)], axis=1).dropna().tail(252)
        if len(pair) >= 60 and pair.iloc[:, 1].var() > 0:
            beta = float(pair.iloc[:, 0].cov(pair.iloc[:, 1]) / pair.iloc[:, 1].var())
        if len(bench.dropna()) > 126 and bench.iloc[-127] > 0:
            rs = ret(126) - (float(bench.iloc[-1]) / float(bench.iloc[-127]) - 1)

    feat = {
        "last_date": d.index[-1], "bars": n, "close": float(df["close"].iloc[-1]),
        "dollar_volume_20": float((df["close"] * volume).tail(20).mean()),
        "volume_ratio": float(volume.iloc[-1] / volume.iloc[-21:-1].mean()) if n > 21 and volume.iloc[-21:-1].mean() > 0 else NAN,
        "ret_1d": ret(1), "ret_5d": ret(5), "ret_21d": ret(21), "ret_63d": ret(63),
        "ret_126d": ret(126), "ret_252d": ret(252),
        "mom_12_1": float(close.iloc[-22] / close.iloc[-253] - 1) if n >= 253 else NAN,
        "sma20_dist": dist(20), "sma50_dist": dist(50), "sma200_dist": dist(200),
        "above_sma50": above(50), "above_sma200": above(200),
        "golden_cross": NAN if fast != fast or slow != slow else bool(fast > slow),
        # fewer than five votes (young stock) is scaled back to -5..5 so it ranks against the others
        "trend_score": int(np.sign(trend["score"]) * np.floor(abs(trend["score"]) * 5 / votes + 0.5)),
        "trend_label": trend["label"],
        "adx14": _last(adx["adx"]), "trend_r2": trend["r2"],
        "trend_slope_annual": trend["slope_annual_pct"] / 100 if trend["slope_annual_pct"] == trend["slope_annual_pct"] else NAN,
        "rsi14": _last(ind.rsi(close)), "stoch_k": _last(stoch["stoch_k"]),
        "mfi14": _last(ind.mfi(high, low, close, volume)),
        "macd_hist_pct": _last(macd["macd_hist"]) / last,
        "roc_12": _last(ind.roc(close, 12)) / 100,
        "atr_pct": atr / last,
        "hvol_20": _last(ind.historical_volatility(close, 20)) / 100,
        "hvol_60": _last(ind.historical_volatility(close, 60)) / 100,
        "bb_pctb": _last(bb["bb_pctb"]), "bb_bandwidth": _last(bb["bb_bandwidth"]),
        "pct_from_52w_high": last / hi_52 - 1, "pct_from_52w_low": last / lo_52 - 1,
        "new_52w_high": bool(last >= prior_high),
        "drawdown_1y_max": float(dd.min()), "drawdown_now": float(dd.iloc[-1]),
        "beta_1y": beta, "rs_126": rs,
        "support_dist_atr": NAN, "resistance_dist_atr": NAN,
        "support_dist_pct": NAN, "resistance_dist_pct": NAN, "n_zones": 0,
    }

    if include_zones:
        zones = sr_zones.build_zones(d, lookback=min(300, n))
        near = sr_zones.nearest_levels(zones, last)
        feat["n_zones"] = len(zones)
        if near["support"] is not None and atr > 0:
            feat["support_dist_pct"] = near["support_dist_pct"] / 100
            feat["support_dist_atr"] = (last - near["support"].high) / atr
        if near["resistance"] is not None and atr > 0:
            feat["resistance_dist_pct"] = near["resistance_dist_pct"] / 100
            feat["resistance_dist_atr"] = (near["resistance"].low - last) / atr
    return feat


def _compute_chunk(args: tuple) -> list[tuple[str, dict | None]]:
    chunk, benchmarks, include_zones = args
    out = []
    for symbol, frame, bench_key in chunk:
        try:
            out.append((symbol, compute_features(frame, benchmarks.get(bench_key), include_zones=include_zones)))
        except Exception as exc:  # one malformed symbol must not sink the run
            log.warning("features failed for %s: %s", symbol, exc)
            out.append((symbol, None))
    return out


def compute_universe_features(
    store: PriceStore,
    symbols: Iterable[str] | None = None,
    interval: str = "1d",
    *,
    last_n: int = 400,
    benchmark: str | None = "SPY",
    workers: int = 0,
    chunk_size: int = 150,
    include_zones: bool = True,
    local_benchmarks: bool = True,
) -> pd.DataFrame:
    """Feature table indexed by symbol for every symbol in the store (or ``symbols``).

    ``beta_1y`` and ``rs_126`` are measured against ``benchmark`` - except for symbols of a market with its own
    benchmark (Thai ``.BK`` stocks use the SET50 ETF ``TDEX.BK``) while ``local_benchmarks`` is on. If that
    benchmark is not in the store they are left empty rather than measured against the wrong market.
    """
    long = store.read_many(symbols, interval, last_n=last_n, columns=READ_COLUMNS)
    if long.empty:
        return pd.DataFrame(columns=list(FEATURE_INFO)).rename_axis("symbol")

    def bench_key(symbol: str) -> str | None:
        local = benchmark_for(symbol)
        return local if local_benchmarks and local != DEFAULT_BENCHMARK else benchmark

    keys = {sym: bench_key(sym) for sym in long["symbol"].unique()}
    benchmarks: dict[str, pd.Series] = {}
    for key in {k for k in keys.values() if k}:
        b = store.read(key, interval)
        if not b.empty:
            benchmarks[key] = b["adj_close"]

    items = [(sym, g.drop(columns="symbol").set_index("ts"), keys[sym]) for sym, g in long.groupby("symbol", sort=False)]
    chunks = [(items[i : i + chunk_size], benchmarks, include_zones) for i in range(0, len(items), chunk_size)]

    if workers and workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = [row for part in pool.map(_compute_chunk, chunks) for row in part]
    else:
        results = [row for c in chunks for row in _compute_chunk(c)]

    rows = {sym: feat for sym, feat in results if feat is not None}
    df = pd.DataFrame.from_dict(rows, orient="index", columns=list(FEATURE_INFO))
    df.index.name = "symbol"
    for column, (_, kind) in FEATURE_INFO.items():
        if kind == "bool":  # True / False / unknown (young stocks have no SMA200)
            df[column] = df[column].astype("boolean")
    # Yahoo's crypto volume is already in USD and index volume is meaningless, so close*volume
    # would pass any liquidity filter for the wrong reason.
    non_traded = [s for s in df.index if infer_asset_class(s) not in (AssetClass.EQUITY, AssetClass.ETF)]
    df.loc[non_traded, "dollar_volume_20"] = NAN
    return df.sort_index()
