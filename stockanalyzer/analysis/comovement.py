"""Co-movement between assets: correlation of returns and cointegration of prices, for stocks and crypto.

Two different questions:

- **correlation**: do the *returns* move together? Measured on log returns, never on price levels (two unrelated
  prices that both trend up show a high price correlation by accident).
- **cointegration**: do the *prices* stay tied together? Two prices are cointegrated when a mix of them,
  ``log y - beta * log x``, keeps returning to a mean even if their daily returns are only loosely correlated.
  That spread is what a pairs trade buys and sells; ``beta`` is the hedge ratio.

Like is compared with like only: stocks with stocks, crypto with crypto (:func:`market_of`; ETFs and stock indices
count as stocks, futures, FX and yields as neither). A pair function or :func:`johansen` given one of each
raises; :func:`correlation_matrix` leaves those cells NaN and :func:`scan_pairs` skips those pairs.

Every function takes prices (dates x symbols, or one series per asset) and aligns them first, because even
within one group:

- **calendars**: a US and a Thai stock close on different holidays. A forward-filled matrix turns each of them
  into a fake 0 % return and pulls correlations toward zero. :func:`align` keeps only the dates on which every
  asset traded, so the return across a holiday covers the same span for both. Pass prices *without*
  forward-filling (``MarketData.close_matrix(ffill=False)``). Crypto trades every day, so a crypto bar is a
  calendar day: a half-life of 10 bars is 10 days, not two trading weeks.
- **closing times**: the SET closes at ~10:00 UTC and the US at 20:00-21:00 UTC, so US news shows up in Thai
  prices a bar later and daily correlations between the two are understated. ``frequency="W"`` (Friday to
  Friday) mostly removes that; :func:`lead_lag` shows it.
- **currency**: PTT.BK in THB against XOM in USD, or BTC-THB against ETH-USD, also carries the USD/THB move.
  Convert to one currency first (``ui.data.base_close_matrix``) when that move should count.

Pairs are aligned pair by pair, so one asset with a short history does not shorten every other pair.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import combinations
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import CollinearityWarning
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from stockanalyzer.domain.assets import AssetClass, infer_asset_class

FREQUENCIES = {"D": None, "W": "W-FRI", "M": "ME"}
METHODS = ("pearson", "spearman", "kendall")
MIN_CORR_OBS = 30
MIN_COINT_OBS = 100  # below ~100 bars the ADF test rarely rejects a unit root even when the spread does revert
# a price "is I(1)" unless ADF rejects a unit root at this level. 1 %, not 5 %: a stock that rallied and then
# went sideways for a year (COIN since late 2023, ADF p = 0.015) still wanders like a price and should be testable
UNIT_ROOT_ALPHA = 0.01
JOHANSEN_MAX_ASSETS = 12  # statsmodels has critical values for at most 12 series

Prices = pd.DataFrame | Mapping[str, pd.Series]


# -- which group a symbol belongs to ---------------------------------------------------------------
_STOCK_CLASSES = {AssetClass.EQUITY, AssetClass.ETF, AssetClass.INDEX}


def market_of(symbol: str) -> str:
    """``"crypto"`` for a crypto pair such as BTC-USD, ``"stock"`` for shares, ETFs and stock indices, else ``"other"``.

    ``"other"`` (futures, FX, yields) is not compared with anything, itself included.
    """
    cls = infer_asset_class(str(symbol))
    if cls is AssetClass.CRYPTO:
        return "crypto"
    return "stock" if cls in _STOCK_CLASSES else "other"


def comparable(a: str, b: str) -> bool:
    """True for two stocks or two crypto."""
    return (m := market_of(a)) == market_of(b) and m != "other"


def same_market(symbols) -> str:
    """The one group all ``symbols`` belong to; raises for stocks mixed with crypto, or for futures/FX/yields."""
    markets = {str(s): market_of(s) for s in symbols}
    other = [s for s, m in markets.items() if m == "other"]
    if other:
        raise ValueError(f"only stocks and crypto can be compared; {other} is neither (futures, FX or a yield)")
    if len(set(markets.values())) > 1:
        crypto = [s for s, m in markets.items() if m == "crypto"]
        stocks = [s for s, m in markets.items() if m == "stock"]
        raise ValueError(f"compare stocks with stocks and crypto with crypto, got crypto {crypto} "
                         f"with stocks {stocks}")
    return next(iter(markets.values()), "stock")


# -- alignment ------------------------------------------------------------------------------------
def align(prices: Prices, frequency: str = "D") -> pd.DataFrame:
    """Prices on the dates every asset traded, optionally sampled at the last common close of each week/month.

    Non-positive prices are treated as missing (a log price needs a positive number).
    """
    if frequency not in FREQUENCIES:
        raise ValueError(f"frequency must be one of {list(FREQUENCIES)}, got {frequency!r}")
    frame = pd.DataFrame(dict(prices)) if isinstance(prices, Mapping) else prices.copy()
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.where(frame > 0).sort_index().dropna(how="any")
    rule = FREQUENCIES[frequency]
    if rule and not frame.empty:
        # last() takes each column's last value in the bucket; all columns share the same dates after dropna
        frame = frame.resample(rule).last().dropna(how="any")
    return frame


def log_returns(aligned: pd.DataFrame) -> pd.DataFrame:
    """Log returns between consecutive aligned dates (a gap of several days is one return)."""
    return np.log(aligned).diff().iloc[1:]


def _pair(a: pd.Series, b: pd.Series, frequency: str) -> pd.DataFrame:
    a_name, b_name = _name(a, "a"), _name(b, "b")
    same_market([a_name, b_name])
    if a_name == b_name:
        b_name = f"{b_name}_2"
    return align({a_name: a, b_name: b}, frequency)


def _name(series: pd.Series, default: str) -> str:
    return str(series.name) if series.name is not None else default


# -- correlation ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class CorrelationResult:
    a: str
    b: str
    corr: float
    p_value: float  # H0: no correlation. Optimistic for returns with volatility clustering (crypto especially)
    ci_low: float
    ci_high: float
    n_obs: int
    method: str
    frequency: str


def _corr_test(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float]:
    if method == "pearson":
        res = stats.pearsonr(x, y)
    elif method == "spearman":
        res = stats.spearmanr(x, y)
    elif method == "kendall":
        res = stats.kendalltau(x, y)
    else:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    return float(res[0]), float(res[1])


def _fisher_ci(r: float, n: int, method: str, level: float) -> tuple[float, float]:
    """Confidence interval through Fisher's z; Spearman and Kendall use the Fieller et al. (1957) variances."""
    var = {"pearson": 1.0 / (n - 3), "spearman": 1.06 / (n - 3), "kendall": 0.437 / (n - 4)}[method]
    if not np.isfinite(r) or n <= 4:
        return float("nan"), float("nan")
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    half = stats.norm.ppf(0.5 + level / 2) * np.sqrt(var)
    return float(np.tanh(z - half)), float(np.tanh(z + half))


def correlation(a: pd.Series, b: pd.Series, *, method: str = "pearson", frequency: str = "D",
                level: float = 0.95, min_obs: int = MIN_CORR_OBS) -> CorrelationResult:
    """Return correlation of two price series on their shared dates, with a p-value and a confidence interval.

    Use ``method="spearman"`` for crypto or small caps: a few huge days dominate Pearson.
    """
    pair = _pair(a, b, frequency)
    names = list(pair.columns)
    rets = log_returns(pair)
    n = len(rets)
    if n < min_obs:
        return CorrelationResult(names[0], names[1], float("nan"), float("nan"), float("nan"), float("nan"),
                                 n, method, frequency)
    x, y = rets.iloc[:, 0].to_numpy(), rets.iloc[:, 1].to_numpy()
    if np.std(x) == 0 or np.std(y) == 0:
        r, p = float("nan"), float("nan")
    else:
        r, p = _corr_test(x, y, method)
    lo, hi = _fisher_ci(r, n, method, level)
    return CorrelationResult(names[0], names[1], r, p, lo, hi, n, method, frequency)


def correlation_matrix(prices: pd.DataFrame, *, method: str = "pearson", frequency: str = "D",
                       min_obs: int = MIN_CORR_OBS) -> pd.DataFrame:
    """Return correlation of every pair, each pair on its own shared dates.

    NaN for a stock against a crypto and for pairs with fewer than ``min_obs`` shared returns.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    cols = list(prices.columns)
    out = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i, j in combinations(range(len(cols)), 2):
        if not comparable(cols[i], cols[j]):
            out.iat[i, j] = out.iat[j, i] = np.nan
            continue
        rets = log_returns(align(prices[[cols[i], cols[j]]], frequency))
        value = rets.iloc[:, 0].corr(rets.iloc[:, 1], method=method) if len(rets) >= min_obs else np.nan
        out.iat[i, j] = out.iat[j, i] = value
    return out


def rolling_correlation(a: pd.Series, b: pd.Series, *, window: int = 60, frequency: str = "D") -> pd.Series:
    """Pearson correlation of returns over a moving window of ``window`` shared bars."""
    rets = log_returns(_pair(a, b, frequency))
    return rets.iloc[:, 0].rolling(window).corr(rets.iloc[:, 1]).dropna().rename("correlation")


def lead_lag(a: pd.Series, b: pd.Series, *, max_lag: int = 3, method: str = "pearson",
             frequency: str = "D") -> pd.Series:
    """Correlation of ``a``'s return at t with ``b``'s at t + lag, for lag in [-max_lag, max_lag].

    A peak at lag +1 means ``a`` moves first and ``b`` follows a bar later; for a US stock against a Thai one
    that is usually the closing-time gap, not one company leading the other.
    """
    rets = log_returns(_pair(a, b, frequency))
    ra, rb = rets.iloc[:, 0], rets.iloc[:, 1]
    lags = range(-max_lag, max_lag + 1)
    return pd.Series([ra.corr(rb.shift(-k), method=method) for k in lags], index=pd.Index(lags, name="lag"),
                     name="correlation")


# -- cointegration --------------------------------------------------------------------------------
@dataclass(frozen=True)
class CointegrationResult:
    """Engle-Granger test of ``log y = intercept + hedge_ratio * log x + spread``."""

    y: str
    x: str
    hedge_ratio: float  # units of log x per unit of log y; the dollar hedge is hedge_ratio * value of y
    intercept: float
    adf_stat: float
    p_value: float  # MacKinnon p-value; H0: no cointegration
    crit_values: dict[str, float]  # 1% / 5% / 10%
    half_life: float  # bars for the spread to close half its gap to the mean; inf = it does not revert
    zscore: float  # latest spread, in standard deviations from its mean
    n_obs: int
    frequency: str
    y_integrated: bool  # the test assumes both log prices have a unit root (wander like prices do)
    x_integrated: bool
    spread: pd.Series = field(repr=False, compare=False)

    def is_cointegrated(self, alpha: float = 0.05) -> bool:
        return bool(self.p_value < alpha and self.y_integrated and self.x_integrated)


def has_unit_root(series: pd.Series, alpha: float = UNIT_ROOT_ALPHA) -> bool:
    """True when ADF cannot reject a unit root, i.e. the series wanders like a price (is I(1) or more)."""
    s = series.dropna()
    if len(s) < 20 or s.std() == 0:
        return False
    return bool(adfuller(s.to_numpy(), autolag="AIC", result_object=True).pvalue > alpha)


def half_life(spread: pd.Series) -> float:
    """Half-life (in bars) of an Ornstein-Uhlenbeck fit: ``d spread = lambda * spread[t-1] + c``."""
    s = spread.dropna()
    if len(s) < 3:
        return float("nan")
    lagged, delta = s.shift(1).iloc[1:].to_numpy(), s.diff().iloc[1:].to_numpy()
    lam = np.polyfit(lagged, delta, 1)[0]
    return float(-np.log(2) / lam) if lam < 0 else float("inf")


def _empty_coint(y: str, x: str, n: int, frequency: str) -> CointegrationResult:
    nan = float("nan")
    return CointegrationResult(y, x, nan, nan, nan, nan, {}, nan, nan, n, frequency, False, False,
                               pd.Series(dtype=float, name="spread"))


def _engle_granger(logs: pd.DataFrame, y: str, x: str, frequency: str,
                   integrated: Mapping[str, bool] | None = None) -> CointegrationResult:
    ly, lx = logs[y], logs[x]
    n = len(logs)
    if lx.std() == 0 or ly.std() == 0:
        return _empty_coint(y, x, n, frequency)
    beta, alpha = np.polyfit(lx.to_numpy(), ly.to_numpy(), 1)
    spread = (ly - alpha - beta * lx).rename("spread")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CollinearityWarning)  # near-identical prices: p-value is still returned
        stat, p, crit = coint(ly.to_numpy(), lx.to_numpy(), trend="c", autolag="aic")
    sd = spread.std(ddof=1)
    integrated = integrated or {y: has_unit_root(ly), x: has_unit_root(lx)}
    return CointegrationResult(
        y=y, x=x, hedge_ratio=float(beta), intercept=float(alpha), adf_stat=float(stat), p_value=float(p),
        crit_values=dict(zip(("1%", "5%", "10%"), map(float, crit))), half_life=half_life(spread),
        zscore=float((spread.iloc[-1] - spread.mean()) / sd) if sd > 0 else float("nan"),
        n_obs=n, frequency=frequency, y_integrated=integrated[y], x_integrated=integrated[x], spread=spread,
    )


def engle_granger(y: pd.Series, x: pd.Series, *, frequency: str = "D",
                  min_obs: int = MIN_COINT_OBS) -> CointegrationResult:
    """Engle-Granger cointegration of two price series, regressing log ``y`` on log ``x``.

    The test is not symmetric: swapping ``y`` and ``x`` gives a different p-value. :func:`cointegration`
    runs both directions.
    """
    pair = _pair(y, x, frequency)
    ny, nx = pair.columns
    if len(pair) < min_obs:
        return _empty_coint(ny, nx, len(pair), frequency)
    return _engle_granger(np.log(pair), ny, nx, frequency)


def cointegration(a: pd.Series, b: pd.Series, *, frequency: str = "D",
                  min_obs: int = MIN_COINT_OBS) -> CointegrationResult:
    """Engle-Granger in both directions; keeps the direction with the lower p-value (the usual practice).

    Picking the better of two tests makes false positives a little more likely than ``p_value`` says.
    """
    pair = _pair(a, b, frequency)
    na, nb = pair.columns
    if len(pair) < min_obs:
        return _empty_coint(na, nb, len(pair), frequency)
    logs = np.log(pair)
    integrated = {na: has_unit_root(logs[na]), nb: has_unit_root(logs[nb])}
    ab = _engle_granger(logs, na, nb, frequency, integrated)
    ba = _engle_granger(logs, nb, na, frequency, integrated)
    return ab if not (ba.p_value < ab.p_value) else ba


@dataclass(frozen=True)
class JohansenResult:
    """Johansen trace test on a basket of log prices."""

    symbols: list[str]
    rank: int  # number of independent cointegrating mixes at 95 % (0 = none)
    trace_stats: list[float]  # H0: rank <= r, for r = 0, 1, ...
    trace_crit_95: list[float]
    max_eig_stats: list[float]
    max_eig_crit_95: list[float]
    weights: pd.DataFrame  # one column per cointegrating vector, scaled so the first symbol has weight 1
    n_obs: int
    frequency: str


def johansen(prices: pd.DataFrame, *, frequency: str = "D", det_order: int = 0, k_ar_diff: int = 1,
             min_obs: int = MIN_COINT_OBS) -> JohansenResult:
    """Johansen test for a basket of 2 to 12 stocks or 2 to 12 crypto, on the dates all of them traded.

    ``det_order``: -1 no constant, 0 constant in the spread (default), 1 linear trend.
    ``k_ar_diff``: lagged differences in the VECM; 1 suits daily data, raise it for weekly/monthly.
    """
    symbols = list(prices.columns)
    if not 2 <= len(symbols) <= JOHANSEN_MAX_ASSETS:
        raise ValueError(f"johansen needs 2 to {JOHANSEN_MAX_ASSETS} assets, got {len(symbols)}")
    same_market(symbols)
    logs = np.log(align(prices, frequency))
    if len(logs) < min_obs:
        empty = pd.DataFrame(index=symbols)
        return JohansenResult(symbols, 0, [], [], [], [], empty, len(logs), frequency)
    with warnings.catch_warnings():
        # statsmodels casts tiny imaginary parts of the eigenvalues away; the real parts are the answer
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        res = coint_johansen(logs.to_numpy(), det_order, k_ar_diff)
    # statsmodels can hand back complex arrays with a zero imaginary part (it warned above); keep the real parts
    trace, crit = np.real(res.lr1), res.cvt[:, 1]
    rank = 0
    while rank < len(trace) and trace[rank] > crit[rank]:
        rank += 1
    vecs = np.real(res.evec[:, :rank]).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        vecs = vecs / vecs[0]
    weights = pd.DataFrame(vecs, index=symbols, columns=[f"v{k + 1}" for k in range(rank)])
    return JohansenResult(symbols, rank, list(map(float, trace)), list(map(float, crit)),
                          list(map(float, np.real(res.lr2))), list(map(float, res.cvm[:, 1])), weights, len(logs), frequency)


# -- scanning a universe --------------------------------------------------------------------------
SCAN_COLUMNS = ["a", "b", "market", "corr", "y", "x", "hedge_ratio", "p_value", "p_adj", "half_life", "zscore",
                "n_obs", "both_integrated", "cointegrated"]


def scan_pairs(prices: pd.DataFrame, *, frequency: str = "D", method: str = "pearson",
               min_corr: float | None = None, alpha: float = 0.05,
               min_obs: int = MIN_COINT_OBS) -> pd.DataFrame:
    """Correlation and cointegration of every stock-stock and crypto-crypto pair in ``prices``, best first.

    A universe holding both is scanned as two groups; stock-crypto pairs are not in the table.
    Testing many pairs finds "cointegrated" ones by luck (190 pairs at 5 % gives ~10 by chance), so ``p_adj``
    is the Benjamini-Hochberg adjusted p-value and ``cointegrated`` uses it. Only valid tests count in that
    adjustment: pairs where both prices have a unit root (tested once per asset on its own history). Pairs with
    a stationary member keep their raw ``p_value`` for reference but no ``p_adj``, and sort last.
    ``min_corr`` skips the cointegration test for pairs whose return correlation is below it; fewer tests also
    means a milder adjustment.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    cols = list(prices.columns)
    markets = {c: market_of(c) for c in cols}
    integrated = {c: has_unit_root(np.log(align(prices[[c]], frequency)[c])) for c in cols}
    rows = []
    for a, b in combinations(cols, 2):
        if not comparable(a, b):
            continue
        pair = align(prices[[a, b]], frequency)
        rets = log_returns(pair)
        corr = rets[a].corr(rets[b], method=method) if len(rets) >= MIN_CORR_OBS else np.nan
        row = {"a": a, "b": b, "market": markets[a], "corr": corr, "n_obs": len(pair),
               "both_integrated": integrated[a] and integrated[b]}
        tested = len(pair) >= min_obs and not (min_corr is not None and not abs(corr) >= min_corr)
        if tested:
            logs = np.log(pair)
            ab = _engle_granger(logs, a, b, frequency, integrated)
            ba = _engle_granger(logs, b, a, frequency, integrated)
            best = ab if not (ba.p_value < ab.p_value) else ba
            row.update(y=best.y, x=best.x, hedge_ratio=best.hedge_ratio, p_value=best.p_value,
                       half_life=best.half_life, zscore=best.zscore)
        rows.append(row)
    out = pd.DataFrame(rows).reindex(columns=SCAN_COLUMNS)
    if out.empty:
        return out
    out["both_integrated"] = out["both_integrated"].astype(bool)
    valid = out["p_value"].notna() & out["both_integrated"]
    if valid.any():
        out.loc[valid, "p_adj"] = multipletests(out.loc[valid, "p_value"], method="fdr_bh")[1]
    out["cointegrated"] = out["p_adj"] < alpha
    out["_rank"] = out["p_adj"].fillna(2.0 + out["p_value"].fillna(1.0))  # valid tests, then invalid, then untested
    out["_corr"] = -out["corr"].abs()
    return out.sort_values(["_rank", "_corr"], na_position="last").drop(columns=["_rank", "_corr"]) \
        .reset_index(drop=True)
