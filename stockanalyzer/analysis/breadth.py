"""Market breadth: how many stocks take part in a move, not only how far the index went.

An index can climb on a handful of giants while most stocks fall; breadth measures say which of the two is happening.
Everything here works on daily counts (see ``data.breadth``) and an optional benchmark (the market's index ETF):

- participation: share of stocks above their 50- and 200-day averages;
- new 52-week highs and lows (closing basis) and their 10-day balance;
- advances and declines: the A/D line, McClellan oscillator and summation index, Zweig's breadth thrust;
- the average stock against the index (equal weight vs capitalisation weight);
- a breadth regime (index trend × participation) and what the index did next in each regime.

None of it is a forecast. The regime table shows how the index behaved after each state on ~9 years of data, with how
many separate episodes that rests on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

COUNT_COLUMNS = ["n", "n50", "above50", "n200", "above200", "n252", "new_highs", "new_lows", "adv", "dec", "unch",
                 "ret_sum", "ret_n", "ret63_sum", "ret63_n", "n63", "beat63"]
RETURN_CLIP = (-0.5, 1.0)  # a daily move outside this is a bad print or a corporate action, not the market
RETURN_CLIP_3M = (-0.95, 3.0)  # same for a 3-month return
MIN_COUNT = 20  # fewer stocks than this with a full average and a share is not worth plotting
PARTIAL_SHARE = 0.6  # a day with fewer stocks than this share of the recent norm was only half updated
THIN_START = 0.3  # history starts on the first day with at least this share of the most stocks ever counted
THRUST_LOW, THRUST_HIGH, THRUST_DAYS = 0.40, 0.615, 10  # Zweig: 10-day EMA of advancers from <40% to >61.5% in 10 days
INDEX_BAND, BREADTH_BAND = 0.01, 0.02  # hysteresis around the 200-day average and the 50% line, so states do not flicker
BROAD_LINE = 0.50
HORIZONS = {"1 month": 21, "3 months": 63, "6 months": 126}

REGIMES = ("Healthy uptrend", "Narrow uptrend", "Improving underneath", "Downtrend")
REGIME_TEXT = {
    "Healthy uptrend": "the index is above its 200-day average and so are most stocks",
    "Narrow uptrend": "the index is above its 200-day average but most stocks are not - a few large names carry it",
    "Improving underneath": "the index is below its 200-day average but most stocks are already above theirs",
    "Downtrend": "the index and most stocks are below their 200-day averages",
}


def complete_days(counts: pd.DataFrame, *, share: float = PARTIAL_SHARE, window: int = 20, thin: float = THIN_START) -> pd.DataFrame:
    """Drop the thin start of the history (a few symbols reach back decades) and days only part of the market had been
    updated for (a stock opened in the app gets today's bar while the rest wait for the nightly run)."""
    if counts.empty:
        return counts
    n = counts["n"].astype("float64")
    first = n.index[n >= thin * n.max()]
    counts = counts.loc[first[0]:] if len(first) else counts
    n = counts["n"].astype("float64")
    norm = n.rolling(window, min_periods=5).median().shift(1)
    keep = norm.isna() | (n >= share * norm)
    return counts.loc[keep]


def _share(part: pd.Series, whole: pd.Series) -> pd.Series:
    return (part / whole.where(whole >= MIN_COUNT)).astype("float64")


def _hysteresis(value: pd.Series, upper: float | pd.Series, lower: float | pd.Series) -> pd.Series:
    """True once ``value`` rises above ``upper`` and until it falls below ``lower`` (NaN before the first decision)."""
    state = pd.Series(np.where(value > upper, 1.0, np.where(value < lower, 0.0, np.nan)), index=value.index)
    state[value.isna()] = np.nan
    return state.ffill()


def indicators(counts: pd.DataFrame, benchmark: pd.Series | None = None) -> pd.DataFrame:
    """Daily breadth readings from :func:`data.breadth.daily_counts` (and the benchmark's adjusted close)."""
    c = counts.astype("float64")
    out = pd.DataFrame(index=counts.index)
    out["stocks"] = c["n"]
    out["pct_above_50"] = _share(c["above50"], c["n50"])
    out["pct_above_200"] = _share(c["above200"], c["n200"])
    out["new_highs"], out["new_lows"] = c["new_highs"], c["new_lows"]
    out["new_high_share"] = _share(c["new_highs"], c["n252"])
    out["new_low_share"] = _share(c["new_lows"], c["n252"])
    out["net_highs"] = c["new_highs"] - c["new_lows"]
    out["net_highs_10"] = out["net_highs"].rolling(10, min_periods=5).mean()
    hl = c["new_highs"] / (c["new_highs"] + c["new_lows"]).replace(0, np.nan)
    out["high_low_index"] = hl.rolling(10, min_periods=5).mean()
    out["adv"], out["dec"] = c["adv"], c["dec"]
    moved = (c["adv"] + c["dec"]).replace(0, np.nan)
    out["ad_ratio"] = c["adv"] / moved
    out["ad_line"] = (c["adv"] - c["dec"]).cumsum()
    ratio_adjusted = (c["adv"] - c["dec"]) / moved * 1000
    out["mcclellan"] = (ratio_adjusted.ewm(alpha=0.10, adjust=False).mean() - ratio_adjusted.ewm(alpha=0.05, adjust=False).mean())
    out["summation"] = out["mcclellan"].cumsum()
    out["thrust_ema"] = out["ad_ratio"].ewm(span=THRUST_DAYS, adjust=False).mean()
    # daily-rebalanced equal weight: only a stand-in for the index when there is none (it is flattered by small-cap bid-ask
    # bounce and by survivorship when compounded over years, so the page never shows it as a performance record)
    ew = (c["ret_sum"] / c["ret_n"].replace(0, np.nan)).fillna(0.0)
    ew.iloc[0] = 0.0
    out["ew_level"] = 100 * (1 + ew).cumprod()
    # the average stock's own 3-month return (no daily compounding) and the share of stocks that beat the index over it
    out["avg_return_3m"] = c["ret63_sum"] / c["ret63_n"].where(c["ret63_n"] >= MIN_COUNT)
    out["beat_share_3m"] = _share(c["beat63"], c["n63"])

    if benchmark is not None and not benchmark.dropna().empty:
        b = benchmark.dropna()
        b.index = pd.DatetimeIndex(b.index).tz_localize(None) if getattr(b.index, "tz", None) else pd.DatetimeIndex(b.index)
        sma = b.rolling(200, min_periods=200).mean()
        high = b.rolling(252, min_periods=126).max()
        out["index"] = b.reindex(out.index).ffill(limit=5)
        out["index_sma200"] = sma.reindex(out.index).ffill(limit=5)
        out["index_off_high"] = (b / high - 1).reindex(out.index).ffill(limit=5)
        out["index_return_3m"] = (b / b.shift(63) - 1).reindex(out.index).ffill(limit=5)
        trend_ref = out["index"] / out["index_sma200"]
    else:  # no benchmark: the average stock stands in for the index
        sma = out["ew_level"].rolling(200, min_periods=200).mean()
        out["index"], out["index_sma200"], out["index_return_3m"] = np.nan, np.nan, np.nan
        out["index_off_high"] = out["ew_level"] / out["ew_level"].rolling(252, min_periods=126).max() - 1
        trend_ref = out["ew_level"] / sma
    up = _hysteresis(trend_ref, 1 + INDEX_BAND, 1 - INDEX_BAND)
    broad = _hysteresis(out["pct_above_200"], BROAD_LINE + BREADTH_BAND, BROAD_LINE - BREADTH_BAND)
    regime = pd.Series(pd.NA, index=out.index, dtype="object")
    known = up.notna() & broad.notna()
    regime[known & (up == 1) & (broad == 1)] = REGIMES[0]
    regime[known & (up == 1) & (broad == 0)] = REGIMES[1]
    regime[known & (up == 0) & (broad == 1)] = REGIMES[2]
    regime[known & (up == 0) & (broad == 0)] = REGIMES[3]
    out["regime"] = regime
    return out


def zweig_thrusts(ind: pd.DataFrame) -> pd.DatetimeIndex:
    """Days the 10-day EMA of the advancing share crossed above 61.5% after being below 40% within the 10 days before."""
    ema = ind["thrust_ema"]
    above = ema > THRUST_HIGH
    cross = above & ~above.shift(1, fill_value=True)
    recent_low = ema.shift(1).rolling(THRUST_DAYS, min_periods=1).min() < THRUST_LOW
    return ind.index[cross & recent_low]


def forward_returns(level: pd.Series, horizon: int) -> pd.Series:
    """Return from each day's close to the close ``horizon`` trading days later (NaN where the future is unknown)."""
    return level.shift(-horizon) / level - 1


def episodes(labels: pd.Series) -> pd.DataFrame:
    """Contiguous runs of the same label: ``label, start, end, days``."""
    s = labels.dropna()
    if s.empty:
        return pd.DataFrame(columns=["label", "start", "end", "days"])
    run = (s != s.shift()).cumsum()
    g = s.groupby(run)
    return pd.DataFrame({"label": g.first(), "start": g.apply(lambda x: x.index[0]), "end": g.apply(lambda x: x.index[-1]),
                         "days": g.size()}).reset_index(drop=True)


def regime_outcomes(ind: pd.DataFrame, horizons: dict[str, int] = HORIZONS) -> pd.DataFrame:
    """For each regime: days in it, separate episodes, and the index's average / median forward return and share of
    positive outcomes per horizon, next to the same figures for every day. Windows overlap, so the day counts overstate
    how much independent evidence there is - the episode count is the honest sample size."""
    level = ind["index"] if ind["index"].notna().any() else ind["ew_level"]
    runs = episodes(ind["regime"])
    rows = []
    for name in [*REGIMES, "All days"]:
        mask = ind["regime"].notna() if name == "All days" else (ind["regime"] == name)
        row: dict = {"regime": name, "days": int(mask.sum()),
                     "episodes": int(len(runs)) if name == "All days" else int((runs["label"] == name).sum())}
        for label, h in horizons.items():
            fwd = forward_returns(level, h)[mask].dropna()
            row[f"avg {label}"] = float(fwd.mean()) if len(fwd) else np.nan
            row[f"median {label}"] = float(fwd.median()) if len(fwd) else np.nan
            row[f"positive {label}"] = float((fwd > 0).mean()) if len(fwd) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("regime")


def event_outcomes(ind: pd.DataFrame, days: pd.DatetimeIndex, horizons: dict[str, int] = HORIZONS) -> pd.DataFrame:
    """The index's return over each horizon after each event day."""
    level = ind["index"] if ind["index"].notna().any() else ind["ew_level"]
    out = pd.DataFrame(index=pd.DatetimeIndex(days, name="date"))
    for label, h in horizons.items():
        out[label] = forward_returns(level, h).reindex(out.index)
    return out


@dataclass
class Reading:
    tone: str  # "good" | "bad" | "neutral"
    text: str


@dataclass
class Breadth:
    ind: pd.DataFrame
    has_benchmark: bool = True
    _thrusts: pd.DatetimeIndex | None = field(default=None, repr=False)

    @property
    def asof(self) -> pd.Timestamp:
        return self.ind.index[-1]

    @property
    def first(self) -> pd.Timestamp:
        return self.ind["pct_above_200"].first_valid_index() or self.ind.index[0]

    @property
    def thrusts(self) -> pd.DatetimeIndex:
        if self._thrusts is None:
            self._thrusts = zweig_thrusts(self.ind)
        return self._thrusts

    def at(self, column: str, days_back: int = 0) -> float:
        s = self.ind[column].dropna()
        return float(s.iloc[-1 - days_back]) if len(s) > days_back else np.nan

    def regime_since(self) -> tuple[str | None, pd.Timestamp | None]:
        runs = episodes(self.ind["regime"])
        if runs.empty:
            return None, None
        last = runs.iloc[-1]
        return last["label"], last["start"]

    def readings(self) -> list[Reading]:
        return readings(self)


def _percentile(series: pd.Series, value: float) -> float:
    s = series.dropna()
    return float((s <= value).mean()) if len(s) else np.nan


def readings(b: Breadth) -> list[Reading]:
    """Plain-language observations about the latest day, most important first."""
    ind = b.ind
    out: list[Reading] = []
    p200, p200_m = b.at("pct_above_200"), b.at("pct_above_200", 21)
    if p200 == p200:
        rank = _percentile(ind["pct_above_200"], p200)
        move = "" if p200_m != p200_m else f" ({'up' if p200 >= p200_m else 'down'} from {p200_m:.0%} a month ago)"
        tone = "good" if p200 >= 0.6 else "bad" if p200 < 0.4 else "neutral"
        out.append(Reading(tone, f"**{p200:.0%}** of stocks are above their 200-day average{move}, more than on {rank:.0%} of days "
                                 f"since {b.first:%b %Y}."))

    off_high = b.at("index_off_high")
    peak = ind["pct_above_200"].iloc[-252:].dropna()
    who = "The index" if b.has_benchmark else "The average stock"
    if off_high == off_high and off_high > -0.03 and len(peak):
        best, best_day = float(peak.max()), peak.idxmax()
        if p200 < BROAD_LINE:
            out.append(Reading("bad", f"{who} is within {abs(off_high):.0%} of its 52-week high, yet fewer than half the stocks are above "
                                      "their 200-day average: a narrow rally carried by a few large names."))
        elif best - p200 >= 0.15:
            out.append(Reading("bad", f"{who} is near its 52-week high, but participation has faded from {best:.0%} "
                                      f"({best_day:%d %b %Y}) to {p200:.0%}: fewer stocks are joining each new high."))
        else:
            out.append(Reading("good", f"{who} is near its 52-week high with broad participation behind it."))
    elif off_high == off_high and off_high < -0.10 and p200 == p200 and p200_m == p200_m and p200 - p200_m >= 0.10:
        out.append(Reading("good", f"{who} is {abs(off_high):.0%} below its high, but the share of stocks above their 200-day average rose "
                                   f"{(p200 - p200_m) * 100:.0f} points in a month: improvement is starting underneath."))

    nh, nl = ind["new_highs"].iloc[-10:].mean(), ind["new_lows"].iloc[-10:].mean()
    if nh == nh and nl == nl and nh + nl > 0:
        if nh > 1.5 * nl:
            out.append(Reading("good", f"New 52-week highs outnumber new lows (10-day average {nh:.0f} vs {nl:.0f})."))
        elif nl > 1.5 * nh:
            out.append(Reading("bad", f"New 52-week lows outnumber new highs (10-day average {nl:.0f} vs {nh:.0f})."))
        else:
            out.append(Reading("neutral", f"New highs and lows are roughly balanced (10-day average {nh:.0f} highs vs {nl:.0f} lows)."))
    nl_share = b.at("new_low_share")
    if nl_share == nl_share and nl_share >= 0.10:
        out.append(Reading("neutral", f"{nl_share:.0%} of stocks closed at a 52-week low on the last day - a selling climax on this scale "
                                      "has often come near a low, but not always at it."))

    ad = ind["ad_line"].dropna()
    if len(ad) > 252 and off_high == off_high and off_high > -0.02:
        ad_high = float(ad.iloc[-252:].max())
        if ad.iloc[-1] < ad_high - 0.02 * abs(ad_high) and ad.iloc[-1] < ad.iloc[-21:].max():
            out.append(Reading("bad", f"{who} is at or near a 52-week high but the advance / decline line is not: more stocks have been "
                                      "falling than rising on the way up."))

    osc = b.at("mcclellan")
    if osc == osc:
        osc_prev = b.at("mcclellan", 5)
        direction = "rising" if osc_prev == osc_prev and osc > osc_prev else "falling"
        out.append(Reading("good" if osc > 0 else "bad",
                           f"McClellan oscillator {osc:+.0f} and {direction}: short-term breadth momentum is "
                           f"{'positive' if osc > 0 else 'negative'}."))

    recent = [d for d in b.thrusts if d >= ind.index[max(0, len(ind) - 30)]]
    if recent:
        out.append(Reading("good", f"Breadth thrust on {recent[-1]:%d %b %Y}: the share of advancing stocks went from under "
                                   f"{THRUST_LOW:.0%} to over {THRUST_HIGH:.1%} within {THRUST_DAYS} days - a rare burst of buying "
                                   "(see the Advance / decline tab for what followed earlier ones)."))

    avg3, ix3, beat = b.at("avg_return_3m"), b.at("index_return_3m"), b.at("beat_share_3m")
    if b.has_benchmark and avg3 == avg3 and ix3 == ix3 and beat == beat:
        detail = f"the average stock returned {avg3:+.1%} against {ix3:+.1%} for the index, and {beat:.0%} of stocks beat it"
        if beat < 0.40:
            out.append(Reading("bad", f"Over 3 months {detail}: the index is carried by a minority of its largest names."))
        elif beat > 0.55:
            out.append(Reading("good", f"Over 3 months {detail}: leadership is broad."))
        else:
            out.append(Reading("neutral", f"Over 3 months {detail}."))

    p50 = b.at("pct_above_50")
    if p50 == p50 and p50 < 0.15:
        out.append(Reading("neutral", f"Only {p50:.0%} of stocks are above their 50-day average: the market is washed out in the short term."))
    elif p50 == p50 and p50 > 0.85:
        out.append(Reading("neutral", f"{p50:.0%} of stocks are above their 50-day average: short-term stretched, though strong markets "
                                      "can stay there for weeks."))
    return out


def group_breadth(states: pd.DataFrame, labels, *, min_stocks: int = 3) -> pd.DataFrame:
    """Participation per group (sector, industry or basket) on the latest day, with the change over the last month."""
    cols = ["stocks", "pct_above_50", "pct_above_200", "pct_above_200_before", "change_200", "new_highs", "new_lows",
            "median_return_month"]
    if states.empty:
        return pd.DataFrame(columns=cols)
    df = states.copy()
    df["group"] = [labels.get(s) for s in df.index]
    df = df.dropna(subset=["group"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    def mean_bool(s: pd.Series) -> float:
        s = s.dropna()
        return float(s.astype(bool).mean()) if len(s) else np.nan

    g = df.groupby("group")
    out = pd.DataFrame({
        "stocks": g.size(),
        "pct_above_50": g["above50"].agg(mean_bool),
        "pct_above_200": g["above200"].agg(mean_bool),
        "pct_above_200_before": g["above200_before"].agg(mean_bool),
        "new_highs": g["new_high"].agg(lambda s: int(s.fillna(False).astype(bool).sum())),
        "new_lows": g["new_low"].agg(lambda s: int(s.fillna(False).astype(bool).sum())),
        "median_return_month": g["ret_month"].median(),
    })
    out["change_200"] = out["pct_above_200"] - out["pct_above_200_before"]
    out = out[out["stocks"] >= min_stocks]
    return out[cols].sort_values(["pct_above_200", "stocks"], ascending=[False, False])
