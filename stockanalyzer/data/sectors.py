"""Rotation universes (US sector ETFs, asset classes, Thai sector baskets) and their price history.

A *sector* is one ETF or an equal-weight basket of stocks, so every universe is analysed the same way.
Yahoo has no SET sector indices, hence the Thai baskets: large, liquid stocks with at least five years of
history, rebalanced daily to equal weight. They are a proxy, not the official SET sector indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stockanalyzer.analysis import rotation
from stockanalyzer.data.store import PriceStore


@dataclass(frozen=True)
class RotationUniverse:
    key: str
    label: str
    benchmark: str  # symbol sectors are measured against; "" = the equal-weight average of the sectors themselves
    benchmark_name: str
    sectors: dict[str, tuple[str, ...]]  # sector name -> member symbols (one ETF, or a basket of stocks)
    offensive: tuple[str, ...] = ()  # sector names that lead when investors take risk
    defensive: tuple[str, ...] = ()  # ... and the ones they hide in
    note: str = ""

    @property
    def members(self) -> list[str]:
        return list(dict.fromkeys(s for group in self.sectors.values() for s in group))

    @property
    def symbols(self) -> list[str]:
        """Everything that must be in the price store: the benchmark (if it is a symbol) first, then every member."""
        return list(dict.fromkeys([*([self.benchmark] if self.benchmark else []), *self.members]))


US_SECTORS = RotationUniverse(
    key="us_sectors", label="US sectors (SPDR ETFs)", benchmark="SPY", benchmark_name="S&P 500 (SPY)",
    sectors={
        "Technology": ("XLK",), "Financials": ("XLF",), "Energy": ("XLE",), "Health Care": ("XLV",),
        "Consumer Discretionary": ("XLY",), "Consumer Staples": ("XLP",), "Industrials": ("XLI",),
        "Materials": ("XLB",), "Utilities": ("XLU",), "Real Estate": ("XLRE",), "Communication Services": ("XLC",),
    },
    offensive=("Technology", "Consumer Discretionary", "Financials", "Industrials", "Materials", "Communication Services"),
    defensive=("Consumer Staples", "Utilities", "Health Care"),
    note="Energy and Real Estate follow commodities and interest rates more than the risk cycle, so they sit out of the "
         "offensive / defensive split.",
)

ASSET_CLASSES = RotationUniverse(
    key="asset_classes", label="Asset classes (ETFs)", benchmark="SPY", benchmark_name="S&P 500 (SPY)",
    sectors={
        "Nasdaq 100": ("QQQ",), "US small caps": ("IWM",), "Developed ex-US": ("EFA",), "Emerging markets": ("EEM",),
        "Long Treasuries": ("TLT",), "Investment-grade credit": ("LQD",), "High-yield credit": ("HYG",),
        "Gold": ("GLD",), "Commodities": ("DBC",), "US dollar": ("UUP",), "US REITs": ("VNQ",), "Bitcoin": ("BTC-USD",),
    },
    offensive=("Nasdaq 100", "US small caps", "Developed ex-US", "Emerging markets", "High-yield credit", "Bitcoin"),
    defensive=("Long Treasuries", "Investment-grade credit", "Gold", "US dollar"),
    note="Where money goes between stocks, bonds, gold, the dollar and crypto, measured against the S&P 500. "
         "Commodities and REITs are left out of the offensive / defensive split.",
)

THAI_SECTORS = RotationUniverse(
    key="th_sectors", label="Thai sectors (large-cap baskets)", benchmark="", benchmark_name="average of the 12 sectors",
    sectors={
        "Banks": ("BBL.BK", "KBANK.BK", "KTB.BK", "SCB.BK", "TTB.BK"),
        "Energy": ("PTT.BK", "PTTEP.BK", "TOP.BK", "BCP.BK", "IRPC.BK"),
        "Utilities": ("GPSC.BK", "BGRIM.BK", "EGCO.BK", "RATCH.BK"),
        "Technology & telecom": ("ADVANC.BK", "TRUE.BK", "DELTA.BK", "KCE.BK", "HANA.BK"),
        "Commerce": ("CPALL.BK", "CRC.BK", "HMPRO.BK", "COM7.BK", "GLOBAL.BK"),
        "Health care": ("BDMS.BK", "BH.BK", "BCH.BK", "CHG.BK"),
        "Property": ("CPN.BK", "LH.BK", "AWC.BK", "WHA.BK", "AMATA.BK", "SPALI.BK"),
        "Transport": ("AOT.BK", "BTS.BK", "BEM.BK"),
        "Food & agro": ("CPF.BK", "TU.BK", "CBG.BK", "OSP.BK"),
        "Materials & petrochem": ("SCC.BK", "SCGP.BK", "PTTGC.BK"),
        "Consumer finance": ("MTC.BK", "SAWAD.BK", "KTC.BK", "TISCO.BK"),
        "Tourism": ("MINT.BK", "CENTEL.BK", "ERW.BK"),
    },
    offensive=("Banks", "Energy", "Commerce", "Property", "Transport", "Materials & petrochem", "Consumer finance", "Tourism"),
    defensive=("Utilities", "Health care", "Food & agro"),
    note="Yahoo has no SET sector indices, so each sector is an equal-weight basket of its large, liquid stocks - "
         "a proxy for the official SET sector indices. They are measured against the average of the sectors: against the "
         "cap-weighted SET50 ETF nearly every equal-weight basket would look like a winner whenever the mega-caps lag "
         "(and Yahoo serves only one bar of the SET index itself). Technology & telecom mixes telecoms (defensive) with "
         "electronics (cyclical), so it sits out of the offensive / defensive split.",
)

UNIVERSES: dict[str, RotationUniverse] = {u.key: u for u in (US_SECTORS, THAI_SECTORS, ASSET_CLASSES)}


def all_symbols() -> list[str]:
    """Every symbol the nightly job keeps fresh for the rotation pages."""
    return list(dict.fromkeys(s for u in UNIVERSES.values() for s in u.symbols))


@dataclass
class SectorData:
    universe: RotationUniverse
    close: pd.DataFrame  # one column per sector: equal-weight total-return level, aligned to the benchmark's days
    benchmark: pd.Series
    mfv: pd.DataFrame = field(default_factory=pd.DataFrame)  # per member: daily money-flow volume
    volume: pd.DataFrame = field(default_factory=pd.DataFrame)  # per member: daily volume
    coverage: dict[str, tuple[int, int]] = field(default_factory=dict)  # sector -> (members with data, members defined)

    @property
    def asof(self) -> pd.Timestamp:
        return self.close.index[-1]

    @property
    def first(self) -> pd.Timestamp:
        return self.close.index[0]

    def money_flow(self, start, end) -> pd.Series:
        """Chaikin money flow per sector over ``(start, end]``: the average of its members' values."""
        if self.mfv.empty:
            return pd.Series(np.nan, index=self.close.columns, dtype="float64")
        per_member = rotation.chaikin_flow(self.mfv, self.volume, start, end)
        return pd.Series({name: per_member.reindex([m for m in self.universe.sectors[name] if m in per_member.index]).mean()
                          for name in self.close.columns}, dtype="float64")

    def scoreboard(self, start, end) -> pd.DataFrame:
        """Return, relative return, rank and money flow of every sector over ``(start, end]`` and the equal-length window before."""
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        prev = start - (end - start)
        return rotation.scoreboard(self.close, self.benchmark, start, end,
                                   flow=self.money_flow(start, end), prev_flow=self.money_flow(prev, start))


def load_sector_data(store: PriceStore, universe: RotationUniverse, *, last_n: int = 2800) -> SectorData | None:
    """Sector levels from the local store (never the network); ``None`` if the benchmark or every sector is missing."""
    long = store.read_many(universe.symbols, last_n=last_n,
                           columns=["high", "low", "close", "adj_close", "volume"])
    if long.empty or (universe.benchmark and universe.benchmark not in set(long["symbol"])):
        return None

    def wide(column: str) -> pd.DataFrame:
        return long.pivot(index="ts", columns="symbol", values=column).sort_index()

    adj, close, high, low, volume = (wide(c) for c in ("adj_close", "close", "high", "low", "volume"))
    # the benchmark's trading days (weekend crypto prints and odd holidays drop out), else every day any member traded
    days = adj[universe.benchmark].dropna().index if universe.benchmark else adj.dropna(how="all").index
    adj = adj.reindex(days).ffill()
    members = [m for m in universe.members if m in adj.columns]
    if not members:
        return None

    levels, sector_returns, coverage = {}, {}, {}
    returns = adj[members].pct_change(fill_method=None)
    for name, group in universe.sectors.items():
        have = [m for m in group if m in adj.columns]
        coverage[name] = (len(have), len(group))
        if have:
            levels[name] = rotation.equal_weight_level(returns[have])
            sector_returns[name] = returns[have].mean(axis=1)
    if not levels:
        return None
    sector_levels = pd.DataFrame(levels).dropna(how="all")
    benchmark = (adj[universe.benchmark].reindex(sector_levels.index) if universe.benchmark
                 else rotation.equal_weight_level(pd.DataFrame(sector_returns).reindex(sector_levels.index)))

    day_range = (high - low).reindex(days)
    clv = (((close - low) - (high - close)).reindex(days) / day_range.where(day_range > 0)).fillna(0.0)
    vol = volume.reindex(days).fillna(0.0)
    return SectorData(universe, sector_levels, benchmark, (clv * vol)[members], vol[members], coverage)
