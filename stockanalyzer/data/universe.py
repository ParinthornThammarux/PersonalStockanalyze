"""Symbol universes: the packaged US stock list, cross-asset presets, user CSV/XLSX lists."""

from __future__ import annotations

import json
import re
from importlib import resources
from pathlib import Path

import pandas as pd

from stockanalyzer.domain.assets import AssetClass, infer_asset_class, parse_market_cap

UNIVERSE_COLUMNS = ["symbol", "listed_symbol", "name", "industry", "market_cap", "asset_class"]
SPAC_INDUSTRY = "Shell Companies"

# Yahoo's 11 sectors and the industries under each (Yahoo's own taxonomy; "Consumer Discretionary" is not a Yahoo industry
# but appears in the packaged list)
SECTOR_INDUSTRIES: dict[str, tuple[str, ...]] = {
    "Technology": ("Communication Equipment", "Computer Hardware", "Consumer Electronics", "Electronic Components",
                   "Electronics & Computer Distribution", "Information Technology Services", "Scientific & Technical Instruments",
                   "Semiconductor Equipment & Materials", "Semiconductors", "Software - Application", "Software - Infrastructure",
                   "Solar"),
    "Communication Services": ("Advertising Agencies", "Broadcasting", "Electronic Gaming & Multimedia", "Entertainment",
                               "Internet Content & Information", "Publishing", "Telecom Services"),
    "Consumer Cyclical": ("Apparel Manufacturing", "Apparel Retail", "Auto & Truck Dealerships", "Auto Manufacturers", "Auto Parts",
                          "Consumer Discretionary", "Department Stores", "Footwear & Accessories",
                          "Furnishings, Fixtures & Appliances", "Gambling", "Home Improvement Retail", "Internet Retail", "Leisure",
                          "Lodging", "Luxury Goods", "Packaging & Containers", "Personal Services", "Recreational Vehicles",
                          "Residential Construction", "Resorts & Casinos", "Restaurants", "Specialty Retail", "Textile Manufacturing",
                          "Travel Services"),
    "Consumer Defensive": ("Beverages - Brewers", "Beverages - Non-Alcoholic", "Beverages - Wineries & Distilleries", "Confectioners",
                           "Discount Stores", "Education & Training Services", "Farm Products", "Food Distribution",
                           "Grocery Stores", "Household & Personal Products", "Packaged Foods", "Tobacco"),
    "Healthcare": ("Biotechnology", "Diagnostics & Research", "Drug Manufacturers - General", "Drug Manufacturers - Specialty & Generic",
                   "Health Information Services", "Healthcare Plans", "Medical Care Facilities", "Medical Devices",
                   "Medical Distribution", "Medical Instruments & Supplies", "Pharmaceutical Retailers"),
    "Financial Services": ("Asset Management", "Asset Management - Income", "Banks - Diversified", "Banks - Regional",
                           "Capital Markets", "Credit Services", "Financial - Credit Services", "Financial Conglomerates",
                           "Financial Data & Stock Exchanges", "Insurance - Diversified", "Insurance - Life",
                           "Insurance - Property & Casualty", "Insurance - Reinsurance", "Insurance - Specialty", "Insurance Brokers",
                           "Mortgage Finance", SPAC_INDUSTRY),
    "Industrials": ("Aerospace & Defense", "Airlines", "Airports & Air Services", "Building Products & Equipment",
                    "Business Equipment & Supplies", "Conglomerates", "Consulting Services", "Electrical Equipment & Parts",
                    "Engineering & Construction", "Farm & Heavy Construction Machinery", "Industrial Distribution",
                    "Infrastructure Operations", "Integrated Freight & Logistics", "Marine Shipping", "Metal Fabrication",
                    "Pollution & Treatment Controls", "Railroads", "Rental & Leasing Services", "Security & Protection Services",
                    "Specialty Business Services", "Specialty Industrial Machinery", "Staffing & Employment Services",
                    "Tools & Accessories", "Trucking", "Waste Management"),
    "Basic Materials": ("Agricultural Inputs", "Aluminum", "Building Materials", "Chemicals", "Coking Coal", "Copper", "Gold",
                        "Lumber & Wood Production", "Other Industrial Metals & Mining", "Other Precious Metals & Mining",
                        "Paper & Paper Products", "Silver", "Specialty Chemicals", "Steel"),
    "Energy": ("Oil & Gas Drilling", "Oil & Gas E&P", "Oil & Gas Exploration & Production", "Oil & Gas Equipment & Services",
               "Oil & Gas Integrated", "Oil & Gas Midstream", "Oil & Gas Refining & Marketing", "Thermal Coal", "Uranium"),
    "Utilities": ("Utilities - Diversified", "Utilities - Independent Power Producers", "Utilities - Regulated Electric",
                  "Utilities - Regulated Gas", "Utilities - Regulated Water", "Utilities - Renewable"),
    "Real Estate": ("REIT - Diversified", "REIT - Healthcare Facilities", "REIT - Hotel & Motel", "REIT - Industrial", "REIT - Mortgage",
                    "REIT - Office", "REIT - Residential", "REIT - Retail", "REIT - Specialty", "Real Estate", "Real Estate - Development",
                    "Real Estate - Diversified", "Real Estate Services"),
}
_SECTOR_OF = {industry: sector for sector, industries in SECTOR_INDUSTRIES.items() for industry in industries}


def sector_of(industry: str | None) -> str | None:
    """Yahoo sector of a Yahoo industry label; ``None`` when unknown or blank."""
    return _SECTOR_OF.get((industry or "").strip())


def us_listed_to_yahoo(symbol: str) -> str:
    """Map exchange notation to Yahoo's: ``BRK.B -> BRK-B``, ``ALB.PRA -> ALB-PA``."""
    s = symbol.strip().upper()
    preferred = re.fullmatch(r"([A-Z]+)\.PR([A-Z])", s)
    if preferred:
        return f"{preferred.group(1)}-P{preferred.group(2)}"
    return s.replace(".", "-")


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=UNIVERSE_COLUMNS)
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    return df


def load_us_stocks() -> pd.DataFrame:
    """~5.5k US-listed tickers with industry and market cap (packaged resource)."""
    raw = json.loads(
        resources.files("stockanalyzer.resources").joinpath("us_stocks.json").read_text("utf-8")
    )
    rows = [
        {
            "symbol": s["yahoo"],
            "listed_symbol": s["symbol"],
            "name": s["name"],
            "industry": s["industry"],
            "market_cap": s["market_cap"],
            "asset_class": AssetClass.EQUITY.value,
        }
        for s in raw["stocks"]
    ]
    return _frame(rows)


def is_thai_common_stock(symbol: str, short_name: str | None = None) -> bool:
    """A SET / mai share, as opposed to what Yahoo lists beside it: depositary receipts of foreign stocks (``NVDA80.BK``,
    named ``NVDA80_DR ...``), NVDRs (``PTT-R.BK``), warrants (``AAV-W1.BK``) and preferred / foreign-board lines (``-P``, ``-F``)."""
    s = symbol.strip().upper()
    if not s.endswith(".BK"):
        return False
    code = s[:-3]
    if not re.fullmatch(r"[A-Z0-9&]+", code):
        return False
    return not (short_name and short_name.upper().startswith(f"{code}_DR"))


def load_th_stocks() -> pd.DataFrame:
    """Every Thai share the app knows (the SET and mai, snapshot of Yahoo's list) with a US-dollar market cap and, where known, an industry."""
    raw = json.loads(
        resources.files("stockanalyzer.resources").joinpath("th_stocks.json").read_text("utf-8")
    )
    rows = [
        {"symbol": s["yahoo"], "listed_symbol": s["symbol"], "name": s["name"], "industry": s["industry"],
         "market_cap": s["market_cap"], "asset_class": AssetClass.EQUITY.value}
        for s in raw["stocks"]
    ]
    return _frame(rows)


# (symbol, name, asset class) -- all valid Yahoo Finance tickers.
CROSS_ASSETS: list[tuple[str, str, AssetClass]] = [
    ("^GSPC", "S&P 500", AssetClass.INDEX),
    ("^IXIC", "Nasdaq Composite", AssetClass.INDEX),
    ("^DJI", "Dow Jones Industrial", AssetClass.INDEX),
    ("^RUT", "Russell 2000", AssetClass.INDEX),
    ("^VIX", "VIX Volatility", AssetClass.INDEX),
    ("^SET.BK", "SET Index (Thailand)", AssetClass.INDEX),
    ("TDEX.BK", "SET50 ETF (Thailand)", AssetClass.ETF),
    ("^N225", "Nikkei 225", AssetClass.INDEX),
    ("^HSI", "Hang Seng", AssetClass.INDEX),
    ("^FTSE", "FTSE 100", AssetClass.INDEX),
    ("^GDAXI", "DAX", AssetClass.INDEX),
    ("^TNX", "US 10Y Yield", AssetClass.RATE),
    ("GC=F", "Gold", AssetClass.COMMODITY),
    ("SI=F", "Silver", AssetClass.COMMODITY),
    ("CL=F", "WTI Crude Oil", AssetClass.COMMODITY),
    ("NG=F", "Natural Gas", AssetClass.COMMODITY),
    ("HG=F", "Copper", AssetClass.COMMODITY),
    ("DX-Y.NYB", "US Dollar Index", AssetClass.FX),
    ("THB=X", "USD/THB", AssetClass.FX),
    ("EURUSD=X", "EUR/USD", AssetClass.FX),
    ("JPY=X", "USD/JPY", AssetClass.FX),
    ("BTC-USD", "Bitcoin", AssetClass.CRYPTO),
    ("ETH-USD", "Ethereum", AssetClass.CRYPTO),
    ("SPY", "SPDR S&P 500 ETF", AssetClass.ETF),
    ("QQQ", "Invesco QQQ (Nasdaq 100)", AssetClass.ETF),
    ("IWM", "iShares Russell 2000 ETF", AssetClass.ETF),
    ("TLT", "iShares 20+ Year Treasury ETF", AssetClass.ETF),
    ("GLD", "SPDR Gold Shares", AssetClass.ETF),
    ("EEM", "iShares MSCI Emerging Markets ETF", AssetClass.ETF),
    ("VNQ", "Vanguard Real Estate ETF", AssetClass.ETF),
]

# FRED series (no API key needed); routed by the "FRED:" prefix.
MACRO_SERIES: dict[str, str] = {
    "FRED:DGS10": "10Y Treasury yield",
    "FRED:DGS2": "2Y Treasury yield",
    "FRED:T10Y2Y": "10Y-2Y spread (curve)",
    "FRED:FEDFUNDS": "Fed funds rate",
    "FRED:CPIAUCSL": "CPI (all urban consumers)",
    "FRED:UNRATE": "Unemployment rate",
    "FRED:VIXCLS": "VIX close",
    "FRED:DCOILWTICO": "WTI crude oil",
    "FRED:DTWEXBGS": "Broad USD index",
}


def cross_asset_universe() -> pd.DataFrame:
    rows = [
        {"symbol": s, "listed_symbol": s, "name": n, "industry": c.value, "market_cap": None,
         "asset_class": c.value}
        for s, n, c in CROSS_ASSETS
    ]
    return _frame(rows)


def load_symbol_file(path: str | Path) -> pd.DataFrame:
    """Load a user list (CSV/XLSX) with a ``symbol`` column; ``name``/``industry`` optional."""
    p = Path(path)
    raw = pd.read_excel(p, dtype=str, keep_default_na=False) if p.suffix.lower() in {".xlsx", ".xls"} \
        else pd.read_csv(p, dtype=str, keep_default_na=False)
    cols = {c.strip().lower().replace(" ", "_"): c for c in raw.columns}
    sym_col = cols.get("symbol") or cols.get("ticker")
    if sym_col is None:
        raise ValueError("file needs a 'symbol' (or 'ticker') column")
    symbols = raw[sym_col].str.strip().str.upper()
    rows = [
        {
            "symbol": s,
            "listed_symbol": s,
            "name": raw[cols["name"]].iat[i] if "name" in cols else "",
            "industry": raw[cols["industry"]].iat[i] if "industry" in cols else "",
            "market_cap": parse_market_cap(raw[cols["market_cap"]].iat[i]) if "market_cap" in cols else None,
            "asset_class": infer_asset_class(s).value,
        }
        for i, s in enumerate(symbols)
        if s
    ]
    return _frame(rows).drop_duplicates("symbol", ignore_index=True)


def search(universe: pd.DataFrame, query: str, limit: int = 25) -> pd.DataFrame:
    """Exact symbol first, then symbol prefix, then company-name substring."""
    q = query.strip()
    if not q:
        return universe.head(limit)
    qu, ql = q.upper(), q.lower()
    sym = universe["symbol"].str.upper()
    name = universe["name"].str.lower()
    rank = pd.Series(3, index=universe.index)
    rank[name.str.contains(re.escape(ql), na=False)] = 2
    rank[sym.str.startswith(qu)] = 1
    rank[sym == qu] = 0
    hits = universe[rank < 3].assign(_rank=rank[rank < 3])
    return hits.sort_values(["_rank", "symbol"]).drop(columns="_rank").head(limit)


def filter_universe(
    universe: pd.DataFrame,
    *,
    min_market_cap: float | None = None,
    max_market_cap: float | None = None,
    industries: list[str] | None = None,
    exclude_spacs: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    df = universe
    if min_market_cap is not None:
        df = df[df["market_cap"] >= min_market_cap]
    if max_market_cap is not None:
        df = df[df["market_cap"] <= max_market_cap]
    if industries:
        df = df[df["industry"].isin(industries)]
    if exclude_spacs:
        df = df[df["industry"] != SPAC_INDUSTRY]
    if limit is not None:
        df = df.sort_values("market_cap", ascending=False, na_position="last").head(limit)
    return df.reset_index(drop=True)
