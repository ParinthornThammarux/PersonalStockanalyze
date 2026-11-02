"""Rebuild stockanalyzer/resources/pools.json: index membership lists copied from public pages.

    python scripts/build_pools.py

Each list is a snapshot of the day it is built - indexes are reconstituted every few months, so rerun this to refresh.

    S&P 500 / 400 / 600   Wikipedia "List of S&P ... companies"
    Nasdaq-100            Nasdaq's public quote API
    Dow Jones 30          State Street SPDR DIA daily holdings (the ETF holds exactly the index)
    SET50                 Wikipedia "SET50 Index and SET100 Index"

SET100 is not included: set.or.th refuses scripted access, and this script does not try to get around that. Import it
yourself as a CSV/XLSX with a ``symbol`` column (Data Manager) if you need it.
"""

from __future__ import annotations

import io
import json
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stockanalyzer.data.universe import us_listed_to_yahoo  # noqa: E402

OUTPUT = ROOT / "stockanalyzer" / "resources" / "pools.json"
AGENT = "stockanalyzer-pool-builder/1.0"  # says what this is; carries no personal information


def fetch(url: str, accept: str = "text/html") -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": AGENT, "Accept": accept})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def wikipedia_table(page: str, column: str) -> pd.DataFrame:
    html = fetch(f"https://en.wikipedia.org/wiki/{page}").decode("utf-8")
    for table in pd.read_html(io.StringIO(html)):
        if column in table.columns and len(table) >= 20:
            return table
    raise ValueError(f"no table with a {column!r} column on Wikipedia page {page}")


def us_symbols(raw) -> list[str]:
    out = [us_listed_to_yahoo(str(s)) for s in raw if isinstance(s, str) and re.fullmatch(r"[A-Za-z]{1,5}([.\-/][A-Za-z]{1,2})?", s.strip())]
    return list(dict.fromkeys(s.replace("/", "-") for s in out))


def sp(page: str) -> list[str]:
    return us_symbols(wikipedia_table(page, "Symbol")["Symbol"])


def nasdaq100() -> list[str]:
    rows = json.loads(fetch("https://api.nasdaq.com/api/quote/list-type/nasdaq100", "application/json"))["data"]["data"]["rows"]
    return us_symbols(r["symbol"] for r in rows)


def dow30() -> tuple[list[str], str]:
    raw = pd.read_excel(io.BytesIO(fetch("https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-dia.xlsx")),
                        header=None, dtype=str)
    as_of = pd.to_datetime(raw.iat[2, 1].replace("As of", "").strip(), format="%d-%b-%Y").date().isoformat()
    head = raw.index[raw[0] == "Name"][0]
    table = raw.iloc[head + 1:].set_axis(raw.iloc[head], axis=1)
    return us_symbols(table["Ticker"]), as_of


def set50() -> list[str]:
    return [f"{s.strip().upper()}.BK" for s in wikipedia_table("SET50_Index_and_SET100_Index", "Symbol")["Symbol"].astype(str)]


def build(output: Path = OUTPUT) -> dict[str, int]:
    today = date.today().isoformat()
    dow, dow_date = dow30()
    pools = [
        ("sp500", "S&P 500", "Large-cap US stocks", sp("List_of_S%26P_500_companies"), "Wikipedia", today),
        ("sp400", "S&P MidCap 400", "Mid-cap US stocks", sp("List_of_S%26P_400_companies"), "Wikipedia", today),
        ("sp600", "S&P SmallCap 600", "Small-cap US stocks", sp("List_of_S%26P_600_companies"), "Wikipedia", today),
        ("ndx", "Nasdaq-100", "The 100 largest non-financial Nasdaq stocks", nasdaq100(), "Nasdaq API", today),
        ("dow", "Dow Jones 30", "30 blue chips, price-weighted", dow, "SPDR DIA holdings", dow_date),
        ("set50", "SET50 (Thailand)", "The 50 largest, most liquid SET stocks", set50(), "Wikipedia", today),
    ]
    out = [{"key": k, "label": label, "description": desc, "source": src, "as_of": as_of, "symbols": syms}
           for k, label, desc, syms, src, as_of in pools]
    output.parent.mkdir(parents=True, exist_ok=True)
    body = ",\n".join("    " + json.dumps(p, ensure_ascii=False) for p in out)
    output.write_text(f'{{\n  "metadata": {json.dumps({"built": today})},\n  "pools": [\n{body}\n  ]\n}}\n', "utf-8")
    return {p["key"]: len(p["symbols"]) for p in out}


if __name__ == "__main__":
    for key, n in build().items():
        print(f"{key:6s} {n:4d} symbols")
    print(f"wrote {OUTPUT}")
