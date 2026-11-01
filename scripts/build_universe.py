"""Rebuild stockanalyzer/resources/us_stocks.json from an Excel/CSV symbol list.

    python scripts/build_universe.py [scripts/Stock_Symbols.xlsx]

Expected columns: Symbol, Company Name, Industry, Market Cap (e.g. "34.44B").
Cells are read as text on purpose: pandas would otherwise turn the tickers "NA"
into NaN and "TRUE" into a boolean.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stockanalyzer.data.universe import us_listed_to_yahoo  # noqa: E402
from stockanalyzer.domain.assets import parse_market_cap  # noqa: E402

DEFAULT_SOURCE = ROOT / "scripts" / "Stock_Symbols.xlsx"
OUTPUT = ROOT / "stockanalyzer" / "resources" / "us_stocks.json"


def build(source: Path, output: Path = OUTPUT) -> int:
    reader = pd.read_excel if source.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv
    df = reader(source, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()

    stocks, seen = [], set()
    for _, row in df.iterrows():
        symbol = row["Symbol"].strip()
        if symbol in {"True", "False"}:  # Excel stored the ticker TRUE as a boolean cell
            symbol = symbol.upper()
        symbol = symbol.upper()
        yahoo = us_listed_to_yahoo(symbol)
        if not symbol or yahoo in seen:
            continue
        seen.add(yahoo)
        stocks.append(
            {
                "symbol": symbol,
                "yahoo": yahoo,
                "name": row["Company Name"].strip(),
                "industry": row["Industry"].strip(),
                "market_cap": parse_market_cap(row["Market Cap"]),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ",\n".join("    " + json.dumps(s, ensure_ascii=False) for s in stocks)
    meta = json.dumps({"count": len(stocks), "source": source.name, "built": date.today().isoformat()})
    output.write_text(f'{{\n  "metadata": {meta},\n  "stocks": [\n{lines}\n  ]\n}}\n', "utf-8")
    return len(stocks)


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    print(f"wrote {build(src)} symbols to {OUTPUT}")
