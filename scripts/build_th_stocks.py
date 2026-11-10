"""Rebuild stockanalyzer/resources/th_stocks.json: every share listed on the SET and mai, from Yahoo Finance.

    python scripts/build_th_stocks.py

Yahoo's screener lists the Thai market (about 2,300 instruments) in a handful of requests. Depositary receipts of foreign
stocks, NVDRs, warrants and preferred lines are dropped (`universe.is_thai_common_stock`), what is left is the shares
(plus the REITs, funds and ETFs that trade like them). The market cap is converted to US dollars at the day's rate so it
can sit next to the packaged US list.

The screener has no industry, so industries are kept from the previous file where a symbol was already in it (the SET50 and
the sector baskets, taken from each company's Yahoo profile) and are otherwise left empty - the Screener fills them from
the fundamentals snapshot once it has been fetched. It is a snapshot: rerun it when new companies list.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stockanalyzer.data import pools, sectors, universe as uni  # noqa: E402
from stockanalyzer.data.providers.yfinance_provider import YFinanceProvider  # noqa: E402

OUTPUT = ROOT / "stockanalyzer" / "resources" / "th_stocks.json"


def main() -> None:
    previous = {r["yahoo"]: r for r in json.loads(OUTPUT.read_text("utf-8"))["stocks"]} if OUTPUT.exists() else {}
    usd_per_thb = 1.0 / float(yf.Ticker("THB=X").history(period="5d")["Close"].iloc[-1])
    quotes = YFinanceProvider().market_quotes("th")
    core = set(pools.load_pools()["set50"].symbols) | set(sectors.THAI_SECTORS.members)
    rows = {}
    for q in quotes:
        symbol = str(q.get("symbol") or "").upper()
        if not uni.is_thai_common_stock(symbol, q.get("shortName")):
            continue
        cap = q.get("marketCap")
        name = q.get("longName") or (q.get("shortName") or symbol).split("_", 1)[-1].title()
        rows[symbol] = {"symbol": symbol.removesuffix(".BK"), "yahoo": symbol, "name": name,
                        "industry": previous.get(symbol, {}).get("industry", ""),
                        "market_cap": round(cap * usd_per_thb) if cap else None}
    missing = sorted(core - set(rows))
    for symbol in missing:  # core names the screener did not list: keep what the previous file knew
        if symbol in previous:
            rows[symbol] = previous[symbol]
    ordered = sorted(rows.values(), key=lambda r: r["yahoo"])
    OUTPUT.write_text(json.dumps({"metadata": {"count": len(ordered), "source": "Yahoo Finance screener", "built": date.today().isoformat(),
                                               "currency": "USD"}, "stocks": ordered}, ensure_ascii=False, indent=1), "utf-8")
    sized = sum(1 for r in ordered if r["market_cap"])
    print(f"wrote {len(ordered)} stocks ({sized} with a market cap, {sum(1 for r in ordered if r['industry'])} with an industry) "
          f"from {len(quotes)} listed instruments to {OUTPUT}" + (f"; core names not listed by Yahoo: {', '.join(missing)}" if missing else ""))


if __name__ == "__main__":
    main()
