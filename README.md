# Stock Analyzer

A personal, multi-asset analysis platform: stocks, ETFs, indices, commodities, FX and crypto.
It keeps a local price store fresh, analyses one symbol in depth, screens thousands at once,
backtests ideas honestly and tracks portfolio risk.

```
data store (Parquet + DuckDB)  ->  indicators / fundamentals  ->  screener / alerts
          |                               |                          |
   incremental updates             deep-dive pages            backtest · portfolio
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install -e .                                       # add ".[dev]" for pytest

streamlit run app/streamlit_app.py                     # the web app
python -m stockanalyzer nightly --limit 500            # fill the store with the 500 largest US stocks + cross-assets
```

First load of 500 stocks (10 years of daily bars) took **34 s** on a laptop; later runs only fetch new bars.
You can also just open **Stock Overview** and type any ticker - it is downloaded and cached on first use.
Tickers use Yahoo notation: `AAPL`, `BRK-B`, `PTT.BK`, `^GSPC`, `GC=F`, `BTC-USD`, `THB=X`.

## What is in the app

| Page | What it does |
|---|---|
| **Stock Overview** | Quote, candlesticks + SMAs, trend classification, recent signals, returns, volatility, drawdown, VaR, beta |
| **Technical Analysis** | 19 indicators (trend / momentum / volatility / volume) with editable parameters, multi-panel charts, candlestick patterns, signal log |
| **Support / Resistance** | ATR-adaptive zones from confirmed pivots, Fibonacci levels, nearest support/resistance |
| **Fundamentals & Valuation** | Statements (annual/quarterly), ratios, Piotroski F-score, Altman Z, PEG, **valuation bands** (P/E, P/B, EV/EBITDA and dividend yield against the company's own history, with the price each ±1 / ±2 SD level implies), DCF in plain units with bear/base/bull, sensitivity grid, reverse-DCF and ADR currency conversion |
| **Screener** | 45 precomputed features for the whole universe, 7 presets, editable rules, composite score, CSV/Excel export |
| **Watchlists & Alerts** | Favorites and any number of sub-lists; price / RSI / cross / 52-week / volume alerts with cooldown |
| **Calendar & News** | Earnings dates (estimate, actual, surprise and the price move each report caused), ex-dividend and payment dates, dividend yield and history, splits, key economic releases (Fed / Bank of Thailand rates, CPI, payrolls, GDP ...) in your timezone, a month view, and headlines (Thai news for SET shares) |
| **Markets & Macro** | Cross-asset performance, correlations (rolling too), FRED rates and yield-curve spread |
| **Sector Rotation** | Where relative strength - and with it money - moves between sectors, for any period you pick: a scoreboard with rank change and money flow, a Relative Rotation Graph with every quadrant change, month-by-month leaders and a rank history, and offensive-vs-defensive risk appetite. US sector ETFs, asset classes (stocks, bonds, gold, dollar, crypto) and Thai sector baskets |
| **Backtest** | 8 strategies with costs and next-open execution, parameter sweeps with in-/out-of-sample split, one strategy on one or many assets held at once (pick from index pools or the largest US stocks, shared capital, your own weights, per-asset P&L), dividends shown and either reinvested or paid out as cash, momentum rotation |
| **Portfolio & Risk** | Transactions (form / CSV / Excel), average-cost or FIFO P&L, allocation, TWR risk stats, stress tests, optimiser, rebalance orders |
| **Data Manager** | Update the store, refresh fundamentals, see failures and pipeline runs |

## Command line

```bash
python -m stockanalyzer status                          # what is stored, when, and recent runs
python -m stockanalyzer nightly --fundamentals 300      # update prices -> features -> fundamentals -> events -> alerts
python -m stockanalyzer calendar --days 14              # earnings, dividends and US/TH releases for your symbols
python -m stockanalyzer rotation th_sectors --months 3  # which sectors gained or lost ground (also us_sectors, asset_classes)
python -m stockanalyzer update --symbols GC=F PTT.BK    # extra symbols (also --macro for FRED series)
python -m stockanalyzer screen "Golden cross + momentum" --top 20 --csv hits.csv
python -m stockanalyzer backtest SPY sma_cross --param fast=20 --param slow=100
python -m stockanalyzer alerts check
```

Run it unattended after the US close, in your machine's timezone. In Bangkok that is early morning,
Tuesday to Saturday (the US closes 16:00 New York = 03:00/04:00 Bangkok), which also catches the previous SET session:

```cron
30 5 * * 2-6  cd /path/to/PersonalStockanalyze && .venv/bin/python -m stockanalyzer nightly --fundamentals 300
```

Event times are shown in the machine's timezone; set `STOCKANALYZER_TZ` (e.g. `Asia/Bangkok`) to override.

## Architecture

```
stockanalyzer/
├── domain/        Asset, AssetClass, FinancialStatements
├── data/
│   ├── providers/ yfinance (prices, statements, info, calendars, news), FRED (macro), Google News RSS
│   ├── store.py   Parquet per symbol, queried across symbols with DuckDB
│   ├── updater.py incremental, batched, retried, split-aware
│   ├── service.py MarketData: the one entry point for prices (store-first, live bar overlay)
│   ├── fundamentals.py  cached snapshots for screening + live statements (merged with the archive)
│   ├── statement_archive.py  every statement period ever fetched, kept after Yahoo drops it
│   ├── events.py  earnings / dividend / economic calendar (cached) + headlines
│   ├── sectors.py rotation universes (US sector ETFs, asset classes, Thai baskets) + sector levels from the store
│   ├── universe.py      5.5k US tickers, cross-asset presets, CSV/XLSX lists
│   ├── pools.py   index pools (S&P 500/400/600, Nasdaq-100, Dow 30, SET50) + the largest US stocks, overall or per industry
│   └── quality.py, market_clock.py
├── analysis/
│   ├── technical/ indicators, registry, signals, patterns, sr_zones
│   ├── fundamental/ ratios, scores (Piotroski, Altman), valuation (DCF, WACC, PEG), bands (multiples through time)
│   ├── rotation.py  scoreboard, relative rotation graph, period leaders, risk appetite
│   └── risk/      returns, drawdown, VaR/CVaR, beta, scenarios, liquidity
├── screener/      features (parallel), rules + presets, scoring, feature store
├── alerts/        watchlists, alert rules and evaluation
├── backtest/      engine (single + portfolio), strategies, sweeps
├── portfolio/     ledger, valuation/TWR, optimiser, rebalance
├── pipelines/     nightly job      cli.py   command line
├── storage/state.py   SQLite: update log, watchlists, alerts, transactions, fundamentals cache
└── ui/            Streamlit app (pages hold no analysis logic - they call the packages above)
```

Local data lives in `./data` of a source checkout (git-ignored), else `~/.stockanalyzer`; override with
`STOCKANALYZER_HOME`. Prices are one Parquet file per symbol (`data/prices/1d/AAPL.parquet`), app state is SQLite.

## Design decisions worth knowing

- **No look-ahead in backtests.** A signal computed at bar *t*'s close first affects the position at bar *t+1* (default: traded at the next open, so the old position still earns the overnight gap). Tests prove that an oracle that peeks one bar ahead beats the honest version by orders of magnitude, and that every strategy's decisions are unchanged when future bars are removed.
- **Adjusted prices, incrementally.** Raw OHLC + `adj_close` are stored. When a split or dividend re-adjusts history, the overlap no longer matches the stored data and that symbol is re-downloaded in full instead of silently mixing adjustment bases.
- **Only completed bars are stored.** Yahoo returns today's half-finished bar during the session; persisting it would freeze a wrong candle. The chart overlays the live bar in memory only.
- **Indicators match TA-Lib** (RSI, ATR, ADX, MACD, MFI, CCI ... to ~1e-12) but are pure pandas, so nothing needs a C library or a Windows wheel. TA-Lib is only a test oracle.
- **No price forecasting, on purpose.** A properly validated test (forward returns, walk-forward folds, baselines, significance test, costs) found no skill for gold, the S&P 500, Bitcoin or crude oil, so the feature was removed instead of shipping a chart that merely tracks price (details in [docs/ROADMAP.md](docs/ROADMAP.md#decisions)).
- **Rotation is relative strength, not fund flows.** Yahoo has no historical creations/redemptions or institutional flows, so "where money moved" is measured by how sectors ranked against the benchmark and the window before, plus Chaikin money flow (volume-weighted buying vs selling pressure). The Relative Rotation Graph uses the open RS-Ratio / RS-Momentum idea with settings measured on ten years of data (about 4-5 quadrant changes per sector a year; the first guess flipped every five weeks), not JdK's proprietary formula, and the risk-on / risk-off flag needs a 1% band so it does not flip every few days.
- **Valuation bands only use what the market already knew.** A period's EPS / book value counts from its announcement date (the cached earnings date, else period end + 45 days for a quarter / 60 for a year-end), never before; trailing EPS is a sum of four *consecutive* quarters, and a quarter Yahoo leaves blank is rebuilt from net income and the neighbouring share count (without that, PTT's P/E froze at 13.3 against Yahoo's 9.7). EV includes minority interest because consolidated EBITDA counts subsidiaries in full. Checked against Yahoo's own trailing P/E, P/B and EV/EBITDA for AAPL, NVDA, PTT.BK and KBANK.BK.
- **Screening is precomputed.** Features for 5.5k symbols are one DuckDB read (1.3 s for 400 bars each) plus parallel pandas; the page then filters a table instead of recomputing indicators.

## Performance (measured)

| Task | Result |
|---|---|
| Initial load, 520 symbols x 10 years (1.2 M rows) | 34 s end to end |
| Read 400 bars of 5,500 symbols from the store | 1.3 s |
| Latest timestamp of every symbol | 0.3 s |
| Features for 515 symbols | 1.7 s |

## Tests

```bash
pip install -e ".[dev,talib]"      # TA-Lib is optional; its tests are skipped without it
pytest                             # ~280 tests, ~40 s, fully offline
pytest -m network                  # 4 live checks against Yahoo / FRED (catch upstream format changes)
```

Covered: providers/normalisation, store, updater edge cases (retry, backoff, splits, delisted tickers), market clock,
indicators vs TA-Lib, signals, zones, fundamentals, risk, screener, alerts, event calendar (timezones, caching, report
reactions), sector rotation (maths on data with known answers), pipeline, backtest timing/costs, multi-asset holding and dividend handling, portfolio accounting and optimisation, and headless runs of every Streamlit page.
Verified on Python 3.11 with pandas 2.2, Python 3.12 with pandas 3.1 and Python 3.14 with pandas 3.1.

## Limits you should know about

- Data comes from Yahoo Finance through `yfinance` (unofficial, rate-limited, occasionally wrong). The provider interface is small if you want to plug in Tiingo/Polygon/FMP.
- The ticker list is a **snapshot**: delisted, renamed or acquired companies are missing or fail with "no data", and past backtests over it carry survivorship bias.
- Non-positive prices are dropped as invalid, which removes the real WTI crude print of -$37 on 2020-04-20.
- Altman Z is not meaningful for banks and insurers, and Yahoo's fundamentals are as-reported, not point-in-time (do not backtest value factors with them).
- Yahoo serves only about 4-5 annual and 5-6 quarterly periods, so a valuation band starts 3-4 years back. Every company you open is archived (`statements_archive` in the state database), so the history lengthens by one period per report from then on - periods Yahoo dropped before you first opened the company are gone. Bands use one exchange rate for the whole history for ADRs, and are indicative for companies whose per-share figures are on another share basis than the quoted price (the page warns when its P/E differs from Yahoo's by over 25%).
- The price store recorded **no splits** before this fix (`Stock Splits` was never mapped to the `splits` column). New downloads and the automatic full re-download after a split are correct; to backfill an existing symbol, delete its file under `data/prices/1d/` and run `update` again.
- Calendar data is Yahoo's: upcoming dividend *amounts* and SET payment dates are often missing until announced,
  and Yahoo publishes economic releases only about five weeks ahead. Headlines come from Yahoo (US symbols) and
  Google News RSS (SET symbols, topic search); both are unofficial feeds.
- Yahoo has no SET sector indices, and serves a single bar for the SET and SET50 indices themselves (`^SET.BK`, `^SET50.BK`; the SET row is therefore missing from Markets & Macro). The Thai sectors are equal-weight baskets of large liquid stocks, measured against the average of the sectors - a proxy, not the official indices. The basket members are a fixed list: delisted or renamed stocks need editing in `data/sectors.py`.
- Alerts are logged, not pushed. Nothing here places orders or gives advice.

See [docs/ROADMAP.md](docs/ROADMAP.md) for what is done and what is next.
