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
python -m stockanalyzer nightly --thai all            # fill the store with the whole US and Thai market (10-15 min the first time)
python -m stockanalyzer nightly --limit 500           # or just the 500 largest US stocks + cross-assets (about 40 s)
```

First load of 500 stocks (10 years of daily bars) took **34 s** on a laptop; the whole market (about 5,700 stocks with prices) took about
10 minutes and 350 MB, and later runs only fetch new bars. In the app: **Data Manager -> Whole market**, or the button on the Screener.
You can also just open **Stock Overview** and type any ticker - it is downloaded and cached on first use.
Tickers use Yahoo notation: `AAPL`, `BRK-B`, `PTT.BK`, `^GSPC`, `GC=F`, `BTC-USD`, `THB=X`.

## What is in the app

| Page | What it does |
|---|---|
| **Stock Overview** | Quote, candlesticks + SMAs, trend classification, recent signals, returns, volatility, drawdown, VaR, beta |
| **Technical Analysis** | 19 indicators (trend / momentum / volatility / volume) with editable parameters, multi-panel charts, candlestick patterns, signal log |
| **Support / Resistance** | ATR-adaptive zones from confirmed pivots, Fibonacci levels, nearest support/resistance |
| **Fundamentals & Valuation** | Statements (annual/quarterly), ratios, Piotroski F-score, Altman Z, PEG, **valuation bands** (P/E, P/B, P/S, EV/EBITDA, EV/Sales and dividend yield against the company's own history, with the price each ±1 / ±2 SD level implies), DCF in plain units with bear/base/bull and **years at full speed** before growth fades, a second **revenue × margin DCF** for growth companies whose free cash flow is small or negative (growth, the lasting margin and years to reach it, stock pay charged as a cost on request, reverse DCF for the growth and for the margin a price needs), **analyst consensus** (EPS and revenue growth, forward P/E and PEG, price target, EPS revisions, saved daily) compared with what the price implies, a **Growth profile** tab (Rule of 40, EV/Sales, EV/Sales per point of growth), a discount rate built from the stock's own market (US vs Thai risk-free rate, premium, tax and a beta that is not trusted below 0.8 outside the US), sensitivity grid, reverse-DCF and ADR currency conversion, a **Graham value** tab (Benjamin Graham's Graham Number, his growth formula with the AAA bond-yield adjustment and the original 1962 form, net current asset value, a margin-of-safety buy zone and the seven defensive-investor tests, each marked when it was judged on a shorter record than Graham asked for), an **Earnings quality** tab (is profit backed by cash: operating cash flow vs net income and Sloan's accruals ratio; receivable / inventory / payable days and receivables or inventory outgrowing sales; Beneish's M-score with the variable that drives it; ROIC against the cost of capital, economic profit, incremental ROIC and the growth reinvestment can fund; DuPont split of ROE; stock pay, dilution and one-off items - all summed up as red / amber / green flags), a **Dividends** tab (the record by year with streaks, cuts and growth; payout from earnings and free cash flow and years paid from borrowing; and the **XD day** measured on the stock's own history next to every liquid payer in its market with a similar dividend: how much of the dividend comes off the price, how many days the gap takes to fill, the run-up before the XD, the drift after it and whether dividend capture pays after tax and costs), and a **bank mode** for lenders (return on equity, capital and income mix, a justified price-to-book valuation with bear/base/bull and a sensitivity grid instead of the FCF DCF, Altman Z and EV/EBITDA) |
| **Screener** | 45 precomputed features for **the whole market** (about 5,600 US and Thai stocks); 10 ready-made screens as cards (including *Growth that pays: Rule of 40* and *Growth at a reasonable price*) that show how many stocks match now, criteria you build from a list of over 70 metrics (percentages typed as percentages, each row showing how many stocks it leaves), composite score; click a result for its chart, score breakdown and why it matched; save screens by name and send results to a watchlist; filter by market, size and liquidity - market cap and traded value are compared in US dollars, so Thai and US stocks share one filter, with a separate traded-value floor per market (US$5M a day for the US, US$0.3M, about 10 million baht, for Thailand; presets carry no liquidity rule of their own); P/E, P/B, market cap and yield for every stock from one market-wide snapshot, full fundamentals (ROE, margins ...) loaded on demand for the stocks a screen could still match; CSV/Excel export of every match |
| **Watchlists & Alerts** | Favorites and any number of sub-lists; price / RSI / cross / 52-week / volume alerts that fire when the condition *starts* to hold, re-arm when it ends and replay the bars of any gap in updates |
| **Calendar & News** | Earnings dates (estimate, actual, surprise and the price move each report caused), ex-dividend and payment dates, dividend yield and history, splits, key economic releases (Fed / Bank of Thailand rates, CPI, payrolls, GDP ...) in your timezone, a month view, and headlines (Thai news for SET shares) |
| **Markets & Macro** | Cross-asset performance, correlations (rolling too), FRED rates and yield-curve spread, and a **Pairs** tab: correlation matrix, cointegration scan with multiple-testing correction, pair inspector (spread z-score, half-life, lead-lag) and Johansen baskets. Stocks are only compared with stocks and crypto with crypto (`analysis/comovement.py`) |
| **Market Breadth** | Is the whole market moving, or only the index? For all US or all Thai stocks (with a traded-value floor): the share above their 50- and 200-day averages, new 52-week highs and lows, the advance / decline line, McClellan oscillator and summation index, Zweig breadth thrusts with what the index did after each, the average stock's 3-month return and the share of stocks beating the index, participation by sector (or US industry) with its change over a month, and a **breadth regime** (index trend × participation) with the index's forward returns in each regime and how many separate episodes that rests on. Plain-language readings sum up the latest day |
| **Sector Rotation** | Where relative strength - and with it money - moves between sectors, for any period you pick: a scoreboard with rank change and money flow, a Relative Rotation Graph with every quadrant change, month-by-month leaders and a rank history, and offensive-vs-defensive risk appetite. US sector ETFs, asset classes (stocks, bonds, gold, dollar, crypto) and Thai sector baskets |
| **Backtest** | 8 strategies with costs and next-open execution, parameter sweeps with in-/out-of-sample split, one strategy on one or many assets held at once (pick from index pools or the largest US stocks, shared capital, your own weights, per-asset P&L), dividends shown and either reinvested or paid out as cash, momentum rotation |
| **Portfolio & Risk** | Buys, sells, **dividends, deposits and withdrawals** (form / CSV / Excel), everything valued in one **base currency** (THB or USD ...) with each purchase converted at its own day's rate; average-cost or FIFO P&L split into stock price, **currency** and dividends; **XIRR** and money multiple next to the time-weighted return and a benchmark; cash balance; dividends worked out from ex-dates; allocation by holding / class / industry / currency; risk stats, stress tests, optimiser and rebalance orders in 100-share SET lots |
| **Monte Carlo** | Thousands of possible futures for your ledger portfolio or a custom list, with **monthly saving (DCA)** that can rise each year or monthly withdrawals, rebalancing, fees and inflation, through **market regimes** (Calm / Stressed / Crisis from a Markov-switching model on SPY, TDEX or Bitcoin weekly returns); start in today's state or a chosen one, make stress longer or more frequent, or **replay a past crisis** (dot-com, 2008, COVID, 2022 ...) first; how today's portfolio went through each crisis; a one-click download of the full history (SPY since 1993); and a walk-forward check of the regime model against a plain bootstrap |
| **Data Manager** | Load the **whole market** (all US and Thai stocks) as a background job with progress, stop and resume; or the largest N, or just your lists; market-wide P/E, P/B snapshot in one click; refresh full fundamentals; see failures and pipeline runs |

## Command line

```bash
python -m stockanalyzer status                          # what is stored, when, and recent runs
python -m stockanalyzer nightly --fundamentals 300      # update prices -> features -> market snapshot -> fundamentals -> events -> alerts
python -m stockanalyzer nightly --thai all --patience 8 # every share on the SET and mai too; wait and carry on if Yahoo throttles (--thai none|core|all)
python -m stockanalyzer calendar --days 14              # earnings, dividends and US/TH releases for your symbols
python -m stockanalyzer rotation th_sectors --months 3  # which sectors gained or lost ground (also us_sectors, asset_classes)
python -m stockanalyzer update --symbols GC=F PTT.BK    # extra symbols (also --macro for FRED series)
python -m stockanalyzer screen "Golden cross + momentum" --top 20 --csv hits.csv
python -m stockanalyzer backtest SPY sma_cross --param fast=20 --param slow=100
python -m stockanalyzer montecarlo SPY TLT PTT.BK --weights 50 30 20 --base THB --initial 1000000 --monthly 10000 --years 20 \
    --backfill --crisis gfc --check                     # DCA plan through simulated regimes, 2008 replayed first, model check
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
│   ├── fundamentals.py  cached snapshots for screening (light: one request per 250 stocks; full: one per stock) + live statements (merged with the archive) + analyst estimates (one snapshot a day)
│   ├── statement_archive.py  every statement period ever fetched, kept after Yahoo drops it
│   ├── events.py  earnings / dividend / economic calendar (cached) + headlines
│   ├── sectors.py rotation universes (US sector ETFs, asset classes, Thai baskets) + sector levels from the store
│   ├── breadth.py daily breadth counts of a whole market, counted inside DuckDB (whole US in ~2 s, Thailand in 0.5 s)
│   ├── dividend_events.py  every ex-dividend day with the prices around it (whole US market in ~2 s)
│   ├── universe.py      5.5k US tickers + the Thai list (th_stocks.json), cross-asset presets, CSV/XLSX lists
│   ├── pools.py   index pools (S&P 500/400/600, Nasdaq-100, Dow 30, SET50) + the largest US stocks, overall or per industry
│   └── quality.py, market_clock.py
├── analysis/
│   ├── technical/ indicators, registry, signals, patterns, sr_zones
│   ├── fundamental/ ratios, scores (Piotroski, Altman), valuation (DCF, revenue × margin DCF, WACC, PEG), dcf_inputs (suggested inputs), estimates (consensus), bands (multiples through time), bank (justified P/B), graham (Graham Number, growth formula, net current asset value, defensive tests), earnings_quality (accruals, working capital, Beneish, ROIC vs WACC, DuPont, flags)
│   ├── rotation.py  scoreboard, relative rotation graph, period leaders, risk appetite
│   ├── breadth.py   participation, highs / lows, A/D, McClellan, Zweig thrust, breadth regime and its forward returns
│   ├── dividends.py dividend record and streaks, coverage, the XD-day study (drop, gap fill, run-up, capture)
│   ├── regimes.py   Markov-switching regimes on an index's weekly returns (states, durations, transition matrix)
│   └── risk/      returns, drawdown, VaR/CVaR, beta, scenarios, liquidity
├── screener/      features (parallel), rules + presets, scoring, feature store
├── alerts/        watchlists, alert rules and evaluation
├── backtest/      engine (single + portfolio), strategies, sweeps
├── simulation/    Monte Carlo: pool (history weeks, estimated before a holding listed), montecarlo (regime paths, block bootstrap, DCA, crisis replay), calibration (walk-forward check)
├── portfolio/     ledger (buys, sells, dividends, cash), currency, valuation/TWR, returns (XIRR, attribution), dividends, optimiser, rebalance
├── pipelines/     nightly job, jobs.py (the same job as a detached background process the app can watch)      cli.py   command line
├── storage/state.py   SQLite: update log, watchlists, alerts, transactions, fundamentals cache
└── ui/            Streamlit app (pages hold no analysis logic - they call the packages above)
```

Local data lives in `./data` of a source checkout (git-ignored), else `~/.stockanalyzer`; override with
`STOCKANALYZER_HOME`. Prices are one Parquet file per symbol (`data/prices/1d/AAPL.parquet`), app state is SQLite.

## Design decisions worth knowing

- **No look-ahead in backtests.** A signal computed at bar *t*'s close first affects the position at bar *t+1* (default: traded at the next open, so the old position still earns the overnight gap). Tests prove that an oracle that peeks one bar ahead beats the honest version by orders of magnitude, and that every strategy's decisions are unchanged when future bars are removed.
- **Adjusted prices, incrementally.** Raw OHLC + `adj_close` are stored. When a split or dividend re-adjusts history, the overlap no longer matches the stored data and that symbol is re-downloaded in full instead of silently mixing adjustment bases - as far back as its stored history goes, so a symbol whose full history was downloaded (`backfill`, Monte Carlo page) does not shrink back to 10 years at its next dividend.
- **Only completed bars are stored.** Yahoo returns today's half-finished bar during the session; persisting it would freeze a wrong candle. The chart overlays the live bar in memory only.
- **Indicators match TA-Lib** (RSI, ATR, ADX, MACD, MFI, CCI ... to ~1e-12) but are pure pandas, so nothing needs a C library or a Windows wheel. TA-Lib is only a test oracle.
- **No price forecasting, on purpose.** A properly validated test (forward returns, walk-forward folds, baselines, significance test, costs) found no skill for gold, the S&P 500, Bitcoin or crude oil, so the feature was removed instead of shipping a chart that merely tracks price (details in [docs/ROADMAP.md](docs/ROADMAP.md#decisions)).
- **Rotation is relative strength, not fund flows.** Yahoo has no historical creations/redemptions or institutional flows, so "where money moved" is measured by how sectors ranked against the benchmark and the window before, plus Chaikin money flow (volume-weighted buying vs selling pressure). The Relative Rotation Graph uses the open RS-Ratio / RS-Momentum idea with settings measured on ten years of data (about 4-5 quadrant changes per sector a year; the first guess flipped every five weeks), not JdK's proprietary formula, and the risk-on / risk-off flag needs a 1% band so it does not flip every few days.
- **Valuation bands only use what the market already knew.** A period's EPS / book value counts from its announcement date (the cached earnings date, else period end + 45 days for a quarter / 60 for a year-end), never before; trailing EPS is a sum of four *consecutive* quarters, and a quarter Yahoo leaves blank is rebuilt from net income and the neighbouring share count (without that, PTT's P/E froze at 13.3 against Yahoo's 9.7). EV includes minority interest because consolidated EBITDA counts subsidiaries in full. Checked against Yahoo's own trailing P/E, P/B and EV/EBITDA for AAPL, NVDA, PTT.BK and KBANK.BK.
- **Growth companies get their own DCF, and their own honesty.** A DCF that starts from today's free cash flow says nothing about a company that is spending its cash on growth: measured on NVDA on 2026-11-08 it gave 66 a share against a price of 187, while revenue × margin (margin heading to 40%) gave 173. So the page offers both and opens on the revenue model when free cash flow is negative or thin next to fast growth. Growth is held at its year-1 rate for *years at full speed* (suggested 1 / 2 / 3 from 0 / 15% / 30% growth - a rule of thumb, not a forecast) and then fades in a straight line; the suggested growth is kept within 0-40% (it was 0-25%) and says when it was cut. The lasting margin is the most important input and is a placeholder when the company has no profit (a quarter of the gross margin, flagged); a result at or below zero is shown as "worth nothing at these inputs" rather than as a negative price. The reverse DCF solves for the growth *and* for the margin a price needs and sets the growth beside what analysts expect. Rule of 40 (growth + free-cash-flow margin) and EV/Sales per point of growth are screening rules of thumb, not valuations; both flatter a company whose free cash flow is boosted by stock pay.
- **One currency per portfolio, converted on the day.** A purchase is converted at the exchange rate of its own date and kept that way, so a gain splits exactly into what the stock did (in its own currency, at the sale or today's rate) and what the exchange rate added (`cost x (rate then - rate at purchase)`); the parts add up exactly to the total, and the tests check it. Average cost and prices stay in the stock's currency like a broker statement. Cash is held in the base currency only: a purchase of US stock by a baht investor draws baht at that day's USDTHB, so no conversion rows are needed.
- **Graham's rules need long records; the app says when it has less.** Graham asked for 10 years of profits, 20 of dividends and three-year averages for growth, but Yahoo gives 4-5 annual statements. Those tests are still judged on what exists and marked "shorter record than Graham asked for"; growth is measured between two-year averages when there are fewer than six years and must beat 4.2% a year (one third over the seven years between ten-year averages). The dividend test reads the price store, which for most symbols starts around 2015: JNJ, which has paid for decades, first showed "fail: 11 straight years" on real data, so a run that reaches the first year on file is now left undecided instead of failed. The growth formula's 4.4 is the 1962 US AAA yield, so the bond yield input is pre-filled as the market's risk-free rate + 0.8 points (an assumption, not market data) and the original formula is shown beside it: with low Thai yields the adjusted value is much higher (SCB.BK: 151 vs 114).
- **Two honest return measures.** The time-weighted return (prices are dividend-adjusted, so dividends count as reinvested) judges the holdings; the XIRR judges when *you* put money in. With deposits and withdrawals in the ledger only those are money in/out and idle cash dilutes the XIRR; without them every purchase is money in and every sale or dividend money out.
- **Each stock is measured against its own market.** Beta and relative strength of a `.BK` stock use the SET50 ETF (`TDEX.BK`, history since 2008) instead of the S&P 500; if it is not in the store they stay empty rather than silently using the wrong index.
- **Banks are valued on equity, not cash flow.** Bank mode uses P/B = (ROE - g) / (cost of equity - g). Yahoo's beta for Thai banks is far below 1 (KBANK 0.15 and SCB 0.13 when checked on 2026-11-02; thin trading or the wrong index), which would imply a cost of equity near the risk-free rate, so beta is kept at 1.0 or above; the risk-free rate and premium per currency are rough starting points shown on screen and editable.
- **The whole market in two tiers.** Prices of ~5,700 stocks come from the usual incremental updater. Fundamentals come in two kinds: a *light* snapshot (P/E, P/B, market cap, yield) for every listed stock from Yahoo's screener, 250 stocks per request, so the whole market takes about 15 seconds instead of one request per stock (400 stocks took 15 s when measured, 0.04 s each, though Yahoo throttles very long runs); and a *full* snapshot (ROE, margins, free cash flow) one request per stock, fetched for a shortlist - the Screener offers it for exactly the stocks that pass a screen's other criteria, largest first. A full fetch that finds nothing leaves the stock's light snapshot alone. A light snapshot never replaces a full one and a full fetch replaces a light one. Presets that need only P/E, P/B and yield ("Cheap with a dividend") work on the whole market at once; "Quality compounders" needs full snapshots.
- **A long download waits out Yahoo.** After about 4,200 symbols in 3.5 minutes Yahoo rate-limits the downloads (and the screener's crumb request, which shows up as HTTP 401). The updater stops cleanly and keeps what it stored; with `--patience N` (the app's background job uses 8) the pipeline waits 5 minutes and carries on from where it stopped instead of reporting a failure.
- **An alert fires on the crossing, not on every check.** Each alert remembers the last bar it examined and whether its condition held there, so a price that stays above its level notifies once, a cross that happened between two nightly runs is still found (up to 60 bars are replayed), and the quiet period only stops a price that flips around its level. A resumed alert starts afresh.
- **The score does not count one signal several times.** Three overlapping return/volatility columns were dropped (they correlate 0.7-0.94 with ones that stay), the trend-quality column is R² signed by the slope (a plain R² rewards a smooth decline), and value and quality are ranked against peers - the same industry in the same market, else the whole market - because a market-wide rank made banks (average value score 75) look cheap and semiconductors (22) expensive, and Thai stocks cheaper than the rest (68 vs 49) just for being Thai. After the change every industry averages about 50.
- **The Screener starts from what you want to find, not from a rule table.** Each ready-made screen is a card with the number of stocks that match it *right now* (the counts follow the market, size and liquidity filters), each criterion is a row - metric, condition, value - with the input that suits the column (a % box for returns and yields, millions for market cap, yes / no for flags, a list for trend labels) and a bar under it saying how many stocks are left after it and how many it would keep alone, with where most of the pool sits to pick a threshold against; and the first result is explained without a click: chart, the score broken into momentum / trend / risk / value / quality, and each criterion with this stock's value and a tick or cross. A screen that needs ROE or margins no longer returns an empty list: the card says "needs fundamentals" and the page offers to load them for the stocks that already pass the other criteria (with a time estimate from the measured speed). Screens, with their sort order and universe filters, can be saved by name, and the shown results added to any watchlist in one click.
- **A missing moving average is unknown, not bearish.** A stock with fewer than 200 bars has no SMA200: its trend label uses the votes that exist (3 of 3 agreeing is an uptrend) and `above_sma200` is empty rather than False.
- **Breadth counts the market as it was, with one honest omission.** Each day only stocks that traded at least the chosen amount over the previous 20 days count, and the counting happens in DuckDB window functions, so there is no cache to go stale. Days when only part of the store had been updated (a stock opened in the app gets today's bar before the nightly run) and the thin start of the history (a few symbols reach back to 2000) are dropped. Delisted companies are missing, so past breadth leans a little positive; the page says so. A long-run "equal weight vs index" line was tried and dropped: compounding a daily-rebalanced average of thousands of small stocks showed the average US stock beating the S&P 500 by 13 points over 2023-26, the opposite of what happened, because bid-ask bounce and the missing delisted stocks inflate it. The page shows the average stock's own 3-month return and the share of stocks beating the index instead (US: 46% on an average day since 2017, 38% on 2026-10-30). The regime table is descriptive: on Thai stocks a healthy uptrend was followed by +3.1% over 3 months on average against -0.6% for a narrow one, while on US stocks a narrow uptrend did *better* (+5.0% vs +2.9%) - so no rule is built on it.
- **Earnings quality compares like with like.** Operating cash flow covers the whole group, so it is set against net income *including* minority shareholders' part (PTT's ratio was 361% against the parent's share, 273% against the group's). Receivables or inventory outgrowing sales is only flagged when the item is at least 5% of assets (CP ALL's receivables are two days of sales; Apple's inventory 2% of assets). Beneish's M-score flags NVDA at -1.13 almost entirely through sales growth (SGI 1.65-2.26), so a score driven mainly by growth is shown as amber with that explanation, not red. Incremental ROIC is only shown when invested capital grew at least 5% (PTT's barely moved and gave -388%). Capital spending against depreciation carries a lease note: under IFRS / TFRS 16 leased stores and planes are depreciated but not capitalised as spending.
- **The XD study compares a stock with payers of a similar dividend.** A big dividend takes longer to fill than a small one, so the peer column holds only XDs whose dividend was 0.5-1.5× the stock's usual share of the price. Drop ratios are only computed for dividends of at least 0.5% of the price: below that the day's normal move swamps them (Apple's quarterly dividend gave ratios from -48 to +24). Measured on the store (2026-11-08): liquid Thai payers gave up a median 87% of the dividend on the XD day (93% for dividends of 2% or more - close to the 90% the 10% withholding tax predicts), beat the market by 1.2% in the 10 days before and took a median 6 trading days to fill the gap (17 for big dividends); buying the day before to collect the dividend lost 0.2% on average after tax and 0.3% costs. US payers gave up about 100% and capture lost 0.3%.
- **The Monte Carlo describes risk, it does not predict.** Regimes are fitted on *weekly* index returns because daily fits only find short volatility bursts (SPY states lasted 53 / 21 days); weekly on SPY since 1993 gives Calm spells of ~37 weeks (11% volatility) and Stressed ones of ~14 weeks (27%), 28-29 episodes each, with Stressed covering 87% of the 2008 weeks and all of 2022. Three states turn the third into a rare crash state (5 episodes) and single fits of it landed in junk optima in 2 of 5 seeds, so 2 states are the default and 3-state fits keep the best of five. The average return inside a state is noise (Thai stocks' and Bitcoin's high-volatility state had a *positive* mean), so the expected return is a stated assumption: by default each holding earns the cash rate plus its beta times the index's excess return (CAPM), because bootstrapping each holding's own decade projected NVDA at 56% a year; the cash rate keeps a bond fund from being assumed to earn nothing. Weeks before a holding listed are estimated from the index (beta in that state + a residual from one of its own weeks in the same state, the same residual week for holdings that traded together). The walk-forward check (every 4 weeks since 2003, re-fitted yearly, Diebold-Mariano on CRPS with Newey-West) found the regime model **no better than a plain block bootstrap** at 3, 6 and 12 months on SPY and TDEX (score differences under 1%, p 0.38-0.99); what it does fix is the range after a stressed start (SPY 3-month 90% band held the real outcome 91% of the time vs 82%) and the too-wide 50% band (51% vs 57%). It stays the default because it scores the same and makes the scenario controls possible; the page shows the check.
- **Screening is precomputed.** Features for 5.5k symbols are one DuckDB read (1.3 s for 400 bars each) plus parallel pandas; the page then filters a table instead of recomputing indicators.

## Performance (measured)

| Task | Result |
|---|---|
| Initial load, 520 symbols x 10 years (1.2 M rows) | 34 s end to end |
| Read 400 bars of 5,500 symbols from the store | 1.3 s |
| Whole market first load: 6,284 symbols selected, 5,711 with prices (353 MB) | ~4 min until Yahoo throttled at ~4,200 symbols, ~1 min after a pause |
| Features for 5,635 symbols | 12 s |
| P/E, P/B, market cap, yield of 4,771 US + 887 Thai stocks | 14 s |
| Latest timestamp of every symbol | 0.3 s |
| Features for 515 symbols | 1.7 s |
| Full fundamentals (ROE, margins, cash flow) of the 400 largest stocks, 4 requests in parallel | 15 s |

## Tests

```bash
pip install -e ".[dev,talib]"      # TA-Lib is optional; its tests are skipped without it
pytest                             # ~600 tests, ~2 min, fully offline
pytest -m network                  # 4 live checks against Yahoo / FRED (catch upstream format changes)
```

Covered: providers/normalisation, store, updater edge cases (retry, backoff, splits, delisted tickers), market clock,
indicators vs TA-Lib, signals, zones, fundamentals, risk, screener, alerts, event calendar (timezones, caching, report
reactions), sector rotation (maths on data with known answers), market breadth (counts on markets of straight-line stocks, partial days, liquidity floor, McClellan / Zweig / regime definitions), earnings quality (accruals, working-capital days, Beneish M-score and ROIC on hand-checked statements), the dividend record and XD study (drop, gap fill and capture on known price paths), pipeline, backtest timing/costs, multi-asset holding and dividend handling, portfolio accounting (currencies, dividends, cash, XIRR, the price / currency split) and optimisation, bank valuation, Graham valuation (formulas on hand-checked numbers, the defensive tests and their short-record handling), Thai-market handling (benchmarks, dollar-comparable screening), the regime model (recovering a known 2-state chain) and the Monte Carlo (DCA arithmetic on constant returns, fees, withdrawals running out, same-state block draws, crisis replays, estimated weeks keeping beta and co-movement, the full-history download keeping its depth), and headless runs of every Streamlit page.
Verified on Python 3.11 with pandas 2.2, Python 3.12 with pandas 3.1 and Python 3.14 with pandas 3.1.

## Limits you should know about

- Data comes from Yahoo Finance through `yfinance` (unofficial, rate-limited, occasionally wrong). The provider interface is small if you want to plug in Tiingo/Polygon/FMP.
- The ticker list is a **snapshot**: delisted, renamed or acquired companies are missing or fail with "no data", and past backtests over it carry survivorship bias.
- Non-positive prices are dropped as invalid, which removes the real WTI crude print of -$37 on 2020-04-20.
- Altman Z is not meaningful for banks and insurers, and Yahoo's fundamentals are as-reported, not point-in-time (do not backtest value factors with them).
- Analyst consensus covers a year or two ahead, is thin for Thai stocks (a handful of analysts, no quarterly revenue) and leans optimistic; Yahoo's `growth_estimates` table is ignored because its index column is the S&P 500 for every stock. Estimate history starts the day you first open a company (Yahoo keeps only 7 / 30 / 60 / 90-day snapshots). Full fundamentals saved before the growth columns existed (`rule_of_40`, `ev_sales`, `target_upside` ...) are refetched the next time they are requested.
- Yahoo serves only about 4-5 annual and 5-6 quarterly periods, so a valuation band starts 3-4 years back. Every company you open is archived (`statements_archive` in the state database), so the history lengthens by one period per report from then on - periods Yahoo dropped before you first opened the company are gone. Bands use one exchange rate for the whole history for ADRs, and are indicative for companies whose per-share figures are on another share basis than the quoted price (the page warns when its P/E differs from Yahoo's by over 25%).
- The price store recorded **no splits** before this fix (`Stock Splits` was never mapped to the `splits` column). New downloads and the automatic full re-download after a split are correct; to backfill an existing symbol, delete its file under `data/prices/1d/` and run `update` again.
- Calendar data is Yahoo's: upcoming dividend *amounts* and SET payment dates are often missing until announced,
  and Yahoo publishes economic releases only about five weeks ahead. Headlines come from Yahoo (US symbols) and
  Google News RSS (SET symbols, topic search); both are unofficial feeds.
- Yahoo has no SET sector indices, and serves a single bar for the SET and SET50 indices themselves (`^SET.BK`, `^SET50.BK`; the SET row is therefore missing from Markets & Macro). The Thai sectors are equal-weight baskets of large liquid stocks, measured against the average of the sectors - a proxy, not the official indices. The basket members are a fixed list: delisted or renamed stocks need editing in `data/sectors.py`.
- Portfolio: exchange rates are Yahoo's daily closes, not your broker's (which adds a spread), so the currency part of a gain is an estimate within a percent or so. Dividends are only as good as the ledger: the page lists ex-dates you held but did not record, with gross amounts, and you set the tax withheld. Cash earns no interest, withholding tax is not reclaimed, and no tax is computed (see the roadmap).
- Bank mode: Yahoo has no bad-loan ratio or regulatory capital ratio, and its loan figures are patchy; the cost of equity is a rule of thumb, which is why the page shows how much the value moves with it. Multi-asset backtests of assets in different currencies are not converted (the page says so).
- The stock lists are snapshots: the US list (5,504 tickers, 2026-10-31) misses companies listed since, and about 575 of its tickers have no prices on Yahoo any more (delisted, renamed) - the screener counts them out. The Thai list (887 shares on the SET and mai, 2026-11-02) comes from Yahoo's screener minus depositary receipts, NVDRs and warrants (`scripts/build_th_stocks.py`); only the 62 SET50 / sector-basket names carry an industry until a full snapshot is fetched. Most Thai small caps trade under US$0.1M a day, so with the default liquidity filter about 60 Thai stocks qualify.
- Light snapshots carry no ROE, margins, free cash flow or debt: screens on those return nothing for a stock until its full snapshot is fetched (the Screener has a button for the shortlist). Yahoo's P/E for a company with negative earnings is empty, so loss-makers never match a P/E rule.
- Alerts are logged, not pushed. Nothing here places orders or gives advice.

See [docs/ROADMAP.md](docs/ROADMAP.md) for what is done and what is next.
- Monte Carlo: the paths are made of history - a crisis worse or different from those in the pool cannot appear, and with the default 10-year store the only real crash is COVID (download the full history on the page). Weeks before a holding listed, and before a currency's rate history starts (THB=X: Dec 2003), are estimated or converted at the first known rate. The market-implied return uses beta to *stocks*, so long bonds and gold get about the cash rate (less with a negative beta) - pick "Halfway" or "Own history" for bond-heavy portfolios. No taxes, no change of holdings over time, the cash balance is left out.
