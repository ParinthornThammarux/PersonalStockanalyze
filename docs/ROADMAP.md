# Roadmap

Status of the original wish list (`note.md`) and what comes next.

## Original requirements

| # | Requirement | Status | Where |
|---|---|---|---|
| 1 | Favorite system and sub-favorites | Done | Watchlists & Alerts (`Favorites` + any number of lists) |
| 2 | Time range | Done | Period and bar-size selectors (hourly / daily / weekly / monthly) |
| 3 | Additional indicators | Done | 19 in `analysis/technical/registry.py` |
| 4 | Excel support | Done | Screener export, transaction import, universe import (XLSX/CSV) |
| 5 | Optional Plotly interactive chart | Done | All charts are Plotly |
| 6 | Multi-select | Done | Indicators, assets, symbols, industries |
| 7 | Auto refresh | Partial | TTL caches + nightly job; no live tick streaming |
| 8 | Pin bar | Done | Hammer / shooting-star detectors with chart markers |
| 9 | Dashboard and summary | Done | Home and Stock Overview |
| 10 | Prediction mode | **Removed** | Properly tested, no model beat a random walk for gold, the S&P 500, Bitcoin or WTI (see Decisions) |
| 11 | Index search | Done | Search by ticker prefix or company name over 5.5k symbols |
| 12 | Detect trend | Done | `signals.trend_state`, regression slope/R², S/R zones |
| 13 | Filter and sort stock | Done | Screener |
| 14 | Tax calculator | **Not done** | Needs your jurisdiction's rules. The ledger already produces FIFO lots with holding days, and records dividends with the tax withheld |
| 15 | Simulation portfolio | Done | Transaction ledger (paper trades work too), multi-asset strategy backtest and rotation backtest, Monte Carlo with DCA through market regimes and crisis replays |
| 16 | Risk evaluation | Done | Portfolio & Risk |
| 17 | Finance and fundamental analyser | Done | Fundamentals & Valuation |
| 18 | Group indicator | Done | Registry categories + multi-panel charts |
| - | Slow PEG ratio | Fixed | No HTML scraping: computed from Yahoo data, cached 7 days |

## Decisions

- **Price forecasting removed (2026-10-31).** The original gold model looked accurate because it predicted price *levels* from
  lagged prices. Evaluated properly on 2016-2026 data (forward returns, walk-forward folds with a purge gap equal to the horizon,
  no-change / historical-mean / momentum baselines, Diebold-Mariano test, long/flat trading test after costs), Ridge,
  HistGradientBoosting and XGBoost all lost to "no change" on gold: out-of-sample R² versus a random walk of -4% to -18%,
  direction hit-rate 48-51%, negative rank IC, and the tree models were significantly *worse* (p < 0.001).
  The same test on the S&P 500 (-23% at 7 days, -44% at 21), Bitcoin (-3% / -9%) and WTI crude (-13% / -36%) found no skill
  either, so the whole ML Forecast page, its CLI command and the `ml` package were removed rather than shipping forecasts nobody
  should trade on.
- **Monte Carlo regimes are a risk tool, not a forecast (2026-11-09).** The walk-forward check (`simulation/calibration.py`) found the
  regime simulation no better than a plain block bootstrap at scoring the index's next 3 / 6 / 12 months (CRPS within 1%, Diebold-Mariano
  p 0.38-0.99 on SPY since 1993 and TDEX since 2008); it only tightens the range after a stressed start. It stays the default engine for
  its scenario controls, and the page shows the check.
- If you revisit forecasting, make a new idea clear that same bar first: predict returns (never levels), validate walk-forward with a
  purge gap, compare with no-change / historical-mean baselines, test significance with an overlap-aware test, and charge costs.

## Next

**Data**
- Thai market: done - every share on the SET and mai (887) is in `th_stocks.json` and loads with `--thai all`. SET100 membership is still not available (set.or.th blocks scripted access). Industries for the ~825 non-SET50 shares fill in from full snapshots. Index pools are snapshots from `scripts/build_pools.py`; rerun it to refresh, and a point-in-time membership history would remove the survivorship bias of testing over today's members.
- Keep the stock lists current: the US list is a packaged snapshot (new listings are missing), a refresh could come from Yahoo's screener the way the Thai list does.
- Point-in-time fundamentals and a survivorship-bias-free universe (needs a paid source) before any factor backtests.
- Intraday bars in the store; corporate-action table. (Earnings, dividend and economic calendars: done, see Calendar & News.)
- Calendar follow-ups: alerts before an earnings / XD date, SET dividend amounts and payment dates from SET itself,
  company names for SET symbols.
- A second price provider (Tiingo / Polygon / FMP behind the existing `PriceProvider` protocol) as a fallback for Yahoo outages.

**Analysis**
- Fundamental / valuation (in progress): valuation bands (P/E, P/B, P/S, EV/EBITDA, EV/Sales, yield), the statement archive, the bank mode
  (justified P/B), the growth-company tools (years at full speed, revenue × margin DCF, analyst consensus with daily snapshots, Rule of 40,
  EV/Sales per point of growth, two growth screens), the Graham value tab (Graham Number, growth formula, net current asset value,
  defensive-investor tests), the Earnings quality tab (cash backing, accruals, working-capital days, Beneish M-score, ROIC vs WACC,
  incremental ROIC, DuPont, stock pay / dilution / one-offs as flags) and the Dividends tab (record and streaks, coverage, the XD-day
  study against similar payers in the same market) are done.
  Next: peer comparison with relative valuation, Magic Formula, then a football-field summary. Earnings-quality follow-ups: the flags
  as screener columns (needs statements per stock - one request each, so for a shortlist), a lease-adjusted capital spending figure.
  Dividend follow-ups: alerts before an XD, the XD study as a screen ("fills its gap fast"), SET's own dividend amounts. Growth follow-ups: nightly fetch of estimates for the watchlist and portfolio (so the history grows without opening each
  page), a screener preset on rising EPS estimates (needs that nightly fetch - estimates are one request per stock), exit-multiple
  terminal value for the revenue model. Bank mode follow-ups: NPL / capital-ratio inputs (Yahoo has neither), a bank screener preset
  (P/B vs ROE), insurers. Ownership (insiders, institutions) is parked: Yahoo has it for US stocks only.
- Market breadth (done): participation, highs / lows, A/D, McClellan, Zweig thrust, the average stock vs the index, sector
  participation and a breadth regime with forward returns, for US and Thai stocks. Follow-ups: an alert when the regime changes or a
  thrust fires, a `breadth` command for the CLI and the nightly report, Thai industries for every share (only fetched full snapshots
  carry one), a point-in-time universe so delisted stocks count in the past.
- Ichimoku, Supertrend, volume profile; multi-timeframe confirmation in the screener.
- Sector rotation (done): US sector ETFs, asset classes, Thai baskets. Follow-ups: alerts when a sector changes quadrant or
  the risk regime flips; official SET sector membership (needs a SET source); industries below the 11 US sectors; money flow
  from ETF shares outstanding (needs a paid source - Yahoo only has the current figure).
- Portfolio (done: currencies, dividends, cash, XIRR, price / currency split). Follow-ups: a tax calculator (needs your rules; the
  ledger has FIFO lots, holding days and dividends with tax withheld), broker-specific CSV importers, a benchmark chosen per
  holding, a currency hedge view, interest on cash.
- Monte Carlo (done): ledger or custom portfolio, DCA with yearly raises or withdrawals, rebalancing, fees, inflation; 2-3 weekly
  Markov-switching regimes on SPY / TDEX / BTC; start state, longer or more frequent stress, crisis replays (dot-com to the 2025 tariff
  shock); crisis table for today's weights; full-history download; walk-forward check. Follow-ups: a return override per holding (bonds and
  gold under the stock-beta default), interest on the cash balance and including it, regime lengths drawn from the real spells
  (semi-Markov: geometric lengths understate long calm years), a rules-based regime option (trend x volatility), running a Backtest strategy
  through the simulated paths, a Thai driver before 2008 (no SET index on Yahoo; a basket of long-listed SET50 names from 2000), taxes.
- Factor exposure (market / size / value / momentum) in Portfolio & Risk.
- Walk-forward strategy optimisation (re-fit parameters on a rolling window) instead of one in-/out-of-sample split.

**Platform**
- Push notifications for alerts (email / LINE / Telegram) behind a notifier interface.
- Docker image and a scheduled run (cron container or GitHub Actions) for the nightly job.
- If it ever becomes multi-user: split a FastAPI service from the Streamlit UI and move SQLite state to Postgres.
