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
| 14 | Tax calculator | **Not done** | Needs your jurisdiction's rules. The ledger already produces FIFO lots with holding days |
| 15 | Simulation portfolio | Done | Transaction ledger (paper trades work too), multi-asset strategy backtest and rotation backtest |
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
- If you revisit forecasting, make a new idea clear that same bar first: predict returns (never levels), validate walk-forward with a
  purge gap, compare with no-change / historical-mean baselines, test significance with an overlap-aware test, and charge costs.

## Next

**Data**
- Thai market universe beyond SET50 (SET100 / mai: set.or.th blocks scripted access, so import a CSV -> `load_symbol_file`; the `.BK` close time is already handled). Index pools are snapshots from `scripts/build_pools.py`; rerun it to refresh, and a point-in-time membership history would remove the survivorship bias of testing over today's members.
- Point-in-time fundamentals and a survivorship-bias-free universe (needs a paid source) before any factor backtests.
- Intraday bars in the store; corporate-action table. (Earnings, dividend and economic calendars: done, see Calendar & News.)
- Calendar follow-ups: alerts before an earnings / XD date, SET dividend amounts and payment dates from SET itself,
  company names for SET symbols.
- A second price provider (Tiingo / Polygon / FMP behind the existing `PriceProvider` protocol) as a fallback for Yahoo outages.

**Analysis**
- Fundamental / valuation (in progress): valuation bands and the statement archive are done. Next: peer comparison with
  relative valuation, analyst consensus and EPS-estimate revisions, a bank mode (justified P/B instead of FCF DCF), Beneish
  M-score / accruals / Magic Formula, dividend quality, then a football-field summary and screener columns for all of it.
  Ownership (insiders, institutions) is parked: Yahoo has it for US stocks only.
- Ichimoku, Supertrend, volume profile; multi-timeframe confirmation in the screener.
- Sector rotation (done): US sector ETFs, asset classes, Thai baskets. Follow-ups: alerts when a sector changes quadrant or
  the risk regime flips; official SET sector membership (needs a SET source); industries below the 11 US sectors; money flow
  from ETF shares outstanding (needs a paid source - Yahoo only has the current figure).
- Factor exposure (market / size / value / momentum) in Portfolio & Risk.
- Walk-forward strategy optimisation (re-fit parameters on a rolling window) instead of one in-/out-of-sample split.

**Platform**
- Push notifications for alerts (email / LINE / Telegram) behind a notifier interface.
- Docker image and a scheduled run (cron container or GitHub Actions) for the nightly job.
- If it ever becomes multi-user: split a FastAPI service from the Streamlit UI and move SQLite state to Postgres.
