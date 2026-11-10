"""Command line: ``python -m stockanalyzer <command>`` (or the ``stockanalyzer`` script)."""

from __future__ import annotations

import argparse
import sys
import warnings

import pandas as pd

from stockanalyzer import __version__
from stockanalyzer.config import get_settings, setup_logging


def _md():
    from stockanalyzer.data.service import MarketData

    return MarketData(settings=get_settings())


def _add_universe_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--universe", choices=["us", "none"], default="us", help="symbol list to maintain")
    p.add_argument("--limit", type=int, help="only the N largest companies by market cap")
    p.add_argument("--min-cap", type=float, dest="min_cap", help="minimum market cap in USD")
    p.add_argument("--symbols", nargs="*", default=[], help="extra symbols (e.g. GC=F PTT.BK)")
    p.add_argument("--no-cross", action="store_true", help="skip indices/commodities/FX/crypto presets")
    p.add_argument("--no-sectors", action="store_true", dest="no_sectors", help="skip the sector / asset-class rotation symbols")
    p.add_argument("--thai", choices=["none", "core", "all"], default="core",
                   help="Thai stocks to maintain: the SET50 + sector baskets (core), every share on the SET and mai (all), or none")
    p.add_argument("--patience", type=int, default=0, metavar="N",
                   help="when Yahoo rate-limits the download, wait and carry on up to N times instead of giving up")
    p.add_argument("--patience-minutes", type=float, default=5.0, dest="patience_minutes", metavar="M", help="minutes per wait (default 5)")
    p.add_argument("--no-snapshot", action="store_true", dest="no_snapshot",
                   help="skip the market-wide P/E, P/B, market cap and yield snapshot (a few dozen requests)")
    p.add_argument("--macro", action="store_true", help="also store FRED macro series")
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--start", help="history start for symbols new to the store (default: 10 years)")
    p.add_argument("--force", action="store_true", help="re-fetch even if the store looks up to date")


def _options(args, **extra):
    from stockanalyzer.pipelines.nightly import PipelineOptions

    return PipelineOptions(
        universe=args.universe, limit=args.limit, min_market_cap=args.min_cap, symbols=args.symbols,
        include_cross_assets=not args.no_cross, include_sectors=not args.no_sectors, thai=args.thai, market_snapshot=not args.no_snapshot, rate_limit_waits=args.patience,
        rate_limit_wait_minutes=args.patience_minutes, include_macro=args.macro, benchmark=args.benchmark,
        start=args.start, force=args.force, **extra,
    )


def cmd_status(args) -> int:
    from stockanalyzer.alerts.alerts import AlertManager
    from stockanalyzer.alerts.watchlist import WatchlistRepo
    from stockanalyzer.pipelines.nightly import last_runs
    from stockanalyzer.screener.store import FeatureStore

    md = _md()
    stats = md.store.stats(deep=True)
    print(f"data home : {md.settings.home}")
    print(f"prices    : {stats['symbols']} symbols, {stats.get('rows', 0):,} rows, {stats['size_mb']} MB")
    if stats["symbols"]:
        print(f"coverage  : {stats['first_ts']:%Y-%m-%d} .. {stats['last_ts']:%Y-%m-%d}")
    fs = FeatureStore(md.settings.features_dir)
    print(f"features  : {len(fs.dates())} snapshot(s), latest {fs.dates()[-1] if fs.dates() else '-'}")
    print(f"watchlists: {WatchlistRepo(md.state).summary() or '-'}")
    print(f"alerts    : {len(AlertManager(md.state).list(active_only=True))} active")
    runs = last_runs(md)
    print("last runs :\n" + (runs.to_string(index=False) if not runs.empty else "  none"))
    return 0


def cmd_update(args) -> int:
    from stockanalyzer.pipelines.nightly import resolve_symbols

    md = _md()
    opts = _options(args, compute_features=False, check_alerts=False)
    symbols = resolve_symbols(opts, md)
    print(f"updating {len(symbols)} symbols ...")
    last = {"pct": -1}

    def show(done, total, msg):
        pct = int(100 * done / max(total, 1))
        if pct != last["pct"] and pct % 5 == 0:
            print(f"  {pct:3d}%  {done}/{total}  {msg}", flush=True)
            last["pct"] = pct

    report = md.update(symbols, start=args.start, force=args.force, progress=show)
    print(report.summary())
    for s, err in list(report.failed.items())[:10]:
        print(f"  failed {s}: {err[:100]}")
    return 1 if report.aborted else 0


def cmd_features(args) -> int:
    from stockanalyzer.screener.features import compute_universe_features
    from stockanalyzer.screener.store import FeatureStore

    md = _md()
    frame = compute_universe_features(md.store, None, benchmark=args.benchmark, workers=args.workers)
    if frame.empty:
        print("no price data - run `update` first", file=sys.stderr)
        return 1
    path = FeatureStore(md.settings.features_dir).save(frame)
    print(f"{len(frame)} symbols -> {path}")
    return 0


def cmd_fundamentals(args) -> int:
    from stockanalyzer.data import universe as uni
    from stockanalyzer.data.fundamentals import FundamentalsService

    md = _md()
    svc = FundamentalsService(settings=md.settings, state=md.state)
    stocks = uni.load_us_stocks()
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        stored = set(md.store.symbols())
        symbols = [s for s in stocks.sort_values("market_cap", ascending=False)["symbol"] if s in stored]
    print(svc.refresh(symbols, limit=args.limit, workers=args.workers).summary())
    return 0


def _screen_table(md):
    from stockanalyzer.data import universe as uni
    from stockanalyzer.data.fundamentals import FundamentalsService
    from stockanalyzer.screener.store import FeatureStore
    from stockanalyzer.screener.table import build_screen_table

    features = FeatureStore(md.settings.features_dir).load_latest()
    if features is None:
        return None
    from stockanalyzer.data import fx as fx_data

    universe = pd.concat([uni.load_us_stocks(), uni.load_th_stocks(), uni.cross_asset_universe()], ignore_index=True)
    fundamentals = FundamentalsService(settings=md.settings, state=md.state).snapshots_frame()
    return build_screen_table(features, universe, fundamentals, usd_rate=lambda ccy: fx_data.fx_rate(md, ccy, "USD"))


def cmd_screen(args) -> int:
    from stockanalyzer.screener.rules import PRESETS
    from stockanalyzer.screener.scoring import score_table
    from stockanalyzer.screener.table import liquidity_mask

    if args.list or not args.preset:
        for name, screen in PRESETS.items():
            print(f"- {name}: {screen.description}")
        return 0
    if args.preset not in PRESETS:
        print(f"unknown preset {args.preset!r}; use --list", file=sys.stderr)
        return 2
    table = _screen_table(_md())
    if table is None:
        print("no features yet - run `nightly` or `features`", file=sys.stderr)
        return 1
    table = table[liquidity_mask(table, args.min_traded * 1e6, args.min_traded_th * 1e6)]
    hits = score_table(PRESETS[args.preset].apply(table)).sort_values("score", ascending=False)
    cols = [c for c in ("name", "close", "ret_21d", "ret_126d", "rsi14", "trend_label", "dollar_volume_20", "score") if c in hits]
    out = hits[cols].head(args.top)
    print(f"{len(hits)} matches for '{args.preset}' (showing {len(out)})")
    print(out.to_string(float_format=lambda v: f"{v:,.3f}"))
    if args.csv:
        hits.to_csv(args.csv)
        print(f"saved {len(hits)} rows to {args.csv}")
    return 0


def cmd_alerts(args) -> int:
    from stockanalyzer.alerts.alerts import AlertManager

    md = _md()
    mgr = AlertManager(md.state)
    if args.action == "list":
        for a in mgr.list():
            print(f"[{a.id}] {'on ' if a.active else 'off'} {a.describe()}  last={a.last_triggered_at or '-'}")
        return 0
    fired = mgr.evaluate(md.store.read)
    for e in fired:
        print(f"ALERT {e.symbol}: {e.message}")
    print(f"{len(fired)} alert(s) fired")
    return 0


def cmd_nightly(args) -> int:
    from stockanalyzer.pipelines.jobs import JobFile
    from stockanalyzer.pipelines.nightly import run_pipeline

    job = JobFile(args.job_file, args.job_title) if args.job_file else None  # the app watches this file
    try:
        result = run_pipeline(_options(args, fundamentals_limit=args.fundamentals, workers=args.workers,
                                       refresh_events=not args.no_events), progress=job.update if job else None)
    except BaseException as exc:
        if job:
            job.finish(False, {"error": {"error": f"{type(exc).__name__}: {exc}"}})
        raise
    if job:
        job.finish(result.ok, result.steps)
    print(result.summary())
    return 0 if result.ok else 1


def cmd_calendar(args) -> int:
    from stockanalyzer.config import display_timezone
    from stockanalyzer.data import events as ev
    from stockanalyzer.pipelines.nightly import tracked_symbols

    md = _md()
    svc = ev.EventsService(store=md.store, state=md.state, settings=md.settings)
    symbols = [s.upper() for s in args.symbols] if args.symbols else tracked_symbols(md.state)
    report = svc.refresh(symbols, max_age_hours=0 if args.refresh else ev.COMPANY_TTL_HOURS)
    if report.failed:
        print(f"could not load: {', '.join(report.failed)}", file=sys.stderr)
    try:
        svc.refresh_economic(max_age_hours=0 if args.refresh else ev.ECONOMIC_TTL_HOURS)
    except Exception as exc:
        print(f"economic calendar unavailable: {exc}", file=sys.stderr)

    tz = display_timezone()
    today = svc.today(tz)
    agenda = svc.agenda(symbols, today, today + pd.Timedelta(days=args.days), regions=args.regions,
                        key_only=not args.all_releases, tz=tz)
    print(f"{len(symbols)} symbols, regions {' '.join(args.regions) or '-'}, {today:%Y-%m-%d} + {args.days} days "
          f"(times UTC{pd.Timestamp.now(tz=tz):%z})")
    for r in agenda.itertuples(index=False):
        clock = ""
        if pd.notna(r.at):  # company days are exchange-local: a US evening report is the next morning in Bangkok
            local = r.at.tz_convert(tz)
            shift = (local.normalize().tz_localize(None) - r.day).days
            clock = f"{local:%H:%M}" + (f"{shift:+d}d" if shift else "")
        who = r.region if r.kind == ev.ECONOMIC else r.symbol
        what = r.title if r.kind == ev.ECONOMIC else ev.KIND_LABELS[r.kind]
        print(f"{r.day:%Y-%m-%d %a} {clock:<8s} {who:<10s} {what:<32s} {ev.describe(r.kind, r.data)}")
    if agenda.empty:
        print("nothing scheduled")
    return 0


def cmd_rotation(args) -> int:
    import pandas as pd

    from stockanalyzer.analysis import rotation as rot
    from stockanalyzer.data import sectors

    md = _md()
    universe = sectors.UNIVERSES[args.universe]
    if args.update:
        print(md.update(universe.symbols).summary())
    data = sectors.load_sector_data(md.store, universe)
    if data is None:
        print(f"no stored prices for {universe.label}; run with --update (or: nightly)", file=sys.stderr)
        return 1
    end = data.asof
    start = end - pd.DateOffset(months=args.months)
    sb = data.scoreboard(start, end)
    bench = rot.window_returns(data.benchmark, start, end)
    print(f"{universe.label}  {start:%Y-%m-%d} .. {end:%Y-%m-%d}  benchmark {universe.benchmark_name} {bench:+.1%}")
    ratio, mom = rot.rrg(data.close, data.benchmark)
    quad = rot.quadrants(ratio, mom).iloc[-1]
    print(f"{'':25s}{'return':>8s}{'vs bench':>9s}{'rank':>6s}{'was':>5s}{'flow':>7s}  quadrant")
    for name, r in sb.iterrows():
        flow = f"{r['flow']:+.2f}" if pd.notna(r["flow"]) else "-"
        was = f"{r['prev_rank']:.0f}" if pd.notna(r["prev_rank"]) else "-"
        print(f"{name:25s}{r['ret']:>+8.1%}{r['rel']:>+9.1%}{r['rank']:>6.0f}{was:>5s}{flow:>7s}  {quad.get(name) or '-'}")
    od = rot.offensive_defensive(data.close, universe.offensive, universe.defensive)
    if len(od) and od["regime"].iloc[-1]:
        print(f"risk appetite: {od['regime'].iloc[-1]}")
    changes = rot.quadrant_changes(rot.quadrants(ratio, mom), since=start).rename(columns={"from": "was"})  # 'from' is a keyword
    for r in changes.itertuples(index=False):
        print(f"  {r.date:%Y-%m-%d}  {r.sector}: {r.was} -> {r.to}")
    return 0


def _kv(pairs: list[str]) -> dict:
    out = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--param expects name=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = int(v) if v.lstrip("-").isdigit() else float(v)
    return out


def cmd_backtest(args) -> int:
    from stockanalyzer.backtest.engine import BacktestConfig
    from stockanalyzer.backtest.strategies import STRATEGIES, run_strategy

    if args.list or not args.strategy:
        for name, s in STRATEGIES.items():
            print(f"- {name}({', '.join(f'{p.name}={p.default}' for p in s.params)}): {s.description}")
        return 0
    if args.strategy not in STRATEGIES:
        print(f"unknown strategy {args.strategy!r}; use --list", file=sys.stderr)
        return 2
    df = _md().prices(args.symbol, period=args.period)
    if df.empty:
        print(f"no data for {args.symbol}", file=sys.stderr)
        return 1
    cfg = BacktestConfig(commission_bps=args.commission, slippage_bps=args.slippage, execution=args.execution,
                         allow_short=args.short)
    res = run_strategy(df, args.strategy, _kv(args.param), cfg)
    m = res.metrics
    print(f"{args.symbol} {args.strategy} {df.index[0]:%Y-%m-%d}..{df.index[-1]:%Y-%m-%d} ({len(df)} bars, {args.execution})")
    rows = [("total return", "total_return", "buy_hold_return", "{:+.1%}"), ("CAGR", "cagr", "buy_hold_cagr", "{:+.1%}"),
            ("Sharpe", "sharpe", "buy_hold_sharpe", "{:.2f}"), ("max drawdown", "max_drawdown", "buy_hold_max_drawdown", "{:.1%}")]
    print(f"{'':14s}{'strategy':>12s}{'buy & hold':>12s}")
    for label, a, b, fmt in rows:
        print(f"{label:14s}{fmt.format(m[a]):>12s}{fmt.format(m[b]):>12s}")
    print(f"trades {int(m['n_trades'])}, win rate {m['win_rate']:.0%}, exposure {m['exposure']:.0%}, costs {m['total_costs']:.2%} of capital")
    return 0


def cmd_montecarlo(args) -> int:
    from stockanalyzer.analysis import regimes as rg
    from stockanalyzer.data import fx as fx_data
    from stockanalyzer.portfolio import currency as ccy
    from stockanalyzer.simulation import calibration as cal
    from stockanalyzer.simulation import montecarlo as mc
    from stockanalyzer.simulation import pool as pl

    md = _md()
    symbols = list(dict.fromkeys(s.strip().upper() for s in args.symbols))
    raw = args.weights or [1.0] * len(symbols)
    if len(raw) != len(symbols) or sum(raw) <= 0:
        print("give one positive weight per symbol (or none for equal weights)", file=sys.stderr)
        return 2
    weights = pd.Series(raw, index=symbols, dtype=float) / sum(raw)
    base = args.base or ("THB" if any(s.endswith(".BK") for s in symbols) else "USD")
    driver = args.driver or ("TDEX.BK" if weights[[s.endswith(".BK") for s in symbols]].sum() > 0.5 else "SPY")
    currencies = [c for c in ccy.currencies_of(symbols) if c != base]
    if args.backfill:
        todo = [driver, *symbols]
        ccy.fx_table(md, currencies, base)  # stores the exchange-rate symbols it needs, so they can be backfilled too
        todo += [p for c in currencies for p in fx_data.pair_symbols(c, base) if md.store.exists(p)]
        rep = md.backfill(list(dict.fromkeys(todo)))
        print(f"full history downloaded for {rep.updated} of {len(set(todo))} symbols" + (f"; failed: {', '.join(rep.failed)}" if rep.failed else ""))

    dclose = md.close_matrix([driver], period=None, live=False)
    if driver not in dclose:
        print(f"no prices for the driver {driver}", file=sys.stderr)
        return 1
    weekly = rg.weekly_returns(dclose[driver].rename(driver))
    try:
        model = rg.fit(weekly, args.states, driver=driver)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    close = md.close_matrix(symbols, period=None, live=False)
    close = ccy.to_base(close, ccy.fx_table(md, ccy.currencies_of(close.columns), base)) if not close.empty else close
    pool = pl.build_pool(model, pl.weekly_frame(close))
    for s in symbols:
        if s in pool.dropped:
            print(f"left out {s}: {pool.dropped[s]}")
        elif s not in pool.assets:
            print(f"left out {s}: no prices in {base}")
    if not pool.assets:
        return 1
    w = weights.reindex(pool.assets)
    start = {"calm": 0, "stressed": 1, "crisis": 2}.get(args.start, args.start)
    if isinstance(start, int) and start >= model.k:
        start = model.k - 1
    stressed = range(1, model.k)
    scenario = mc.Scenario(start=start, stretch={i: args.stretch for i in stressed}, replay=args.crisis,
                           replay_after=52 if args.recovery else 0)
    plan = mc.Plan(initial=args.initial, monthly=args.monthly, step_up=args.step_up / 100, years=args.years,
                   rebalance=args.rebalance, cost=args.fee / 100, inflation=args.inflation / 100)
    rf = (args.cash_rate if args.cash_rate is not None else (1.5 if base == "THB" else 3.5)) / 100
    index_return = args.index_return / 100 if args.index_return is not None else None
    try:
        res = mc.simulate(pool, w, model, plan, scenario, returns_mode=args.returns, rf=pl.to_weekly(rf),
                          index_return=None if index_return is None else pl.to_weekly(index_return),
                          n_paths=args.paths, regimes=not args.plain, seed=args.seed)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    st = model.stats()
    print(f"Monte Carlo: {' · '.join(f'{s} {v:.0%}' for s, v in (w / w.sum()).items())} in {base}, {args.years} years, "
          f"{args.paths} paths, {'plain bootstrap' if args.plain else f'{model.k} regimes'} on {driver} "
          f"(weekly since {weekly.index[0]:%Y})")
    print(f"market now: {model.current_state} ({model.current.max():.0%}); average stay "
          + ", ".join(f"{n} {st.loc[n, 'avg_weeks']:.0f} wk" for n in model.names))
    mix = model.long_run
    used = pool.targets(args.returns, mix, rf=pl.to_weekly(rf),
                        index_return=None if index_return is None else pl.to_weekly(index_return))
    print(f"assumed returns ({pl.RETURN_MODES[args.returns]}, cash {rf:.1%}): "
          + ", ".join(f"{s} {a:+.1%}/yr" for s, a in zip(pool.assets, pl.to_annual(used))))
    if args.crisis:
        print(f"every path starts with the {mc.CRISES[args.crisis][0]}")
    ms = res.milestones()
    print(f"\n{'':>7s}{'money in':>13s}{'5%':>13s}{'25%':>13s}{'median':>13s}{'75%':>13s}{'95%':>13s}{'below in':>10s}")
    for y, r in ms.iterrows():
        print(f"{y:>4g} yr" + "".join(f"{r[c]:>13,.0f}" for c in ("invested", "p5", "p25", "p50", "p75", "p95"))
              + f"{r['below_invested']:>10.0%}")
    s = res.summary()
    print(f"\nmedian return {s['cagr_median']:+.1%}/yr (5-95%: {s['cagr_p5']:+.1%} to {s['cagr_p95']:+.1%}); worst fall: median "
          f"{-s['drawdown_median']:.0%}, 1 in 20 {-s['drawdown_p95']:.0%}; median in today's money {s['median_real']:,.0f}"
          + (f"; money runs out in {s['ran_out']:.0%} of paths" if args.monthly < 0 else ""))
    crises = mc.crisis_table(pool, w)
    if not crises.empty:
        print("\nthis portfolio, bought and held through past crises:")
        for _, r in crises.iterrows():
            rec = ("not back by the end of the data" if pd.isna(r["recovery_weeks"]) else "ended above its start"
                   if r["recovery_weeks"] == 0 else f"back in {r['recovery_weeks'] / 52:.1f} yr")
            est = f", {r['estimated']:.0%} estimated" if r["estimated"] >= 0.005 else ""
            print(f"  {r['crisis']:52s} {driver} {r['index']:+6.0%}  portfolio {r['portfolio']:+6.0%}  {rec}{est}")
    if args.check:
        c = cal.calibrate(weekly, args.states, driver=driver, min_train=max(rg.MIN_WEEKS, min(cal.MIN_TRAIN, len(weekly) // 2)))
        print(f"\nwalk-forward check of {driver}, {c.first:%Y-%m} .. {c.last:%Y-%m}:")
        print(c.table().to_string(float_format=lambda v: f"{v:.3f}"))
        print(c.verdict().to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nA range of outcomes under these assumptions, not a forecast.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stockanalyzer", description="Multi-asset analysis platform")
    parser.add_argument("--version", action="version", version=f"stockanalyzer {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="show store, snapshots, watchlists and recent runs").set_defaults(fn=cmd_status)

    p = sub.add_parser("update", help="incrementally update prices")
    _add_universe_args(p)
    p.set_defaults(fn=cmd_update)

    p = sub.add_parser("features", help="recompute the feature table from stored prices")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--benchmark", default="SPY")
    p.set_defaults(fn=cmd_features)

    p = sub.add_parser("fundamentals", help="refresh cached fundamental snapshots")
    p.add_argument("--limit", type=int, default=200, help="max snapshots to fetch this run")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--symbols", nargs="*", default=[])
    p.set_defaults(fn=cmd_fundamentals)

    p = sub.add_parser("screen", help="run a preset screen on the latest features")
    p.add_argument("preset", nargs="?")
    p.add_argument("--list", action="store_true")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--csv")
    p.add_argument("--min-traded", type=float, default=5.0, help="min average traded value, US$ millions a day (US and other stocks)")
    p.add_argument("--min-traded-th", type=float, default=0.3, help="the same for Thai (.BK) stocks")
    p.set_defaults(fn=cmd_screen)

    p = sub.add_parser("alerts", help="list or evaluate alerts")
    p.add_argument("action", choices=["check", "list"], nargs="?", default="check")
    p.set_defaults(fn=cmd_alerts)

    p = sub.add_parser("backtest", help="backtest a rule-based strategy on one symbol")
    p.add_argument("symbol", nargs="?", default="SPY")
    p.add_argument("strategy", nargs="?")
    p.add_argument("--list", action="store_true", help="list strategies")
    p.add_argument("--period", default="10y")
    p.add_argument("--param", action="append", help="strategy parameter, e.g. --param fast=20 --param slow=100")
    p.add_argument("--commission", type=float, default=1.0, help="bps per side")
    p.add_argument("--slippage", type=float, default=2.0, help="bps per side")
    p.add_argument("--execution", choices=["next_open", "close"], default="next_open")
    p.add_argument("--short", action="store_true", help="allow short positions")
    p.set_defaults(fn=cmd_backtest)

    p = sub.add_parser("montecarlo", help="simulate a portfolio with monthly saving (DCA) through market regimes and crises")
    p.add_argument("symbols", nargs="+", help="holdings, e.g. SPY TLT PTT.BK")
    p.add_argument("--weights", nargs="*", type=float, help="relative weights in the same order (default: equal)")
    p.add_argument("--base", help="currency of the values (default THB if any Thai holding, else USD)")
    p.add_argument("--initial", type=float, default=100_000.0, help="starting amount")
    p.add_argument("--monthly", type=float, default=0.0, help="added on the first week of each month; negative = withdrawn")
    p.add_argument("--step-up", type=float, default=0.0, dest="step_up", help="raise of the monthly amount each year, in %%")
    p.add_argument("--years", type=int, default=20)
    p.add_argument("--rebalance", choices=["none", "monthly", "quarterly", "yearly"], default="yearly")
    p.add_argument("--fee", type=float, default=0.1, help="fee per trade in %% (default 0.1)")
    p.add_argument("--inflation", type=float, default=2.0, help="%%/yr, for the today's-money figure")
    p.add_argument("--paths", type=int, default=5000)
    p.add_argument("--driver", help="index whose regimes are simulated (default SPY, TDEX.BK for a mostly Thai portfolio)")
    p.add_argument("--states", type=int, choices=[2, 3], default=2)
    p.add_argument("--returns", choices=["market", "half", "own"], default="market",
                   help="market-implied (cash + beta x index, default), halfway, or each holding's own history")
    p.add_argument("--cash-rate", type=float, dest="cash_rate", help="%%/yr (default 1.5 for THB, 3.5 otherwise)")
    p.add_argument("--index-return", type=float, dest="index_return", help="the driver's %%/yr (default: its own history)")
    p.add_argument("--start", default="now", choices=["now", "long_run", "calm", "stressed", "crisis"])
    p.add_argument("--crisis", choices=["dotcom", "gfc", "euro", "taper", "oil", "q4_2018", "covid", "rates", "tariff"],
                   help="replay this crisis at the start of every path")
    p.add_argument("--recovery", action="store_true", help="with --crisis: replay the year after it too")
    p.add_argument("--stretch", type=float, default=1.0, help="make stressed spells last this many times longer")
    p.add_argument("--plain", action="store_true", help="no regimes: a plain block bootstrap of all past weeks")
    p.add_argument("--backfill", action="store_true", help="first download the full history of the driver, holdings and FX")
    p.add_argument("--check", action="store_true", help="also run the walk-forward check of the regime model")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=cmd_montecarlo)

    p = sub.add_parser("nightly", help="update + features + (fundamentals) + events + alerts")
    _add_universe_args(p)
    p.add_argument("--fundamentals", type=int, default=0, help="refresh up to N stale snapshots")
    p.add_argument("--workers", type=int, default=0, help="processes for the feature step")
    p.add_argument("--no-events", action="store_true", dest="no_events",
                   help="skip earnings/dividend dates and the economic calendar")
    p.add_argument("--job-file", dest="job_file", help=argparse.SUPPRESS)  # progress file the app watches (see pipelines/jobs.py)
    p.add_argument("--job-title", dest="job_title", default="", help=argparse.SUPPRESS)
    p.set_defaults(fn=cmd_nightly)

    p = sub.add_parser("rotation", help="which sectors / asset classes gained or lost ground, and the rotations in between")
    p.add_argument("universe", nargs="?", default="us_sectors", choices=["us_sectors", "th_sectors", "asset_classes"])
    p.add_argument("--months", type=int, default=3, help="length of the window to analyse (default 3)")
    p.add_argument("--update", action="store_true", help="download / refresh the universe's prices first")
    p.set_defaults(fn=cmd_rotation)

    p = sub.add_parser("calendar", help="upcoming earnings, dividends and economic releases")
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--symbols", nargs="*", help="default: your watchlists and portfolio holdings")
    p.add_argument("--regions", nargs="*", default=["US", "TH"], help="economic release regions, e.g. US TH CN EU")
    p.add_argument("--all-releases", action="store_true", dest="all_releases", help="not just the key releases")
    p.add_argument("--refresh", action="store_true", help="re-fetch even if cached today")
    p.set_defaults(fn=cmd_calendar)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "WARNING")
    warnings.filterwarnings("ignore", message=".*inplace.*")  # yfinance vs pandas 3 deprecation noise
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
