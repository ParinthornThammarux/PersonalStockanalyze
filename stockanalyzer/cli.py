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
    p.add_argument("--macro", action="store_true", help="also store FRED macro series")
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--start", help="history start for symbols new to the store (default: 10 years)")
    p.add_argument("--force", action="store_true", help="re-fetch even if the store looks up to date")


def _options(args, **extra):
    from stockanalyzer.pipelines.nightly import PipelineOptions

    return PipelineOptions(
        universe=args.universe, limit=args.limit, min_market_cap=args.min_cap, symbols=args.symbols,
        include_cross_assets=not args.no_cross, include_macro=args.macro, benchmark=args.benchmark,
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
    universe = pd.concat([uni.load_us_stocks(), uni.cross_asset_universe()], ignore_index=True)
    fundamentals = FundamentalsService(settings=md.settings, state=md.state).snapshots_frame()
    return build_screen_table(features, universe, fundamentals)


def cmd_screen(args) -> int:
    from stockanalyzer.screener.rules import PRESETS
    from stockanalyzer.screener.scoring import score_table

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
    from stockanalyzer.pipelines.nightly import run_pipeline

    result = run_pipeline(_options(args, fundamentals_limit=args.fundamentals, workers=args.workers))
    print(result.summary())
    return 0 if result.ok else 1


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

    p = sub.add_parser("nightly", help="update + features + (fundamentals) + alerts")
    _add_universe_args(p)
    p.add_argument("--fundamentals", type=int, default=0, help="refresh up to N stale snapshots")
    p.add_argument("--workers", type=int, default=0, help="processes for the feature step")
    p.set_defaults(fn=cmd_nightly)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "WARNING")
    warnings.filterwarnings("ignore", message=".*inplace.*")  # yfinance vs pandas 3 deprecation noise
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
