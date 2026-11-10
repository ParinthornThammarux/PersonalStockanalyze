"""Monte Carlo: a portfolio's possible futures with monthly saving (DCA) or withdrawals, through market regimes and crises."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import streamlit as st

from stockanalyzer.analysis import regimes as rg
from stockanalyzer.data import fx as fx_data
from stockanalyzer.domain.assets import AssetClass, infer_asset_class
from stockanalyzer.portfolio import currency as ccy
from stockanalyzer.portfolio import valuation as val
from stockanalyzer.portfolio.ledger import build_book
from stockanalyzer.simulation import calibration as cal
from stockanalyzer.simulation import montecarlo as mc
from stockanalyzer.simulation import pool as pl
from stockanalyzer.ui import charts
from stockanalyzer.ui import data as ui_data
from stockanalyzer.ui.context import services
from stockanalyzer.ui.format import money, pct
from stockanalyzer.ui.widgets import show_chart, show_table

DRIVERS = {
    "SPY": "SPY - S&P 500 ETF (US stocks, from 1993, dividends included)",
    "TDEX.BK": "TDEX - SET50 ETF (Thai stocks, from 2008)",
    "BTC-USD": "BTC-USD - Bitcoin (from 2014)",
    "^GSPC": "^GSPC - S&P 500 index (from 1927, price only: no dividends)",
}
BASES = ["THB", "USD", "EUR", "SGD", "JPY", "GBP", "HKD"]
PATHS = [1000, 2000, 5000, 10000]
SAMPLE_PATHS = 12
OLD_CRISIS = pd.Timestamp("2007-10-01")  # a history starting later misses 2008 (and 2000-02)
START_LABELS = {"now": "Today's state (from the model)", "long_run": "Long-run mix of states", 0: "Calm", 1: "Stressed", 2: "Crisis (3 states)"}
ENGINES = {"regimes": "Regimes", "plain": "Plain bootstrap (no regimes)"}
NOT_A_FORECAST = ("A range of outcomes under the assumptions on this page, **not a forecast**: price forecasting failed every honest "
                  "test in this project, and the walk-forward check (last tab) shows the regime model does not predict either.")


# -- data -----------------------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def _driver_weekly(driver: str) -> pd.Series:
    close = ui_data.close_matrix((driver,), None, False)
    return rg.weekly_returns(close[driver].rename(driver)) if driver in close else pd.Series(dtype="float64")


@st.cache_data(ttl=600, show_spinner=False)
def _asset_weekly(symbols: tuple[str, ...], base: str) -> pd.DataFrame:
    return pl.weekly_frame(ui_data.base_close_matrix(symbols, None, base))


@st.cache_resource(max_entries=8, show_spinner=False)
def _model(driver: str, k: int, last_week: str) -> rg.RegimeModel:  # last_week: refit when new data arrives
    return rg.fit(_driver_weekly(driver), k, driver=driver)


@st.cache_resource(max_entries=8, show_spinner=False)
def _pool(driver: str, k: int, last_week: str, symbols: tuple[str, ...], base: str) -> pl.ReturnPool:
    return pl.build_pool(_model(driver, k, last_week), _asset_weekly(symbols, base))


@st.cache_resource(max_entries=6, show_spinner=False)
def _run(driver: str, k: int, last_week: str, symbols: tuple[str, ...], base: str, weights: tuple[float, ...], plan_fields: tuple,
         scenario_key: tuple, mode: str, rf: float, index_return: float, n_paths: int, engine: str) -> mc.SimResult:
    plan = mc.Plan(*plan_fields)
    start, stretch, freq, replay, after = scenario_key
    model, pool = _model(driver, k, last_week), _pool(driver, k, last_week, symbols, base)
    stressed = range(1, k)
    scenario = mc.Scenario(name="Scenario", start=start if not isinstance(start, int) else min(start, k - 1),
                           stretch={i: stretch for i in stressed}, frequency={i: freq for i in stressed},
                           replay=replay, replay_after=52 if after else 0)
    return mc.simulate(pool, pd.Series(weights, index=list(symbols)), model, plan, scenario, returns_mode=mode,
                       rf=pl.to_weekly(rf), index_return=pl.to_weekly(index_return), n_paths=n_paths,
                       regimes=engine == "regimes", seed=7)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _calibration(driver: str, k: int, last_week: str) -> cal.Calibration:
    weekly = _driver_weekly(driver)
    return cal.calibrate(weekly, k, driver=driver, min_train=max(rg.MIN_WEEKS, min(cal.MIN_TRAIN, len(weekly) // 2)))


def _clear_caches() -> None:
    for fn in (ui_data.close_matrix, ui_data.base_close_matrix, ui_data.fx_series, _driver_weekly, _asset_weekly, _calibration):
        fn.clear()
    for fn in (_model, _pool, _run):
        fn.clear()


# -- inputs ---------------------------------------------------------------------------------------
def _ledger_weights(pf: str, base: str) -> tuple[pd.Series | None, float, str | None]:
    """Current holdings of a ledger portfolio as weights, and their value in ``base``."""
    txs = services().transactions.list(pf)
    if not txs:
        return None, 0.0, f"Portfolio '{pf}' has no transactions yet - add them on Portfolio & Risk, or use a custom list."
    table = ui_data.fx_series(tuple(sorted({t.currency for t in txs} | {base})), base)
    if ccy.missing(table):
        return None, 0.0, f"No exchange rate into {base} for: {', '.join(ccy.missing(table))}. Pick another currency."
    try:
        book = build_book(txs, "average", base=base, fx_at=ccy.lookup(table))
    except ValueError as exc:
        return None, 0.0, str(exc)
    if book.positions.empty:
        return None, 0.0, "That portfolio has no open positions."
    prices = {}
    for s in book.positions.index:
        df = ui_data.prices(s, "1mo", "1d", False)
        if not df.empty:
            prices[s] = float(df["close"].iloc[-1])
    marked = val.mark_to_market(book.positions, prices, fx=ccy.latest(table))
    mv = marked["market_value"].dropna()
    mv = mv[mv > 0]
    if mv.empty:
        return None, 0.0, "No prices for the holdings of that portfolio."
    return mv / mv.sum(), float(mv.sum()), None


def _portfolio_inputs() -> tuple[pd.Series | None, float | None, str, str]:
    names = services().transactions.portfolios()
    sources = (["My portfolio"] if names else []) + ["Custom list"]
    c1, c2, c3 = st.columns([1, 2.4, 0.8])
    src = c1.radio("Simulate", sources, key="mc_src")
    if src == "My portfolio":
        pf = c2.selectbox("Portfolio", names, key="mc_pf")
        thai = any(t.symbol.endswith(".BK") for t in services().transactions.list(pf) if t.symbol)
        base = c3.selectbox("Currency", BASES, index=BASES.index("THB" if thai else "USD"), key="mc_base_pf")
        weights, value, err = _ledger_weights(pf, base)
        if err:
            st.info(err)
            return None, None, base, src
        c2.caption("Current holdings at today's prices (cash balance left out): "
                   + " · ".join(f"{s} {w:.0%}" for s, w in weights.sort_values(ascending=False).items()))
        return weights, value, base, src
    typed = c2.text_input("Symbols (comma separated)", "SPY, TLT", key="mc_symbols",
                          help="Yahoo format: AAPL, PTT.BK, TDEX.BK, GLD, BTC-USD")
    symbols = list(dict.fromkeys(s.strip().upper() for s in typed.split(",") if s.strip()))
    base = c3.selectbox("Currency", BASES, index=BASES.index("THB" if any(s.endswith(".BK") for s in symbols) else "USD"),
                        key="mc_base_custom")
    if not symbols:
        st.info("Enter at least one symbol.")
        return None, None, base, src
    start = pd.DataFrame({"symbol": symbols, "weight %": [round(100 / len(symbols), 2)] * len(symbols)})
    edited = st.data_editor(start, hide_index=True, disabled=["symbol"], key=f"mc_w_{'_'.join(symbols)}",
                            column_config={"weight %": st.column_config.NumberColumn(min_value=0.0, max_value=1000.0, step=1.0)})
    w = pd.Series(edited["weight %"].to_numpy(dtype=float), index=edited["symbol"]).clip(lower=0)
    if w.sum() <= 0:
        st.error("Give at least one holding a weight above zero.")
        return None, None, base, src
    w = w / w.sum()
    st.caption("Target weights: " + " · ".join(f"{s} {v:.0%}" for s, v in w.items()) + " (weights are relative: 2 / 1 / 1 = 50 / 25 / 25 %)")
    return w, None, base, src


def _default_driver(weights: pd.Series) -> str:
    thai = float(weights[[s.endswith(".BK") for s in weights.index]].sum())
    crypto = float(weights[[infer_asset_class(s) is AssetClass.CRYPTO for s in weights.index]].sum())
    return "TDEX.BK" if thai > 0.5 else "BTC-USD" if crypto > 0.5 else "SPY"


def _history_panel(driver: str, symbols: tuple[str, ...], base: str) -> None:
    md = services().md
    if msg := st.session_state.pop("mc_backfill_msg", None):
        st.success(msg)
    first = md.first_timestamps([driver, *symbols])
    d0 = first.get(driver)
    short = d0 is None or d0 > OLD_CRISIS
    label = f"Price history: {driver} from {d0:%Y}" if d0 is not None else f"Price history: no {driver} prices yet"
    with st.expander(label + (" - 2008 and earlier crises are not in it" if short else ""), expanded=short):
        rows = [{"symbol": s, "role": "regime driver" if s == driver else "holding",
                 "prices from": first[s].date() if s in first else None} for s in dict.fromkeys([driver, *symbols])]
        show_table(pd.DataFrame(rows), hide_index=True)
        st.caption("The store keeps 10 years per symbol by default, which holds one real crash (COVID). The full history adds the "
                   "dot-com bust and 2008 for the regime model and the crisis replays: SPY reaches back to 1993, TDEX to 2008, "
                   "most large stocks to 2000 or earlier. Weeks before a holding listed are estimated from the index (see Assumptions).")
        if st.button("Download full history", key="mc_backfill",
                     help="One request per symbol (the driver, the holdings and their exchange rates); replaces their stored history."):
            fx_syms = [p for c in ccy.currencies_of(symbols) if c != base for p in fx_data.pair_symbols(c, base) if md.store.exists(p)]
            todo = list(dict.fromkeys([driver, *symbols, *fx_syms]))
            bar = st.progress(0.0, text="Downloading ...")
            rep = md.backfill(todo, progress=lambda i, n, s: bar.progress(i / n, text=f"{s} ({i}/{n})"))
            _clear_caches()
            fails = f" Failed: {', '.join(rep.failed)}." if rep.failed else ""
            st.session_state["mc_backfill_msg"] = f"Full history downloaded for {rep.updated} of {len(todo)} symbols.{fails}"
            st.rerun()


def _settings_form(base: str, initial: float, src: str, driver: str, weekly: pd.Series, index_hist: float) -> dict | None:
    crises = mc.crises_within(weekly.index)
    thb = base == "THB"
    with st.form("mc_form"):
        st.markdown("**Plan**")
        c = st.columns(4)
        years = c[0].slider("Years", 1, 40, 20, key="mc_years")
        start_amount = c[1].number_input(f"Starting amount ({base})", 0.0, 1e13, float(initial), step=10_000.0, format="%.0f",
                                         key=f"mc_initial_{src}_{base}")
        monthly = c[2].number_input(f"Monthly saving ({base})", -1e11, 1e11, 10_000.0 if thb else 500.0, step=1_000.0 if thb else 100.0,
                                    format="%.0f", key=f"mc_monthly_{base}",
                                    help="Invested on the first week of every month at the target weights (DCA). "
                                         "A negative amount is a monthly withdrawal instead.")
        step_up = c[3].number_input("Raise the saving each year (%)", 0.0, 50.0, 0.0, 1.0, key="mc_stepup",
                                    help="3 = the monthly amount grows 3 % every year (pay rises).")
        c = st.columns(4)
        rebalance = c[0].selectbox("Rebalance to the target weights", list(mc.REBALANCE), index=3, format_func=mc.REBALANCE.get,
                                   key="mc_reb")
        fee = c[1].number_input("Fee per trade (%)", 0.0, 3.0, 0.16 if thb else 0.05, 0.01, key=f"mc_fee_{base}",
                                help="Charged on every amount bought or sold: the monthly buys, rebalancing and withdrawals.")
        inflation = c[2].number_input("Inflation (%/yr)", -5.0, 30.0, 1.5 if thb else 2.5, 0.25, key=f"mc_infl_{base}",
                                      help="Only used to show the result in today's money.")
        n_paths = c[3].selectbox("Paths", PATHS, index=2, key="mc_paths")

        st.markdown("**Expected returns**")
        c = st.columns([2, 1, 1])
        mode = c[0].selectbox("Each holding earns", list(pl.RETURN_MODES), format_func=pl.RETURN_MODES.get, key="mc_mode",
                              help="Market-implied: the cash rate plus the holding's beta times the index's excess return - no "
                                   "assumption that a stock will keep beating the market (NVDA's last 10 years were ~56 %/yr). "
                                   "Own history: what the holding did in the pooled weeks. Halfway: the average of the two.")
        rf = c[1].number_input("Cash rate (%/yr)", -2.0, 20.0, 1.5 if thb else 3.5, 0.25, key=f"mc_rf_{base}",
                               help="What a holding with no stock-market risk earns (deposit / T-bill rate).")
        idx = c[2].number_input(f"{driver} return (%/yr)", -30.0, 80.0, round(index_hist * 100, 1), 0.5, key=f"mc_idx_{driver}",
                                help=f"The index's average yearly return the market-implied returns are built on. Default: its "
                                     f"own average since {weekly.index[0]:%Y}.")

        st.markdown("**Market regimes and scenario**")
        c = st.columns(4)
        engine = c[0].radio("Engine", list(ENGINES), format_func=ENGINES.get, key="mc_engine",
                            help="Regimes: calm and stressed spells follow each other as they did in the past, and each week is drawn "
                                 "from history in the same state. Plain: any past week can follow any other.")
        k = c[0].radio("States", [2, 3], horizontal=True, key="mc_k",
                       help="3 adds a rare Crisis state (SPY since 1993: 5 episodes, 2 % of weeks). 2 is the safer fit.")
        start = c[1].selectbox("Start in", list(START_LABELS), format_func=START_LABELS.get, key="mc_start")
        replay = c[2].selectbox("Play a crisis first", [None, *crises], format_func=lambda x: "None" if x is None else crises[x],
                                key=f"mc_replay_{driver}",
                                help=f"The crisis weeks of {driver} replayed on every path (your holdings' returns of those same "
                                     "weeks), then the simulation carries on from the state at the bottom.")
        after = c[2].checkbox("... and the year after it", key="mc_after", help="Replays the 52 weeks of recovery that followed too.")
        stretch = c[3].slider("Stressed spells last (x)", 0.5, 3.0, 1.0, 0.25, key="mc_stretch",
                              help="2 = every stressed spell lasts twice as long on average.")
        freq = c[3].slider("Stressed spells come (x as often)", 0.5, 3.0, 1.0, 0.25, key="mc_freq")
        compare = st.checkbox("Also run the base case (today's state, no changes) to compare", True, key="mc_compare")
        if not st.form_submit_button("Run simulation", type="primary"):
            return None
    plan = mc.Plan(initial=start_amount, monthly=monthly, step_up=step_up / 100, years=years, rebalance=rebalance,
                   cost=fee / 100, inflation=inflation / 100)
    return {"plan": plan, "mode": mode, "rf": rf / 100, "index": idx / 100, "engine": engine, "k": int(k), "n_paths": int(n_paths),
            "scenario": (start, float(stretch), float(freq), replay, bool(after)), "compare": compare}


# -- results --------------------------------------------------------------------------------------
def _kpis(res: mc.SimResult, base: str) -> None:
    s = res.summary()
    years = res.plan.years
    k = st.columns(6)
    k[0].metric(f"Median after {years} years", money(s["median"]), f"{s['median'] / s['invested']:.1f}x the money put in"
                if s["invested"] > 0 else None, delta_color="off",
                help=f"Half the paths end above, half below. Money put in: {money(s['invested'])} {base}.")
    k[1].metric("Range (5 - 95 %)", f"{money(s['p5'], 1)} - {money(s['p95'], 1)}", help="9 paths in 10 end inside this range.")
    if res.plan.monthly < 0:
        k[2].metric("Money runs out", pct(s["ran_out"]), help="Share of paths where the withdrawals emptied the portfolio.")
    else:
        k[2].metric("Ends below the money put in", pct(s["below_invested"]))
    k[3].metric("Median return per year", pct(s["cagr_median"]),
                help=f"Time-weighted (the saving does not distort it). 5 - 95 %: {pct(s['cagr_p5'])} to {pct(s['cagr_p95'])}.")
    k[4].metric("Worst fall on the way (median)", pct(-s["drawdown_median"]),
                help=f"Largest peak-to-trough drop of each path. One path in 20 falls more than {pct(s['drawdown_p95'])}.")
    k[5].metric("Median in today's money", money(s["median_real"]), help=f"Deflated at {res.plan.inflation:.1%} a year.")


def _comparison(base_res: mc.SimResult, scen: mc.SimResult) -> pd.DataFrame:
    rows = {"Median end value": "median", "5 % worst end value": "p5", "95 % best end value": "p95",
            "Ends below money put in": "below_invested", "Median return per year": "cagr_median",
            "Bad-case return per year (5 %)": "cagr_p5", "Median worst fall": "drawdown_median", "1-in-20 worst fall": "drawdown_p95"}
    if base_res.plan.monthly < 0:
        rows["Money runs out"] = "ran_out"
    a, b = base_res.summary(), scen.summary()
    out = pd.DataFrame({"Base case": [a[v] for v in rows.values()], "Scenario": [b[v] for v in rows.values()]}, index=list(rows))
    return out


def _fmt_compare(df: pd.DataFrame) -> pd.DataFrame:
    def f(label, v):
        if "value" in label:
            return money(v)
        if "fall" in label:
            return pct(-v)
        return pct(v)
    return pd.DataFrame({c: [f(i, v) for i, v in df[c].items()] for c in df.columns}, index=df.index)


def _outcomes_tab(res: mc.SimResult, base_res: mc.SimResult | None, base: str, pool: pl.ReturnPool, weights: pd.Series) -> None:
    _kpis(res, base)
    bands = res.bands()
    rng = np.random.default_rng(0)
    pick = rng.choice(res.value.shape[0], size=min(SAMPLE_PATHS, res.value.shape[0]), replace=False)
    samples = pd.DataFrame(res.value[pick].T, index=res.dates)
    invested = pd.Series(res.invested, index=res.dates)
    compare = base_res.bands()["p50"] if base_res is not None else None
    log = st.toggle("Log scale", False, key="mc_log")
    show_chart(charts.fan_figure(bands, invested, samples=samples, compare=compare, currency=base, log=log))
    sc = res.scenario
    if sc.replay:
        row = mc.crisis_table(pool, weights).set_index("key").loc[sc.replay]
        st.caption(f"Every path starts with the **{row['crisis']}** ({int(row['weeks'])} weeks{', plus the year after' if sc.replay_after else ''}): "
                   "your holdings' real returns of those weeks (estimated where one had not listed yet), so all paths are identical at "
                   "first, then fan out. "
                   f"Index over the window {pct(row['index'], signed=True)}, this portfolio {pct(row['portfolio'], signed=True)}"
                   + (f" ({row['estimated']:.0%} of it estimated: holdings not yet listed)" if row["estimated"] > 0.005 else "") + ".")
    st.caption(NOT_A_FORECAST)

    if base_res is not None:
        st.markdown("**Scenario against the base case**")
        show_table(_fmt_compare(_comparison(base_res, res)))
        st.caption("Both runs use the same random draws, so the difference comes from the scenario, not from luck.")

    st.markdown("**Milestones**")
    ms = res.milestones()
    view = pd.DataFrame({"date": ms["date"].dt.date, "money put in": ms["invested"].map(money),
                         "5 %": ms["p5"].map(money), "25 %": ms["p25"].map(money), "median": ms["p50"].map(money),
                         "75 %": ms["p75"].map(money), "95 %": ms["p95"].map(money),
                         "below money put in": ms["below_invested"].map(pct), "median, today's money": ms["median_real"].map(money)}
                        ).set_axis([f"{y:g} yr" for y in ms.index])
    show_table(view)
    if len(res.names) > 1:
        bs = res.by_stress()
        if not bs.empty:
            st.markdown(f"**Outcomes by time spent in {res.names[-1]}**")
            show_table(bs.style.format({"median CAGR": "{:.1%}", "bad case CAGR (5%)": "{:.1%}", "median max drawdown": "{:.0%}"}),
                       hide_index=True)
            st.caption("Paths split into thirds by how much of the time the market spent stressed. The spread between the rows "
                       "is how much the luck of the regime sequence matters for this plan.")


def _crises_tab(pool: pl.ReturnPool, weights: pd.Series, driver: str) -> None:
    table = mc.crisis_table(pool, weights)
    if table.empty:
        st.info(f"The {driver} history (from {pool.dates[0]:%Y}) does not cover any of the listed crises. Download the full history "
                "(Price history, above).")
        return
    view = pd.DataFrame({
        "weeks": table["weeks"], driver: table["index"].map(lambda v: pct(v, signed=True)),
        "this portfolio": table["portfolio"].map(lambda v: pct(v, signed=True)), "worst week": table["worst week"].map(pct),
        "max drawdown": table["max drawdown"].map(lambda v: pct(-v)),
        "back to start after": table["recovery_weeks"].map(lambda w: "not by the end of the data" if pd.isna(w) else
                                                         ("ended above its start" if w == 0 else f"{w / 52:.1f} years")),
        "estimated share": table["estimated"].map(lambda v: "-" if v < 0.005 else f"{v:.0%}"),
    }).set_axis(table["crisis"].to_list())
    show_table(view)
    st.caption(f"Today's weights bought at the start of each {driver} peak-to-trough window and held (no saving, no rebalancing), "
               "in weekly steps, with what happened - not the return assumption. **Estimated share**: the part of the portfolio "
               "whose holdings had not listed yet, filled in as index x beta + the holding's own noise (see Assumptions). "
               "Any of these can be played at the start of the simulation (scenario settings).")
    missing = [v[0] for k, v in mc.CRISES.items() if k not in set(table["key"])]
    if missing:
        st.caption(f"Not covered by the {driver} history from {pool.dates[0]:%Y}: " + "; ".join(missing) + ".")


def _regimes_tab(model: rg.RegimeModel) -> None:
    cur = model.current
    k = st.columns(model.k + 1)
    k[0].metric("Market state now", model.current_state, help=f"Most likely state of the latest week ({model.weekly.index[-1]:%d %b %Y}), "
                "from the weeks up to it only.")
    for i, name in enumerate(model.names):
        k[i + 1].metric(f"Chance it is {name}", pct(cur[i], 0))
    level = (1 + model.weekly).cumprod() * 100
    show_chart(charts.regime_figure(level, model.labels.map(dict(enumerate(model.names))), name=f"{model.driver} (weekly, rebased)"))
    s = model.stats()
    show_table(s.style.format({"volatility": "{:.0%}", "mean": "{:+.0%}", "avg_weeks": "{:.0f}", "share": "{:.0%}",
                               "long_run_share": "{:.0%}", "worst_week": "{:.1%}"}))
    st.caption(f"Fitted on {len(model.weekly)} weeks of {model.driver} since {model.weekly.index[0]:%b %Y}. **avg_weeks**: how long a "
               "stay in the state lasts on average; **episodes**: separate stays. The states differ above all in **volatility**; the "
               "**mean** inside a state is noisy (high-volatility weeks with positive means turned up in Thai stocks and Bitcoin), "
               "so it is not a prediction of returns.")
    thin = model.thin_states()
    if thin:
        st.warning(f"{', '.join(thin)} rests on fewer than {rg.MIN_EPISODES} separate episodes: treat what it says about "
                   "length and frequency as rough. Fewer states or a longer history (download it above) helps.")
    P = pd.DataFrame(model.transition, index=[f"from {n}" for n in model.names], columns=[f"to {n}" for n in model.names])
    a, b = st.columns(2)
    with a:
        st.markdown("**Weekly chance of moving between states**")
        show_table(P.style.format("{:.1%}"))
    with b:
        st.markdown("**Episodes**")
        ep = model.episodes().iloc[::-1]
        show_table(ep.assign(start=ep["start"].dt.date, end=ep["end"].dt.date, **{"return": ep["return"].map(lambda v: pct(v, signed=True))}),
                   hide_index=True, height=260)


def _assumptions_tab(pool: pl.ReturnPool, model: rg.RegimeModel, weights: pd.Series, cfg: dict) -> None:
    mix = model.long_run
    rf, idx = pl.to_weekly(cfg["rf"]), pl.to_weekly(cfg["index"])
    own, market = pool.own_mean(mix), pool.market_mean(mix, rf=rf, index_return=idx)
    used = pool.targets(cfg["mode"], mix, rf=rf, index_return=idx)
    vol = pool.returns.std(axis=0) * np.sqrt(rg.WEEKS_PER_YEAR)
    first = pool.first_real()
    t = pd.DataFrame({"weight": weights.reindex(pool.assets).fillna(0) / weights.reindex(pool.assets).fillna(0).sum(),
                      "prices from": first.dt.date, "real weeks": pool.coverage(),
                      **{f"beta {n}": pool.beta[n] for n in pool.names},
                      "own history %/yr": pl.to_annual(own), "market-implied %/yr": pl.to_annual(market),
                      "used %/yr": pl.to_annual(used), "volatility %/yr": vol}, index=pool.assets)
    show_table(t.style.format({"weight": "{:.0%}", "real weeks": "{:.0%}", **{f"beta {n}": "{:.2f}" for n in pool.names},
                               "own history %/yr": "{:+.1%}", "market-implied %/yr": "{:+.1%}", "used %/yr": "{:+.1%}",
                               "volatility %/yr": "{:.0%}"}))
    st.markdown(
        f"- **Pool**: {pool.n_weeks} weeks of {model.driver} since {pool.dates[0]:%b %Y}, each with its state. Returns are in your "
        f"currency, so exchange-rate moves are part of the risk (before a currency's rate history starts, its first known rate is used).\n"
        f"- **Weeks before a holding listed** (real weeks < 100 %) are estimated: {model.driver}'s return that week x the holding's beta "
        f"in that state + a residual from one of the holding's real weeks in the same state. Holdings that traded over the same years "
        f"share the residual week, so they keep moving together.\n"
        f"- **Expected return**: every week of a holding is moved by one constant so its long-run average equals *used %/yr*. "
        f"Market-implied = cash rate {cfg['rf']:.1%} + beta x ({model.driver} {cfg['index']:.1%} - cash rate). The shift is the "
        f"same in every state, so stressed weeks stay worse than calm ones.\n"
        f"- **Draws**: blocks of consecutive history weeks (4 on average) in the state each path needs.\n"
        f"- Not modelled: taxes on dividends or gains, a change in what you hold, the cash balance, and anything the "
        f"{pool.dates[0]:%Y}-{pool.dates[-1]:%Y} history did not contain."
    )
    if pool.dropped:
        st.warning("Left out (too little history): " + "; ".join(f"{s}: {why}" for s, why in pool.dropped.items()))


def _check_tab(driver: str, k: int, last_week: str, weekly: pd.Series) -> None:
    st.markdown(f"Would the simulation have described **{driver}**'s next 3 months, 6 months and 1 year well, using only the "
                "data known at each point? Every 4 weeks from the 10th year of history on, both engines simulate the index and "
                "the real outcome is placed in their range.")
    if not st.button("Run the check (15 - 60 s)", key="mc_check") and not st.session_state.get(f"mc_check_{driver}_{k}"):
        return
    st.session_state[f"mc_check_{driver}_{k}"] = True
    try:
        with st.spinner("Walking forward through the history ..."):
            c = _calibration(driver, k, last_week)
    except ValueError as exc:
        st.info(str(exc))
        return
    v = c.verdict()
    if not v.empty:
        lines = []
        for h, r in v.iterrows():
            if r["p-value"] < 0.05:
                who = "regimes **better**" if r["CRPS change"] < 0 else "plain bootstrap **better**"
            else:
                who = "**no clear difference**"
            lines.append(f"- {h}: {who} (score {r['CRPS change']:+.1%}, p = {r['p-value']:.2f}, ~{int(r['independent windows'])} independent windows)")
        st.markdown(f"Checks from {c.first:%b %Y} to {c.last:%b %Y}:\n" + "\n".join(lines))
    show_table(c.table().style.format({"inside 50%": "{:.0%}", "inside 90%": "{:.0%}", "below 5%": "{:.0%}", "above 95%": "{:.0%}",
                                       "CRPS": "{:.4f}"}))
    st.caption("**inside 50 % / 90 %**: how often the real return fell in the middle 50 % / 90 % of the simulated ones - a good "
               "simulation scores close to 50 % and 90 %. **CRPS**: one score for the whole simulated range against what happened, "
               "lower is better; the p-value allows for the overlap of consecutive windows (Diebold-Mariano, Newey-West).")
    st.markdown("**Split by the state at the start**")
    show_table(c.by_state().style.format({"inside 90%": "{:.0%}", "CRPS": "{:.4f}"}))
    st.caption("Where regimes are meant to help: after a stressed start the plain bootstrap's range tends to be too narrow (real "
               "outcomes fall outside it more often than 10 % of the time) and after a calm start too wide. On SPY since 1993 the "
               "regimes fixed part of that without improving the overall score - they describe risk better after stress, they do "
               "not predict returns.")


def render() -> None:
    st.title("🎲 Monte Carlo")
    st.caption("Thousands of possible futures for a portfolio, with monthly saving (DCA) or withdrawals, through calm and stressed "
               "markets drawn from history - crises included.")
    weights, value, base, src = _portfolio_inputs()
    if weights is None:
        return
    symbols = tuple(weights.index)
    d0 = _default_driver(weights)
    options = list(dict.fromkeys([d0, *DRIVERS]))
    driver = st.selectbox("Regime driver (the market whose calm and stressed spells are simulated)", options,
                          format_func=lambda d: DRIVERS.get(d, d), key=f"mc_driver_{d0}")
    _history_panel(driver, symbols, base)
    weekly = _driver_weekly(driver)
    if len(weekly) < rg.MIN_WEEKS:
        st.error(f"Need at least {rg.MIN_WEEKS} weeks of {driver} prices, have {len(weekly)}. Download its history above.")
        return
    last_week = f"{weekly.index[-1]:%Y-%m-%d}|{len(weekly)}"
    initial = value if value is not None else (1_000_000.0 if base == "THB" else 100_000.0)
    submitted = _settings_form(base, initial, src, driver, weekly, float(pl.to_annual(weekly.mean())))
    key = (src, symbols, base, driver)
    if submitted is not None:
        st.session_state["mc_cfg"] = {**submitted, "key": key}
    cfg = st.session_state.get("mc_cfg")
    if cfg is None or cfg.get("key") != key:
        st.info("Set the plan and press **Run simulation**.")
        return

    k = cfg["k"]
    try:
        with st.spinner("Fitting the regime model and simulating ..."):
            model = _model(driver, k, last_week)
            pool = _pool(driver, k, last_week, symbols, base)
            missing = [s for s in symbols if s not in pool.assets and s not in pool.dropped]
            if missing:
                st.warning("No prices in " + base + " for: " + ", ".join(missing) + " - left out.")
            if not pool.assets:
                st.error("None of the holdings has enough price history to simulate.")
                return
            w = tuple(float(weights.get(s, 0.0)) for s in pool.assets)
            args = (driver, k, last_week, tuple(pool.assets), base, w, dataclasses.astuple(cfg["plan"]))
            rest = (cfg["mode"], cfg["rf"], cfg["index"], cfg["n_paths"], cfg["engine"])
            res = _run(*args, cfg["scenario"], *rest)
            neutral = ("now", 1.0, 1.0, None, False)
            base_res = _run(*args, neutral, *rest) if cfg["compare"] and cfg["scenario"] != neutral else None
    except ValueError as exc:
        st.error(str(exc))
        return
    weights_in_pool = pd.Series(w, index=pool.assets)
    tabs = st.tabs(["Outcomes", "Crises", "Regimes", "Assumptions", "Check model"])
    with tabs[0]:
        _outcomes_tab(res, base_res, base, pool, weights_in_pool)
    with tabs[1]:
        _crises_tab(pool, weights_in_pool, driver)
    with tabs[2]:
        _regimes_tab(model)
    with tabs[3]:
        _assumptions_tab(pool, model, weights_in_pool, cfg)
    with tabs[4]:
        _check_tab(driver, k, last_week, weekly)
