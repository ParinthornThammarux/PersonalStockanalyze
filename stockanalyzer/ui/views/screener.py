"""Screener: pick a screen from a card, see how many stocks fit, then look closer at any of them."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from stockanalyzer.alerts.watchlist import FAVORITES
from stockanalyzer.analysis.technical import indicators as ind
from stockanalyzer.data import universe as uni
from stockanalyzer.domain.assets import infer_asset_class
from stockanalyzer.screener.features import compute_universe_features
from stockanalyzer.screener.readiness import fetch_seconds, readiness
from stockanalyzer.screener.rules import PRESETS, Rule, Screen, explain_row
from stockanalyzer.screener.saved import rules_to_spec, spec_to_rules
from stockanalyzer.screener.scoring import DEFAULT_WEIGHTS, score_table
from stockanalyzer.screener.table import MIN_TRADED_TH, MIN_TRADED_US, build_screen_table, liquidity_mask
from stockanalyzer.ui import charts
from stockanalyzer.ui import criteria as crit
from stockanalyzer.ui import data as ui_data
from stockanalyzer.ui import loader
from stockanalyzer.ui.context import services
from stockanalyzer.ui.format import describe_rule, feature_view, format_feature, label_of
from stockanalyzer.ui.widgets import show_chart

EXCEL_ROWS = 50_000  # far more than the market has symbols; Excel itself stops at about a million rows
RESULT_COLUMNS = ["name", "close", "currency", "score", "ret_21d", "ret_126d", "rsi14", "trend_label", "pct_from_52w_high",
                  "pe", "roe", "market_cap", "industry"]
CUSTOM = "Custom"
SAVED = "★ "  # prefix of a saved screen's name in the "current screen" state
DEFAULT_RULES = [Rule("rsi14", "<", 40)]
NEW_RULE = Rule("ret_126d", ">", 0.0)  # what "Add criterion" starts from: a 6-month return above 0%
MARKET_US, MARKET_TH = "US & other", "Thailand (.BK)"
ICONS = {"Oversold in an uptrend": "🔻", "Breakout near 52-week high": "🚀", "Golden cross + momentum": "✨",
         "Near support": "🧱", "Strong relative strength": "💪", "Value + positive momentum": "💰",
         "Cheap with a dividend": "🪙", "Quality compounders": "🏆", CUSTOM: "✏️"}
CARDS_PER_ROW = 3
MAX_FETCH = 2000  # full snapshots in one go; 400 took 15 s without Yahoo pushing back, a few thousand is untested
SPARKLINE_ROWS = 300  # rows of the table that get a price chart
WATCHLIST_ROWS = 500
NEW_LIST = "➕ New watchlist ..."
COMPONENT_LABELS = {"momentum": "Momentum", "trend": "Trend", "risk": "Low risk", "value": "Value vs peers",
                    "quality": "Quality vs peers", "liquidity": "Liquidity"}


@st.cache_data(ttl=300, show_spinner=False)
def _table(stamp: tuple) -> pd.DataFrame | None:
    sv = services()
    features = sv.features.load_latest()
    if features is None:
        return None
    return build_screen_table(features, ui_data.universe_table(), sv.fundamentals.snapshots_frame(),
                              usd_rate=lambda ccy: ui_data.fx_rate(ccy, "USD"))


def _stamp() -> tuple:
    sv = services()
    latest = Path(sv.settings.features_dir) / "latest.parquet"
    return (latest.stat().st_mtime if latest.exists() else 0, len(sv.fundamentals.snapshots_frame()))


def _to_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="screen")
    return buf.getvalue()




# -- state helpers --------------------------------------------------------------------------------
def _text(value) -> str:
    return value if isinstance(value, str) else ""


def _select(name: str) -> None:
    st.session_state["scr_preset"] = name


def _load_saved(name: str) -> None:
    spec = services().saved_screens.load(name)
    if spec is not None:
        st.session_state["scr_pending"] = (SAVED + name, spec)


def _delete_saved(name: str) -> None:
    services().saved_screens.delete(name)
    if st.session_state.get("scr_preset") == SAVED + name:
        st.session_state["scr_preset"] = next(iter(PRESETS))


def _apply_pending(table: pd.DataFrame) -> None:
    """Make a loaded or just-saved screen the current one: its criteria, sort order and universe filters.

    Runs before any widget is built, which is the only moment their values may be set from the script.
    """
    pending = st.session_state.pop("scr_pending", None)
    if not pending:
        return
    name, spec = pending
    st.session_state["scr_preset"] = name
    st.session_state.pop(_crit_key(name), None)  # the criteria start again from the saved ones
    universe = spec.get("universe", {})
    known = {"industries": set(table["industry"].dropna()) if "industry" in table else set(),
             "classes": set(table["asset_class"].dropna()) if "asset_class" in table else set(),
             "markets": {MARKET_US, MARKET_TH}}
    for key, field in (("scr_markets", "markets"), ("scr_classes", "classes"), ("scr_industries", "industries")):
        if field in universe:
            st.session_state[key] = [v for v in universe[field] if v in known[field]]
    for key, source, field in (("scr_min_cap", universe, "min_cap"), ("scr_min_dv_us", universe, "min_dv_us"),
                               ("scr_min_dv_th", universe, "min_dv_th"), ("scr_spacs", universe, "no_spacs"),
                               ("scr_sort", spec, "sort_by"), ("scr_asc", spec, "ascending"), ("scr_top", spec, "top")):
        if field in source:
            st.session_state[key] = source[field]


def _base_rules(name: str, saved_screens) -> list[Rule]:
    if name in PRESETS:
        return list(PRESETS[name].rules)
    if name.startswith(SAVED):
        spec = saved_screens.load(name[len(SAVED):])
        if spec:
            return spec_to_rules(spec.get("rules", []))
    return list(DEFAULT_RULES)


# -- the criteria builder ------------------------------------------------------------------------
def _crit_key(name: str) -> str:
    return f"scr_crit_{name}"


def _new_item(rule: Rule) -> dict:
    """A criterion with an id of its own, so its widgets keep their state when others are added or removed."""
    st.session_state["scr_crit_seq"] = st.session_state.get("scr_crit_seq", 0) + 1
    return {"id": st.session_state["scr_crit_seq"], "field": rule.field, "op": rule.op, "value": rule.value}


def _items(name: str, base: list[Rule]) -> list[dict]:
    if _crit_key(name) not in st.session_state:
        st.session_state[_crit_key(name)] = [_new_item(r) for r in base]
    return st.session_state[_crit_key(name)]


def _add_item(name: str) -> None:
    st.session_state[_crit_key(name)].append(_new_item(NEW_RULE))


def _remove_item(name: str, item_id: int) -> None:
    st.session_state[_crit_key(name)] = [i for i in st.session_state[_crit_key(name)] if i["id"] != item_id]


def _reset_items(name: str) -> None:
    st.session_state.pop(_crit_key(name), None)


def _seed(key: str, make) -> None:
    """Give a widget its first value through session state (the value it keeps from then on), once."""
    if key not in st.session_state:
        st.session_state[key] = make()


def _value_input(box, item: dict, field: str, kind: str, op: str, series: pd.Series, same: bool):
    """The input that suits the column - % for fractions, millions for dollars, yes / no, a list of labels - and the rule value it gives."""
    base = f"scr_c{item['id']}_{field}"
    original = item["value"] if same and (op == item["op"] or (op == "between") == (item["op"] == "between")) else None

    def start(default):
        if original is None:
            return default
        try:
            return crit.to_ui(original, kind)
        except (TypeError, ValueError):
            return default

    if kind == "bool":
        _seed(base + "_b", lambda: "no" if original is False else "yes")
        return box.selectbox("Value", ["yes", "no"], key=base + "_b", label_visibility="collapsed") == "yes"
    if kind == "text":
        options = crit.text_options(series)
        picked = start(crit.default_value(kind, op, series))
        if op in ("in", "not in"):
            picked = [v for v in (picked if isinstance(picked, (list, tuple)) else [picked]) if isinstance(v, str)]
            options = [*options, *[v for v in picked if v not in options]]
            _seed(base + "_m", lambda: picked)
            return box.multiselect("Value", options, key=base + "_m", label_visibility="collapsed", placeholder="pick one or more")
        picked = picked if isinstance(picked, str) else (picked[0] if picked else "")
        options = options if picked in options or not picked else [*options, picked]
        _seed(base + "_t", lambda: picked or (options[0] if options else ""))
        return box.selectbox("Value", options, key=base + "_t", label_visibility="collapsed")

    step = crit.step_for(series, kind)
    fmt = crit.number_format(step, kind)
    cast = int if kind == "int" else float
    if op == "between":
        low, high = box.columns(2)
        lo0, hi0 = start(crit.default_value(kind, op, series)) if original is None or isinstance(original, (list, tuple)) else crit.default_value(kind, op, series)
        _seed(base + "_lo", lambda: cast(lo0))
        _seed(base + "_hi", lambda: cast(hi0))
        a = low.number_input("From", key=base + "_lo", step=step, format=fmt, label_visibility="collapsed")
        b = high.number_input("To", key=base + "_hi", step=step, format=fmt, label_visibility="collapsed")
        return crit.from_ui((a, b), kind)
    one = start(crit.default_value(kind, op, series)) if not isinstance(original, (list, tuple)) else crit.default_value(kind, op, series)
    _seed(base + "_v", lambda: cast(one))
    return crit.from_ui(box.number_input("Value", key=base + "_v", step=step, format=fmt, label_visibility="collapsed"), kind)


def _criterion(item: dict, name: str, pool: pd.DataFrame, running: pd.Series, fields: list[str]) -> tuple[Rule, pd.Series]:
    """One row: metric, condition, value, a remove button - and under it how many stocks it leaves."""
    options = fields if item["field"] in fields else [*fields, item["field"]]
    field_key = f"scr_c{item['id']}_field"
    if st.session_state.get(field_key) not in options:
        st.session_state[field_key] = item["field"]
    with st.container(border=True):
        c_field, c_op, c_val, c_rm = st.columns([4.2, 2.2, 3.6, 0.6], vertical_alignment="center")
        field = c_field.selectbox("Metric", options, key=field_key, label_visibility="collapsed",
                                  format_func=lambda f: crit.option_label(f, pool[f] if f in pool.columns else None))
        series = pool[field] if field in pool.columns else pd.Series(dtype="float64")
        kind = crit.kind_of(field, series)
        same = field == item["field"]
        ops = crit.ops_for(kind, item["op"] if same else None)
        op_key = f"scr_c{item['id']}_{field}_op"
        if st.session_state.get(op_key) not in dict(ops):
            st.session_state[op_key] = item["op"] if same and item["op"] in dict(ops) else ops[0][0]
        op = c_op.selectbox("Condition", [o for o, _ in ops], key=op_key, label_visibility="collapsed", format_func=dict(ops).get)
        value = _value_input(c_val, item, field, kind, op, series, same)
        c_rm.button("✕", key=f"scr_c{item['id']}_rm", help="Remove this criterion", on_click=_remove_item, args=(name, item["id"]))

        rule = Rule(field, op, value)
        alone = rule.mask(pool)
        running = running & alone
        total, left = len(pool), int(running.sum())
        if not crit.has_data(series):
            st.progress(0.0, text=f"No stock in the pool has {crit.option_label(field).split(' · ', 1)[1]} yet - load fundamentals below to use it")
        else:
            hint = crit.typical(series, field)
            st.progress(left / total if total else 0.0,
                        text=f"{left:,} of {total:,} left · this one alone keeps {int(alone.sum()):,}" + (f" · {hint}" if hint else ""))
    return rule, running


def _criteria_builder(name: str, base: list[Rule], pool: pd.DataFrame) -> list[Rule]:
    items = _items(name, base)
    plural = "" if len(items) == 1 else "s"
    title = f"Criteria · {len(items)} rule{plural}, all must hold" if items else "Criteria · none"
    rules: list[Rule] = []
    with st.expander(title, expanded=True):
        if name in PRESETS:
            st.caption(PRESETS[name].description)
        running, fields = pd.Series(True, index=pool.index), crit.offered_fields(pool)
        for item in list(items):
            rule, running = _criterion(item, name, pool, running, fields)
            rules.append(rule)
        if not items:
            st.info("No criteria: every stock in the pool is listed. Add one to narrow it down.")
        add, reset, _ = st.columns([1.2, 1.4, 3])
        add.button("➕ Add criterion", key="scr_add", on_click=_add_item, args=(name,), width="stretch")
        reset.button("↺ Back to the defaults", key="scr_reset", on_click=_reset_items, args=(name,), width="stretch",
                     help="Drops your changes to this screen's criteria.")
    return rules


# -- pieces of the page ---------------------------------------------------------------------------
def _cards(pool: pd.DataFrame, current: str) -> None:
    entries = []
    for name, screen in PRESETS.items():
        entries.append((name, screen.description, readiness(screen.rules, pool), int(screen.mask(pool).sum())))
    entries.append((CUSTOM, "Write your own criteria from any column of the table.", None, None))
    for start in range(0, len(entries), CARDS_PER_ROW):
        for col, (i, (name, description, ready, matches)) in zip(
                st.columns(CARDS_PER_ROW), enumerate(entries[start:start + CARDS_PER_ROW], start)):
            chosen = name == current
            with col.container(border=True):
                st.markdown(f"**{ICONS.get(name, '🧮')} {name}**")
                st.caption(description)
                if matches is None:
                    st.markdown("##### build your own")
                elif ready.blocked:
                    st.markdown("##### needs fundamentals")
                    st.caption(f"None of the {len(pool):,} stocks has them loaded yet - one click loads them.")
                else:
                    st.markdown(f"##### {matches:,} match" + ("" if matches == 1 else "es"))
                    if ready.partial:
                        st.caption(f"Fundamentals are loaded for {ready.covered:,} of {ready.total:,} stocks.")
                st.button("Showing" if chosen else "Show", key=f"scr_card_{i}", type="primary" if chosen else "secondary",
                          disabled=chosen, on_click=_select, args=(name,), width="stretch")


def _saved_row(saved_screens, names: list[str]) -> None:
    if not names:
        return
    with st.expander(f"★ My saved screens ({len(names)})"):
        pick = st.selectbox("Saved screen", names, key="scr_saved_pick", label_visibility="collapsed")
        a, b, _ = st.columns([1, 1, 3])
        a.button("Load", key="scr_saved_load", on_click=_load_saved, args=(pick,), width="stretch")
        b.button("Delete", key="scr_saved_delete", on_click=_delete_saved, args=(pick,), width="stretch")


def _fetch_fundamentals(symbols: list[str]) -> None:
    """Fetch full snapshots with a progress bar; the next run shows the outcome and rebuilds the table."""
    bar = st.progress(0.0, text="starting ...")
    report = services().fundamentals.refresh(symbols, workers=4,
                                             progress=lambda d, t, m: bar.progress(min(d / max(t, 1), 1.0), text=f"{d}/{t} · {m}"))
    bar.empty()
    st.session_state["scr_notice"] = f"Fundamentals loaded: {report.summary()}"
    _table.clear()
    st.rerun()


def _missing_fundamentals_panel(rules: list[Rule], ready) -> None:
    """The way out of 'this screen needs ROE / margins and none are loaded': fetch them for the stocks that could still match."""
    wanted = ", ".join(sorted({label_of(r.field) for r in ready.needs_full}))
    with st.container(border=True):
        st.markdown(f"**{len(ready.candidates):,} stocks pass your other criteria but have no fundamentals loaded yet**")
        st.caption(f"{wanted} come from each company's full fundamentals, which are fetched one stock at a time; "
                   f"{ready.covered:,} of {ready.total:,} stocks in the pool have them so far. "
                   "Loading starts with the largest companies by market cap.")
        cap = min(len(ready.candidates), MAX_FETCH)
        count = int(st.number_input("Load fundamentals for the largest", 1, cap, min(cap, 1000), 50, key="scr_fetch_n"))
        if st.button(f"Load fundamentals for {count:,} stocks (about {fetch_seconds(count)} s)", type="primary", key="scr_fetch_full"):
            _fetch_fundamentals(ready.candidates[:count])


def _sparklines(symbols: list[str]) -> dict[str, list[float]]:
    closes = ui_data.stored_closes(tuple(symbols[:SPARKLINE_ROWS]), 64)
    return {s: closes[s].dropna().tolist() for s in closes.columns}


def _results_table(hits: pd.DataFrame):
    view, config = feature_view(hits, [c for c in RESULT_COLUMNS if c in hits.columns])
    lines = _sparklines(list(hits.index))
    view.insert(min(1, len(view.columns)), "chart", [lines.get(s) or None for s in view.index])
    config["chart"] = st.column_config.LineChartColumn("3 months", width="small")
    return st.dataframe(view, column_config=config, width="stretch", height=560, on_select="rerun",
                        selection_mode="single-row", key="scr_table")


def _detail(symbol: str, scored: pd.DataFrame, rules: list[Rule], hint: str) -> None:
    row = scored.loc[[symbol]]
    r = row.iloc[0]
    st.divider()
    st.subheader(f"{symbol} · {_text(r.get('name')) or symbol}")
    full = r.get("fundamentals") == "full"
    st.caption(" · ".join(b for b in (_text(r.get("industry")), _text(r.get("currency")),
                                      "full fundamentals loaded" if full else "market-wide figures only (P/E, P/B, yield)", hint) if b))
    for col, (label, field) in zip(st.columns(6), [("Last", "close"), ("1 month", "ret_21d"), ("6 months", "ret_126d"),
                                                   ("From 52w high", "pct_from_52w_high"), ("P/E", "pe"), ("Market cap", "market_cap")]):
        col.metric(label, format_feature(field, r.get(field)))

    left, right = st.columns([3, 2], gap="large")
    with left:
        frame = ui_data.prices(symbol, "2y", "1d", True, False)  # the local store only, never the network
        if len(frame) < 30:
            st.caption("No stored prices to chart.")
        else:
            overlays = [(f"SMA {n}", ind.sma(frame["close"], n).tail(252)) for n in (50, 200) if len(frame) > n]
            show_chart(charts.price_figure(frame.tail(252), title="1 year", overlays=overlays, volume=False, height=340,
                                           skip_weekends=not infer_asset_class(symbol).trades_weekends))
    with right:
        parts = [(label, r.get(f"score_{key}")) for key, label in COMPONENT_LABELS.items() if f"score_{key}" in scored.columns]
        st.markdown(f"**Score {format_feature('score', r.get('score'))} / 100**")
        st.dataframe(pd.DataFrame(parts, columns=["Part", "Rank"]), hide_index=True, width="stretch",
                     column_config={"Rank": st.column_config.ProgressColumn("Rank in pool", min_value=0, max_value=100, format="%.0f")})
        st.markdown("**Why it is on this list**")
        if not rules:
            st.caption("No criteria: every stock in the pool is listed.")
        else:
            checks = explain_row(rules, row)
            st.dataframe(pd.DataFrame({"Criterion": [describe_rule(rule) for rule in checks["rule"]],
                                       "This stock": [format_feature(rule.field, v) for rule, v in zip(checks["rule"], checks["value"])],
                                       "": ["✓" if ok else "✗" for ok in checks["passed"]]}),
                         hide_index=True, width="stretch")
    a, b, _ = st.columns([1, 1, 2])
    if a.button(f"Open {symbol}", key="scr_open", width="stretch"):
        st.session_state["symbol"] = symbol
        from stockanalyzer.ui.nav import PAGES
        st.switch_page(PAGES["overview"])
    if b.button(f"Add {symbol} to Favorites", key="scr_fav", width="stretch"):
        services().watchlists.add(FAVORITES, symbol)
        st.toast(f"{symbol} added to {FAVORITES}")


def _save_panel(sv, hits: pd.DataFrame, spec: dict) -> None:
    with st.expander("Keep these results"):
        left, right = st.columns(2, gap="large")
        with left:
            shown = list(hits.index[:WATCHLIST_ROWS])
            st.markdown(f"**Add {len(shown):,} shown symbol{'' if len(shown) == 1 else 's'} to a watchlist**")
            lists = sv.watchlists.names() or [FAVORITES]
            target = st.selectbox("Watchlist", [*lists, NEW_LIST], key="scr_wl_target")
            new_name = st.text_input("New watchlist name", key="scr_wl_new") if target == NEW_LIST else ""
            destination = new_name.strip() if target == NEW_LIST else target
            if len(hits) > WATCHLIST_ROWS:
                st.caption(f"The first {WATCHLIST_ROWS:,} of {len(hits):,} are added; narrow the criteria for the rest.")
            if st.button(f"Add to {destination or '...'}", key="scr_wl_add", disabled=not shown or not destination):
                added = sv.watchlists.add_many(destination, shown)
                st.toast(f"{added} new symbol{'' if added == 1 else 's'} added to {destination} ({len(shown) - added} already there)")
        with right:
            st.markdown("**Save this screen to use again**")
            name = st.text_input("Name", key="scr_save_name", placeholder="e.g. Thai dividend picks",
                                 help="Keeps the criteria, the sort order and the universe filters in the sidebar. "
                                      "Saving under an existing name replaces it.")
            if st.button("Save screen", key="scr_save", disabled=not name.strip()):
                sv.saved_screens.save(name, spec)
                st.session_state["scr_pending"] = (SAVED + name.strip(), spec)
                st.session_state["scr_notice"] = f"Saved “{name.strip()}”. Find it under My saved screens."
                st.rerun()


def render() -> None:
    st.title("🧮 Screener")
    sv = services()
    table = _table(_stamp())

    if table is None:
        st.info("No feature table yet. It is computed from the local price store (a few seconds for hundreds of symbols).")
        stored = sv.md.store.symbols()
        st.caption(f"{len(stored)} symbols in the store.")
        if st.button("Compute features now", disabled=not stored, type="primary"):
            with st.spinner(f"Computing features for {len(stored)} symbols ..."):
                frame = compute_universe_features(sv.md.store, workers=0)
                if frame.empty:
                    st.error("None of the stored symbols has enough history (60+ daily bars).")
                else:
                    sv.features.save(frame)
                    _table.clear()
                    ui_data.store_overview.clear()
                    st.rerun()
        if not stored:
            st.caption("Fill the store first in **Data Manager** (pick *Whole market*), or run `python -m stockanalyzer nightly --thai all`.")
        return

    _apply_pending(table)
    if notice := st.session_state.pop("scr_notice", None):
        st.success(notice)
    asof = table["last_date"].max()
    has_fund = "pe" in table.columns
    full = int((table["fundamentals"] == "full").sum()) if "fundamentals" in table else 0
    st.caption(f"{len(table):,} symbols · data as of {asof:%Y-%m-%d}"
               + (f" · P/E for {int(table['pe'].notna().sum()):,}, full fundamentals (ROE, margins, cash flow) for {full:,}" if has_fund
                  else " · no fundamentals cached yet (value/quality presets stay empty)"))
    loader.coverage_banner(loader.coverage(), key="scr_load_market")

    # -- universe filters --------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### Universe")
        classes = sorted(table["asset_class"].dropna().unique()) if "asset_class" in table else []
        chosen = st.multiselect("Asset class", classes, default=[c for c in classes if c == "equity"] or classes, key="scr_classes")
        thai_rows = table.index.str.endswith(".BK")
        markets = st.multiselect("Market", [MARKET_US, MARKET_TH], default=[MARKET_US, MARKET_TH], key="scr_markets",
                                 help=f"{int((~thai_rows).sum()):,} US and other symbols, {int(thai_rows.sum()):,} Thai (.BK) symbols are stored.")
        min_cap = st.number_input("Min market cap (US$ M)", 0, 5_000_000, 0, 100, key="scr_min_cap",
                                  help="Thai and other non-dollar stocks are converted at today's exchange rate.")
        st.caption("Minimum average traded value (20-day price x volume, in US$ M per day)")
        min_dv_us = st.number_input("US & other", 0.0, 100_000.0, MIN_TRADED_US / 1e6, 1.0, key="scr_min_dv_us",
                                    help="Applied to every symbol except Thai ones. 0 keeps everything, including indices and crypto.")
        min_dv_th = st.number_input("Thailand (.BK)", 0.0, 100_000.0, MIN_TRADED_TH / 1e6, 0.1, format="%.2f", key="scr_min_dv_th",
                                    help="Its own floor, because US$5M a day is only met by about 60 of the 883 Thai stocks. "
                                         "The default US$0.3M is about 10 million baht a day (about 170 stocks); most Thai small caps trade under US$0.05M.")
        no_spacs = st.checkbox("Exclude shell companies (SPACs)", value=True, key="scr_spacs")
        industries = (st.multiselect("Industries (optional)", sorted(table["industry"].dropna().unique()), key="scr_industries")
                      if "industry" in table else [])

    pool = table
    if len(markets) == 1:
        pool = pool[pool.index.str.endswith(".BK") == (markets[0] == MARKET_TH)]
    elif not markets:
        pool = pool.iloc[0:0]
    if chosen and "asset_class" in pool:
        pool = pool[pool["asset_class"].isin(chosen)]
    if min_cap and "market_cap" in pool:
        pool = pool[pool["market_cap"] >= min_cap * 1e6]
    pool = pool[liquidity_mask(pool, min_dv_us * 1e6, min_dv_th * 1e6)]
    if no_spacs and "industry" in pool:
        pool = pool[pool["industry"] != uni.SPAC_INDUSTRY]
    if industries:
        pool = pool[pool["industry"].isin(industries)]
    st.markdown(f"**{len(pool):,}** symbols in the investable pool")

    # -- 1 · pick a screen -------------------------------------------------------------------
    saved_names = sv.saved_screens.names()
    current = st.session_state.get("scr_preset")
    if current not in {*PRESETS, CUSTOM, *(SAVED + n for n in saved_names)}:
        current = st.session_state["scr_preset"] = next(iter(PRESETS))
    _cards(pool, current)
    _saved_row(sv.saved_screens, saved_names)
    if current.startswith(SAVED):
        st.info(f"Showing your saved screen **{current[len(SAVED):]}**.")

    # -- 2 · criteria --------------------------------------------------------------------------
    rules = _criteria_builder(current, _base_rules(current, sv.saved_screens), pool)

    ready = readiness(rules, pool)
    if ready.needs_full and ready.candidates:
        _missing_fundamentals_panel(rules, ready)

    # -- 3 · results ---------------------------------------------------------------------------
    c1, c2, c3 = st.columns([2, 1, 1])
    sort_options = ["score", *[c for c in RESULT_COLUMNS if c in pool.columns and c not in ("name", "industry", "trend_label", "score")]]
    if st.session_state.get("scr_sort", "score") not in sort_options:
        st.session_state["scr_sort"] = "score"
    sort_by = c1.selectbox("Sort by", sort_options, key="scr_sort")
    ascending = c2.toggle("Ascending", value=False, key="scr_asc")
    top = c3.number_input("Show top", 10, 10_000, 100, 10, key="scr_top")

    scored = score_table(pool)
    screen = Screen(current, rules, sort_by=sort_by, ascending=ascending, limit=int(top))
    hits = screen.apply(scored)
    n_matches = int(screen.mask(scored).sum())
    st.subheader(f"{n_matches:,} match" + ("" if n_matches == 1 else "es"))
    if n_matches > len(hits):
        st.caption(f"Showing the top {len(hits):,} by {sort_by}; raise *Show top* or export to get all {n_matches:,}.")
    elif ready.blocked and not ready.candidates:
        st.caption("This screen needs company fundamentals, and no stock passes its other criteria to load them for.")

    if hits.empty:
        picked = []
    else:
        event = _results_table(hits)
        picked = [hits.index[i] for i in event.selection.rows] if event and event.selection.rows else []

    everything = Screen(current, rules, sort_by=sort_by, ascending=ascending).apply(scored)  # not cut at "Show top"
    a, b, _ = st.columns([1, 1, 2])
    with a:
        st.download_button(f"CSV ({len(everything):,})", everything.to_csv().encode(), file_name="screen.csv", mime="text/csv", width="stretch")
    with b:
        st.download_button(f"Excel ({len(everything):,})", _to_excel(everything.head(EXCEL_ROWS)), file_name="screen.xlsx", width="stretch",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # -- 4 · look closer -----------------------------------------------------------------------
    if not hits.empty:
        symbol = picked[0] if picked else hits.index[0]
        _detail(symbol, scored, rules, "" if picked else "the first result - click another row to switch")
        spec = {"rules": rules_to_spec(rules), "sort_by": sort_by, "ascending": bool(ascending), "top": int(top),
                "universe": {"markets": markets, "classes": chosen, "min_cap": int(min_cap), "min_dv_us": float(min_dv_us),
                             "min_dv_th": float(min_dv_th), "no_spacs": bool(no_spacs), "industries": industries}}
        _save_panel(sv, hits, spec)

    light = list(hits.index[hits["fundamentals"].ne("full")]) if "fundamentals" in hits else []
    if light and not (ready.needs_full and ready.candidates):
        with st.expander(f"Get full fundamentals (ROE, margins, free cash flow) for these matches - {len(light):,} still have only P/E, P/B and yield"):
            take = st.number_input("For the top", 1, min(len(light), MAX_FETCH), min(len(light), 100), 10, key="scr_detail_n",
                                   help="One Yahoo request per stock; about 20 stocks a second with 4 in parallel when measured.")
            if st.button(f"Load (about {fetch_seconds(int(take))} s)", key="scr_detail"):
                _fetch_fundamentals(light[:int(take)])
    with st.expander("How the score works"):
        st.markdown("Each component is the average **percentile rank within the investable pool** (0-100) of a few columns; "
                    "the composite weights the components that have data. Default weights: "
                    + ", ".join(f"{k} {v:.0%}" for k, v in DEFAULT_WEIGHTS.items()) + ".\n\n"
                    "- **Momentum** is the 6-month and 12-1-month return; **trend** is the trend score and the 60-day fit quality "
                    "(R² signed by the slope, so a smooth decline scores low); **risk** is 60-day volatility and the worst 1-year drawdown.\n"
                    "- **Value and quality are ranked against peers**: a stock is compared with the others in its industry and market "
                    "(or its whole market when fewer than 10 peers have the figure), so banks are not 'cheap' just for having a low "
                    "P/B and Thai stocks are not 'cheap' just for being Thai.\n\n"
                    "Ranks are relative, so the score says *which symbols look strongest now*, not that they will keep doing so.")
