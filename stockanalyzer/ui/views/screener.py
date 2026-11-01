"""Screener: filter, score and rank the whole universe from the precomputed feature table."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from stockanalyzer.data import universe as uni
from stockanalyzer.screener.features import FEATURE_INFO, compute_universe_features
from stockanalyzer.screener.rules import OPERATORS, PRESETS, Rule, Screen
from stockanalyzer.screener.scoring import DEFAULT_WEIGHTS, score_table
from stockanalyzer.screener.table import build_screen_table
from stockanalyzer.ui import data as ui_data
from stockanalyzer.ui.context import services
from stockanalyzer.ui.format import feature_view
from stockanalyzer.ui.widgets import show_table

RESULT_COLUMNS = ["name", "industry", "market_cap", "close", "ret_21d", "ret_126d", "rsi14", "adx14", "trend_label",
                  "volume_ratio", "pct_from_52w_high", "pe", "roe", "score"]
CUSTOM = "Custom"


def parse_value(op: str, text) -> object:
    """Text cell -> rule value: numbers, booleans, 'a, b' pairs for between / lists for in."""
    def scalar(t: str):
        t = t.strip()
        if t.lower() in ("true", "false"):
            return t.lower() == "true"
        try:
            return float(t)
        except ValueError:
            return t
    parts = [scalar(p) for p in str(text).strip("()[] ").split(",") if p.strip() != ""]
    if op == "between":
        if len(parts) != 2:
            raise ValueError("'between' needs two numbers, e.g. 0, 15")
        return tuple(parts)
    if op in ("in", "not in"):
        return parts
    if len(parts) != 1:
        raise ValueError("expected a single value")
    return parts[0]


def rules_to_frame(rules: list[Rule]) -> pd.DataFrame:
    def show(v):
        return ", ".join(str(x) for x in v) if isinstance(v, (tuple, list)) else str(v)
    return pd.DataFrame([{"field": r.field, "op": r.op, "value": show(r.value)} for r in rules],
                        columns=["field", "op", "value"])


def frame_to_rules(frame: pd.DataFrame) -> tuple[list[Rule], list[str]]:
    rules, errors = [], []
    for i, row in frame.reset_index(drop=True).iterrows():
        if not row.get("field") or not row.get("op") or pd.isna(row.get("value")):
            continue
        try:
            rules.append(Rule(row["field"], row["op"], parse_value(row["op"], row["value"])))
        except ValueError as exc:
            errors.append(f"rule {i + 1}: {exc}")
    return rules, errors


@st.cache_data(ttl=300, show_spinner=False)
def _table(stamp: tuple) -> pd.DataFrame | None:
    sv = services()
    features = sv.features.load_latest()
    if features is None:
        return None
    return build_screen_table(features, ui_data.universe_table(), sv.fundamentals.snapshots_frame())


def _stamp() -> tuple:
    sv = services()
    latest = Path(sv.settings.features_dir) / "latest.parquet"
    return (latest.stat().st_mtime if latest.exists() else 0, len(sv.fundamentals.snapshots_frame()))


def _to_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="screen")
    return buf.getvalue()


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
            st.caption("Fill the store first in **Data Manager**, or run `python -m stockanalyzer nightly --limit 500`.")
        return

    asof = table["last_date"].max()
    has_fund = "pe" in table.columns
    st.caption(f"{len(table):,} symbols · data as of {asof:%Y-%m-%d}"
               + (f" · fundamentals for {int(table['pe'].notna().sum()):,}" if has_fund else " · no fundamentals cached yet (value/quality presets stay empty)"))

    # -- universe filters --------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### Universe")
        classes = sorted(table["asset_class"].dropna().unique()) if "asset_class" in table else []
        chosen = st.multiselect("Asset class", classes, default=[c for c in classes if c == "equity"] or classes)
        min_cap = st.number_input("Min market cap ($M)", 0, 5_000_000, 0, 100)
        min_dv = st.number_input("Min avg $ volume ($M/day)", 0.0, 100_000.0, 5.0, 1.0)
        no_spacs = st.checkbox("Exclude shell companies (SPACs)", value=True)
        industries = st.multiselect("Industries (optional)", sorted(table["industry"].dropna().unique())) if "industry" in table else []

    pool = table
    if chosen and "asset_class" in pool:
        pool = pool[pool["asset_class"].isin(chosen)]
    if min_cap and "market_cap" in pool:
        pool = pool[pool["market_cap"] >= min_cap * 1e6]
    if min_dv and "dollar_volume_20" in pool:
        pool = pool[pool["dollar_volume_20"] >= min_dv * 1e6]
    if no_spacs and "industry" in pool:
        pool = pool[pool["industry"] != uni.SPAC_INDUSTRY]
    if industries:
        pool = pool[pool["industry"].isin(industries)]
    st.markdown(f"**{len(pool):,}** symbols in the investable pool")

    # -- rules -----------------------------------------------------------------------------------
    preset_name = st.selectbox("Screen", [*PRESETS, CUSTOM], key="scr_preset")
    preset = PRESETS.get(preset_name)
    if preset:
        st.caption(preset.description)
    numeric_cols = [c for c in pool.columns if c in FEATURE_INFO or pd.api.types.is_numeric_dtype(pool[c])]
    editor = st.data_editor(
        rules_to_frame(preset.rules if preset else [Rule("rsi14", "<", 40)]),
        key=f"scr_rules_{preset_name}", num_rows="dynamic", hide_index=True, width="stretch",
        column_config={"field": st.column_config.SelectboxColumn("Field", options=sorted(numeric_cols), required=True),
                       "op": st.column_config.SelectboxColumn("Operator", options=OPERATORS, required=True),
                       "value": st.column_config.TextColumn("Value", help="Number, true/false, 'low, high' for between, a, b for in")},
    )
    rules, errors = frame_to_rules(editor)
    for e in errors:
        st.error(e)

    c1, c2, c3 = st.columns([2, 1, 1])
    sort_options = ["score", *[c for c in RESULT_COLUMNS if c in pool.columns and c not in ("name", "industry", "trend_label", "score")]]
    sort_by = c1.selectbox("Sort by", sort_options)
    ascending = c2.toggle("Ascending", value=False)
    top = c3.number_input("Show top", 10, 2000, 100, 10)

    scored = score_table(pool)
    screen = Screen(preset_name, rules, sort_by=sort_by, ascending=ascending, limit=int(top))
    hits = screen.apply(scored)
    st.subheader(f"{int(screen.mask(scored).sum()):,} matches")

    view, config = feature_view(hits, [c for c in RESULT_COLUMNS if c in hits.columns])
    event = st.dataframe(view, column_config=config, width="stretch", height=560, on_select="rerun",
                         selection_mode="single-row", key="scr_table")
    picked = [hits.index[i] for i in event.selection.rows] if event and event.selection.rows else []

    a, b, c, d = st.columns([1, 1, 1, 1])
    with a:
        st.download_button("CSV", hits.to_csv().encode(), file_name="screen.csv", mime="text/csv", width="stretch")
    with b:
        st.download_button("Excel", _to_excel(hits), file_name="screen.xlsx", width="stretch",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if picked:
        sym = picked[0]
        if c.button(f"Add {sym} to Favorites", width="stretch"):
            sv.watchlists.add("Favorites", sym)
            st.toast(f"{sym} added")
        if d.button(f"Open {sym}", width="stretch"):
            st.session_state["symbol"] = sym
            from stockanalyzer.ui.nav import PAGES
            st.switch_page(PAGES["overview"])

    with st.expander("Rule funnel - how many symbols each rule removes"):
        if rules:
            show_table(Screen("funnel", rules).explain(scored), hide_index=True)
    with st.expander("How the score works"):
        st.markdown("Each component is the average **percentile rank within the investable pool** (0-100) of a few columns; "
                    "the composite weights the components that have data. Default weights: "
                    + ", ".join(f"{k} {v:.0%}" for k, v in DEFAULT_WEIGHTS.items()) + ". "
                    "Ranks are relative, so the score says *which symbols look strongest now*, not that they will keep doing so.")
