"""Watchlists (favourites and sub-lists) and price/indicator alerts."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from stockanalyzer.alerts.alerts import KINDS
from stockanalyzer.alerts.watchlist import FAVORITES
from stockanalyzer.analysis.technical import indicators as ind
from stockanalyzer.ui import data as ui_data
from stockanalyzer.ui.context import services
from stockanalyzer.ui.widgets import show_table


def _snapshot(symbols: list[str]) -> pd.DataFrame:
    closes = ui_data.stored_closes(tuple(symbols), 300)
    rows = []
    for s in symbols:
        c = closes[s].dropna() if s in closes else pd.Series(dtype=float)
        if len(c) > 22:
            rows.append({"symbol": s, "last": c.iloc[-1], "1d %": (c.iloc[-1] / c.iloc[-2] - 1) * 100,
                         "1m %": (c.iloc[-1] / c.iloc[-22] - 1) * 100,
                         "3m %": (c.iloc[-1] / c.iloc[-64] - 1) * 100 if len(c) > 63 else float("nan"),
                         "RSI 14": ind.rsi(c, 14).iloc[-1],
                         "vs SMA50 %": (c.iloc[-1] / c.tail(50).mean() - 1) * 100 if len(c) >= 50 else float("nan")})
        else:
            rows.append({"symbol": s})
    return pd.DataFrame(rows).set_index("symbol")


def _lists_tab() -> None:
    sv = services()
    repo = sv.watchlists
    if FAVORITES not in repo.names():
        repo.create(FAVORITES)
    names = repo.names()
    left, right = st.columns([1, 3])
    with left:
        current = st.radio("Lists", names, key="wl_current")
        new_name = st.text_input("New list", placeholder="e.g. Dividend ideas", key="wl_new")
        if st.button("Create list", disabled=not new_name.strip()):
            repo.create(new_name)
            st.rerun()
        if current != FAVORITES and st.button(f"Delete '{current}'"):
            repo.delete(current)
            st.rerun()
    with right:
        c1, c2 = st.columns([3, 1])
        symbol = c1.text_input("Add symbol", placeholder="AAPL, PTT.BK, GC=F ...", key="wl_symbol").strip().upper()
        note = c1.text_input("Note", key="wl_note")
        if c2.button("Add", disabled=not symbol, width="stretch"):
            repo.add(current, symbol, note)
            st.rerun()
        items = repo.items(current)
        if not items:
            st.caption("This list is empty.")
            return
        symbols = [i["symbol"] for i in items]
        snap = _snapshot(symbols)
        snap["note"] = [i["note"] for i in items]
        missing = [s for s in symbols if pd.isna(snap.loc[s].get("last", float("nan")))]
        show_table(snap.style.format({c: "{:,.2f}" for c in snap.columns if c != "note"}, na_rep="-"))
        if missing:
            st.caption(f"No stored prices yet for: {', '.join(missing)}. Update them in **Data Manager** "
                       "or open them once on **Stock Overview**.")
        gone = st.multiselect("Remove", symbols, key="wl_remove")
        if st.button("Remove selected", disabled=not gone):
            for s in gone:
                repo.remove(current, s)
            st.rerun()


def _alerts_tab() -> None:
    sv = services()
    mgr = sv.alerts
    st.markdown("**New alert**")
    c1, c2 = st.columns([1, 2])
    symbol = c1.text_input("Symbol", key="al_symbol").strip().upper()
    kind = c2.selectbox("Condition", list(KINDS), format_func=lambda k: KINDS[k].label, key="al_kind")
    params = {}
    spec = KINDS[kind]
    if spec.defaults:
        cols = st.columns(len(spec.defaults))
        for col, (name, default) in zip(cols, spec.defaults.items()):
            params[name] = col.number_input(name, value=float(default) if not isinstance(default, int) else int(default),
                                            key=f"al_{kind}_{name}")
    cooldown = st.number_input("Mute after firing (hours)", 1.0, 720.0, 24.0, key="al_cd")
    if st.button("Create alert", disabled=not symbol, type="primary"):
        try:
            mgr.add(symbol, kind, params, cooldown)
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    alerts = mgr.list()
    st.markdown("**Alerts**")
    if not alerts:
        st.caption("No alerts yet.")
    for a in alerts:
        c1, c2, c3 = st.columns([6, 1, 1])
        c1.write(("🟢 " if a.active else "⚪ ") + a.describe() + (f" · last fired {a.last_triggered_at} UTC" if a.last_triggered_at else ""))
        if c2.button("Pause" if a.active else "Resume", key=f"al_t{a.id}"):
            mgr.set_active(a.id, not a.active)
            st.rerun()
        if c3.button("Delete", key=f"al_d{a.id}"):
            mgr.remove(a.id)
            st.rerun()

    if st.button("Check alerts now (uses stored prices)"):
        fired = mgr.evaluate(sv.md.store.read)
        for e in fired:
            st.success(f"{e.symbol}: {e.message}")
        if not fired:
            st.info("Nothing triggered (or alerts are muted/stale - update prices first).")
    events = mgr.events(25)
    if events:
        st.markdown("**History**")
        show_table(pd.DataFrame(events)[["triggered_at", "symbol", "kind", "message"]], hide_index=True)
    st.caption("Alerts are evaluated by the nightly job and by the button above. They are logged here; "
               "this app does not send emails or push notifications.")


def render() -> None:
    st.title("⭐ Watchlists & Alerts")
    tab_lists, tab_alerts = st.tabs(["Watchlists", "Alerts"])
    with tab_lists:
        _lists_tab()
    with tab_alerts:
        _alerts_tab()
