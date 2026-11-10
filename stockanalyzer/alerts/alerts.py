"""Price/indicator alerts evaluated against the local store (no network).

Alerts are checked after each data update (nightly job or "Check now" in the UI). Every kind is a condition on the
latest bar, and an alert fires when that condition *becomes* true - a price crossing its level, an RSI entering its zone,
a cross happening - not on every check while it stays true. It re-arms by itself once the condition ends.

Each alert remembers the last bar it examined, so a gap in updates (a weekend, a skipped night) is replayed bar by bar
instead of missing a cross that happened in between. ``cooldown_hours`` is an extra guard against a price flapping
around its level: after a fire the alert stays quiet that long even if it crosses again. A fired alert is written to
``alert_events``. Delivery is intentionally just the event log + the UI; add a notifier by reading ``AlertManager.events()``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable

import numpy as np
import pandas as pd

from stockanalyzer.analysis.technical import indicators as ind
from stockanalyzer.analysis.technical import signals
from stockanalyzer.storage.state import StateDB, to_iso

Evaluation = tuple[bool, float | None, str]
MAX_CATCH_UP = 60  # bars replayed after a gap; a longer gap is judged by the most recent ones


def _price_above(df, p) -> Evaluation:
    last = float(df["close"].iloc[-1])
    return last >= p["level"], last, f"close {last:,.2f} is at/above {p['level']:,.2f}"


def _price_below(df, p) -> Evaluation:
    last = float(df["close"].iloc[-1])
    return last <= p["level"], last, f"close {last:,.2f} is at/below {p['level']:,.2f}"


def _pct_move(df, p) -> Evaluation:
    days = int(p["days"])
    if len(df) <= days:
        return False, None, ""
    move = float(df["close"].iloc[-1] / df["close"].iloc[-1 - days] - 1) * 100
    return abs(move) >= p["threshold_pct"], move, f"{move:+.1f}% over {days} bar(s) (threshold {p['threshold_pct']}%)"


def _rsi(direction: str) -> Callable:
    def check(df, p) -> Evaluation:
        value = float(ind.rsi(df["close"], int(p["n"])).iloc[-1])
        hit = value >= p["level"] if direction == "above" else value <= p["level"]
        return bool(hit) and value == value, value, f"RSI({int(p['n'])}) = {value:.1f} is {direction} {p['level']}"
    return check


def _cross(direction: int, label: str) -> Callable:
    def check(df, p) -> Evaluation:
        adj = df["adj_close"] if "adj_close" in df.columns else df["close"]
        event = signals.crossover(ind.sma(adj, int(p["fast"])), ind.sma(adj, int(p["slow"]))).iloc[-1]
        return event == direction, None, f"{label} (SMA {int(p['fast'])}/{int(p['slow'])})"
    return check


def _new_extreme(high: bool) -> Callable:
    def check(df, p) -> Evaluation:
        prior = (df["high"] if high else df["low"]).iloc[-253:-1]
        if prior.empty:
            return False, None, ""
        last = float(df["close"].iloc[-1])
        hit = last >= prior.max() if high else last <= prior.min()
        return bool(hit), last, f"close {last:,.2f} is a new 52-week {'high' if high else 'low'}"
    return check


def _volume_spike(df, p) -> Evaluation:
    base = df["volume"].iloc[-21:-1].mean()
    if not base or base != base:
        return False, None, ""
    ratio = float(df["volume"].iloc[-1] / base)
    return ratio >= p["ratio"], ratio, f"volume is {ratio:.1f}x its 20-day average"


@dataclass(frozen=True)
class AlertKind:
    name: str
    label: str
    defaults: dict
    check: Callable[[pd.DataFrame, dict], Evaluation]


KINDS: dict[str, AlertKind] = {k.name: k for k in [
    AlertKind("price_above", "Price at/above level", {"level": 0.0}, _price_above),
    AlertKind("price_below", "Price at/below level", {"level": 0.0}, _price_below),
    AlertKind("pct_move", "Absolute % move over N bars", {"threshold_pct": 5.0, "days": 1}, _pct_move),
    AlertKind("rsi_above", "RSI at/above level", {"level": 70.0, "n": 14}, _rsi("above")),
    AlertKind("rsi_below", "RSI at/below level", {"level": 30.0, "n": 14}, _rsi("below")),
    AlertKind("golden_cross", "SMA fast crosses above slow", {"fast": 50, "slow": 200}, _cross(1, "golden cross")),
    AlertKind("death_cross", "SMA fast crosses below slow", {"fast": 50, "slow": 200}, _cross(-1, "death cross")),
    AlertKind("new_52w_high", "New 52-week high", {}, _new_extreme(True)),
    AlertKind("new_52w_low", "New 52-week low", {}, _new_extreme(False)),
    AlertKind("volume_spike", "Volume spike vs 20-day average", {"ratio": 2.0}, _volume_spike),
]}


@dataclass
class Alert:
    id: int
    symbol: str
    kind: str
    params: dict
    active: bool
    cooldown_hours: float
    created_at: str
    last_triggered_at: str | None = None
    last_bar: str | None = None  # timestamp of the newest bar already examined
    in_zone: bool | None = None  # was the condition true on that bar (None = never examined)

    @property
    def label(self) -> str:
        return KINDS[self.kind].label

    def describe(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.symbol}: {self.label}" + (f" ({args})" if args else "")


@dataclass
class AlertEvent:
    alert_id: int
    symbol: str
    kind: str
    triggered_at: str
    message: str
    value: float | None = None


def _row_to_alert(r: dict) -> Alert:
    return Alert(r["id"], r["symbol"], r["kind"], json.loads(r["params"]), bool(r["active"]),
                 r["cooldown_hours"], r["created_at"], r["last_triggered_at"], r["last_bar"],
                 None if r["in_zone"] is None else bool(r["in_zone"]))


@dataclass
class AlertManager:
    state: StateDB
    now: Callable[[], datetime] = field(default=lambda: datetime.now(timezone.utc))

    def add(self, symbol: str, kind: str, params: dict | None = None, cooldown_hours: float = 24.0) -> Alert:
        if kind not in KINDS:
            raise ValueError(f"unknown alert kind {kind!r}; choose from {sorted(KINDS)}")
        spec = KINDS[kind]
        extra = set(params or {}) - set(spec.defaults)
        if extra:
            raise ValueError(f"{kind} does not take {sorted(extra)}; valid: {sorted(spec.defaults)}")
        merged = {**spec.defaults, **(params or {})}
        if kind.startswith("price_") and merged["level"] <= 0:
            raise ValueError("price alerts need a positive 'level'")
        alert_id = self.state.execute(
            "INSERT INTO alerts(symbol, kind, params, active, cooldown_hours, created_at) VALUES (?,?,?,?,?,?)",
            (symbol.strip().upper(), kind, json.dumps(merged), 1, cooldown_hours, to_iso(self.now())),
        )
        return self.get(alert_id)

    def get(self, alert_id: int) -> Alert:
        row = self.state.query_one("SELECT * FROM alerts WHERE id=?", (alert_id,))
        if row is None:
            raise KeyError(alert_id)
        return _row_to_alert(row)

    def list(self, *, active_only: bool = False, symbol: str | None = None) -> list[Alert]:
        sql, args = "SELECT * FROM alerts WHERE 1=1", []
        if active_only:
            sql += " AND active=1"
        if symbol:
            sql += " AND symbol=?"
            args.append(symbol.strip().upper())
        return [_row_to_alert(r) for r in self.state.query(sql + " ORDER BY symbol, id", args)]

    def remove(self, alert_id: int) -> None:
        self.state.execute("DELETE FROM alerts WHERE id=?", (alert_id,))

    def set_active(self, alert_id: int, active: bool) -> None:
        # a resumed alert starts afresh: it judges today's bar, it does not replay what happened while it was paused
        self.state.execute("UPDATE alerts SET active=?, last_bar=NULL, in_zone=NULL WHERE id=?", (int(active), alert_id))

    def _muted(self, alert: Alert) -> bool:
        if not alert.last_triggered_at:
            return False
        last = datetime.fromisoformat(alert.last_triggered_at).replace(tzinfo=timezone.utc)
        return self.now() - last < timedelta(hours=alert.cooldown_hours)

    def evaluate(self, load_prices: Callable[[str], pd.DataFrame]) -> list[AlertEvent]:
        """Check every active alert against the bars it has not seen; returns (and records) the ones that fired."""
        frames: dict[str, pd.DataFrame] = {}
        fired: list[AlertEvent] = []
        for alert in self.list(active_only=True):
            if alert.symbol not in frames:
                frames[alert.symbol] = load_prices(alert.symbol)
            df = frames[alert.symbol]
            if df is None or len(df) < 2:
                continue
            try:
                scan = self._scan(alert, df)
            except Exception:  # a broken alert must not block the others
                continue
            if scan is None:  # no new bar since the last check
                continue
            crossings, in_zone = scan
            self.state.execute("UPDATE alerts SET last_bar=?, in_zone=? WHERE id=?",
                               (df.index[-1].isoformat(), int(in_zone), alert.id))
            if not crossings or self._muted(alert):
                continue
            ts, value, message = crossings[-1]
            if ts != df.index[-1]:
                message = f"{ts:%Y-%m-%d}: {message}"
            at = to_iso(self.now())
            self.state.execute(
                "INSERT INTO alert_events(alert_id, symbol, triggered_at, message, value) VALUES (?,?,?,?,?)",
                (alert.id, alert.symbol, at, message, value),
            )
            self.state.execute("UPDATE alerts SET last_triggered_at=? WHERE id=?", (at, alert.id))
            fired.append(AlertEvent(alert.id, alert.symbol, alert.kind, at, message, value))
        return fired

    @staticmethod
    def _scan(alert: Alert, df: pd.DataFrame) -> tuple[list[tuple[pd.Timestamp, float | None, str]], bool] | None:
        """Replay the unseen bars: the bars where the condition turned true, and whether it holds on the last one.

        The first look at an alert judges only the latest bar. ``None`` when there is nothing new to look at.
        """
        if alert.last_bar is None:
            todo = [len(df) - 1]
        else:
            todo = [int(i) for i in np.flatnonzero(df.index > pd.Timestamp(alert.last_bar))][-MAX_CATCH_UP:]
            if not todo:
                return None
        check = KINDS[alert.kind].check
        holding, crossings = bool(alert.in_zone), []
        for i in todo:
            hit, value, message = check(df.iloc[: i + 1], alert.params)
            hit = bool(hit)
            if hit and not holding:
                crossings.append((df.index[i], value, message))
            holding = hit
        return crossings, holding

    def events(self, limit: int = 100) -> list[dict]:
        return self.state.query(
            "SELECT e.id, e.symbol, e.triggered_at, e.message, e.value, a.kind FROM alert_events e "
            "LEFT JOIN alerts a ON a.id=e.alert_id ORDER BY e.id DESC LIMIT ?", (limit,))
