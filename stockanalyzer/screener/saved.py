"""Screens you named and kept: criteria, sort order and the universe filters, stored in SQLite."""

from __future__ import annotations

import json

from stockanalyzer.screener.rules import Rule
from stockanalyzer.storage.state import StateDB, utcnow_iso


def rules_to_spec(rules: list[Rule]) -> list[dict]:
    return [{"field": r.field, "op": r.op, "value": list(r.value) if isinstance(r.value, tuple) else r.value} for r in rules]


def spec_to_rules(spec: list[dict]) -> list[Rule]:
    """Rules back from JSON; a ``between`` pair is a tuple again. Entries that no longer make sense are skipped."""
    rules = []
    for item in spec:
        try:
            value = tuple(item["value"]) if item["op"] == "between" else item["value"]
            rules.append(Rule(item["field"], item["op"], value))
        except (KeyError, TypeError, ValueError):
            continue
    return rules


class SavedScreens:
    def __init__(self, state: StateDB):
        self.state = state

    def save(self, name: str, spec: dict) -> None:
        """Create or overwrite. ``spec`` holds ``rules`` (see :func:`rules_to_spec`) plus anything else the page wants back."""
        name = name.strip()
        if not name:
            raise ValueError("a saved screen needs a name")
        self.state.execute(
            "INSERT INTO saved_screens(name, spec, saved_at) VALUES (?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET spec=excluded.spec, saved_at=excluded.saved_at",
            (name, json.dumps(spec), utcnow_iso()),
        )

    def load(self, name: str) -> dict | None:
        row = self.state.query_one("SELECT spec FROM saved_screens WHERE name=?", (name,))
        return json.loads(row["spec"]) if row else None

    def names(self) -> list[str]:
        return [r["name"] for r in self.state.query("SELECT name FROM saved_screens ORDER BY saved_at DESC, name")]

    def delete(self, name: str) -> None:
        self.state.execute("DELETE FROM saved_screens WHERE name=?", (name,))
