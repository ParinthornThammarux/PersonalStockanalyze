"""Named watchlists ("Favorites" and any number of sub-lists), stored in SQLite."""

from __future__ import annotations

from stockanalyzer.storage.state import StateDB, utcnow_iso

FAVORITES = "Favorites"


class WatchlistRepo:
    def __init__(self, state: StateDB):
        self.state = state

    def create(self, name: str) -> int:
        name = name.strip()
        if not name:
            raise ValueError("watchlist name cannot be empty")
        row = self.state.query_one("SELECT id FROM watchlists WHERE name=?", (name,))
        if row:
            return row["id"]
        return self.state.execute("INSERT INTO watchlists(name, created_at) VALUES (?,?)", (name, utcnow_iso()))

    def delete(self, name: str) -> None:
        self.state.execute("DELETE FROM watchlists WHERE name=?", (name,))

    def rename(self, old: str, new: str) -> None:
        if self.state.query_one("SELECT id FROM watchlists WHERE name=?", (new.strip(),)):
            raise ValueError(f"a watchlist named {new!r} already exists")
        self.state.execute("UPDATE watchlists SET name=? WHERE name=?", (new.strip(), old))

    def names(self) -> list[str]:
        return [r["name"] for r in self.state.query("SELECT name FROM watchlists ORDER BY name")]

    def summary(self) -> list[dict]:
        return self.state.query(
            "SELECT w.name, COUNT(i.symbol) AS symbols FROM watchlists w "
            "LEFT JOIN watchlist_items i ON i.watchlist_id = w.id GROUP BY w.id ORDER BY w.name"
        )

    def add(self, name: str, symbol: str, note: str = "") -> None:
        wid = self.create(name)
        self.state.execute(
            "INSERT INTO watchlist_items(watchlist_id, symbol, note, added_at) VALUES (?,?,?,?) "
            "ON CONFLICT(watchlist_id, symbol) DO UPDATE SET note=excluded.note",
            (wid, symbol.strip().upper(), note, utcnow_iso()),
        )

    def remove(self, name: str, symbol: str) -> None:
        self.state.execute(
            "DELETE FROM watchlist_items WHERE symbol=? AND watchlist_id=(SELECT id FROM watchlists WHERE name=?)",
            (symbol.strip().upper(), name),
        )

    def items(self, name: str) -> list[dict]:
        return self.state.query(
            "SELECT i.symbol, i.note, i.added_at FROM watchlist_items i "
            "JOIN watchlists w ON w.id=i.watchlist_id WHERE w.name=? ORDER BY i.symbol",
            (name,),
        )

    def symbols(self, name: str | None = None) -> list[str]:
        """Symbols of one list, or the union of every list when ``name`` is None."""
        if name is None:
            rows = self.state.query("SELECT DISTINCT symbol FROM watchlist_items ORDER BY symbol")
        else:
            rows = self.items(name)
        return [r["symbol"] for r in rows]

    def lists_containing(self, symbol: str) -> list[str]:
        rows = self.state.query(
            "SELECT w.name FROM watchlists w JOIN watchlist_items i ON i.watchlist_id=w.id "
            "WHERE i.symbol=? ORDER BY w.name",
            (symbol.strip().upper(),),
        )
        return [r["name"] for r in rows]
