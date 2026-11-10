"""Columnar price store: one Parquet file per symbol, queried across symbols with DuckDB.

Layout: ``<root>/<interval>/<url-quoted symbol>.parquet`` with columns
``ts, symbol, open, high, low, close, adj_close, volume, dividends, splits``.

Per-symbol files make incremental updates cheap (rewrite only that symbol, atomically)
while DuckDB reads many files at once for screening and portfolio work.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Iterable
from urllib.parse import quote, unquote

import duckdb
import pandas as pd

from stockanalyzer.data.providers.base import CANONICAL_COLUMNS, empty_ohlcv, normalize_ohlcv


# Parquet files per DuckDB query; keeps open descriptors well under a 1024 soft limit.
_FILE_BATCH = 200
# Streamlit reruns (and background jobs) hit the store from several threads at once; without
# this, N concurrent scans would each hold _FILE_BATCH descriptors and blow the limit again.
# DuckDB already parallelises inside a query, so serialising queries costs little.
_SCAN_LOCK = threading.Lock()


def _empty_long(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"symbol": pd.Series(dtype="object"), "ts": pd.Series(dtype="datetime64[ns]"),
         **{c: pd.Series(dtype="float64") for c in columns}}
    )


class PriceStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    # -- paths ------------------------------------------------------------------------------
    def _dir(self, interval: str) -> Path:
        return self.root / interval

    def path_for(self, symbol: str, interval: str = "1d") -> Path:
        return self._dir(interval) / (quote(symbol.strip().upper(), safe="-._") + ".parquet")

    def exists(self, symbol: str, interval: str = "1d") -> bool:
        return self.path_for(symbol, interval).exists()

    def symbols(self, interval: str = "1d") -> list[str]:
        d = self._dir(interval)
        return sorted(unquote(p.stem) for p in d.glob("*.parquet")) if d.exists() else []

    def _files(self, symbols: Iterable[str] | None, interval: str) -> list[str]:
        if symbols is None:
            d = self._dir(interval)
            return sorted(str(p) for p in d.glob("*.parquet")) if d.exists() else []
        return [str(p) for s in symbols if (p := self.path_for(s, interval)).exists()]

    # -- single symbol ----------------------------------------------------------------------
    def read(
        self, symbol: str, interval: str = "1d", *, start=None, end=None
    ) -> pd.DataFrame:
        path = self.path_for(symbol, interval)
        if not path.exists():
            return empty_ohlcv()
        df = pd.read_parquet(path, columns=["ts", *CANONICAL_COLUMNS])
        df["ts"] = df["ts"].astype("datetime64[ns]")
        df = df.set_index("ts")
        if start is not None or end is not None:
            df = df.loc[start:end]
        return df

    def write(self, symbol: str, df: pd.DataFrame, interval: str = "1d", *, replace: bool = False) -> int:
        """Merge ``df`` into the symbol's file (new rows win on equal timestamps).

        Returns the number of *new* timestamps added.
        """
        symbol = symbol.strip().upper()
        new = normalize_ohlcv(df, interval=interval)
        if new.empty:
            return 0
        old = empty_ohlcv() if replace else self.read(symbol, interval)
        if old.empty:
            merged = new
        else:
            merged = pd.concat([old.loc[~old.index.isin(new.index)], new]).sort_index()
        added = len(merged) - len(old)

        path = self.path_for(symbol, interval)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        merged.reset_index().assign(symbol=symbol).to_parquet(
            tmp, engine="pyarrow", compression="zstd", index=False
        )
        os.replace(tmp, path)
        return added

    def delete(self, symbol: str, interval: str = "1d") -> bool:
        path = self.path_for(symbol, interval)
        if path.exists():
            path.unlink()
            return True
        return False

    # -- many symbols -----------------------------------------------------------------------
    @staticmethod
    def _query(sql: str, params: list) -> pd.DataFrame:
        con = duckdb.connect()
        try:
            return con.execute(sql, params).df()
        finally:
            con.close()

    def _query_files(self, sql: str, files: list[str], params: list | None = None) -> list[pd.DataFrame]:
        """Run ``sql`` (whose first ``?`` is the file list) over ``files`` in batches.

        DuckDB keeps one file descriptor open per Parquet file in a ``read_parquet([...])``
        list, so a whole-market store (thousands of files) exceeds the usual 1024 soft limit
        ("Too many open files"). Each file holds exactly one symbol, so per-batch results
        can simply be concatenated / re-aggregated by the caller.
        """
        extra = params or []
        out = []
        for i in range(0, len(files), _FILE_BATCH):
            with _SCAN_LOCK:
                out.append(self._query(sql, [files[i : i + _FILE_BATCH], *extra]))
        return out

    def read_many(
        self,
        symbols: Iterable[str] | None = None,
        interval: str = "1d",
        *,
        start=None,
        end=None,
        last_n: int | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Long frame ``symbol, ts, <columns>`` sorted by symbol then time.

        ``last_n`` keeps only the most recent N bars of *each* symbol.
        """
        cols = columns or CANONICAL_COLUMNS
        files = self._files(symbols, interval)
        if not files:
            return _empty_long(cols)

        sql = f"SELECT symbol, ts, {', '.join(cols)} FROM read_parquet(?, union_by_name=true)"
        params: list = []
        clauses = []
        if start is not None:
            clauses.append("ts >= ?")
            params.append(pd.Timestamp(start).to_pydatetime())
        if end is not None:
            clauses.append("ts <= ?")
            params.append(pd.Timestamp(end).to_pydatetime())
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        if last_n is not None:
            sql += " QUALIFY row_number() OVER (PARTITION BY symbol ORDER BY ts DESC) <= ?"
            params.append(int(last_n))
        sql += " ORDER BY symbol, ts"

        frames = self._query_files(sql, files, params)
        df = frames[0] if len(frames) == 1 else (
            pd.concat(frames, ignore_index=True).sort_values(["symbol", "ts"], kind="stable", ignore_index=True)
        )
        df["ts"] = df["ts"].astype("datetime64[ns]")
        return df

    def last_timestamps(
        self, symbols: Iterable[str] | None = None, interval: str = "1d"
    ) -> dict[str, pd.Timestamp]:
        files = self._files(symbols, interval)
        if not files:
            return {}
        frames = self._query_files(
            "SELECT symbol, max(ts) AS last_ts FROM read_parquet(?, union_by_name=true) GROUP BY symbol",
            files,
        )
        return {r.symbol: pd.Timestamp(r.last_ts) for df in frames for r in df.itertuples()}

    def stats(self, interval: str = "1d", *, deep: bool = False) -> dict:
        files = self._files(None, interval)
        info: dict = {
            "symbols": len(files),
            "size_mb": round(sum(os.path.getsize(f) for f in files) / 1e6, 2),
        }
        if deep and files:
            parts = pd.concat(
                self._query_files(
                    "SELECT count(*) AS rows, min(ts) AS first_ts, max(ts) AS last_ts "
                    "FROM read_parquet(?, union_by_name=true)",
                    files,
                ),
                ignore_index=True,
            )
            info.update(rows=int(parts["rows"].sum()), first_ts=pd.Timestamp(parts["first_ts"].min()),
                        last_ts=pd.Timestamp(parts["last_ts"].max()))
        return info
