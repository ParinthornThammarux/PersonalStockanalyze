"""Persisted feature snapshots: ``latest.parquet`` plus one dated file per run."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class FeatureStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def save(self, df: pd.DataFrame, asof: str | pd.Timestamp | None = None) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        if asof is None:
            asof = df["last_date"].max() if "last_date" in df.columns and len(df) else pd.Timestamp.today()
        day = pd.Timestamp(asof).strftime("%Y-%m-%d")
        dated = self.root / f"asof={day}.parquet"
        df.to_parquet(dated, compression="zstd")
        df.to_parquet(self.root / "latest.parquet", compression="zstd")
        return dated

    def load_latest(self) -> pd.DataFrame | None:
        path = self.root / "latest.parquet"
        return pd.read_parquet(path) if path.exists() else None

    def dates(self) -> list[str]:
        return sorted(p.stem.removeprefix("asof=") for p in self.root.glob("asof=*.parquet"))

    def load(self, asof: str) -> pd.DataFrame:
        return pd.read_parquet(self.root / f"asof={asof}.parquet")
