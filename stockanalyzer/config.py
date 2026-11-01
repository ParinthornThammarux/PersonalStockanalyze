"""Runtime settings: where data lives and how logging is set up.

The data directory is resolved in this order:

1. ``$STOCKANALYZER_HOME``
2. ``<repo>/data`` when running from a source checkout
3. ``~/.stockanalyzer``

so cron jobs, the CLI and Streamlit all see the same store regardless of the
current working directory.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

ENV_HOME = "STOCKANALYZER_HOME"


def default_home() -> Path:
    env = os.environ.get(ENV_HOME)
    if env:
        return Path(env).expanduser()
    repo = Path(__file__).resolve().parents[1]
    if (repo / "pyproject.toml").exists():
        return repo / "data"
    return Path.home() / ".stockanalyzer"


@dataclass(frozen=True)
class Settings:
    home: Path

    @property
    def prices_dir(self) -> Path:
        return self.home / "prices"

    @property
    def features_dir(self) -> Path:
        return self.home / "features"

    @property
    def reports_dir(self) -> Path:
        return self.home / "reports"

    @property
    def state_db(self) -> Path:
        return self.home / "state.db"

    def ensure_dirs(self) -> "Settings":
        for p in (self.home, self.prices_dir, self.features_dir, self.reports_dir):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_settings(home: str | Path | None = None) -> Settings:
    return Settings(home=Path(home).expanduser() if home else default_home())


def setup_logging(level: int | str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # yfinance logs every failed ticker at ERROR level; the updater reports these itself.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
