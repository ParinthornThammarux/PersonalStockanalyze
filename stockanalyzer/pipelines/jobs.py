"""Run the data pipeline as a detached background process, so a 10-minute whole-market load survives a page refresh.

The child process (``python -m stockanalyzer nightly ... --job-file F``) keeps ``F`` up to date through ``JobFile``;
the app only ever reads it, which also makes the progress visible to a second browser tab or after a restart.
One job runs at a time per data home.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from stockanalyzer.config import ENV_HOME, Settings
from stockanalyzer.storage.state import to_iso

if TYPE_CHECKING:  # nightly imports a lot, and the CLI imports this module before it needs any of it
    from stockanalyzer.pipelines.nightly import PipelineOptions

JOB_FILE = "job.json"
LOG_FILE = "job.log"
WRITE_EVERY = 1.0  # seconds between progress writes: the updater reports after every batch of 100 symbols


def nightly_args(opts: "PipelineOptions", *, fundamentals: int = 0, events: bool = True) -> list[str]:
    """The ``nightly`` command line that does what ``opts`` says (watchlists are always included by the command)."""
    args = ["nightly", "--universe", opts.universe, "--thai", opts.thai]
    if opts.limit:
        args += ["--limit", str(opts.limit)]
    if opts.min_market_cap:
        args += ["--min-cap", str(opts.min_market_cap)]
    if opts.symbols:
        args += ["--symbols", *opts.symbols]
    for flag, wanted in (("--no-cross", opts.include_cross_assets), ("--no-sectors", opts.include_sectors),
                         ("--no-snapshot", opts.market_snapshot), ("--no-events", events)):
        if not wanted:
            args.append(flag)
    if opts.include_macro:
        args.append("--macro")
    if opts.start:
        args += ["--start", str(opts.start)]
    if opts.force:
        args.append("--force")
    if opts.rate_limit_waits:
        args += ["--patience", str(opts.rate_limit_waits), "--patience-minutes", f"{opts.rate_limit_wait_minutes:g}"]
    return args + ["--benchmark", opts.benchmark, "--fundamentals", str(fundamentals)]


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, belongs to someone else
        return True
    try:  # a zombie answers kill(0) too
        return Path(f"/proc/{pid}/stat").read_text().split(") ")[-1][0] != "Z"
    except OSError:
        return True


@dataclass
class JobState:
    pid: int | None = None
    title: str = ""
    started_at: str = ""
    step: str = ""
    done: int = 0
    total: int = 0
    message: str = ""
    finished_at: str | None = None
    ok: bool | None = None
    summary: dict = field(default_factory=dict)
    updated_at: str = ""

    @property
    def running(self) -> bool:
        return self.finished_at is None and _alive(self.pid)

    @property
    def interrupted(self) -> bool:
        """It never wrote a finish: the process was stopped or died."""
        return self.finished_at is None and not _alive(self.pid)

    @property
    def fraction(self) -> float:
        return min(self.done / self.total, 1.0) if self.total else 0.0


class JobFile:
    """Written by the job itself; reading it needs no process handle."""

    def __init__(self, path: str | Path, title: str = ""):
        self.path = Path(path)
        self.state = JobState(pid=os.getpid(), title=title, started_at=to_iso(datetime.now(timezone.utc)))
        self._last = 0.0
        self.write()

    def write(self) -> None:
        self.state.updated_at = to_iso(datetime.now(timezone.utc))
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self.state)), "utf-8")
        tmp.replace(self.path)  # readers never see a half-written file
        self._last = time.monotonic()

    def update(self, step: str, done: int, total: int, message: str = "") -> None:
        """Progress callback for ``run_pipeline``. A new step is written at once, progress within one at most once a second."""
        new_step = step != self.state.step
        if not total and not new_step:  # a message only (e.g. "waiting for Yahoo"): keep the count
            done, total = self.state.done, self.state.total
        self.state.step, self.state.done, self.state.total, self.state.message = step, done, total, message
        if new_step or time.monotonic() - self._last >= WRITE_EVERY:
            self.write()

    def finish(self, ok: bool, summary: dict) -> None:
        self.state.finished_at, self.state.ok, self.state.summary = to_iso(datetime.now(timezone.utc)), ok, summary
        self.write()


def read_job(path: str | Path) -> JobState | None:
    try:
        raw = json.loads(Path(path).read_text("utf-8"))
        return JobState(**{k: v for k, v in raw.items() if k in JobState.__dataclass_fields__})
    except (OSError, ValueError, TypeError):
        return None


class Jobs:
    """Start, watch and stop the one background job of a data home."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def path(self) -> Path:
        return self.settings.home / JOB_FILE

    @property
    def log_path(self) -> Path:
        return self.settings.home / LOG_FILE

    def status(self) -> JobState | None:
        return read_job(self.path)

    def start(self, args: list[str], title: str = "") -> JobState:
        """Run ``python -m stockanalyzer <args> --job-file ...`` detached; refuses while another job runs."""
        current = self.status()
        if current is not None and current.running:
            raise RuntimeError(f"a job is already running ({current.title or 'pipeline'}, started {current.started_at} UTC)")
        self.settings.home.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, ENV_HOME: str(self.settings.home), "PYTHONUNBUFFERED": "1"}
        with self.log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [sys.executable, "-m", "stockanalyzer", *args, "--job-file", str(self.path), "--job-title", title],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        state = JobState(pid=proc.pid, title=title, started_at=to_iso(datetime.now(timezone.utc)), message="starting")
        written = self.status()
        if written is None or written.pid != proc.pid:  # the child writes its own once it is up; never overwrite that
            self.path.write_text(json.dumps(asdict(state)), "utf-8")
        return state

    def stop(self) -> bool:
        """Ask the job (and anything it started) to stop; what it already stored stays, and the next run resumes."""
        state = self.status()
        if state is None or not state.running:
            return False
        try:
            os.killpg(os.getpgid(state.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return False
        return True

    def log_tail(self, lines: int = 12) -> str:
        try:
            return "\n".join(self.log_path.read_text("utf-8", errors="replace").splitlines()[-lines:])
        except OSError:
            return ""
