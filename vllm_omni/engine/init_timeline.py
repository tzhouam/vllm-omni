"""Initialization timeline: phase timers for multi-stage engine start-up.

Two roles write into one timeline:

* the **engine** process (``AsyncOmniEngine`` + ``StageRuntime``) records phases
  directly through :meth:`InitTimeline.phase`;
* **worker/engine-core** subprocesses append JSON lines to
  ``$VLLM_OMNI_INIT_TIMELINE_DIR/worker_<pid>.jsonl`` through :func:`worker_phase`
  / :func:`worker_mark` (they inherit the env at spawn time, so no plumbing
  through constructor arguments is needed). The engine ingests those files
  once start-up finishes (or times out) with :meth:`InitTimeline.ingest_workers`.

Enable with ``VLLM_OMNI_INIT_TIMELINE=/path/to/timeline.json`` (a JSON dump is
written there) or ``log_stats=True`` (table logged at INFO only). Disabled
timelines cost one attribute check per phase.
"""

from __future__ import annotations

import functools
import json
import math
import os
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV_OUT = "VLLM_OMNI_INIT_TIMELINE"
ENV_DIR = "VLLM_OMNI_INIT_TIMELINE_DIR"
ENV_STAGE_ID = "VLLM_OMNI_STAGE_ID"
ENV_REPLICA_ID = "VLLM_OMNI_REPLICA_ID"
SCHEMA_VERSION = 1

DEFAULT_STAGE_INIT_TIMEOUT_S = 300
DEFAULT_INIT_TIMEOUT_S = 600


@dataclass
class InitPhase:
    name: str
    start: float
    end: float
    stage_id: int | None = None
    replica_id: int = 0
    pid: int = 0
    role: str = "engine"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def dur_s(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dur_s"] = self.dur_s
        return d


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


class InitTimeline:
    """Process-local collector of :class:`InitPhase` records."""

    _current: InitTimeline | None = None
    _lock = threading.Lock()

    def __init__(self, enabled: bool, run_dir: Path | None = None, t0: float | None = None):
        self.enabled = bool(enabled)
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.t0 = t0 if t0 is not None else time.time()
        self.t_end: float | None = None
        self.phases: list[InitPhase] = []
        self._phase_lock = threading.Lock()

    # ------------------------------------------------------------ lifecycle
    @classmethod
    def start(cls, enabled: bool, run_dir: Path | str | None = None) -> InitTimeline:
        """Create the process-global timeline (idempotent per process)."""
        with cls._lock:
            run_path: Path | None = None
            if enabled:
                run_path = Path(run_dir) if run_dir else Path(tempfile.mkdtemp(prefix="omni_init_timeline_"))
                run_path.mkdir(parents=True, exist_ok=True)
                os.environ[ENV_DIR] = str(run_path)
            tl = cls(enabled=enabled, run_dir=run_path)
            cls._current = tl
            return tl

    @classmethod
    def current(cls) -> InitTimeline | None:
        return cls._current

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._current = None

    def finish(self) -> None:
        self.t_end = time.time()

    # ------------------------------------------------------------- recording
    def record(self, phase: InitPhase) -> None:
        if not self.enabled:
            return
        with self._phase_lock:
            self.phases.append(phase)

    @contextmanager
    def phase(
        self, name: str, stage_id: int | None = None, replica_id: int = 0, **extra: Any
    ) -> Iterator[dict[str, Any]]:
        """Time a block; the yielded dict can receive extra fields."""
        info: dict[str, Any] = dict(extra)
        if not self.enabled:
            yield info
            return
        start = time.time()
        try:
            yield info
        finally:
            self.record(
                InitPhase(
                    name=name,
                    start=start,
                    end=time.time(),
                    stage_id=stage_id,
                    replica_id=replica_id,
                    pid=os.getpid(),
                    role="engine",
                    extra=info,
                )
            )

    def mark(self, name: str, stage_id: int | None = None, replica_id: int = 0, **extra: Any) -> None:
        now = time.time()
        self.record(
            InitPhase(
                name=name,
                start=now,
                end=now,
                stage_id=stage_id,
                replica_id=replica_id,
                pid=os.getpid(),
                role="engine",
                extra=dict(extra),
            )
        )

    # ------------------------------------------------------------- ingestion
    def ingest_workers(self) -> int:
        """Merge worker JSONL files from ``run_dir``; returns the number of phases added."""
        if not self.enabled or self.run_dir is None or not self.run_dir.exists():
            return 0
        added = 0
        seen = {(p.pid, p.name, p.start) for p in self.phases}
        for path in sorted(self.run_dir.glob("worker_*.jsonl")):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    phase = InitPhase(
                        name=str(d["name"]),
                        start=float(d["start"]),
                        end=float(d["end"]),
                        stage_id=d.get("stage_id"),
                        replica_id=int(d.get("replica_id", 0) or 0),
                        pid=int(d.get("pid", 0) or 0),
                        role=str(d.get("role", "worker")),
                        extra=dict(d.get("extra") or {}),
                    )
                except (KeyError, ValueError, TypeError):
                    continue
                key = (phase.pid, phase.name, phase.start)
                if key in seen:
                    continue
                seen.add(key)
                self.phases.append(phase)
                added += 1
        self.phases.sort(key=lambda p: (p.start, p.end))
        return added

    # ------------------------------------------------------------- reporting
    @property
    def total_s(self) -> float:
        end = self.t_end if self.t_end is not None else time.time()
        return max(0.0, end - self.t0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "t0": self.t0,
            "t_end": self.t_end,
            "total_s": self.total_s,
            "phases": [p.to_dict() for p in sorted(self.phases, key=lambda p: (p.start, p.end))],
        }

    def dump(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)

    def stage_spans(self) -> dict[int | None, tuple[float, float]]:
        spans: dict[int | None, tuple[float, float]] = {}
        for p in self.phases:
            lo, hi = spans.get(p.stage_id, (p.start, p.end))
            spans[p.stage_id] = (min(lo, p.start), max(hi, p.end))
        return spans

    def format_table(self) -> str:
        rows = sorted(self.phases, key=lambda p: (p.start, p.end))
        header = f"{'stage':>5} {'role':<7} {'pid':>8} {'phase':<28} {'start_s':>9} {'dur_s':>9}"
        lines = [f"[init_timeline] total={self.total_s:.2f}s phases={len(rows)}", header, "-" * len(header)]
        for p in rows:
            stage = "-" if p.stage_id is None else str(p.stage_id)
            lines.append(
                f"{stage:>5} {p.role:<7} {p.pid:>8} {p.name[:28]:<28} {p.start - self.t0:>9.2f} {p.dur_s:>9.2f}"
            )
        return "\n".join(lines)

    def suggest_timeouts(self, factor: float = 1.5) -> dict[str, int]:
        """Timeouts (seconds) that would cover this run with ``factor`` headroom."""
        stage_max = 0.0
        for stage_id, (lo, hi) in self.stage_spans().items():
            if stage_id is None:
                continue
            stage_max = max(stage_max, hi - lo)
        return {
            "stage_init_timeout": max(DEFAULT_STAGE_INIT_TIMEOUT_S, int(math.ceil(stage_max * factor))),
            "init_timeout": max(DEFAULT_INIT_TIMEOUT_S, int(math.ceil(self.total_s * factor))),
        }


# ------------------------------------------------------------------ workers
def _worker_run_dir() -> Path | None:
    raw = os.environ.get(ENV_DIR)
    return Path(raw) if raw else None


def _append_worker_phase(phase: InitPhase) -> None:
    run_dir = _worker_run_dir()
    if run_dir is None:
        return
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / f"worker_{os.getpid()}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(phase.to_dict()) + "\n")
    except OSError:  # never let telemetry break init
        pass


def _record_any(phase: InitPhase) -> None:
    """Record into the in-process timeline if this is the engine process, else to JSONL."""
    tl = InitTimeline.current()
    if tl is not None and tl.enabled:
        tl.record(phase)
        return
    _append_worker_phase(phase)


def timeline_enabled() -> bool:
    tl = InitTimeline.current()
    if tl is not None:
        return tl.enabled
    return _worker_run_dir() is not None


@contextmanager
def worker_phase(
    name: str, stage_id: int | None = None, replica_id: int | None = None, **extra: Any
) -> Iterator[dict[str, Any]]:
    """Time a block in any process; no-op unless a timeline is active."""
    info: dict[str, Any] = dict(extra)
    if not timeline_enabled():
        yield info
        return
    start = time.time()
    try:
        yield info
    finally:
        _record_any(
            InitPhase(
                name=name,
                start=start,
                end=time.time(),
                stage_id=stage_id if stage_id is not None else _env_int(ENV_STAGE_ID),
                replica_id=replica_id if replica_id is not None else (_env_int(ENV_REPLICA_ID) or 0),
                pid=os.getpid(),
                role="engine" if InitTimeline.current() is not None else "worker",
                extra=info,
            )
        )


def worker_mark(name: str, **extra: Any) -> None:
    if not timeline_enabled():
        return
    now = time.time()
    _record_any(
        InitPhase(
            name=name,
            start=now,
            end=now,
            stage_id=_env_int(ENV_STAGE_ID),
            replica_id=_env_int(ENV_REPLICA_ID) or 0,
            pid=os.getpid(),
            role="engine" if InitTimeline.current() is not None else "worker",
            extra=dict(extra),
        )
    )


def timed_phase(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator form of :func:`worker_phase` for methods on the init path."""

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with worker_phase(name):
                return fn(*args, **kwargs)

        return wrapper

    return deco
