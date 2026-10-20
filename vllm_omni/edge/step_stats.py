# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Steady-state per-step overhead attribution (edge scheduling, WP8).

Process-local counters that answer "where does a codec frame's 80 ms budget go
outside the model forward?": scheduler ``schedule()`` / ``update_from_output()``
time, the whole engine-core step, the talker -> code2wav chunk hop (enqueue ->
connector put, put -> consumer get, get duration) and the orchestrator dispatch
loop. Enabled by ``VLLM_OMNI_STEP_STATS_DIR=<dir>``; every process dumps
``<dir>/<role>_<pid>.json`` periodically and at shutdown. ``merge_dir`` folds
the files into one attribution table (``benchmarks/tts/stream_latency_bench.py
--step-stats``).

Zero cost when disabled: ``StepStats.get().enabled`` is False and every hook
returns immediately.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ENV_DIR = "VLLM_OMNI_STEP_STATS_DIR"
ENV_SYNC = "VLLM_OMNI_STEP_STATS_SYNC"  # "1": synchronize the accelerator around model-level timers
ENV_STAGE_ID = "VLLM_OMNI_STAGE_ID"
FRAME_BUDGET_MS = 80.0  # one Qwen3-TTS codec frame at 12.5 Hz
_MAX_SAMPLES = 20000
_FLUSH_EVERY = 500


class _Series:
    __slots__ = ("count", "total", "max", "samples")

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.max = 0.0
        self.samples: list[float] = []

    def add(self, v: float) -> None:
        self.count += 1
        self.total += v
        if v > self.max:
            self.max = v
        if len(self.samples) < _MAX_SAMPLES:
            self.samples.append(v)
        else:  # keep a uniform-ish reservoir: overwrite a pseudo-random slot
            self.samples[(self.count * 7919) % _MAX_SAMPLES] = v

    def summary(self) -> dict[str, float | int]:
        s = sorted(self.samples)
        n = len(s)

        def pct(p: float) -> float:
            if not n:
                return 0.0
            return s[min(n - 1, int(round((n - 1) * p)))]

        return {
            "count": self.count,
            "sum_ms": self.total,
            "mean_ms": (self.total / self.count) if self.count else 0.0,
            "p50_ms": pct(0.5),
            "p95_ms": pct(0.95),
            "max_ms": self.max,
        }


class StepStats:
    """Process-local counter registry (thread-safe adds)."""

    _instance: StepStats | None = None
    _instance_lock = threading.Lock()

    def __init__(self, out_dir: str | None, role: str | None = None) -> None:
        self.out_dir = out_dir
        self.enabled = bool(out_dir)
        self.sync = os.environ.get(ENV_SYNC, "0").strip().lower() in ("1", "true", "yes", "on")
        self.role = role or _default_role()
        self.pid = os.getpid()
        self._series: dict[str, _Series] = {}
        self._lock = threading.Lock()
        self._n_since_flush = 0
        self.started_at = time.time()
        if self.enabled:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            atexit.register(self.dump)

    # ------------------------------------------------------------ singleton
    @classmethod
    def get(cls) -> StepStats:
        inst = cls._instance
        if inst is None:
            with cls._instance_lock:
                inst = cls._instance
                if inst is None:
                    inst = cls(os.environ.get(ENV_DIR) or None)
                    cls._instance = inst
        return inst

    @classmethod
    def reset(cls, out_dir: str | None = None, role: str | None = None) -> StepStats:
        """Replace the singleton (tests / explicit enablement in-process)."""
        with cls._instance_lock:
            cls._instance = cls(out_dir, role)
            return cls._instance

    # ------------------------------------------------------------ recording
    def add(self, name: str, value_ms: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            s = self._series.get(name)
            if s is None:
                s = self._series[name] = _Series()
            s.add(float(value_ms))
            self._n_since_flush += 1
            flush = self._n_since_flush >= _FLUSH_EVERY
            if flush:
                self._n_since_flush = 0
        if flush:
            self.dump()

    @contextmanager
    def timed(self, name: str, sync_device: Any = None) -> Iterator[None]:
        """Time a block; with ``sync_device`` (a torch device) and ``VLLM_OMNI_STEP_STATS_SYNC=1``
        the accelerator is synchronized before and after so async kernels are included."""
        if not self.enabled:
            yield
            return
        sync = sync_device is not None and self.sync and getattr(sync_device, "type", "cpu") == "cuda"
        if sync:
            import torch

            torch.accelerator.synchronize(sync_device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if sync:
                torch.accelerator.synchronize(sync_device)
            self.add(name, (time.perf_counter() - t0) * 1000.0)

    def wrap(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Return ``fn`` timed under ``name`` (identity when disabled)."""
        if not self.enabled:
            return fn

        def timed_fn(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.add(name, (time.perf_counter() - t0) * 1000.0)

        timed_fn.__name__ = getattr(fn, "__name__", name)
        timed_fn.__wrapped__ = fn  # type: ignore[attr-defined]
        return timed_fn

    # ------------------------------------------------------------ output
    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            counters = {k: v.summary() for k, v in self._series.items()}
        return {
            "role": self.role,
            "pid": self.pid,
            "started_at": self.started_at,
            "dumped_at": time.time(),
            "counters": counters,
        }

    def dump(self) -> str | None:
        if not self.enabled:
            return None
        path = Path(self.out_dir) / f"{self.role}_{self.pid}.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(self.to_dict(), f, indent=1)
            os.replace(tmp, path)
        except OSError:
            return None
        return str(path)


def _default_role() -> str:
    sid = os.environ.get(ENV_STAGE_ID)
    return f"stage{sid}" if sid not in (None, "") else "orchestrator"


# ---------------------------------------------------------------- hooks
def instrument_engine_core(engine_core: Any) -> None:
    """Time the scheduler and the whole step of a (stage) EngineCore instance."""
    stats = StepStats.get()
    if not stats.enabled:
        return
    sched = getattr(engine_core, "scheduler", None)
    if sched is not None:
        for name in ("schedule", "update_from_output"):
            fn = getattr(sched, name, None)
            if callable(fn) and not hasattr(fn, "__wrapped__"):
                setattr(sched, name, stats.wrap(f"sched.{name}_ms", fn))
    step_fn = getattr(engine_core, "step_fn", None)
    if callable(step_fn) and not hasattr(step_fn, "__wrapped__"):
        engine_core.step_fn = stats.wrap("core.step_ms", step_fn)


def instrument_sampler(sampler: Any) -> None:
    """Time the phases of vLLM's v1 ``Sampler`` in place (idempotent, no-op when disabled)."""
    stats = StepStats.get()
    if not stats.enabled or sampler is None or getattr(sampler, "_omni_step_stats_instrumented", False):
        return
    for name in (
        "apply_logits_processors",
        "apply_penalties",
        "apply_temperature",
        "sample",
        "compute_logprobs",
        "gather_logprobs",
    ):
        fn = getattr(sampler, name, None)
        if callable(fn):
            setattr(sampler, name, stats.wrap(f"sampler.{name}_ms", fn))
    tk = getattr(sampler, "topk_topp_sampler", None)
    if tk is not None and callable(getattr(tk, "forward", None)):
        tk.forward = stats.wrap("sampler.topk_topp_ms", tk.forward)
    sampler._omni_step_stats_instrumented = True


def shm_lockfile_age_ms(key: str, now: float | None = None) -> float | None:
    """Age of the SharedMemoryConnector lock file for ``key`` (written at put time)."""
    try:
        from vllm_omni.windows.paths import shm_path

        st = os.stat(shm_path(f"shm_{key}_lockfile.lock"))
    except OSError:
        return None
    return ((now if now is not None else time.time()) - st.st_mtime) * 1000.0


# ---------------------------------------------------------------- merge
def merge_dir(out_dir: str | Path) -> dict[str, Any]:
    """Merge per-process dumps into ``{role: {counter: summary}}`` plus a share table."""
    roles: dict[str, dict[str, Any]] = {}
    for f in sorted(Path(out_dir).glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        role = d.get("role", f.stem)
        bucket = roles.setdefault(role, {})
        for name, summ in d.get("counters", {}).items():
            cur = bucket.get(name)
            if cur is None:
                bucket[name] = dict(summ)
            else:  # several processes with the same role: combine count/sum/max, keep worst p95
                cnt = cur["count"] + summ["count"]
                cur.update(
                    count=cnt,
                    sum_ms=cur["sum_ms"] + summ["sum_ms"],
                    mean_ms=((cur["sum_ms"] + summ["sum_ms"]) / cnt) if cnt else 0.0,
                    p50_ms=max(cur["p50_ms"], summ["p50_ms"]),
                    p95_ms=max(cur["p95_ms"], summ["p95_ms"]),
                    max_ms=max(cur["max_ms"], summ["max_ms"]),
                )
    return {"roles": roles, "attribution": attribution(roles)}


def attribution(roles: dict[str, dict[str, Any]], frame_budget_ms: float = FRAME_BUDGET_MS) -> list[dict[str, Any]]:
    """Rows of (role, counter, mean, p95, share of the frame budget) sorted by share."""
    rows = []
    for role, counters in roles.items():
        for name, s in counters.items():
            rows.append(
                {
                    "role": role,
                    "counter": name,
                    "count": s["count"],
                    "mean_ms": s["mean_ms"],
                    "p95_ms": s["p95_ms"],
                    "share_of_frame": (s["mean_ms"] / frame_budget_ms) if frame_budget_ms else 0.0,
                }
            )
    rows.sort(key=lambda r: -r["share_of_frame"])
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| role | counter | n | mean ms | p95 ms | share of 80 ms |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['role']} | {r['counter']} | {r['count']} | {r['mean_ms']:.2f} | {r['p95_ms']:.2f} "
            f"| {100 * r['share_of_frame']:.1f}% |"
        )
    return "\n".join(lines)
