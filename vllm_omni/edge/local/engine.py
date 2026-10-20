# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The M0 local text engine: run an admitted plan, and record what happened.

This is thin on purpose. It owns the plan, the session state references, the
bounded output stream, the cancellation epoch and the measurement; it owns no
kernels, no scheduler and no cache. vLLM keeps doing all of that inside the
Omni stage, which is the whole point of evolving the existing runtime instead
of writing a second one.

**On measuring memory here, honestly.** Three counters are sampled and they are
three different things, which the architecture record is emphatic about not
adding together:

* ``nvml_device_used`` is the whole card, every process on it. Under WSL2,
  ``nvmlDeviceGetComputeRunningProcesses`` returns an empty list, so this
  engine *cannot* attribute GPU bytes to its own PIDs on this host. The delta
  across a load is still evidence -- 7.8 GiB appearing while the host RSS does
  not move says where the weights went -- but it is a delta on a shared
  counter, and it is labelled that way.
* ``tree_rss`` / ``tree_pss`` are this process and its stage subprocesses.
  PSS includes a share of file-backed pages, so on an mmap'd safetensors load
  it counts pages that are page cache, not anonymous memory.
* ``host_available`` is the system's, and moves for reasons that are not us.

Sampling is at a fixed interval, so a peak shorter than the interval is missed.
That lower bound is reported with the numbers rather than assumed away.

**On cancellation.** ``cancel`` does three things in order: retire the stream's
epoch so nothing already in flight can be delivered, abort the request in the
backend so it stops producing, and release the session's reservation in the
ledger. Doing them in that order is what makes "取消后旧数据不能重新进入输出流"
true rather than likely.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from vllm.logger import init_logger

from vllm_omni.edge.local.plan import ExecutionPlan
from vllm_omni.edge.local.session import (
    BoundedEventStream,
    ChunkEvent,
    StateHandle,
    new_session_id,
)

logger = init_logger(__name__)

GiB = 2**30
SAMPLE_INTERVAL_S = 0.25
"""Memory sampling period. Anything shorter-lived than this is not seen; the
reported peaks are lower bounds and say so."""

RUNTIME_ENV: dict[str, str] = {
    # FlashInfer's sampler JIT-compiles on first use and needs nvcc, which is
    # not installed here -- nvidia-smi reporting a CUDA version is not a
    # toolkit. Without this the *sampler* takes down the engine at the first
    # token with "Could not find nvcc".
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    # Under WSL2 vLLM's own GPU model runner allocates a UvaBuffer for its
    # input batch and refuses to start with "UVA is not available" unless
    # pinned memory is enabled. Omni's AR model runner does not take that path,
    # so this matters most when something compares the two.
    "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
"""The environment the measured runs used, in one place.

It lives here rather than inline in ``start()`` because more than one entry
point has to apply it: a comparison that configures the reference differently
from the engine is measuring the configuration, not the engine. Applied with
``setdefault``, so a caller who set one of these keeps their value -- and the
run record reports what was actually in force."""


def apply_runtime_env() -> dict[str, str | None]:
    """Set :data:`RUNTIME_ENV` without overriding the caller, and report it."""
    for key, value in RUNTIME_ENV.items():
        os.environ.setdefault(key, value)
    return {key: os.environ.get(key) for key in RUNTIME_ENV}


CANCEL_DRAIN_TIMEOUT_S = 10.0
"""How long ``cancel`` waits for the producer task to finish unwinding before
reporting ``producer_stopped: False`` and returning anyway. The consumer is
released before this wait starts, so exceeding this delays cleanup, not the
caller."""


# ---- measurement -----------------------------------------------------------


def _host_kind() -> str:
    """``windows``, ``wsl2`` or ``linux`` -- the three hosts this code has run on."""
    import sys

    if sys.platform == "win32":
        return "windows"
    try:
        with open("/proc/version", encoding="utf-8", errors="replace") as f:
            if "microsoft" in f.read().lower():
                return "wsl2"
    except OSError:
        pass
    return "linux"


def _host_compat_report() -> dict[str, Any] | None:
    """``vllm_omni.windows.report()`` on Windows, so a record says what made the run possible."""
    from vllm_omni.windows import is_windows, report

    if not is_windows():
        return None
    try:
        return report()
    except Exception as exc:  # the audit must never take the engine down
        return {"error": f"{type(exc).__name__}: {exc}"}


def gpu_attribution_for_tree() -> dict[str, Any]:
    """Try to attribute GPU bytes to this process tree, and say why when it fails.

    [edge-infer W4] The earlier version of this report hardcoded a WSL2
    explanation for an empty ``nvmlDeviceGetComputeRunningProcesses`` list.
    The W1 acceptance run on *native Windows* then carried that WSL2 sentence
    verbatim -- the conclusion (placement unverified) was right, the reason
    offered to the operator was wrong. So this now measures instead of
    assuming: it asks NVML, intersects with the process tree, and only when
    that comes back empty does it name the host-specific mechanism (WSL2's
    GPU-PV and Windows' WDDM driver model both hide per-process usage from
    NVML; a bare Linux host normally reports it).
    """
    host = _host_kind()
    try:
        import psutil
        import pynvml

        proc = psutil.Process()
        tree = {proc.pid} | {c.pid for c in proc.children(recursive=True)}
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            running = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        finally:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()
    except Exception as exc:
        return {
            "per_process_gpu_attribution": None,
            "per_process_gpu_attribution_reason": (
                f"NVML per-process query unavailable on this {host} host: {exc}. "
                "The load delta is a whole-card measurement."
            ),
        }
    ours = [
        {"pid": int(r.pid), "used_gpu_memory": getattr(r, "usedGpuMemory", None)}
        for r in running
        if int(r.pid) in tree
    ]
    if ours:
        return {
            "per_process_gpu_attribution": ours,
            "per_process_gpu_attribution_reason": None,
        }
    mechanism = {
        "wsl2": "under WSL2 the GPU-PV driver reports no per-process compute list",
        "windows": "under the Windows WDDM driver model NVML reports no per-process "
                   "compute list (the same NVML call that is empty under WSL2's GPU-PV)",
        "linux": "no compute process from this tree appears in NVML's list",
    }[host]
    return {
        "per_process_gpu_attribution": None,
        "per_process_gpu_attribution_reason": (
            f"nvmlDeviceGetComputeRunningProcesses returned {len(running)} process(es), "
            f"none from this engine's tree of {len(tree)} PID(s): {mechanism}, so GPU "
            "bytes cannot be attributed to this engine's PIDs on this host. The load "
            "delta is a whole-card measurement."
        ),
    }



@dataclass
class MemorySample:
    unix: float
    nvml_device_used: int | None
    nvml_device_total: int | None
    tree_rss: int
    tree_pss: int
    host_available: int
    host_total: int


class MemorySampler:
    """Background sampler over the process tree, the host and the GPU.

    A thread rather than a task: the engine's event loop is the thing being
    measured, and a sampler that stops whenever the loop is busy would miss
    exactly the moments worth sampling.
    """

    def __init__(self, interval_s: float = SAMPLE_INTERVAL_S) -> None:
        self.interval_s = interval_s
        self.samples: list[MemorySample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = None
        self._handle = None

    def _init_nvml(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as exc:  # no GPU, or no NVML in this venv
            logger.debug("NVML unavailable, GPU memory will not be sampled: %s", exc)
            self._nvml = None

    def sample(self) -> MemorySample:
        import psutil

        proc = psutil.Process()
        rss = pss = 0
        for p in [proc] + proc.children(recursive=True):
            try:
                info = p.memory_full_info()
                rss += info.rss
                pss += getattr(info, "pss", 0) or 0
            except (psutil.Error, OSError):
                continue
        vm = psutil.virtual_memory()
        used = total = None
        if self._nvml is not None:
            try:
                mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                used, total = int(mem.used), int(mem.total)
            except Exception:
                used = total = None
        return MemorySample(
            unix=time.time(), nvml_device_used=used, nvml_device_total=total,
            tree_rss=rss, tree_pss=pss,
            host_available=vm.available, host_total=vm.total,
        )

    def start(self) -> None:
        self._init_nvml()

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    self.samples.append(self.sample())
                except Exception as exc:  # never take the engine down
                    logger.debug("memory sample failed: %s", exc)
                self._stop.wait(self.interval_s)

        self._thread = threading.Thread(target=loop, name="local-mem-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._nvml is not None:
            with contextlib.suppress(Exception):
                self._nvml.nvmlShutdown()

    def peaks(self, since: float = 0.0) -> dict[str, Any]:
        window = [s for s in self.samples if s.unix >= since]
        if not window:
            return {"samples": 0, "interval_s": self.interval_s}
        gpu = [s.nvml_device_used for s in window if s.nvml_device_used is not None]
        return {
            "samples": len(window),
            "interval_s": self.interval_s,
            "note": (
                "peaks are lower bounds: anything shorter than interval_s is missed. "
                "nvml_device_used is the whole card including other processes; "
                "tree_pss includes a share of file-backed pages. These three "
                "counters are different accountings and must not be summed."
            ),
            "nvml_device_used_peak": max(gpu) if gpu else None,
            "nvml_device_used_min": min(gpu) if gpu else None,
            "tree_rss_peak": max(s.tree_rss for s in window),
            "tree_pss_peak": max(s.tree_pss for s in window),
            "host_available_min": min(s.host_available for s in window),
        }


# ---- per-request record ----------------------------------------------------


@dataclass
class RequestRecord:
    """Timing for one request, in the units the validation plan asks for."""

    request_id: str
    session_id: str
    epoch: int
    prompt_chars: int
    prompt_tokens: int | None = None
    output_tokens: int = 0
    submitted_unix: float = 0.0
    first_token_unix: float | None = None
    last_token_unix: float | None = None
    finished: bool = False
    cancelled: bool = False
    error: str | None = None
    token_unix: list[float] = field(default_factory=list)
    text: str = ""
    output_token_ids: list[int] = field(default_factory=list)
    """The sampled ids, kept so a run can be compared against a reference
    decode token-for-token. Text is not enough for that: two different id
    sequences can detokenize to the same string."""

    @property
    def ttft_s(self) -> float | None:
        """Submission to first token. Includes admission and scheduling, which
        is what a caller experiences; it is not an isolated prefill kernel
        time and is never reported as one."""
        if self.first_token_unix is None:
            return None
        return self.first_token_unix - self.submitted_unix

    @property
    def decode_tok_per_s(self) -> float | None:
        """Tokens after the first, over the time between first and last.

        ``None`` below two tokens rather than a number computed from one
        interval.
        """
        if self.first_token_unix is None or self.last_token_unix is None:
            return None
        if self.output_tokens < 2:
            return None
        span = self.last_token_unix - self.first_token_unix
        return (self.output_tokens - 1) / span if span > 0 else None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ttft_s"] = self.ttft_s
        d["decode_tok_per_s"] = self.decode_tok_per_s
        return d


# ---- the engine ------------------------------------------------------------


class PlanNotAdmitted(RuntimeError):
    """Raised when someone tries to run a plan that refused."""


class LocalTextEngine:
    """One admitted plan, one Omni stage, streamed text with a cancel epoch."""

    def __init__(self, plan: ExecutionPlan, *, extra_engine_kwargs: dict[str, Any] | None = None) -> None:
        if not plan.admitted:
            raise PlanNotAdmitted(plan.summary())
        self.plan = plan
        self.extra_engine_kwargs = dict(extra_engine_kwargs or {})
        self.sampler = MemorySampler()
        self.records: dict[str, RequestRecord] = {}
        self._omni: Any = None
        self._sessions: dict[str, StateHandle] = {}
        self._streams: dict[str, BoundedEventStream] = {}
        self._pumps: dict[str, asyncio.Task] = {}
        self._inflight: dict[str, str] = {}  # request_id -> session_id
        self.load_seconds: float | None = None
        self.load_memory: dict[str, Any] = {}
        self._closed = False

    # -- lifecycle ----------------------------------------------------------
    async def start(self) -> None:
        """Load the model according to the plan, and price the load."""
        from vllm_omni import AsyncOmni

        apply_runtime_env()
        self.sampler.start()
        before = self.sampler.sample()
        started = time.perf_counter()
        kwargs = {**self.plan.engine_kwargs, **self.extra_engine_kwargs}
        logger.info(
            "[local] loading %s on %s with %s",
            self.plan.manifest.model_dir, self.plan.selected.device_id, kwargs,
        )
        self._omni = AsyncOmni(model=self.plan.manifest.model_dir, **kwargs)
        self.load_seconds = time.perf_counter() - started
        after = self.sampler.sample()

        self.load_memory = {
            "before": asdict(before),
            "after": asdict(after),
            "nvml_device_used_delta": (
                (after.nvml_device_used - before.nvml_device_used)
                if (after.nvml_device_used is not None and before.nvml_device_used is not None)
                else None
            ),
            "tree_rss_delta": after.tree_rss - before.tree_rss,
            "peaks_during_load": self.sampler.peaks(since=before.unix),
        }
        self._record_actuals()

    def _record_actuals(self) -> None:
        """Turn the load into a bound on the reservations, without overclaiming.

        The NVML delta across the load is **not** "the weights". It is the
        weights plus the CUDA context, the allocator's arenas and whatever else
        the backend set up, measured on a counter shared with every other
        process on the card. So it is recorded as an *upper bound* on the
        weights reservation rather than as its actual value: a ratio above 1.0
        here means the budget was under the bound, not that the weights were
        bigger than their files.
        """
        pool = self.plan.selected.memory_pool if self.plan.selected else "vram"
        if pool == "vram":
            delta = self.load_memory.get("nvml_device_used_delta")
            unit = "on the device (whole-card delta)"
        else:
            delta = self.load_memory.get("tree_rss_delta")
            unit = "of process-tree RSS"
        if not delta or delta <= 0:
            return
        for reservation in self.plan.reservations:
            if reservation.purpose == "weights" and reservation.pool == pool:
                reservation.actual_upper_bound_bytes = int(delta)
                reservation.note += (
                    f"; the load moved +{delta / GiB:.2f} GiB {unit}, which bounds "
                    "weights + backend context together and is not an attribution "
                    "to the weights alone"
                )

    def measured_peak(self) -> dict[str, Any]:
        """Budgeted peak against the measured one, on the selected pool.

        The measured number subtracts a pre-load baseline, because the counter
        is shared: on this host the card already held 2.75 GiB before the
        engine started, and reporting that as the engine's would be the same
        mistake the architecture record calls out about summing WSL ``used``,
        Windows ``used`` and PSS.
        """
        if self.plan.selected is None:
            return {}
        pool = self.plan.selected.memory_pool
        peaks = self.sampler.peaks()
        before = self.load_memory.get("before") or {}
        if pool == "vram":
            baseline = before.get("nvml_device_used")
            peak = peaks.get("nvml_device_used_peak")
            counter = "nvml_device_used"
        else:
            baseline = before.get("tree_rss")
            peak = peaks.get("tree_rss_peak")
            counter = "tree_rss"
        attributable = (peak - baseline) if (peak is not None and baseline is not None) else None
        budget = self.plan.peak_bytes
        return {
            "pool": pool,
            "counter": counter,
            "baseline_bytes": baseline,
            "peak_bytes": peak,
            "attributable_peak_bytes": attributable,
            "budgeted_peak_bytes": budget,
            "budget_over_measured": (budget / attributable) if attributable else None,
            "admitted_without_oom": True,
            "caveats": (
                "sampled every %.2f s, so the peak is a lower bound; the counter is "
                "shared with other processes and the baseline subtraction only "
                "removes what was there before the load, not what arrived during it. "
                "A budget above the measured peak is the intended outcome -- this "
                "budget is a safe upper bound for admission, not a prediction."
                % self.sampler.interval_s
            ),
        }

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task in list(self._pumps.values()):
            task.cancel()
        for task in list(self._pumps.values()):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._pumps.clear()
        for stream in list(self._streams.values()):
            with contextlib.suppress(Exception):
                await stream.close()
        self._streams.clear()
        if self._omni is not None:
            for name in ("shutdown", "close"):
                closer = getattr(self._omni, name, None)
                if callable(closer):
                    with contextlib.suppress(Exception):
                        result = closer()
                        if asyncio.iscoroutine(result):
                            await result
                    break
        self.sampler.stop()

    async def __aenter__(self) -> LocalTextEngine:
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # -- sessions -----------------------------------------------------------
    def open_session(self) -> StateHandle:
        handle = StateHandle(
            session_id=new_session_id(),
            backend=self.plan.backend or "unknown",
            artifact_id=self.plan.manifest.artifact_id,
        )
        self._sessions[handle.session_id] = handle
        return handle

    def session(self, session_id: str) -> StateHandle:
        return self._sessions[session_id]

    def close_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    # -- generation ---------------------------------------------------------
    async def submit(
        self,
        handle: StateHandle,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        ignore_eos: bool = False,
        request_id: str | None = None,
        max_chunks: int = 64,
        max_bytes: int = 1 << 20,
    ) -> tuple[str, BoundedEventStream]:
        """Start one request; return its id and its bounded event stream.

        Returning the stream rather than yielding from here keeps cancellation
        possible from outside the consumer: ``cancel`` can retire the epoch
        while the caller is still awaiting an event.
        """
        if self._omni is None:
            raise RuntimeError("engine not started")
        live = self._sessions.get(handle.session_id)
        if live is None or not live.accepts(handle):
            raise RuntimeError(
                f"state handle for session {handle.session_id} is stale "
                f"(epoch {handle.epoch}, live {live.epoch if live else 'closed'}); "
                "open a new session rather than reusing retired state"
            )
        from vllm import SamplingParams

        request_id = request_id or f"{handle.session_id}-{len(self.records)}"
        stream = BoundedEventStream(max_chunks=max_chunks, max_bytes=max_bytes)
        if handle.epoch:
            # A session that has been cancelled before starts its new
            # stream already fenced against everything older.
            stream.retire_epoch(handle.epoch - 1)
        record = RequestRecord(
            request_id=request_id,
            session_id=handle.session_id,
            epoch=handle.epoch,
            prompt_chars=len(prompt),
            submitted_unix=time.time(),
        )
        self.records[request_id] = record
        self._streams[request_id] = stream
        self._inflight[request_id] = handle.session_id

        params = SamplingParams(
            temperature=temperature, max_tokens=max_tokens, ignore_eos=ignore_eos
        )
        self._pumps[request_id] = asyncio.create_task(
            self._pump(request_id, handle, prompt, params, stream, record),
            name=f"local-pump-{request_id}",
        )
        return request_id, stream

    async def _pump(
        self,
        request_id: str,
        handle: StateHandle,
        prompt: str,
        params: Any,
        stream: BoundedEventStream,
        record: RequestRecord,
    ) -> None:
        """Adapt Omni's cumulative outputs into incremental ChunkEvents.

        Omni yields the output-so-far, like vLLM. Re-emitting that verbatim
        would make a dropped chunk invisible to the consumer, so only the delta
        goes on the wire and ``seq`` is what detects loss.
        """
        seq = 0
        emitted_text = ""
        try:
            async for out in self._omni.generate(
                prompt, request_id=request_id, sampling_params_list=[params]
            ):
                completion = _first_completion(out)
                if completion is None:
                    continue
                text = completion.get("text", "")
                token_ids = completion.get("token_ids", ())
                if record.prompt_tokens is None:
                    record.prompt_tokens = completion.get("prompt_tokens")
                delta = text[len(emitted_text):] if text.startswith(emitted_text) else text
                n_tokens = len(token_ids)
                if n_tokens <= record.output_tokens and not delta:
                    continue
                now = time.time()
                if record.first_token_unix is None:
                    record.first_token_unix = now
                record.last_token_unix = now
                record.output_tokens = max(record.output_tokens, n_tokens)
                record.token_unix.append(now)
                emitted_text = text if text.startswith(emitted_text) else emitted_text + delta
                if n_tokens >= len(record.output_token_ids):
                    record.output_token_ids = list(token_ids)
                seq += 1
                await stream.put(ChunkEvent(
                    request_id=request_id, stage_id=0, seq=seq, epoch=handle.epoch,
                    kind="token", payload={"text": delta, "n_tokens": n_tokens},
                    started_unix=record.submitted_unix, emitted_unix=now,
                    input_watermark=record.prompt_tokens or 0,
                ))
            record.text = emitted_text
            record.finished = True
            await stream.put(ChunkEvent(
                request_id=request_id, stage_id=0, seq=seq + 1, epoch=handle.epoch,
                kind="done", payload={"output_tokens": record.output_tokens},
                started_unix=record.submitted_unix, emitted_unix=time.time(), final=True,
            ))
        except asyncio.CancelledError:
            record.cancelled = True
            record.text = emitted_text
            raise
        except Exception as exc:
            record.error = str(exc)
            record.text = emitted_text
            with contextlib.suppress(Exception):
                await stream.put(ChunkEvent(
                    request_id=request_id, stage_id=0, seq=seq + 1, epoch=handle.epoch,
                    kind="error", payload=None, started_unix=record.submitted_unix,
                    emitted_unix=time.time(), final=True, error=str(exc),
                ))
        finally:
            self._inflight.pop(request_id, None)
            with contextlib.suppress(Exception):
                await stream.close()

    async def cancel(self, request_id: str) -> dict[str, Any]:
        """Retire the epoch, stop the backend, release the reservation.

        The order matters and is the contract: by the time this returns, no
        event produced before the cancel can still reach the consumer.
        """
        record = self.records.get(request_id)
        stream = self._streams.get(request_id)
        session_id = self._inflight.get(request_id) or (record.session_id if record else None)
        before = self.sampler.sample()

        if stream is not None:
            stream.retire_epoch(record.epoch if record else 0)

        aborted = False
        if self._omni is not None:
            for name in ("abort", "abort_request", "abort_requests_async"):
                fn = getattr(self._omni, name, None)
                if callable(fn):
                    with contextlib.suppress(Exception):
                        result = fn(request_id)
                        if asyncio.iscoroutine(result):
                            await result
                        aborted = True
                        break

        # Close the stream *before* waiting on the producer. The consumer is
        # blocked in ``get()`` and the sentinel is what releases it; making
        # that depend on the backend's unwind would mean a backend that hangs
        # hangs the caller too, which is the opposite of what cancel is for.
        if stream is not None:
            with contextlib.suppress(Exception):
                await stream.close()

        task = self._pumps.pop(request_id, None)
        pump_stopped = True
        if task is not None:
            task.cancel()
            # Bounded: tearing down an async generator runs the backend's own
            # cleanup, and that cleanup can wait on a request this method has
            # already aborted. Waiting forever for it would turn a cancel into
            # a deadlock, so the wait has a deadline and the outcome is
            # reported rather than assumed.
            done, pending = await asyncio.wait({task}, timeout=CANCEL_DRAIN_TIMEOUT_S)
            pump_stopped = not pending
            if done:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    task.result()

        if record is not None:
            record.cancelled = True
        if session_id and session_id in self._sessions:
            # A cancelled session's state is retired, not reused: the next
            # request on it must start from a fresh epoch.
            self._sessions[session_id] = self._sessions[session_id].next_epoch()

        # Give the backend a moment to free blocks, then look.
        await asyncio.sleep(SAMPLE_INTERVAL_S * 2)
        after = self.sampler.sample()
        return {
            "request_id": request_id,
            "aborted_in_backend": aborted,
            "producer_stopped": pump_stopped,
            "epoch_retired": stream.epoch if stream is not None else None,
            "dropped_stale_events": stream.dropped_stale if stream is not None else 0,
            "output_tokens_at_cancel": record.output_tokens if record else 0,
            "session_epoch_now": self._sessions[session_id].epoch if session_id in self._sessions else None,
            "inflight_after": len(self._inflight),
            "nvml_device_used_before": before.nvml_device_used,
            "nvml_device_used_after": after.nvml_device_used,
            "note": (
                "a cancelled request's KV blocks return to vLLM's pool, which is "
                "pre-allocated; the device counter is therefore expected NOT to "
                "fall. What must fall is the in-flight count, and no retired-epoch "
                "event may be delivered."
            ),
        }

    # -- reporting ----------------------------------------------------------
    def report_placement(self) -> dict[str, Any]:
        """Where the stage is actually configured and running.

        Two claims of different strength: the *configured* device, read back
        from the stage's own ``VllmConfig`` rather than from the plan that
        requested it, and the *observed* device memory delta across the load.
        Neither is a per-process attribution, because NVML cannot give one
        here; that limitation is in the output, not only in the docstring.
        """
        configured: dict[str, Any] = {}
        try:
            configs = getattr(self._omni.engine, "stage_vllm_configs", None) or {}
            cfg = configs[0] if isinstance(configs, (list, tuple)) else configs.get(0)
            device_config = getattr(cfg, "device_config", None)
            model_config = getattr(cfg, "model_config", None)
            cache_config = getattr(cfg, "cache_config", None)
            configured = {
                "device": str(getattr(device_config, "device", "")),
                "device_type": str(getattr(device_config, "device_type", "")),
                "dtype": str(getattr(model_config, "dtype", "")),
                "quantization": getattr(model_config, "quantization", None),
                "max_model_len": getattr(model_config, "max_model_len", None),
                "enforce_eager": getattr(model_config, "enforce_eager", None),
                "kv_cache_memory_bytes": getattr(cache_config, "kv_cache_memory_bytes", None),
                "num_gpu_blocks": getattr(cache_config, "num_gpu_blocks", None),
                "block_size": getattr(cache_config, "block_size", None),
            }
        except Exception as exc:
            configured = {"error": f"stage config unavailable: {exc}"}

        return {
            "planned_device": self.plan.selected.device_id if self.plan.selected else None,
            "planned_backend": self.plan.backend,
            "configured": configured,
            "observed": {
                "nvml_device_used_delta_at_load": self.load_memory.get("nvml_device_used_delta"),
                "tree_rss_delta_at_load": self.load_memory.get("tree_rss_delta"),
                **gpu_attribution_for_tree(),
            },
            # [edge-infer] On native Windows, which compatibility layer carried this
            # process (shims, patches, wheel-provided sites). None elsewhere.
            "host_compat": _host_compat_report(),
        }

    def report_usage(self) -> dict[str, Any]:
        """Budgeted vs measured, per reservation, plus the sampled peaks."""
        return {
            "load_seconds": self.load_seconds,
            "reservations": [
                {
                    **r.to_dict(),
                    "upper_bound_over_budget": (
                        r.actual_upper_bound_bytes / r.bytes
                        if (r.actual_upper_bound_bytes and r.bytes)
                        else None
                    ),
                }
                for r in self.plan.reservations
            ],
            "budgeted_peak_bytes": self.plan.peak_bytes,
            "measured_peak": self.measured_peak(),
            "load_memory": self.load_memory,
            "peaks": self.sampler.peaks(),
            "requests": [r.to_dict() for r in self.records.values()],
            "streams": {rid: s.stats() for rid, s in self._streams.items()},
        }


def _first_completion(out: Any) -> dict[str, Any] | None:
    """Pull ``(text, token_ids, prompt_tokens)`` out of an Omni output.

    Kept in one function because the shape is the thing most likely to move
    between vLLM/Omni versions, and a version bump should break here, loudly,
    rather than silently start reporting zero tokens.
    """
    outputs = getattr(out, "outputs", None)
    if not outputs:
        return None
    first = outputs[0]
    text = getattr(first, "text", None)
    token_ids = getattr(first, "token_ids", None)
    if text is None and token_ids is None:
        return None
    prompt_ids = getattr(out, "prompt_token_ids", None)
    return {
        "text": text or "",
        "token_ids": tuple(token_ids or ()),
        "prompt_tokens": len(prompt_ids) if prompt_ids is not None else None,
    }
