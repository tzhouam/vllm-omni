# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Planning and opening a stage that runs on the 890M or the NPU.

Same contract as :mod:`vllm_omni.edge.local.plan`, one device further out: pick
a device, budget it, and either place the work or refuse with a code and a
remedy. Three things differ, and each is the reason for a refusal code that a
checkpoint-on-vLLM plan never needs.

**These devices execute graphs, not checkpoints.** The gate is the exported
artifact's format against the device's, so "no export exists yet" is its own
answer (:data:`~vllm_omni.edge.local.plan.REFUSE_NO_ARTIFACT`) rather than a
vague unsupported -- the work it implies is an export, not a port.

**Placement has to be measured after the fact.** A VitisAI session lists the EP
in ``get_providers()`` whether or not it took a single node, and a graph it
declined runs entirely on the CPU and returns bit-identical numbers. So neither
the provider list nor the outputs can distinguish a real NPU run from a silent
fallback; only ORT's per-node assignments can, and those exist only after
something has run. :meth:`ExternalStage.open` therefore loads, runs once under
the profiler, and refuses on the node split. An *unverified* placement is
refused too: ``None`` is not rounded to "fine".

**The memory is not on a private pool.** Both devices allocate out of the same
host RAM the CPU stage is budgeted against, and the worker is a different
process -- on Windows, a different OS -- so the engine's psutil sampler cannot
see it at all. The worker reports its own resident set and it lands in the same
ledger; an unreported worker is a hole in the budget, not a rounding error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np

from vllm_omni.edge.local.capabilities import MASK_ENV, DeviceCapability
from vllm_omni.edge.local.external import launch as _launch
from vllm_omni.edge.local.external.client import ExternalWorker, LoadReport, RunTiming
from vllm_omni.edge.local.manifest import GraphArtifact
from vllm_omni.edge.local.plan import (
    EXTERNAL_RESERVE_BYTES,
    MIN_FRACTION_ON_TARGET,
    REFUSE_CAPACITY,
    REFUSE_EP_PLACEMENT,
    REFUSE_NO_ARTIFACT,
    REFUSE_NO_DEVICE,
    REFUSE_ROUTE,
    MemoryReservation,
    Refusal,
)

MiB = 2**20

WORKER_BASE_BYTES: dict[str, int] = {
    # Measured on this laptop, 2026-09-15: a bare Windows 3.12 interpreter is
    # 28.5 MiB resident, and the same process holding one VitisAI session on a
    # 2 MiB graph is 270 MiB. Nearly all of that is the execution provider, not
    # the graph, so it is charged per worker rather than scaled by artifact
    # size. E, with a measured basis; ``open()`` replaces it with the worker's
    # own number as soon as there is one.
    _launch.ROUTE_VITISAI: 320 * MiB,
    # Measured 369 MiB holding the 27B vision tower, against an earlier 256 MiB
    # guess -- the budget has to bound the measurement, so it was raised rather
    # than the overrun explained away.
    _launch.ROUTE_DML: 512 * MiB,
    _launch.ROUTE_CPU: 128 * MiB,
    # torch-directml drags a whole torch build in beside the D3D12 runtime.
    _launch.ROUTE_TORCH_DML: 768 * MiB,
}

WORKER_WORKSPACE_BYTES = 256 * MiB
"""Activations and the EP's scratch. E: not separable from the worker's RSS by
any counter available here, so it is budgeted and then subsumed by the measured
total rather than checked on its own."""


class PlacementRefused(RuntimeError):
    """The stage could not be placed. Carries the refusal, not just a message."""

    def __init__(self, refusal: Refusal, report: LoadReport | None = None) -> None:
        super().__init__(f"[{refusal.code}] {refusal.message}")
        self.refusal = refusal
        self.report = report


@dataclass
class StagePlan:
    """Where one exported graph would run, and what it would cost."""

    artifact: GraphArtifact
    device: DeviceCapability | None
    route: _launch.Route | None
    reservations: list[MemoryReservation] = field(default_factory=list)
    refusals: list[Refusal] = field(default_factory=list)
    min_fraction_on_target: float = MIN_FRACTION_ON_TARGET
    worker_peak_rss_hint_bytes: int | None = None
    observed_worker_peak_rss_bytes: int | None = None
    report: LoadReport | None = None

    @property
    def admitted(self) -> bool:
        return self.device is not None and self.route is not None

    @property
    def budget_bytes(self) -> int:
        return sum(r.bytes for r in self.reservations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact.to_dict(),
            "device": self.device.to_dict() if self.device else None,
            "route": self.route.to_dict() if self.route else None,
            "reservations": [r.to_dict() for r in self.reservations],
            "refusals": [r.to_dict() for r in self.refusals],
            "min_fraction_on_target": self.min_fraction_on_target,
            "worker_peak_rss_hint_bytes": self.worker_peak_rss_hint_bytes,
            "observed_worker_peak_rss_bytes": self.observed_worker_peak_rss_bytes,
            "budget_bytes": self.budget_bytes,
            "admitted": self.admitted,
            "report": self.report.to_dict() if self.report else None,
        }

    def summary(self) -> str:
        head = f"{self.artifact.component} ({self.artifact.fmt}, {self.artifact.bytes / MiB:.1f} MiB)"
        if not self.admitted:
            lines = [f"{head}: REFUSED"]
            lines += [f"  [{r.code}] {r.device_id}: {r.message}\n      remedy: {r.remedy}" for r in self.refusals]
            return "\n".join(lines)
        assert self.device is not None and self.route is not None
        lines = [
            f"{head} -> {self.device.device_id} via {self.route.name}",
            f"  budget {self.budget_bytes / MiB:.0f} MiB on {self.device.memory_pool} "
            f"(shared with every other device on this pool)",
        ]
        for reservation in self.reservations:
            actual = ""
            if reservation.actual_upper_bound_bytes is not None:
                measured = reservation.actual_upper_bound_bytes
                # A budget that does not bound its measurement is a failed
                # budget. Saying so here is the whole discipline: the number is
                # supposed to be a provable ceiling, not a guess that is usually
                # close.
                verdict = "measured <=" if measured <= reservation.bytes else "OVER, measured"
                actual = f", {verdict} {measured / MiB:.0f} MiB"
            lines.append(
                f"    {reservation.purpose:<18} {reservation.bytes / MiB:>7.0f} MiB  "
                f"[{reservation.evidence}] {reservation.lifetime}{actual}"
            )
        if self.report is not None:
            lines.append(f"  {self.report.summary()}")
        for refusal in self.refusals:
            lines.append(f"  also considered: [{refusal.code}] {refusal.device_id}: {refusal.message}")
        return "\n".join(lines)


def _reservations_for(
    artifact: GraphArtifact,
    route: _launch.Route,
    pool: str,
    worker_peak_rss_hint_bytes: int | None = None,
) -> list[MemoryReservation]:
    # Working-set peak includes resident graph pages. The separate graph-file
    # reservation already charges those bytes, so charge the remaining peak to
    # the worker. A measured hint is device/EP-specific and gets 10% headroom.
    runtime_bytes = WORKER_BASE_BYTES.get(route.name, 256 * MiB)
    if worker_peak_rss_hint_bytes is not None:
        runtime_bytes = max(
            runtime_bytes,
            max(0, ceil(worker_peak_rss_hint_bytes * 1.10) - artifact.bytes),
        )
    return [
        MemoryReservation(
            pool=pool,
            purpose="graph weights",
            bytes=artifact.bytes,
            lifetime="session",
            reclaimable="no",
            evidence="D",
            note=f"the exported file, {artifact.fmt}, sha {artifact.sha256[:12]}",
        ),
        MemoryReservation(
            pool=pool,
            purpose="worker runtime",
            bytes=runtime_bytes,
            lifetime="session",
            reclaimable="on_release",
            evidence="E",
            note=(
                "interpreter plus execution provider in the worker process; "
                "measured 270 MiB for a VitisAI session on a 2 MiB graph and "
                "369 MiB for a DirectML session on the 27B vision tower. "
                "NOTE: on an integrated GPU the device allocations come out of "
                "this same system RAM and do not appear in the worker's working "
                "set, so the measured figure is a lower bound for this pool, "
                "not a total"
                + (
                    f"; includes 10% headroom on measured worker peak "
                    f"{worker_peak_rss_hint_bytes} bytes after subtracting graph file bytes"
                    if worker_peak_rss_hint_bytes is not None else ""
                )
            ),
        ),
        MemoryReservation(
            pool=pool,
            purpose="worker workspace",
            bytes=WORKER_WORKSPACE_BYTES,
            lifetime="request",
            reclaimable="on_release",
            evidence="E",
            note="activations and EP scratch; not separable by any counter here",
        ),
        MemoryReservation(
            pool=pool,
            purpose="external reserve",
            bytes=EXTERNAL_RESERVE_BYTES,
            lifetime="session",
            reclaimable="no",
            evidence="E",
            note="host RAM this engine does not control; the Windows side included",
        ),
    ]


def host_pool_capacity(devices: list[DeviceCapability]) -> int:
    """Bytes available on the shared host pool, taken from the row that owns it.

    The iGPU and the NPU report ``memory_bytes == 0`` on purpose: all three
    devices allocate out of the CPU's RAM, and carrying the number on each row
    would let a caller add three copies of the same memory together. So the
    capacity a stage is checked against is the host row's, which is also what
    the CPU stage is being budgeted against at the same time.
    """
    for device in devices:
        if device.memory_pool == "host_ram" and device.memory_bytes:
            available = int((device.extra or {}).get("ram_available_bytes") or 0)
            return available or device.memory_bytes
    for device in devices:
        hinted = int((device.extra or {}).get("host_pool_available_bytes") or 0)
        if hinted:
            return hinted
    return 0


def plan_external_stage(
    artifact: GraphArtifact,
    devices: list[DeviceCapability],
    *,
    prefer: str | None = None,
    require: str | None = None,
    min_fraction_on_target: float = MIN_FRACTION_ON_TARGET,
    pool_capacity_bytes: int | None = None,
    worker_peak_rss_hint_bytes: int | None = None,
) -> StagePlan:
    """Choose a device for one exported graph, or refuse with reasons.

    ``prefer`` names a ``device_id`` to try first and falls through to the
    others if it is refused. ``require`` names the only one that may be used,
    so a refusal is the answer rather than a quieter substitution -- which is
    what you want when the question is "does the NPU run this", not "does
    anything run this". Answering the first question with the iGPU would be the
    silent-fallback mistake this module exists to prevent, one level up.

    Every device that was rejected leaves a refusal behind even when a later
    one succeeds, so the record says what was considered, not only what won.
    """
    if worker_peak_rss_hint_bytes is not None:
        if worker_peak_rss_hint_bytes <= 0:
            raise ValueError("worker peak RSS hint must be positive")
        if require is None:
            raise ValueError("worker peak RSS hint is device-specific; require a device")
    plan = StagePlan(
        artifact=artifact, device=None, route=None,
        min_fraction_on_target=min_fraction_on_target,
        worker_peak_rss_hint_bytes=worker_peak_rss_hint_bytes,
    )

    candidates = [d for d in devices if d.kind in ("gpu_integrated", "npu")]
    if require:
        candidates = [d for d in candidates if d.device_id == require]
        if not candidates:
            plan.refusals.append(
                Refusal(
                    REFUSE_NO_DEVICE,
                    require,
                    f"{require!r} was required and is not among the enumerated "
                    f"external devices ({[d.device_id for d in devices]})",
                    "run `python -m vllm_omni.edge.local devices` to see what this "
                    "process can reach, and why anything missing is missing",
                )
            )
            return plan
    elif prefer:
        candidates.sort(key=lambda d: d.device_id != prefer)
    if not candidates:
        plan.refusals.append(
            Refusal(
                REFUSE_NO_DEVICE,
                "-",
                "no integrated GPU or NPU was enumerated on this machine",
                "run `python -m vllm_omni.edge.local devices`; if the hardware is "
                "present but absent here, the probe could not open it and its "
                "reason field says why",
            )
        )
        return plan

    for device in candidates:
        if not device.runnable:
            plan.refusals.append(
                Refusal(
                    REFUSE_ROUTE,
                    device.device_id,
                    device.reason or "no worker route resolves from this process",
                    _route_remedy(device),
                )
            )
            continue
        if artifact.fmt not in device.weight_formats:
            plan.refusals.append(
                Refusal(
                    REFUSE_NO_ARTIFACT,
                    device.device_id,
                    f"{device.device_id} executes {sorted(device.weight_formats)}, "
                    f"and this graph is {artifact.fmt}",
                    _format_remedy(artifact, device),
                )
            )
            continue

        route_name = (device.extra or {}).get("worker_route")
        route = _launch.resolve(str(route_name)) if route_name else None
        if route is None or not route.available:
            plan.refusals.append(
                Refusal(
                    REFUSE_ROUTE,
                    device.device_id,
                    f"the route recorded for this device ({route_name!r}) no longer resolves",
                    "re-run device enumeration; an interpreter was removed between "
                    "enumeration and planning",
                )
            )
            continue

        reservations = _reservations_for(
            artifact, route, device.memory_pool, worker_peak_rss_hint_bytes
        )
        capacity = (
            pool_capacity_bytes
            if pool_capacity_bytes is not None
            else host_pool_capacity(devices)
        )
        budget = sum(r.bytes for r in reservations)
        if capacity and budget > capacity:
            plan.refusals.append(
                Refusal(
                    REFUSE_CAPACITY,
                    device.device_id,
                    f"the stage budgets {budget / MiB:.0f} MiB and the shared host "
                    f"pool has {capacity / MiB:.0f} MiB available; this is the same "
                    "pool the CPU stage is budgeted against, not a pool of its own",
                    "free host memory, or keep this stage on the device that "
                    "already holds the model so the graph is not resident twice",
                )
            )
            continue

        plan.device = device
        plan.route = route
        plan.reservations = reservations
        break

    return plan


def _route_remedy(device: DeviceCapability) -> str:
    if device.masked:
        # A masked device is unreachable *by request*. Explaining the MCDM
        # forwarding rules here would send someone after a problem they do not
        # have; the fix is one flag.
        return (
            f"drop {device.kind!r} from --mask / {MASK_ENV} to re-enable it. The "
            "device itself was not tested -- masking is how the no-accelerator "
            "deployment class is exercised without removing hardware"
        )
    routes = (device.extra or {}).get("routes") or []
    missing = [r for r in routes if not r.get("available")]
    if device.kind == "npu":
        return (
            "the NPU is not reachable from inside WSL at all -- it is an MCDM "
            "device and GPU-PV forwards display adapters only. It needs a "
            "native-Windows interpreter with onnxruntime >= 1.25 (not "
            "onnxruntime-directml, which is pinned at 1.24.4 and cannot load the "
            "VitisAI EP). Point VLLM_OMNI_EXTERNAL_PYTHON_ORT_VITISAI at one."
            + (f" Tried: {[r['reason'] for r in missing]}" if missing else "")
        )
    return (
        "the iGPU needs either a native-Windows venv with onnxruntime-directml "
        "(VLLM_OMNI_EXTERNAL_PYTHON_ORT_DML) or a WSL venv with torch-directml, "
        "which pins torch 2.4.1 and so cannot share Omni's environment "
        "(VLLM_OMNI_EXTERNAL_PYTHON_TORCH_DML)."
        + (f" Tried: {[r['reason'] for r in missing]}" if missing else "")
    )


def _format_remedy(artifact: GraphArtifact, device: DeviceCapability) -> str:
    if device.kind == "npu":
        return (
            f"export {artifact.component} as onnx:a16w8 at opset >= 21 -- 16-bit "
            "activations with 8-bit weights, via npu_ryzenai.quantization_kwargs(). "
            "A8W8, which is what quantize_static does by default, and fp32 are "
            "both declined by the XDNA2 overlays with no error and every node "
            "left on the CPU."
        )
    return (
        f"export {artifact.component} as onnx:fp16 or onnx:fp32; DirectML has no "
        "path for this graph's precision"
    )


class ExternalStage:
    """A planned stage, opened on its worker, with the placement gate enforced."""

    def __init__(self, plan: StagePlan) -> None:
        if not plan.admitted:
            raise PlacementRefused(
                plan.refusals[0]
                if plan.refusals
                else Refusal(REFUSE_NO_DEVICE, "-", "no device was selected", "see the plan")
            )
        self.plan = plan
        assert plan.route is not None
        self._worker = ExternalWorker(plan.route)
        self._open = False

    def open(
        self,
        example_inputs: dict[str, np.ndarray],
        *,
        profile_dir: str | Path | None = None,
    ) -> LoadReport:
        """Start the worker, load the graph, and verify where it actually ran.

        ``example_inputs`` are required, not optional: ORT writes node
        assignments only for nodes that have executed, so without a run there
        is no placement to check and the gate would be passing on faith.
        """
        if not example_inputs:
            raise ValueError(
                "example_inputs are required: placement is verified from a "
                "profiled run, and without one the node assignments do not exist"
            )
        self._worker.start()
        self._open = True
        prefix = Path(profile_dir) / f"{self.plan.artifact.component}_" if profile_dir else None
        report = self._worker.load(
            self.plan.artifact.path, example_inputs=example_inputs, profile_prefix=prefix
        )
        self.plan.report = report

        refusal = verify_placement(self.plan, report)
        if refusal is not None:
            self.plan.refusals.append(refusal)
            self.close()
            raise PlacementRefused(refusal, report)

        # A settled RSS can be much smaller than VitisAI's compilation/load
        # peak. Check the worker's own high-water mark before admitting runs.
        try:
            stats = self._worker.stats()
        except BaseException:
            self.close()
            raise
        peak = int(stats.get("peak_rss_bytes") or 0)
        self.plan.observed_worker_peak_rss_bytes = peak or None
        runtime = next(r for r in self.plan.reservations if r.purpose == "worker runtime")
        graph = next(r for r in self.plan.reservations if r.purpose == "graph weights")
        if peak:
            runtime.actual_upper_bound_bytes = max(0, peak - graph.bytes)
        if not peak or peak > runtime.bytes + graph.bytes:
            refusal = Refusal(
                REFUSE_CAPACITY,
                self.plan.device.device_id if self.plan.device else "-",
                (
                    f"worker load peak {peak / MiB:.0f} MiB exceeds graph+runtime "
                    f"reservation {(runtime.bytes + graph.bytes) / MiB:.0f} MiB"
                    if peak else "worker load peak RSS was not reported"
                ),
                "record a device/EP-specific load peak, reserve it with "
                "worker_peak_rss_hint_bytes, and recheck shared host RAM before launch",
            )
            self.plan.refusals.append(refusal)
            self.close()
            raise PlacementRefused(refusal, report)

        # The measured peak is now bounded by the pre-load plan.
        return report

    def run(self, inputs: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], RunTiming]:
        if not self._open:
            raise RuntimeError("stage is not open; call open() first")
        return self._worker.run(inputs)

    def stats(self) -> dict[str, Any]:
        return self._worker.stats()

    def close(self) -> None:
        self._open = False
        self._worker.close()

    def __enter__(self) -> ExternalStage:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def verify_placement(plan: StagePlan, report: LoadReport) -> Refusal | None:
    """The gate. ``None`` means the device really ran the graph.

    Two failures, deliberately not merged: a graph the EP *declined* and a
    graph whose placement was never *measured* are different situations with
    different remedies, and collapsing the second into the first would let an
    unverified run be reported as a verified one.
    """
    device_id = plan.device.device_id if plan.device else "-"
    if report.requested_provider_missing:
        # The most precise form of the failure, so it is reported before the
        # node count -- "the provider never initialised" is actionable, while
        # "0 of 1171 nodes" leaves the reader to guess whether the graph was
        # declined op by op or the EP was never there at all.
        return Refusal(
            REFUSE_EP_PLACEMENT,
            device_id,
            report.fallback_note or f"{report.requested_provider_missing} did not initialise",
            _placement_remedy(plan, report),
        )
    if report.fraction_on_target is None:
        return Refusal(
            REFUSE_EP_PLACEMENT,
            device_id,
            "placement was not measured: no profiled run produced node "
            f"assignments ({report.note or 'no run'})",
            "call open() with example inputs so the graph executes once under "
            "the ORT profiler; an unverified placement is refused rather than "
            "assumed, because a session that ran entirely on the CPU looks "
            "identical from every cheaper angle",
        )
    if report.fraction_on_target < plan.min_fraction_on_target:
        split = ", ".join(f"{k}={v}" for k, v in sorted(report.node_counts.items()))
        return Refusal(
            REFUSE_EP_PLACEMENT,
            device_id,
            f"{report.ep} took {report.target_nodes} of {report.total_nodes} nodes "
            f"({report.fraction_on_target:.0%}, below the {plan.min_fraction_on_target:.0%} "
            f"required); the split was {split}",
            _placement_remedy(plan, report),
        )
    return None


def _placement_remedy(plan: StagePlan, report: LoadReport) -> str:
    if plan.device is not None and plan.device.kind == "npu":
        return (
            "the graph is not in a shape XDNA2 partitions. Check, in this order: "
            "the quantization is A16W8 (QUInt16 activations, QInt8 weights) at "
            "opset >= 21; the EP loaded is onnxruntime_vitisai_ep.dll and not the "
            "RyzenAI *Light* provider beside it, which registers cleanly and "
            "claims zero nodes from every graph; and that the graph is not mostly "
            "requantization, since the partitioner leaves dequantize nodes on the "
            "CPU by its own account. Running it anyway would be a CPU fallback "
            "reported as NPU execution."
        )
    return (
        "DirectML declined most of this graph. Check for ops with no DML kernel "
        "(integer Gather in an RVQ codebook lookup is the one this project has "
        "hit) and consider exporting the variant that takes the embedding as an "
        "input instead."
    )
