# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Decide where a text session runs, and be able to say why -- before loading.

This is the M0 deliverable the proposal calls "一个可审计的 Omni 本地文本模式":
a plan that names the device, the backend, every byte it intends to reserve and
the evidence class of each claim, and that **refuses explicitly** when no
combination works. Refusing is a first-class outcome here, not an exception on
the way to one.

Two refusals are worth naming because they are the reason this module exists
rather than a ``try: load()``:

*The format refusal.* ``models/Spark-X2.5-1.7B-int8`` is 1.9 GB and the card
has 24 GiB, so every capacity check passes. It still cannot run: cutlass has no
int8 ``scaled_mm`` for SM >= 100 and vLLM's dispatch raises in the first
forward. Without this module you find out after ~40 s of engine startup, from a
stack trace in a subprocess. With it you find out in 30 ms, from a sentence
that names the remedy.

*The capacity refusal.* ``Spark-X2.5-4B`` is 7.66 GiB of weights, and the
budget it needs is not 7.66 GiB -- it is weights plus the KV cache for the
context you asked for plus activations plus the loader's transient plus a
margin. Sizing that from ``free VRAM`` at load time is what the proposal's
section 6.2 calls out: the number to check is the sum over the session's
lifetime, formed before anything allocates.

**What is measured and what is arithmetic.** The weight bytes are read off the
files (D). The KV/state bytes come from
:func:`vllm_omni.edge.kv_budget.kv_budget_for`, which reads the real
``layer_types`` rather than charging every layer full length (E, exact
arithmetic on a documented layout). Activations, workspace and the load
transient are **estimates** (E) from the constants below, and they are
deliberately generous: this module's job is to be a safe upper bound that
refuses before an OOM, not to predict the peak.

The estimates are checked rather than believed:
:meth:`vllm_omni.edge.local.engine.LocalTextEngine.measured_peak` reports the
budgeted peak against the sampled one after every run, so a budget that is
wildly off is visible in the record instead of surviving as a constant nobody
re-examined. On Spark-X2.5-4B at a 4096 context this budget is 12.51 GiB
against a measured 8.93 GiB attributable peak -- 1.40x, i.e. safe and loose,
which is the intended direction.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from vllm_omni.edge.hardware_probe import HardwareProfile, load_profile
from vllm_omni.edge.kv_budget import KVBudget, kv_budget_for, text_config
from vllm_omni.edge.local.capabilities import (
    DeviceCapability,
    enumerate_devices,
    parse_mask,
    runs_exported_graphs,
)
from vllm_omni.edge.local.manifest import ArtifactManifest, build_manifest

GiB = 2**30
MiB = 2**20

# ---- estimate constants ----------------------------------------------------
# Kept as module constants rather than env vars, per project policy, and named
# so a plan's arithmetic can be read off the output.

ACTIVATION_BYTES_PER_TOKEN_PER_HIDDEN = 2
"""bf16 activations. A prefill chunk's peak is dominated by a handful of
``[tokens, hidden]`` tensors live at once inside a layer."""

ACTIVATION_LIVE_TENSORS = 12
"""How many ``[tokens, hidden]``-sized tensors to charge for concurrently.
Generous: a gated MLP with a 4x intermediate is the widest point, and 12 covers
it with room for the attention workspace. E."""

LOGITS_BYTES_PER_TOKEN_PER_VOCAB = 4
"""The logits tensor is fp32 in vLLM's sampler. At Spark's 131072 vocab this is
512 KiB per sampled row, which is small per request and stops being small if a
plan ever batches prefill logits."""

BACKEND_WORKSPACE_BYTES = 512 * MiB
"""Attention backend scratch, NCCL/driver context, allocator slack. E."""

GRAPH_CAPTURE_BYTES = 2 * GiB
"""Charged only when CUDA graphs are enabled. The same conservative constant
:func:`vllm_omni.engine.stage_admission.graph_reserve_bytes` uses, kept equal on
purpose so a local plan and a multi-stage admission do not disagree about the
same pool. M0 runs eager, so this is 0 in practice."""

LOAD_TRANSIENT_FRACTION = 0.10
"""Peak above the resident weights while loading, as a fraction of them.

Safetensors is mmap'd and copied shard by shard, so the transient tracks one
shard rather than the whole checkpoint. This fraction is the floor;
:func:`load_transient_bytes` takes the larger of it and one average shard,
which is what covers a single-file checkpoint. E."""

SAFETY_MARGIN_BYTES = 1 * GiB
"""Fragmentation and allocator slack. Matches
``stage_admission._DEFAULT_SAFETY_MARGIN_BYTES``."""

EXTERNAL_RESERVE_BYTES = 1 * GiB
"""Memory this engine does not control: the compositor on a VRAM pool, every
other process on a host_ram pool. Matches
``stage_admission._DEFAULT_EXTERNAL_RESERVE_BYTES``."""


# ---- refusals --------------------------------------------------------------

REFUSE_NO_DEVICE = "no_runnable_device"
REFUSE_FORMAT = "weight_format_unsupported"
REFUSE_CAPACITY = "budget_exceeds_device"
REFUSE_CONFIG = "config_unreadable"

# External-stage refusals (M3). A stage placed on the 890M or the NPU can fail
# for three reasons a checkpoint-on-vLLM plan never has to name.
REFUSE_NO_ARTIFACT = "no_exported_graph"
"""The device runs exported graphs, and no export exists for this stage at
this precision. Distinct from "unsupported": the work is an export, not a port."""
REFUSE_EP_PLACEMENT = "execution_provider_declined_graph"
"""The session opened, ran, and left the graph on the CPU. This is the refusal
that stops a CPU fallback being reported as NPU execution -- and it cannot be
replaced by a cheaper check, because a VitisAI session lists the EP in
``get_providers()`` whether or not it claimed a single node, and when it claims
none the outputs are bit-identical to the CPU's."""
REFUSE_NUMERICS = "numeric_gate_failed"
"""The graph ran on the device and disagreed with the reference beyond
tolerance. Constrains *this artifact*, never the model or the precision in
general -- see AGENTS.md on quantization failures."""
REFUSE_ROUTE = "device_unreachable_from_this_process"
"""The device exists and works, and no interpreter here can drive it: the NPU
from inside WSL, or a route whose venv is not installed."""

MIN_FRACTION_ON_TARGET = 0.5
"""How much of a graph the target EP must actually take for the placement to
count. Not tuned -- chosen as "most of it", because the failure this guards is
bimodal: XDNA2 either partitions a graph or declines it wholesale, and the
observed decline is 0 nodes, not a poor split. Callers override it per stage;
what must not happen is accepting an unverified placement by default."""


@dataclass(frozen=True)
class Refusal:
    """A named reason a device was not chosen, with the action that fixes it."""

    code: str
    device_id: str
    message: str
    remedy: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryReservation:
    """One claim on one pool, budgeted before the load and checked after it.

    ``actual_upper_bound_bytes`` starts ``None`` and stays ``None`` for
    anything the runtime does not bound. Most reservations never get one: this
    host has no counter that isolates a KV pool from an activation buffer, and
    a reservation whose measurement is unknown is reported as unknown rather
    than backfilled with its own estimate. The plan-level check that does
    happen every run is ``LocalTextEngine.measured_peak``.
    """

    pool: str
    """``vram`` or ``host_ram``. Two reservations in the same pool add up;
    reservations in different pools do not."""
    purpose: str
    bytes: int
    lifetime: str
    """``load`` (freed once weights are resident), ``session`` (held until the
    engine closes) or ``request`` (held per in-flight request)."""
    reclaimable: str
    """``no``, ``on_evict`` (a rebuildable cache) or ``on_release``."""
    evidence: str
    """``D`` read from the artifact, ``E`` estimated."""
    note: str = ""
    actual_upper_bound_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionPlan:
    """Everything needed to start the session, and to explain the choice."""

    manifest: ArtifactManifest
    devices: list[DeviceCapability]
    selected: DeviceCapability | None
    backend: str | None
    stage: dict[str, Any]
    engine_kwargs: dict[str, Any]
    reservations: list[MemoryReservation]
    kv_budget: KVBudget | None
    refusals: list[Refusal]
    fallbacks: list[str]
    admitted: bool
    request_limits: dict[str, int]
    notes: list[str] = field(default_factory=list)

    # -- accounting ---------------------------------------------------------
    def reserved_bytes(self, pool: str, *, lifetimes: tuple[str, ...] | None = None) -> int:
        return sum(
            r.bytes
            for r in self.reservations
            if r.pool == pool and (lifetimes is None or r.lifetime in lifetimes)
        )

    @property
    def peak_bytes(self) -> int:
        """The high-water mark on the selected device's pool.

        Load-lifetime reservations and the KV cache do not both peak: vLLM sizes
        the cache *after* the weights are resident and the transient is gone. So
        the peak is the larger of (load path) and (steady state), not the sum --
        which is the arithmetic ``free VRAM at load time`` gets wrong in the
        other direction.
        """
        if self.selected is None:
            return 0
        pool = self.selected.memory_pool
        load = self.reserved_bytes(pool, lifetimes=("load", "session_weights"))
        steady = self.reserved_bytes(pool)
        return max(load, steady)

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "backend": self.backend,
            "selected_device": self.selected.to_dict() if self.selected else None,
            "devices": [d.to_dict() for d in self.devices],
            "stage": self.stage,
            "engine_kwargs": self.engine_kwargs,
            "reservations": [r.to_dict() for r in self.reservations],
            "reserved_total_bytes": self.peak_bytes,
            "kv_budget": _kv_to_dict(self.kv_budget),
            "refusals": [r.to_dict() for r in self.refusals],
            "fallbacks": self.fallbacks,
            "request_limits": self.request_limits,
            "manifest": self.manifest.to_dict(),
            "notes": self.notes,
        }

    def summary(self) -> str:
        if not self.admitted:
            lines = ["REFUSED: no device in this process can run this checkpoint as asked."]
            for r in self.refusals:
                lines.append(f"  [{r.code}] {r.device_id}: {r.message}")
                lines.append(f"      remedy: {r.remedy}")
            return "\n".join(lines)
        assert self.selected is not None
        pool = self.selected.memory_pool
        capacity = _device_capacity(self.selected)
        capacity_note = "available" if pool == "host_ram" else "total"
        lines = [
            f"ADMITTED on {self.selected.device_id} ({self.selected.name}) via {self.backend}",
            f"  artifact  {self.manifest.artifact_id} {self.manifest.weight_format} "
            f"{self.manifest.weight_bytes / GiB:.2f}GiB",
            f"  budget    {self.peak_bytes / GiB:.2f}GiB of {capacity / GiB:.2f}GiB {pool} ({capacity_note})",
        ]
        for r in self.reservations:
            lines.append(
                f"    {r.purpose:<18} {r.bytes / MiB:9.0f} MiB  {r.pool:<9} "
                f"{r.lifetime:<15} {r.evidence}  {r.note}"
            )
        if self.kv_budget is not None:
            lines.append(f"  kv        {self.kv_budget.summary()}")
        for r in self.refusals:
            lines.append(f"  not used: [{r.code}] {r.device_id}: {r.message}")
        return "\n".join(lines)


def _kv_to_dict(kv: KVBudget | None) -> dict[str, Any] | None:
    if kv is None:
        return None
    d = asdict(kv)
    d["summary"] = kv.summary()
    d["saving_vs_flat"] = kv.saving_vs_flat
    return d


# ---- estimates -------------------------------------------------------------


def activation_bytes(hf_config: Any, *, max_num_batched_tokens: int, max_num_seqs: int) -> int:
    """Peak activation working set for one prefill chunk. E."""
    cfg = text_config(hf_config)
    hidden = int(getattr(cfg, "hidden_size", 0) or 0)
    vocab = int(getattr(cfg, "vocab_size", 0) or 0)
    hidden_bytes = (
        max_num_batched_tokens
        * hidden
        * ACTIVATION_BYTES_PER_TOKEN_PER_HIDDEN
        * ACTIVATION_LIVE_TENSORS
    )
    # Logits are charged per *sequence*, not per batched token: vLLM gathers
    # only the last position of each sequence before sampling.
    logits_bytes = max_num_seqs * vocab * LOGITS_BYTES_PER_TOKEN_PER_VOCAB
    return hidden_bytes + logits_bytes


def load_transient_bytes(manifest: ArtifactManifest) -> int:
    """Headroom above the resident weights during the load. E.

    Charged as the larger of a flat fraction of the checkpoint and one average
    shard, because a single-file checkpoint (Spark-1.7B: one 1.9 GB
    ``model.safetensors``) has a transient the fraction alone understates.
    """
    n_shards = max(len(manifest.weight_files), 1)
    per_shard = manifest.weight_bytes // n_shards
    return int(max(manifest.weight_bytes * LOAD_TRANSIENT_FRACTION, per_shard))


# ---- planning --------------------------------------------------------------


def _load_hf_config(manifest: ArtifactManifest) -> Any:
    from transformers import AutoConfig

    # Spark's config class is vendored precisely so this does not need
    # ``trust_remote_code``; registration happens on the omni import path.
    from vllm_omni.engine.arg_utils import _register_omni_hf_configs

    _register_omni_hf_configs()
    return AutoConfig.from_pretrained(manifest.model_dir, trust_remote_code=False)


def _device_capacity(device: DeviceCapability) -> int:
    """Bytes this engine may plan against on the device's pool.

    On ``host_ram`` that is *available*, not total: the OS and everything else
    already hold the difference, and the proposal is explicit that Windows RAM,
    the WSL quota and iGPU/NPU shared memory are separate constraints that must
    not be added together.
    """
    if device.memory_pool == "host_ram":
        available = int(device.extra.get("ram_available_bytes") or 0)
        return available or device.memory_bytes
    return device.memory_bytes


def _candidate_order(devices: list[DeviceCapability]) -> list[DeviceCapability]:
    """Runnable devices, best first.

    "Best" here is only "private memory pool before shared one", which on this
    class of machine means the discrete GPU before the CPU. It is deliberately
    not a performance ranking: the proposal's rule is that placement is decided
    by profiles from measured full chains, and this module has none.
    """
    runnable = [d for d in devices if d.runnable and not runs_exported_graphs(d)]
    return sorted(runnable, key=lambda d: (d.memory_pool != "vram", -d.memory_bytes))


def plan_text_session(
    model_dir: str,
    *,
    profile: HardwareProfile | None = None,
    max_model_len: int = 4096,
    max_num_seqs: int = 1,
    max_num_batched_tokens: int | None = None,
    enforce_eager: bool = True,
    kv_dtype: str = "auto",
    mask: str | frozenset[str] | None = None,
    platform_device_type: str | None = None,
    digest_weights: bool = False,
) -> ExecutionPlan:
    """Plan one single-stage text session, or refuse with reasons.

    The caller gets a plan object either way: ``admitted`` says which, and
    ``refusals`` is never empty when it is ``False``.
    """
    manifest = build_manifest(model_dir, digest_weights=digest_weights)
    profile = profile if profile is not None else load_profile()
    devices = enumerate_devices(
        profile,
        mask=mask if isinstance(mask, frozenset) else parse_mask(mask),
        platform_device_type=platform_device_type,
    )
    batched_tokens = max_num_batched_tokens or min(2048, max_model_len)

    refusals: list[Refusal] = []
    notes: list[str] = []

    try:
        hf_config = _load_hf_config(manifest)
    except Exception as exc:
        return ExecutionPlan(
            manifest=manifest, devices=devices, selected=None, backend=None,
            stage={}, engine_kwargs={}, reservations=[], kv_budget=None,
            refusals=[Refusal(
                REFUSE_CONFIG, "-",
                f"the checkpoint config could not be read as a registered architecture: {exc}",
                "check that the architecture is in vllm_omni.model_executor.models.registry "
                "and its config class in engine.arg_utils._register_omni_hf_configs",
            )],
            fallbacks=[], admitted=False,
            request_limits={}, notes=notes,
        )

    # KV geometry is a property of the model, not the device, so it is computed
    # once and reused for every candidate.
    effective_kv_dtype = manifest.dtype or "bfloat16" if kv_dtype == "auto" else kv_dtype
    kv = kv_budget_for(
        hf_config,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        kv_dtype=effective_kv_dtype,
    )

    for device in devices:
        if not device.runnable:
            refusals.append(Refusal(
                REFUSE_NO_DEVICE, device.device_id,
                device.reason or "not runnable from this process",
                "see the reason; masked devices are re-enabled by dropping them from --mask",
            ))
        elif runs_exported_graphs(device):
            # Runnable, and still not a candidate for *this* plan. Said out
            # loud rather than left out: once these devices gained a backend
            # they stopped appearing in the refusal list at all, which reads as
            # "not considered" when the truth is "cannot host a whole session".
            refusals.append(Refusal(
                REFUSE_NO_ARTIFACT, device.device_id,
                f"{device.device_id} executes exported graphs "
                f"({sorted(device.weight_formats)}), not checkpoints, so a whole "
                "autoregressive text session cannot be placed on it",
                "place a single stage there instead, via "
                "vllm_omni.edge.local.external.stage.plan_external_stage; the "
                "decode loop stays on the CPU or the GPU because both of these "
                "devices read the same system memory it does",
            ))

    for device in _candidate_order(devices):
        if manifest.weight_format not in device.weight_formats:
            refusals.append(Refusal(
                REFUSE_FORMAT, device.device_id,
                f"the checkpoint is {manifest.weight_format} and {device.device_id} "
                f"({device.name}, sm_{''.join(map(str, device.compute_capability or ())) or '-'}) "
                f"executes only {sorted(device.weight_formats)}",
                _format_remedy(manifest, device),
            ))
            continue

        capacity = _device_capacity(device)
        reservations = _reservations_for(
            manifest, hf_config, kv, device,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=batched_tokens,
            enforce_eager=enforce_eager,
        )
        plan = ExecutionPlan(
            manifest=manifest, devices=devices, selected=device,
            backend=device.backend, stage=_stage_spec(manifest),
            engine_kwargs={}, reservations=reservations, kv_budget=kv,
            refusals=list(refusals), fallbacks=[], admitted=False,
            request_limits={
                "max_model_len": max_model_len,
                "max_num_seqs": max_num_seqs,
                "max_num_batched_tokens": batched_tokens,
            },
            notes=notes,
        )
        required = plan.peak_bytes
        if required > capacity:
            refusals.append(Refusal(
                REFUSE_CAPACITY, device.device_id,
                f"the session needs {required / GiB:.2f} GiB on {device.memory_pool} and "
                f"{capacity / GiB:.2f} GiB is available "
                f"({manifest.weight_bytes / GiB:.2f} GiB weights + "
                f"{kv.bytes_total / GiB:.2f} GiB KV at {max_model_len} tokens x {max_num_seqs})",
                _capacity_remedy(kv, max_model_len, required - capacity),
            ))
            continue

        plan.refusals = list(refusals)
        plan.engine_kwargs = _engine_kwargs(
            manifest, device, kv,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=batched_tokens,
            enforce_eager=enforce_eager,
            capacity=capacity,
            required=required,
        )
        plan.fallbacks = _fallbacks(manifest, devices, device)
        plan.admitted = True
        plan.notes = notes + [
            "activations, workspace and the load transient are estimates (E); "
            "engine.py records the measured peak against them after the load",
        ]
        return plan

    return ExecutionPlan(
        manifest=manifest, devices=devices, selected=None, backend=None,
        stage=_stage_spec(manifest), engine_kwargs={}, reservations=[],
        kv_budget=kv, refusals=refusals, fallbacks=[], admitted=False,
        request_limits={
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": batched_tokens,
        },
        notes=notes,
    )


def _stage_spec(manifest: ArtifactManifest) -> dict[str, Any]:
    """The single AR text stage, in the shape ``StagePipelineConfig`` declares.

    M0 is one model and one stage on purpose. The fields are spelled out rather
    than implied so that adding a second stage later is a change to this dict,
    not a change to the engine.
    """
    return {
        "stage_id": 0,
        "model_stage": manifest.model_type,
        "execution_type": "LLM_AR",
        "input_sources": [],
        "final_output": True,
        "final_output_type": "text",
        "owns_tokenizer": True,
        "state": "kv_cache",
        "streaming_granularity": "token",
    }


def _reservations_for(
    manifest: ArtifactManifest,
    hf_config: Any,
    kv: KVBudget,
    device: DeviceCapability,
    *,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    enforce_eager: bool,
) -> list[MemoryReservation]:
    pool = device.memory_pool
    acts = activation_bytes(
        hf_config,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )
    reservations = [
        MemoryReservation(
            pool, "weights", manifest.weight_bytes, "session_weights", "no", "D",
            f"{len(manifest.weight_files)} safetensors shard(s), on-disk size",
        ),
        MemoryReservation(
            pool, "kv_cache", kv.bytes_total, "session", "on_evict", "E",
            f"flat budget at {max_num_seqs}x context; "
            f"{kv.bytes_total_hybrid / MiB:.0f} MiB if the backend accounts "
            f"{kv.sliding_layers} sliding layers separately",
        ),
        MemoryReservation(
            pool, "activations", acts, "request", "on_release", "E",
            f"{max_num_batched_tokens}-token prefill chunk x {ACTIVATION_LIVE_TENSORS} live tensors",
        ),
        MemoryReservation(
            pool, "backend_workspace", BACKEND_WORKSPACE_BYTES, "session", "no", "E",
            "attention scratch, driver/allocator context",
        ),
        MemoryReservation(
            pool, "load_transient", load_transient_bytes(manifest), "load", "on_release", "E",
            "freed once the weights are resident; peaks with the largest shard",
        ),
        MemoryReservation(
            pool, "external_reserve", EXTERNAL_RESERVE_BYTES, "session", "no", "E",
            "memory this engine does not own on a shared pool",
        ),
        MemoryReservation(
            pool, "safety_margin", SAFETY_MARGIN_BYTES, "session", "no", "E",
            "fragmentation and allocator slack",
        ),
    ]
    if not enforce_eager:
        reservations.append(MemoryReservation(
            pool, "graph_capture", GRAPH_CAPTURE_BYTES, "session", "no", "E",
            "CUDA graph pool; zero under enforce_eager",
        ))
    return reservations


def _engine_kwargs(
    manifest: ArtifactManifest,
    device: DeviceCapability,
    kv: KVBudget,
    *,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    enforce_eager: bool,
    capacity: int,
    required: int,
) -> dict[str, Any]:
    """Turn the plan into flags the backend actually has.

    The KV pool is set as an **absolute** byte count, not left to
    ``gpu_memory_utilization``. The proposal's section 6.2 records the failure
    mode this avoids: a 4096-token single-sequence run whose logs still showed
    a far larger cache allocated, because the fraction, not the request limits,
    decided the pool size.
    """
    kwargs: dict[str, Any] = {
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "enforce_eager": enforce_eager,
        "enable_prefix_caching": False,
        # Derived from this plan's own budget, not left at the default. The
        # flag is named for the GPU and is **not** GPU-only: on the CPU backend
        # vLLM reads the same field as the fraction of *host RAM* to reserve.
        # Leaving it alone there is not "no opinion", it is asking for 92% of
        # the machine -- which is how a 6.55 GiB CPU plan turned into
        # "desired CPU memory utilization (0.92, 28.43 GiB)" and refused to
        # start. Setting it on both pools is what makes the plan's number the
        # number the backend uses.
        "gpu_memory_utilization": min(0.95, math.ceil(required / capacity * 100) / 100),
        # Absolute KV pool, so the cache is the size the request limits imply
        # rather than whatever is left over inside the fraction above.
        "kv_cache_memory_bytes": int(kv.bytes_total),
    }
    if manifest.dtype:
        kwargs["dtype"] = manifest.dtype
    return kwargs


def _fallbacks(
    manifest: ArtifactManifest,
    devices: list[DeviceCapability],
    selected: DeviceCapability,
) -> list[str]:
    """Pre-approved alternatives, named so a fallback is never a surprise.

    The proposal forbids silently swapping the model, the precision or the
    context. So the list is descriptive: these are the moves an operator may
    make, not moves the engine will make on its own.
    """
    out: list[str] = []
    for d in devices:
        if d.device_id == selected.device_id or not d.runnable:
            continue
        if manifest.weight_format in d.weight_formats:
            out.append(f"{d.device_id} runs this format; re-plan with --mask {selected.kind}")
    out.append("shorter --max-model-len reduces the KV reservation linearly")
    return out


def _format_remedy(manifest: ArtifactManifest, device: DeviceCapability) -> str:
    from vllm_omni.edge.local.capabilities import FORMAT_FP8, FORMAT_INT8

    if manifest.weight_format == FORMAT_INT8 and FORMAT_FP8 in device.weight_formats:
        return (
            "this device has fp8 tensor cores but no int8 scaled_mm path "
            "(cutlass c3x has none for SM >= 100). Use an fp8 or a dense build "
            "here, or run the int8 build on the CPU backend (.venvs/omni-cpu)."
        )
    return (
        f"build or obtain a {sorted(device.weight_formats)} artifact for this device, "
        "or plan for a device that executes this format"
    )


def _capacity_remedy(kv: KVBudget, max_model_len: int, over_by: int) -> str:
    per_token = max(kv.bytes_per_token_flat, 1)
    shed_tokens = math.ceil(over_by / per_token)
    if shed_tokens < max_model_len:
        return (
            f"dropping --max-model-len to about {max_model_len - shed_tokens} frees "
            f"{over_by / MiB:.0f} MiB of KV. The engine will not do this for you: "
            "silently shortening a user's context is exactly what the design forbids."
        )
    return (
        "the weights alone do not leave room for a usable context on this device. "
        "Use a smaller or more compressed build, or a device with a larger pool."
    )
