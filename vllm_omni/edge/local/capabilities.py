# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""What this *process* can actually execute, which is not what the machine has.

:mod:`vllm_omni.edge.hardware_probe` answers "what is in the box": it
enumerates every CPU, iGPU and NPU and marks which ones a process could open.
That is the right question at install time and the wrong one at load time,
because a vLLM wheel is built for exactly one platform. The CUDA wheel in
``.venvs/omni-cuda`` cannot run a CPU stage no matter how many cores the probe
finds, and the CPU wheel in ``.venvs/omni-cpu`` cannot reach the 5090 sitting
next to it. So a device is *runnable* here only when the probe finds it **and**
``vllm.platforms.current_platform`` is the platform that drives it.

The second thing a device decides is which weight formats it can execute, and
that gate is sharper than "has enough memory". Measured on this laptop
(2026-09-15): loading ``models/Spark-X2.5-1.7B-int8`` -- a compressed-tensors
int8 W8A8 checkpoint -- onto the RTX 5090 Laptop GPU initializes cleanly and
then dies in the first forward with

    dispatch_scaled_mm, csrc/.../c3x/scaled_mm_helper.hpp:34,
    Int8 not supported on SM120. Use FP8 quantization instead, or run on
    older arch (SM < 100).

Twenty seconds and 2 GiB of weight loading to find out. The whole point of
:mod:`vllm_omni.edge.local.plan` is that this is knowable from the checkpoint's
``quantization_config`` and the device's compute capability before anything is
loaded, so the engine refuses with that sentence instead of crashing with it.

Evidence markers follow the proposal's scale: ``D`` detected, ``B`` the backend
executed something here, ``E`` estimated. Nothing in this module claims ``C``
or ``P``; it describes devices, not models.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from vllm_omni.edge.hardware_probe import (
    ACCEL_GPU_DISCRETE,
    ACCEL_GPU_INTEGRATED,
    ACCEL_NPU,
    HardwareProfile,
)

GiB = 2**30

# Weight formats this engine knows how to name. These are the spellings that
# appear in a checkpoint's ``quantization_config.quant_method`` (plus
# ``dense`` for an unquantized checkpoint), not vLLM's ``--quantization``
# argument, which is a different vocabulary with overlapping words.
FORMAT_DENSE = "dense"
"""bf16/fp16/fp32 weights, no quantization_config."""
FORMAT_INT8 = "compressed-tensors:int8"
FORMAT_FP8 = "compressed-tensors:fp8"
FORMAT_GPTQ = "gptq"
FORMAT_AWQ = "awq"
FORMAT_NVFP4 = "modelopt_fp4"

# The external-stage devices do not execute checkpoints at all: they execute an
# exported graph, and the format that gates them is the graph's, not the
# checkpoint's. Kept in the same vocabulary so one ``weight_formats`` set can
# answer "can this device run this artifact" for every device on the row.
FORMAT_ONNX_FP32 = "onnx:fp32"
FORMAT_ONNX_FP16 = "onnx:fp16"
FORMAT_PT_FP32 = "pt:fp32"
FORMAT_PT_FP16 = "pt:fp16"
"""A TorchScript or ``torch.export`` artifact. The 890M has two routes and they
do not eat the same thing: onnxruntime-directml on Windows takes ONNX, while
torch-directml inside WSL takes a torch export -- which is also what
``vllm_omni.edge.decoder_export`` already produces. The device's format set
therefore depends on which route resolved, rather than promising both."""

FORMAT_ONNX_A16W8 = "onnx:a16w8"
"""16-bit activations, 8-bit weights. The **only** door on XDNA2: the overlays
shipped in the EP are named ``8x4_psu_model_a16w8_qdq.xclbin`` and friends, and
A8W8 -- which is what ``quantize_static`` produces by default -- is rejected as
completely as fp32, with no error and every node quietly on the CPU. See
:mod:`vllm_omni.edge.npu_ryzenai`."""

MASK_ENV = "VLLM_OMNI_LOCAL_MASK"
"""Comma-separated device kinds to hide, e.g. ``gpu_discrete``. The
capability-masked start of the M0 acceptance run uses this; it exists so the
"no NVIDIA" deployment class can be exercised without physically removing the
card, and it is honest about being a mask -- ``masked=True`` is reported, the
device is never silently dropped."""


@dataclass(frozen=True)
class DeviceCapability:
    """One device, as this process can use it."""

    device_id: str
    """Stable within a run: ``cuda:0``, ``cpu``. Never a bare index -- the
    proposal's rule that 890M and the 5090 must not both be "GPU 0"."""
    kind: str
    """``gpu_discrete`` | ``cpu`` | ``gpu_integrated`` | ``npu``, from
    :mod:`~vllm_omni.edge.hardware_probe`."""
    vendor: str
    name: str
    backend: str | None
    """The runtime that would execute a stage here, e.g. ``vllm:cuda``,
    ``vllm:cpu``. ``None`` when no backend in this process drives it."""
    runnable: bool
    """Probe found it *and* a backend in this process can drive it."""
    memory_pool: str
    """``vram`` (private) or ``host_ram`` (shared with the OS and every other
    device on this row). Two devices sharing ``host_ram`` must never be given
    two independent budgets."""
    memory_bytes: int
    compute_capability: tuple[int, ...] | None
    weight_formats: frozenset[str]
    """Formats this device can *execute*. Not "can load": see the module
    docstring."""
    evidence: str
    """``D``, ``B`` or ``E`` for the runnable claim."""
    reason: str = ""
    """Why it is not runnable, or how it is limited. The actionable half."""
    masked: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["weight_formats"] = sorted(self.weight_formats)
        d["compute_capability"] = list(self.compute_capability) if self.compute_capability else None
        return d


def cuda_weight_formats(capability: tuple[int, ...] | None) -> frozenset[str]:
    """Formats a CUDA device of this compute capability can execute.

    The int8 exclusion is the one that is not arithmetic: cutlass' ``c3x``
    ``scaled_mm`` has no int8 path for SM >= 100, and vLLM's dispatch says so
    by raising. Blackwell consumer parts (SM 120) are inside that range, so an
    int8 W8A8 checkpoint is a *load-time refusal* on this laptop even though
    it is 2 GiB and the card has 24. B: observed on sm_120, 2026-09-15.
    """
    formats = {FORMAT_DENSE, FORMAT_GPTQ, FORMAT_AWQ}
    if capability is None:
        return frozenset(formats)
    major = capability[0]
    if major >= 9:  # Hopper and later carry the fp8 tensor cores
        formats.add(FORMAT_FP8)
    if major >= 10:  # Blackwell: NVFP4, and no int8 scaled_mm
        formats.add(FORMAT_NVFP4)
    else:
        formats.add(FORMAT_INT8)
    return frozenset(formats)


def cpu_weight_formats(profile: HardwareProfile) -> frozenset[str]:
    """Formats the vLLM CPU platform can execute on this CPU.

    GPTQ is listed for x86 only because that is where the 4-bit CPU kernels
    live (``int4_scaled_mm_cpu`` with AMX, tinygemm W4A16 without); see
    :mod:`vllm_omni.edge.weight_path`, which picks *between* them. E on
    aarch64: no 4-bit CPU path has been measured there in this repo.
    """
    formats = {FORMAT_DENSE, FORMAT_INT8}
    if profile.arch == "x86_64":
        formats.add(FORMAT_GPTQ)
    return frozenset(formats)


def _current_platform_device_type() -> str | None:
    """What ``vllm`` in *this* interpreter will actually execute on.

    Imported lazily and defensively: the probe has to work in a process with no
    CUDA context (that is why ``vllm_omni.__getattr__`` defers ``Omni``), and a
    platform import failure must degrade to "no backend", not to a traceback.
    """
    try:
        from vllm.platforms import current_platform
    except Exception:  # pragma: no cover - vllm always present in practice
        return None
    device_type = getattr(current_platform, "device_type", None)
    return str(device_type) if device_type else None


def parse_mask(mask: str | None = None) -> frozenset[str]:
    """Device kinds to hide, from an explicit argument or :data:`MASK_ENV`."""
    raw = mask if mask is not None else os.environ.get(MASK_ENV, "")
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def enumerate_devices(
    profile: HardwareProfile,
    *,
    mask: frozenset[str] | str | None = None,
    platform_device_type: str | None = None,
) -> list[DeviceCapability]:
    """Every device the probe found, annotated with what this process can do.

    ``platform_device_type`` is injectable so the CUDA-visible and
    capability-masked plans of the M0 acceptance run can both be produced from
    one snapshot, and so tests do not need a GPU.
    """
    masked_kinds = mask if isinstance(mask, frozenset) else parse_mask(mask)
    platform = platform_device_type if platform_device_type is not None else _current_platform_device_type()
    devices: list[DeviceCapability] = []

    for accel in profile.accelerators:
        kind = str(accel.get("kind", ""))
        if kind == ACCEL_GPU_DISCRETE and accel.get("vendor") == "nvidia":
            devices.append(_cuda_device(profile, accel, masked_kinds, platform))
        elif kind in (ACCEL_GPU_INTEGRATED, ACCEL_NPU) and accel.get("vendor") == "amd":
            devices.append(_amd_external_device(profile, accel, kind, masked_kinds))
        elif kind in (ACCEL_GPU_INTEGRATED, ACCEL_NPU):
            devices.append(_unsupported_accelerator(accel, kind, masked_kinds))

    devices.append(_cpu_device(profile, masked_kinds, platform))
    return devices


def _cuda_device(
    profile: HardwareProfile,
    accel: dict[str, Any],
    masked: frozenset[str],
    platform: str | None,
) -> DeviceCapability:
    capability = tuple(profile.gpu_capability) if profile.gpu_capability else None
    is_masked = ACCEL_GPU_DISCRETE in masked
    probe_usable = bool(accel.get("usable"))
    platform_ok = platform == "cuda"
    runnable = probe_usable and platform_ok and not is_masked

    if is_masked:
        reason = f"masked out by {MASK_ENV}/--mask ({ACCEL_GPU_DISCRETE}); the no-NVIDIA deployment class"
    elif not probe_usable:
        reason = str(accel.get("reason") or "the probe could not open this device from this process")
    elif not platform_ok:
        reason = (
            f"vllm.platforms.current_platform is {platform!r}, not 'cuda': this "
            "interpreter's vLLM wheel does not drive the card. Use the CUDA venv "
            "(.venvs/omni-cuda)."
        )
    else:
        reason = ""

    return DeviceCapability(
        device_id="cuda:0",
        kind=ACCEL_GPU_DISCRETE,
        vendor="nvidia",
        name=profile.gpu_name or str(accel.get("name", "")),
        backend="vllm:cuda" if runnable else None,
        runnable=runnable,
        memory_pool="vram",
        memory_bytes=int(profile.gpu_mem_bytes or accel.get("mem_bytes") or 0),
        compute_capability=capability,
        weight_formats=cuda_weight_formats(capability),
        evidence="B" if runnable else "D",
        reason=reason,
        masked=is_masked,
        extra={"sm_count": profile.gpu_sm_count, "source": accel.get("source")},
    )


def _cpu_device(profile: HardwareProfile, masked: frozenset[str], platform: str | None) -> DeviceCapability:
    is_masked = "cpu" in masked
    platform_ok = platform == "cpu"
    runnable = platform_ok and not is_masked
    if is_masked:
        reason = f"masked out by {MASK_ENV}/--mask (cpu)"
    elif not platform_ok:
        reason = (
            f"vllm.platforms.current_platform is {platform!r}, not 'cpu': this "
            "interpreter's vLLM wheel has no CPU executor. The CPU build lives "
            "in a separate venv (.venvs/omni-cpu); a CPU plan from here is a "
            "plan for that interpreter, not something this one can run."
        )
    else:
        reason = ""
    return DeviceCapability(
        device_id="cpu",
        kind="cpu",
        vendor="amd" if "amd" in profile.cpu_model.lower() else "cpu",
        name=profile.cpu_model,
        backend="vllm:cpu" if runnable else None,
        runnable=runnable,
        memory_pool="host_ram",
        memory_bytes=int(profile.ram_total_bytes),
        compute_capability=None,
        weight_formats=cpu_weight_formats(profile),
        evidence="B" if runnable else "D",
        reason=reason,
        masked=is_masked,
        extra={
            "n_physical": profile.n_physical,
            "n_logical": profile.n_logical,
            "ram_available_bytes": profile.ram_available_bytes,
            "bf16": profile.supports_bf16,
        },
    )


def _amd_external_device(
    profile: HardwareProfile,
    accel: dict[str, Any],
    kind: str,
    masked: frozenset[str],
) -> DeviceCapability:
    """The Radeon 890M or the XDNA2 NPU, reached through an external worker.

    Neither runs under vLLM, and neither can run in *this* interpreter:

    * the NPU is an MCDM device. WSL2's GPU-PV forwards WDDM display adapters
      only, so there is no ``/dev/accel`` inside WSL, and VitisAI is a Windows
      DLL besides.
    * the 890M is reachable from WSL through DirectML, but ``torch-directml``
      pins torch 2.4.1 while Omni runs 2.13.

    So ``runnable`` here means "a worker route resolves", not "this process can
    open the device" -- which is why the route is recorded in ``extra`` and the
    reason names the missing interpreter when there is one. Being explicit
    about that is the point: "the NPU is unusable" and "the NPU's venv is not
    installed" call for different work.

    Both draw from ``host_ram``. They must never be given budgets of their own:
    an iGPU allocation and a CPU stage's weights come out of the same pool that
    the Windows host and the WSL quota are already splitting.
    """
    from vllm_omni.edge.local.external import launch as _launch

    is_npu = kind == ACCEL_NPU
    # Same ``kind:vendor`` spelling ``_unsupported_accelerator`` uses, so a
    # device does not change identity the day it gains a backend -- these ids
    # are in the M0 records already.
    device_id = f"{kind}:{accel.get('vendor', 'amd')}"
    # Ordered by preference. The iGPU has two routes and they are not equal --
    # DirectML-under-WSL measured *slower than the CPU it shares memory with*
    # (1.23 TFLOP/s fp32 against 1.54, 5-7.4 GB/s against 20.7) because the
    # D3D12 translation layer costs more than the iGPU wins, so the native
    # route is tried first and M3.1 measures both.
    candidates = (
        (_launch.ROUTE_VITISAI,) if is_npu else (_launch.ROUTE_DML, _launch.ROUTE_TORCH_DML)
    )
    routes = [_launch.resolve(name) for name in candidates]
    chosen = next((r for r in routes if r.available), None)

    is_masked = kind in masked
    runnable = chosen is not None and not is_masked

    if is_masked:
        reason = f"masked out by {MASK_ENV}/--mask ({kind})"
    elif chosen is None:
        reason = "; ".join(f"{r.name}: {r.reason}" for r in routes)
    else:
        reason = str(accel.get("reason") or "")

    if is_npu:
        # A16W8 and nothing else. Measured, not a policy: A8W8 -- the default
        # shape of ``quantize_static`` -- and fp32 are both rejected by the
        # overlays with no error at all, leaving every node on the CPU while
        # the session still reports the EP. See :mod:`vllm_omni.edge.npu_ryzenai`.
        formats = frozenset({FORMAT_ONNX_A16W8})
    elif chosen is not None and chosen.name == _launch.ROUTE_TORCH_DML:
        formats = frozenset({FORMAT_PT_FP16, FORMAT_PT_FP32})
    elif chosen is not None:
        formats = frozenset({FORMAT_ONNX_FP16, FORMAT_ONNX_FP32})
    else:
        # No route resolved, so nothing is executable here. Advertising the
        # union of what *some* route would take would make the format gate pass
        # for a device that cannot run anything.
        formats = frozenset()

    backend = None
    if runnable and chosen is not None:
        backend = "torch:dml" if chosen.name == _launch.ROUTE_TORCH_DML else f"ort:{chosen.ep}"

    return DeviceCapability(
        device_id=device_id,
        kind=kind,
        vendor=str(accel.get("vendor", "amd")),
        name=str(accel.get("name", "")),
        backend=backend,
        runnable=runnable,
        memory_pool="host_ram",
        # Zero, and deliberately. Section 6 forbids giving the iGPU and the NPU
        # budgets of their own: all three devices allocate out of the one pool
        # the CPU row already carries, so repeating the number here would let a
        # caller add three copies of the same RAM together. The external stage
        # planner takes its capacity from the host row for exactly this reason.
        memory_bytes=0,
        compute_capability=None,
        weight_formats=formats,
        evidence="B" if runnable else "D",
        reason=reason,
        masked=is_masked,
        extra={
            "source": accel.get("source"),
            "probe_route": accel.get("route"),
            "host_pool_bytes": int(profile.ram_total_bytes),
            "host_pool_available_bytes": int(profile.ram_available_bytes),
            "worker_route": chosen.name if chosen else None,
            "worker_interpreter": chosen.interpreter if chosen else None,
            "routes": [r.to_dict() for r in routes],
            "executes": "exported graphs, not checkpoints",
            # Dedicated VRAM, for the record. It is not a budget: 512 MiB is a
            # carve-out of the same system RAM, and DirectML grows past it into
            # the shared pool.
            "dedicated_vram_bytes": int(accel.get("mem_bytes") or 0),
        },
    )


def _unsupported_accelerator(accel: dict[str, Any], kind: str, masked: frozenset[str]) -> DeviceCapability:
    """An iGPU or NPU: found, and not a stage backend in this engine yet.

    Reported rather than dropped, because "the machine has an NPU" and "a stage
    can run on it" are different claims and the plan has to be able to say
    which one it is relying on. M3 is where this row grows a backend.
    """
    return DeviceCapability(
        device_id=f"{kind}:{accel.get('vendor', '?')}",
        kind=kind,
        vendor=str(accel.get("vendor", "")),
        name=str(accel.get("name", "")),
        backend=None,
        runnable=False,
        memory_pool="host_ram",
        memory_bytes=0,
        compute_capability=None,
        weight_formats=frozenset(),
        evidence="D",
        reason=(
            str(accel.get("reason") or "")
            or "detected, but this engine has no stage backend for it yet (roadmap M3)"
        ),
        masked=kind in masked,
        extra={"source": accel.get("source"), "route": accel.get("route")},
    )


def runs_exported_graphs(device: DeviceCapability) -> bool:
    """True for a device that executes an exported graph rather than a checkpoint.

    The distinction matters to :func:`~vllm_omni.edge.local.plan.plan_text_session`:
    the 890M and the NPU are runnable, and a whole autoregressive text session
    still cannot be placed on either. They take a stage, from
    :func:`~vllm_omni.edge.local.external.stage.plan_external_stage`.
    """
    # Keyed on the device, not on today's format set. When no worker route is
    # installed the format set is empty -- correctly, since nothing here can
    # run -- and a predicate reading that set would then answer "no", which
    # would quietly drop the device out of the planner's explanation exactly
    # when the caller most needs to be told why it was not used.
    return device.kind in (ACCEL_GPU_INTEGRATED, ACCEL_NPU)


def describe(devices: list[DeviceCapability]) -> str:
    lines = []
    for d in devices:
        mark = "runnable" if d.runnable else ("masked" if d.masked else "not here")
        mem = f"{d.memory_bytes / GiB:.1f}GiB {d.memory_pool}" if d.memory_bytes else d.memory_pool
        tail = f"\n             {d.reason}" if d.reason else ""
        lines.append(f"  [{mark:^8}] {d.device_id:<16} {d.name} ({mem}) backend={d.backend}{tail}")
    return "\n".join(lines) if lines else "  (none)"
