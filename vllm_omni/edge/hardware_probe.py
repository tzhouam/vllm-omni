"""Hardware capability probe for edge deployments.

Produces a :class:`HardwareProfile` from ``/proc``, ``/sys`` and (optionally)
torch, and maps it to a coarse *hardware class* used to select deploy overlays
(``vllm_omni/deploy/edge/hardware/<class>.yaml``) and to derive engine
overrides (see :mod:`vllm_omni.edge.adapt`).

The probe reads from an injectable filesystem root so tests can feed captured
snapshots of other devices, and ``VLLM_OMNI_HW_PROFILE=<json>`` replaces the
probe entirely (used for emulating a device class on a development host).

**Reachability, not presence (2026-09-15).** The first version of this
inventory reported the iGPU and the NPU as simply unreachable under WSL2, which
was true of the *default* setup and false as a statement about the machine.
Both turned out to have a route, and the routes are not equivalent:

* the **Radeon 890M** is reachable from inside WSL today, through DirectML over
  ``/dev/dxg`` (``torch-directml``). It computes correctly -- a 512x512 matmul
  agrees with the CPU to 3.6e-07. It is also, measured, **slower than the CPU it
  shares memory with**: 1.12 TFLOP/s fp32 against the CPU's 1.54, and 5-7.4 GB/s
  against the CPU's 20.7, because the D3D12 translation layer costs more than
  the iGPU wins. Usable, and rarely worth using through this route.
* the **XDNA2 NPU** is *not* reachable from inside WSL at all: it is an MCDM
  device (``IpuMcdmDriver``) and WSL's GPU-PV forwards WDDM display adapters
  only, so no ``/dev/accel`` node exists and DirectML enumerates 2 adapters, not
  3. From a Windows-side process it **works**, at 2.6-5.3x the CPU on GEMMs,
  once three things are right at once: the general VitisAI EP rather than the
  *Light* one beside it, its directory on the DLL search path, and A16W8
  quantization. Each of those failing looks identical -- a session that runs
  correctly with every node on the CPU. The recipe and the measurements are in
  :mod:`vllm_omni.edge.npu_ryzenai`.

So ``usable`` answers "can this process compute on it", and ``route`` says what
would have to be true. Reporting a device as absent because the default path
does not reach it is the mistake this field exists to stop.

**Accelerator inventory (2026-09-14).** ``accelerator`` is one string, chosen
for the one device vLLM will run on. That was enough while the edge target was
a phone or a single-GPU box, and it is not enough on the machines this is now
aimed at: a Ryzen AI laptop has a discrete GPU, an integrated GPU and an XDNA2
NPU at once, and which of the three a process can actually open depends on the
kernel it is running under, not on the silicon. So ``accelerators`` lists every
one that was found, each with ``usable`` and, when it is not, the ``reason``.

The reason field earns its place immediately. Under WSL2 -- which is where this
project's own laptop measurements run -- the host passes through ``/dev/dxg``
and nothing else: CUDA works, while ``/dev/dri`` and ``/dev/accel`` do not
exist, so the iGPU and the NPU are present in the machine and unreachable from
the process. Reporting them as absent would be false, and reporting them as
available would send a deploy planner after a device it cannot open. They are
reported as present, unusable, and told why.
"""

from __future__ import annotations

import json
import os
import platform
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ENV_OVERRIDE = "VLLM_OMNI_HW_PROFILE"

HW_CLASS_JETSON = "jetson"
HW_CLASS_CUDA_DISCRETE = "cuda_discrete"
HW_CLASS_ARM64_CPU = "arm64_cpu"
HW_CLASS_X86_CPU = "x86_cpu"
HW_CLASS_NPU_PHONE = "npu_phone"
HW_CLASSES = (HW_CLASS_JETSON, HW_CLASS_CUDA_DISCRETE, HW_CLASS_ARM64_CPU, HW_CLASS_X86_CPU, HW_CLASS_NPU_PHONE)

HOST_OS_LINUX = "linux"
HOST_OS_WSL2 = "wsl2"

ACCEL_GPU_DISCRETE = "gpu_discrete"
ACCEL_GPU_INTEGRATED = "gpu_integrated"
ACCEL_NPU = "npu"

# PCI vendor ids, for naming a device that has no driver bound to it.
_PCI_VENDORS = {"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel", "0x1022": "amd"}
# PCI device ids of neural accelerators that present as a plain PCI function.
_NPU_PCI_IDS = {
    ("0x1022", "0x17f0"): "AMD XDNA2 NPU (Strix Point)",
    ("0x1022", "0x1502"): "AMD XDNA NPU (Phoenix)",
    ("0x8086", "0x643e"): "Intel NPU (Meteor Lake)",
    ("0x8086", "0x7d1d"): "Intel NPU (Lunar Lake)",
}
_WSL_IGPU_ROUTE = (
    "No /dev/dri under WSL2, but reachable through DirectML over /dev/dxg: "
    "pip install torch-directml in its own venv (it pins torch==2.4.1). "
    "Measured 1.12 TFLOP/s fp32 and 5-7.4 GB/s -- below the CPU on both, so "
    "route work here only when the CPU is the thing you are trying to free."
)
_WSL_NPU_ROUTE = (
    "An MCDM device; WSL2's GPU-PV forwards WDDM display adapters only, so it "
    "has no device node here and nothing installed inside WSL reaches it. It "
    "does work from a Windows-side process -- onnxruntime>=1.25 plus the "
    "WindowsWorkload.EP.AMD.VitisAI.Framework appx, loading "
    "onnxruntime_vitisai_ep.dll (not the RyzenAI *Light* EP beside it) with "
    "its own directory on the DLL search path, against an A16W8-quantized "
    "opset-21 graph. Measured 5.3x the CPU on a 1024x2048x2048 GEMM. See "
    "vllm_omni/edge/npu_ryzenai.py."
)

# Flags we care about, normalized across x86 (``flags``) and ARM (``Features``).
_X86_FLAGS = ("avx2", "avx512f", "avx512_bf16", "avx512_vnni", "amx_bf16", "amx_tile", "amx_int8", "fma", "f16c")
_ARM_FLAGS = ("asimd", "asimdhp", "asimddp", "i8mm", "bf16", "sve", "sve2", "sme")
_FLAG_ALIASES = {"avx512f": "avx512", "asimd": "neon", "asimdhp": "fp16_arith", "asimddp": "dotprod"}


@dataclass
class HardwareProfile:
    arch: str = "unknown"  # x86_64 | aarch64 | ...
    cpu_model: str = ""
    cpu_flags: list[str] = field(default_factory=list)
    n_logical: int = 0
    n_physical: int = 0
    clusters: list[dict[str, Any]] = field(default_factory=list)  # [{"max_khz": int, "cpus": [int, ...]}], big first
    numa_nodes: int = 1
    ram_total_bytes: int = 0
    ram_available_bytes: int = 0
    accelerator: str = "none"  # none | cuda_discrete | cuda_unified | npu:<vendor>
    gpu_name: str = ""
    gpu_mem_bytes: int = 0
    gpu_sm_count: int = 0
    gpu_capability: list[int] = field(default_factory=list)
    supports_bf16: bool = False
    supports_fp16: bool = True
    thermal_zones: int = 0
    host_os: str = HOST_OS_LINUX
    accelerators: list[dict[str, Any]] = field(default_factory=list)
    """Every accelerator found, usable or not. See the module docstring."""
    source: str = "probe"

    # ---------------------------------------------------------------- helpers
    def has(self, *flags: str) -> bool:
        return all(f in self.cpu_flags for f in flags)

    @property
    def big_cpus(self) -> list[int]:
        return list(self.clusters[0]["cpus"]) if self.clusters else list(range(self.n_logical))

    def accelerators_of(self, kind: str, *, usable: bool | None = None) -> list[dict[str, Any]]:
        return [
            a
            for a in self.accelerators
            if a.get("kind") == kind and (usable is None or bool(a.get("usable")) is usable)
        ]

    @property
    def npus(self) -> list[dict[str, Any]]:
        return self.accelerators_of(ACCEL_NPU)

    @property
    def integrated_gpus(self) -> list[dict[str, Any]]:
        return self.accelerators_of(ACCEL_GPU_INTEGRATED)

    @property
    def unusable_accelerators(self) -> list[dict[str, Any]]:
        """Present in the machine, not openable from this process. The reason
        each one carries is the actionable part: usually a different kernel or
        a helper process on the host OS."""
        return [a for a in self.accelerators if not a.get("usable")]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HardwareProfile:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, text: str) -> HardwareProfile:
        return cls.from_dict(json.loads(text))


# ------------------------------------------------------------------ parsing
def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def parse_cpuinfo(text: str) -> tuple[str, list[str], int]:
    """Return (model name, normalized flags, logical cpu count) from /proc/cpuinfo."""
    model = ""
    flags: set[str] = set()
    n_logical = 0
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "processor":
            n_logical += 1
        elif key in ("model name", "hardware", "cpu model") and not model:
            model = value
        elif key in ("flags", "features"):
            for tok in value.split():
                if tok in _X86_FLAGS or tok in _ARM_FLAGS:
                    flags.add(_FLAG_ALIASES.get(tok, tok))
    return model, sorted(flags), n_logical


def parse_meminfo(text: str) -> tuple[int, int]:
    total = avail = 0
    for line in text.splitlines():
        m = re.match(r"^(MemTotal|MemAvailable):\s+(\d+)\s*kB", line)
        if m:
            val = int(m.group(2)) * 1024
            if m.group(1) == "MemTotal":
                total = val
            else:
                avail = val
    return total, avail


def parse_clusters(cpu_max_khz: dict[int, int]) -> list[dict[str, Any]]:
    """Group CPUs by max frequency (descending) -> big.LITTLE clusters."""
    groups: dict[int, list[int]] = {}
    for cpu, khz in cpu_max_khz.items():
        groups.setdefault(khz, []).append(cpu)
    return [{"max_khz": khz, "cpus": sorted(cpus)} for khz, cpus in sorted(groups.items(), key=lambda kv: -kv[0])]


def _sysfs_cpu_dirs(root: Path) -> list[Path]:
    base = root / "sys" / "devices" / "system" / "cpu"
    return sorted((p for p in base.glob("cpu[0-9]*") if p.is_dir()), key=lambda p: int(p.name[3:]))


def _physical_cores(root: Path, n_logical: int) -> int:
    ids: set[tuple[str, str]] = set()
    for cpu in _sysfs_cpu_dirs(root):
        core = _read(cpu / "topology" / "core_id")
        pkg = _read(cpu / "topology" / "physical_package_id")
        if core is not None:
            ids.add(((pkg or "0").strip(), core.strip()))
    return len(ids) if ids else n_logical


def _cpu_max_khz(root: Path) -> dict[int, int]:
    out: dict[int, int] = {}
    for cpu in _sysfs_cpu_dirs(root):
        raw = _read(cpu / "cpufreq" / "cpuinfo_max_freq")
        if raw and raw.strip().isdigit():
            out[int(cpu.name[3:])] = int(raw.strip())
    return out


def _numa_nodes(root: Path) -> int:
    base = root / "sys" / "devices" / "system" / "node"
    nodes = [p for p in base.glob("node[0-9]*")] if base.exists() else []
    return max(1, len(nodes))


def _thermal_zones(root: Path) -> int:
    base = root / "sys" / "class" / "thermal"
    return len(list(base.glob("thermal_zone*"))) if base.exists() else 0


def _is_tegra(root: Path) -> bool:
    return (
        (root / "etc" / "nv_tegra_release").exists()
        or (root / "proc" / "device-tree" / "compatible").exists()
        and "tegra" in (_read(root / "proc" / "device-tree" / "compatible") or "")
    )


def detect_host_os(root: Path) -> str:
    """``wsl2`` when this kernel is Microsoft's, else ``linux``.

    It is read from ``/proc/version`` rather than from an env var so a captured
    snapshot of another machine classifies the same way the machine would.
    """
    return HOST_OS_WSL2 if "microsoft" in (_read(root / "proc" / "version") or "").lower() else HOST_OS_LINUX


def _accel(kind: str, vendor: str, name: str, usable: bool, source: str, reason: str = "", **extra: Any) -> dict[str, Any]:
    entry = {"kind": kind, "vendor": vendor, "name": name, "usable": usable, "source": source}
    if reason:
        entry["reason"] = reason
    entry.update(extra)
    return entry


def _pci_ids(root: Path, device_dir: Path) -> tuple[str, str]:
    vendor = (_read(device_dir / "vendor") or "").strip().lower()
    device = (_read(device_dir / "device") or "").strip().lower()
    return vendor, device


def linux_accelerators(root: Path) -> list[dict[str, Any]]:
    """Accelerators this kernel actually exposes a device node for.

    Three sources, in the order of how much they prove:

    * ``/dev/dri/renderD*`` -- a GPU a process can submit to. Discrete vs
      integrated is **inferred from the PCI vendor** (NVIDIA discrete, AMD and
      Intel integrated), which is right on the laptop and edge-box parts this
      targets and wrong for a discrete Radeon or Arc. Entries say so.
    * ``/dev/accel/accel*`` -- the Linux accel subsystem, which is how both
      ``amdxdna`` (XDNA/XDNA2) and ``intel_vpu`` present an NPU.
    * a PCI sweep, which finds an NPU whose driver is not loaded. Those are
      reported unusable with the driver named, because "no /dev/accel node" and
      "no such hardware" are very different problems to be handed.
    """
    found: list[dict[str, Any]] = []
    seen_npu_slots: set[str] = set()

    for node in sorted((root / "dev" / "dri").glob("renderD*")) if (root / "dev" / "dri").exists() else []:
        dev_link = root / "sys" / "class" / "drm" / node.name / "device"
        vendor_id, _ = _pci_ids(root, dev_link)
        vendor = _PCI_VENDORS.get(vendor_id, "unknown")
        kind = ACCEL_GPU_DISCRETE if vendor == "nvidia" else ACCEL_GPU_INTEGRATED
        found.append(
            _accel(kind, vendor, f"{vendor} GPU ({node.name})", True, f"/dev/dri/{node.name}", inferred_kind=vendor != "nvidia")
        )

    for node in sorted((root / "dev" / "accel").glob("accel*")) if (root / "dev" / "accel").exists() else []:
        driver = ""
        link = root / "sys" / "class" / "accel" / node.name / "device" / "driver"
        try:
            driver = link.resolve().name
        except OSError:
            pass
        slot_dir = root / "sys" / "class" / "accel" / node.name / "device"
        vendor_id, device_id = _pci_ids(root, slot_dir)
        name = _NPU_PCI_IDS.get((vendor_id, device_id), f"NPU ({driver or node.name})")
        seen_npu_slots.add(device_id)
        found.append(
            _accel(ACCEL_NPU, _PCI_VENDORS.get(vendor_id, "unknown"), name, True, f"/dev/accel/{node.name}", driver=driver)
        )

    pci_root = root / "sys" / "bus" / "pci" / "devices"
    for slot in sorted(pci_root.glob("*")) if pci_root.exists() else []:
        vendor_id, device_id = _pci_ids(root, slot)
        name = _NPU_PCI_IDS.get((vendor_id, device_id))
        if not name or device_id in seen_npu_slots:
            continue
        bound = (slot / "driver").exists()
        found.append(
            _accel(
                ACCEL_NPU,
                _PCI_VENDORS.get(vendor_id, "unknown"),
                name,
                False,
                f"pci:{slot.name}",
                reason=(
                    "driver bound but no /dev/accel node" if bound else
                    "no driver bound: load amdxdna (Linux 6.14+) or intel_vpu"
                ),
            )
        )
    return found


def wsl_host_accelerators(timeout_s: float = 5.0) -> list[dict[str, Any]]:
    """What the *Windows* host has, asked over WSL interop.

    Costs one ``powershell.exe`` round trip (~0.7 s) and is only worth making
    on WSL, where the Linux-side probe is guaranteed to find nothing but the
    paravirtualised CUDA device. Everything returned is marked unusable: the
    point is to record that the silicon exists and name what would have to
    change to reach it, not to suggest it can be opened from here.
    """
    import subprocess

    try:
        out = subprocess.run(
            [
                "powershell.exe", "-NoProfile", "-Command",
                "Get-PnpDevice -Class ComputeAccelerator,Display -Status OK | "
                "Select-Object -ExpandProperty FriendlyName",
            ],
            capture_output=True, text=True, timeout=timeout_s, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []

    found: list[dict[str, Any]] = []
    for line in (l.strip() for l in out.splitlines()):
        if not line:
            continue
        low = line.lower()
        if "nvidia" in low:
            continue  # reachable through /dev/dxg; the torch probe owns it
        if "npu" in low or "neural" in low or "ai boost" in low:
            vendor = "amd" if ("amd" in low or "npu compute" in low) else "intel"
            found.append(_accel(
                ACCEL_NPU, vendor, line, False, "wsl-interop",
                reason=_WSL_NPU_ROUTE, route="ryzenai-ep-windows",
            ))
        elif "radeon" in low or "graphics" in low or "arc" in low:
            vendor = "amd" if "radeon" in low else "intel"
            found.append(_accel(
                ACCEL_GPU_INTEGRATED, vendor, line, directml_available(), "wsl-interop",
                reason=_WSL_IGPU_ROUTE, route="directml",
            ))
    return found


def directml_available() -> bool:
    """Whether a DirectML torch device can actually be opened from here.

    ``torch-directml`` pins ``torch==2.4.1``, so it never shares a virtualenv
    with the engine; this only reports True in a venv built for it, which is
    the honest answer to "can *this* process compute on the iGPU".
    """
    try:
        import torch_directml

        return torch_directml.device_count() > 0
    except Exception:
        return False


def _torch_gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False}
    try:
        import torch

        if not torch.cuda.is_available():
            return info
        props = torch.cuda.get_device_properties(0)
        info.update(
            available=True,
            name=props.name,
            mem_bytes=int(props.total_memory),
            sm_count=int(getattr(props, "multi_processor_count", 0)),
            capability=[int(props.major), int(props.minor)],
            bf16=bool(torch.cuda.is_bf16_supported()),
        )
    except Exception:
        return info
    return info


# -------------------------------------------------------------------- probe
def probe(root: str | Path = "/", use_torch: bool = True, probe_host: bool = True) -> HardwareProfile:
    """Probe the current machine (or a snapshot rooted at ``root``).

    ``probe_host`` allows the one WSL-interop round trip that finds the host's
    NPU and iGPU. It only fires on a live WSL probe; a snapshot root never
    shells out, so tests stay hermetic.
    """
    root = Path(root)
    model, flags, n_logical = parse_cpuinfo(_read(root / "proc" / "cpuinfo") or "")
    ram_total, ram_avail = parse_meminfo(_read(root / "proc" / "meminfo") or "")
    arch = platform.machine() if root == Path("/") else _arch_from_flags(flags, model)
    profile = HardwareProfile(
        arch=arch,
        cpu_model=model,
        cpu_flags=flags,
        n_logical=n_logical or os.cpu_count() or 1,
        n_physical=_physical_cores(root, n_logical or os.cpu_count() or 1),
        clusters=parse_clusters(_cpu_max_khz(root)),
        numa_nodes=_numa_nodes(root),
        ram_total_bytes=ram_total,
        ram_available_bytes=ram_avail,
        thermal_zones=_thermal_zones(root),
        host_os=detect_host_os(root),
    )
    gpu = _torch_gpu_info() if (use_torch and root == Path("/")) else {"available": False}
    tegra = _is_tegra(root)
    if gpu.get("available"):
        profile.accelerator = "cuda_unified" if tegra else "cuda_discrete"
        profile.gpu_name = gpu["name"]
        profile.gpu_mem_bytes = gpu["mem_bytes"]
        profile.gpu_sm_count = gpu["sm_count"]
        profile.gpu_capability = gpu["capability"]
        profile.supports_bf16 = gpu["bf16"]
    elif tegra:
        profile.accelerator = "cuda_unified"
        profile.gpu_mem_bytes = ram_total  # unified memory
        profile.supports_bf16 = True
    else:
        profile.supports_bf16 = _cpu_bf16(profile)
    profile.supports_fp16 = (
        True if profile.accelerator.startswith("cuda") else profile.has("fp16_arith") or arch == "x86_64"
    )

    inventory = linux_accelerators(root)
    if gpu.get("available"):
        # torch owns the CUDA device; drop the DRM node that is the same card.
        inventory = [a for a in inventory if a["vendor"] != "nvidia"]
        inventory.insert(
            0,
            _accel(
                ACCEL_GPU_DISCRETE, "nvidia", gpu["name"], True, "torch.cuda",
                mem_bytes=gpu["mem_bytes"],
            ),
        )
    if probe_host and root == Path("/") and profile.host_os == HOST_OS_WSL2:
        inventory += wsl_host_accelerators()
    profile.accelerators = inventory
    return profile


def _arch_from_flags(flags: list[str], model: str) -> str:
    if any(f in flags for f in ("neon", "dotprod", "sve", "i8mm", "fp16_arith")):
        return "aarch64"
    if any(f in flags for f in ("avx2", "avx512", "amx_tile", "fma")):
        return "x86_64"
    return "unknown"


def _cpu_bf16(profile: HardwareProfile) -> bool:
    if profile.arch == "x86_64":
        return profile.has("avx512_bf16") or profile.has("amx_bf16")
    if profile.arch == "aarch64":
        return profile.has("bf16")
    return False


def hardware_class(profile: HardwareProfile) -> str:
    acc = profile.accelerator
    if acc.startswith("npu"):
        return HW_CLASS_NPU_PHONE
    if acc == "cuda_unified":
        return HW_CLASS_JETSON
    if acc == "cuda_discrete":
        return HW_CLASS_CUDA_DISCRETE
    if profile.arch == "aarch64":
        return HW_CLASS_ARM64_CPU
    return HW_CLASS_X86_CPU


def load_profile(path: str | Path | None = None, use_torch: bool = True, probe_host: bool = True) -> HardwareProfile:
    """Env override (``VLLM_OMNI_HW_PROFILE``) or explicit path wins over probing."""
    override = path or os.environ.get(ENV_OVERRIDE)
    if override:
        p = HardwareProfile.from_json(Path(override).read_text(encoding="utf-8"))
        p.source = f"override:{override}"
        return p
    return probe(use_torch=use_torch, probe_host=probe_host)


def describe(profile: HardwareProfile) -> str:
    cls = hardware_class(profile)
    big = profile.big_cpus
    return (
        f"class={cls} arch={profile.arch} cpu='{profile.cpu_model}' "
        f"logical={profile.n_logical} physical={profile.n_physical} "
        f"big_cpus={len(big)} flags={','.join(profile.cpu_flags)} ram={profile.ram_total_bytes / 2**30:.1f}GiB "
        f"accel={profile.accelerator} gpu='{profile.gpu_name}' gpu_mem={profile.gpu_mem_bytes / 2**30:.1f}GiB "
        f"bf16={profile.supports_bf16} host_os={profile.host_os} source={profile.source}"
    )


def describe_accelerators(profile: HardwareProfile) -> str:
    """One line per accelerator, saying which can be opened from here."""
    if not profile.accelerators:
        return "  (none found)"
    lines = []
    for a in profile.accelerators:
        mark = "usable" if a.get("usable") else "not here"
        route = f" route={a['route']}" if a.get("route") else ""
        tail = f"\n           {a['reason']}" if a.get("reason") else ""
        lines.append(
            f"  [{mark:^8}] {a['kind']:<16} {a['vendor']:<8} {a['name']} "
            f"(via {a['source']}){route}{tail}"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - manual use
    prof = load_profile()
    print(describe(prof))
    print(describe_accelerators(prof))
    print(prof.to_json())
