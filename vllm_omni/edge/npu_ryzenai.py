# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Running graphs on a Ryzen AI (XDNA2) NPU, from the Windows side.

The NPU is an **MCDM** device. WSL2's GPU-PV forwards WDDM display adapters
only, so there is no ``/dev/accel`` node and no amount of installing inside
WSL changes that -- this module is meant to be imported by a *Windows* Python
process that a WSL process talks to, not by the engine.

Three things had to be right at once, and getting any one of them wrong looks
identical from the outside: the session builds, runs, returns correct numbers,
and every node silently executes on the CPU.

**1. The right EP.** The appx package ships two.
``onnxruntime_providers_ryzenai.dll`` registers cleanly, enumerates the NPU as
``OrtHardwareDeviceType.NPU``, opens a session -- and claims **zero nodes** from
every graph tried (fp32, INT8 QDQ symmetric and asymmetric, per-channel,
power-of-two scales, MatMul and Conv, and via ahead-of-time
``ModelCompiler``, which produced zero ``EPContext`` nodes). It is the *Light*
EP. The general one is ``onnxruntime_vitisai_ep.dll``.

**2. The DLL search path.** ``onnxruntime_vitisai_ep.dll`` fails to load with a
bare ``register_execution_provider_library`` -- ORT reports ``Error loading
... But no dependency``, which is as unhelpful as it sounds. It needs its own
directory on the search path first (``os.add_dll_directory``), because it pulls
in ``dyn_dispatch_core.dll`` and ``vaiml.dll`` from beside it.

**3. A16W8, not A8W8.** The overlays are named in the DLL:

    4x2_psf_model_a8w8_qdq.xclbin
    8x4_psu_model_a16w8_qdq.xclbin     <- most of them look like this
    2x4x4_bmm_model_a16w16.xclbin

XDNA2 wants **16-bit activations with 8-bit weights**. Quantizing A8W8 -- the
default shape of ``quantize_static`` -- is rejected as completely as fp32, with
no error. Once A16W8 is used the partitioner starts talking, and names its
target: ``AMD_AIE2P_4x8_CMC_Overlay``.

Measured once all three are right, against the same INT8 graph on the CPU
(Ryzen AI 9 HX 370, 12 cores):

    GEMM                    CPU        NPU     speedup   NPU
    256 x 1024 x 1024     1.47 ms    0.56 ms    2.64x    0.97 TOPS
    512 x 2048 x 2048     9.64 ms    2.14 ms    4.51x    2.01 TOPS
    1024 x 2048 x 2048   19.89 ms    3.76 ms    5.29x    2.29 TOPS

with a max relative error of 3.2e-05 against the CPU. It scales the right way:
the bigger the GEMM, the better, because a small one cannot amortise the
dispatch. Dequantize nodes stay on the CPU -- the partitioner says so
explicitly -- so a graph that is mostly requantization will not benefit.

**What this is worth for LLM serving.** Not decode: 2.29 TOPS against the
dGPU's 48.2 TFLOP/s, and it reads the same 20.7 GB/s system memory the CPU
does, so it cannot help a bandwidth-bound step (see
``docs/edge/laptop-heterogeneous.md``). It is worth having for the things that
are compute-bound and off the per-token path -- a vision tower, a draft model
for speculative decoding -- where 5x the CPU on large GEMMs is real.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

APPX_GLOB = "WindowsWorkload.EP.AMD.VitisAI.Framework*"
WINDOWSAPPS = Path(r"C:\Program Files\WindowsApps")

EP_GENERAL = "onnxruntime_vitisai_ep.dll"
"""The EP that partitions general graphs. Not the one whose name looks right."""
EP_LIGHT = "onnxruntime_providers_ryzenai.dll"
"""Enumerates the NPU and claims nothing. Kept named so nobody re-discovers it."""

PARTITION_TARGET = "AMD_AIE2P_4x8_CMC_Overlay"
"""What the partitioner calls XDNA2 when it accepts a subgraph. Seeing this
string in the log is the only positive confirmation that the NPU took work."""


def find_ep_directory(root: Path = WINDOWSAPPS) -> Path:
    """The ``ExecutionProvider`` folder inside the appx package.

    ``WindowsApps`` denies directory listing to a normal process, so a plain
    ``glob`` returns nothing even though the files are readable by full path.
    PowerShell's ``Get-AppxPackage`` is the reliable way to get the path;
    globbing is tried first because it works when the ACL permits it.
    """
    for base in sorted(root.glob(APPX_GLOB)):
        candidate = base / "ExecutionProvider"
        if (candidate / EP_GENERAL).exists():
            return candidate
    import subprocess

    out = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command",
         f"(Get-AppxPackage -Name '{APPX_GLOB}').InstallLocation"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    if out:
        candidate = Path(out) / "ExecutionProvider"
        if (candidate / EP_GENERAL).exists():
            return candidate
    raise FileNotFoundError(
        "Ryzen AI execution provider not found. It ships as the appx package "
        f"{APPX_GLOB!r}; on a machine without it, AMD's Ryzen AI Software "
        "installs the same EP."
    )


def register(ep_dir: Path | None = None, name: str = "vitisai") -> str:
    """Put the EP on the DLL search path and register it. Returns the EP name.

    Order matters: ``add_dll_directory`` before ``register_execution_provider_library``,
    or the load fails with "no dependency".
    """
    import onnxruntime as ort

    if tuple(int(p) for p in ort.__version__.split(".")[:2]) < (1, 25):
        raise RuntimeError(
            f"onnxruntime {ort.__version__} is too old: the EP requests ORT API "
            "25, so 1.25+ is required. Note that onnxruntime-directml is pinned "
            "at 1.24.4 and cannot load it -- use plain onnxruntime."
        )
    ep_dir = Path(ep_dir) if ep_dir else find_ep_directory()
    os.environ["PATH"] = f"{ep_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    os.add_dll_directory(str(ep_dir))
    ort.register_execution_provider_library(name, str(ep_dir / EP_GENERAL))
    return name


def npu_devices(name: str = "vitisai") -> list[Any]:
    import onnxruntime as ort

    return [
        d for d in ort.get_ep_devices()
        if d.ep_name == name and str(d.device.type).endswith("NPU")
    ]


def session_options(name: str = "vitisai") -> Any:
    """SessionOptions bound to the NPU, or a clear error if it is not there."""
    import onnxruntime as ort

    devices = npu_devices(name)
    if not devices:
        raise RuntimeError(
            "no NPU device from the VitisAI EP. Check the driver is present "
            "(Get-PnpDevice -Class ComputeAccelerator) and that register() ran."
        )
    options = ort.SessionOptions()
    options.add_provider_for_devices(devices, {})
    return options


def quantization_kwargs() -> dict[str, Any]:
    """A16W8 -- what the overlays are built for.

    Pass to ``onnxruntime.quantization.quantize_static`` alongside
    ``quant_format=QuantFormat.QDQ``. The model must be opset 21 or later, so
    that 16-bit QDQ is expressible without contrib ops.
    """
    from onnxruntime.quantization import QuantType

    return {
        "activation_type": QuantType.QUInt16,
        "weight_type": QuantType.QInt8,
        "per_channel": False,
    }


def nodes_on_npu(profile_path: str | Path, name: str = "vitisai") -> int:
    """How many nodes the EP actually took, from an ORT profile.

    The only trustworthy check. A session that reports the EP in
    ``get_providers()`` may still be running every node on the CPU, and the
    outputs will be bit-identical to the CPU when it does -- which is exactly
    what a *correct* NPU run does not look like.
    """
    import json

    events = json.loads(Path(profile_path).read_text())
    return sum(
        1 for e in events
        if e.get("cat") == "Node" and (e.get("args") or {}).get("provider") == name
    )
