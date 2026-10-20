# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Which interpreter runs a stage worker, and why one is missing.

Four routes reach the two AMD devices on this laptop, and no two of them can
share an interpreter:

==============  ==========================  ====================================
route           interpreter                 why it is separate
==============  ==========================  ====================================
``ort-vitisai`` native Windows, plain ORT   the NPU is an MCDM device; WSL's
                >= 1.25                     GPU-PV forwards WDDM adapters only,
                                            so ``/dev/accel`` does not exist
``ort-dml``     native Windows,             ``onnxruntime-directml`` is pinned
                onnxruntime-directml        at 1.24.4 and cannot load the
                                            VitisAI EP, which needs >= 1.25
``torch-dml``   WSL, ``.venvs/dml``         ``torch-directml`` pins torch 2.4.1;
                                            Omni runs 2.13
``ort-cpu``     anything with onnxruntime   the reference arm, and what the
                                            tests use on a machine with no AMD
                                            hardware
==============  ==========================  ====================================

Every lookup returns a :class:`Route` either way. A missing interpreter is a
first-class answer with a ``reason`` a person can act on, not an exception and
not silence: "the NPU is unusable" and "the NPU venv is not installed" lead to
different work, and the planner has to be able to tell the caller which it hit.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROUTE_VITISAI = "ort-vitisai"
ROUTE_DML = "ort-dml"
ROUTE_TORCH_DML = "torch-dml"
ROUTE_CPU = "ort-cpu"
ROUTE_WIN_CUDA = "win-cuda"
"""[W4] Native-Windows vLLM on the NVIDIA GPU.

The first route whose far side is **torch**, not onnxruntime, and the first
that exists to escape WSL rather than to reach a device WSL cannot see. It is
declared here so a Windows engine is a row in this table instead of a parallel
universe -- but see :data:`WIN_CUDA_PLACEMENT_EVIDENCE`: it cannot currently
pass the placement gate, and that is deliberate, not an oversight.
"""

ENV_PREFIX = "VLLM_OMNI_EXTERNAL_PYTHON_"
"""``VLLM_OMNI_EXTERNAL_PYTHON_ORT_VITISAI=...`` overrides one route's
interpreter. Route names are upper-cased with ``-`` turned into ``_``."""

WORKER_ENV_PREFIX = "VLLM_OMNI_EXTERNAL_WORKER_"
"""``VLLM_OMNI_EXTERNAL_WORKER_ORT_CPU=...`` overrides one route's worker
script. The tests point this at a fake that speaks the protocol and never
imports onnxruntime, so the client, the framing and the refusal paths are
covered on a machine with no AMD hardware and no ORT at all."""

# Where the routes live on this machine. Defaults, not requirements: the env
# override above is the supported way to point at a different install, and the
# reason string names this list when nothing is found.
_DEFAULT_CANDIDATES: dict[str, tuple[str, ...]] = {
    ROUTE_VITISAI: (
        r"/mnt/c/Users/zhout/npu-ep/Scripts/python.exe",
        r"/mnt/c/Users/zhout/ryzenai/Scripts/python.exe",
    ),
    ROUTE_DML: (
        r"/mnt/c/Users/zhout/dml-ep/Scripts/python.exe",
        r"/mnt/c/Users/zhout/directml/Scripts/python.exe",
    ),
    ROUTE_TORCH_DML: (".venvs/dml/bin/python",),
    ROUTE_CPU: (".venvs/omni-cuda/bin/python", ".venvs/omni-cpu/bin/python"),
    ROUTE_WIN_CUDA: (
        r"/mnt/c/Users/zhout/w0/w0venv/Scripts/python.exe",
        r"/mnt/c/Users/zhout/vllm-win/Scripts/python.exe",
    ),
}

_WORKER_FOR_ROUTE = {
    ROUTE_VITISAI: "worker_ort.py",
    ROUTE_DML: "worker_ort.py",
    ROUTE_CPU: "worker_ort.py",
    ROUTE_TORCH_DML: "worker_dml.py",
    ROUTE_WIN_CUDA: "worker_vllm.py",
}

_EP_FOR_ROUTE = {
    ROUTE_VITISAI: "vitisai",
    ROUTE_DML: "dml",
    ROUTE_CPU: "cpu",
    ROUTE_TORCH_DML: "dml",
    ROUTE_WIN_CUDA: "cuda",
}


WIN_CUDA_PLACEMENT_EVIDENCE: str | None = None
"""What a ``win-cuda`` worker can prove about where its work ran. Currently: nothing.

Every other route in this table earns its placement from a *record the runtime
writes while the graph runs*: onnxruntime reports per-node provider assignments,
and ``torch-directml`` reports the device of the output tensor. M3 made that a
hard admission condition after a VitisAI session listed its EP, took **zero**
nodes, and returned output bit-identical to the CPU's -- so neither the provider
list nor the numbers can tell a real device run from a silent fallback.

**vLLM has no analogue.** It does not emit a per-operation device map, and the
cheap substitutes are exactly the ones M3 proved worthless: ``current_platform``
is a *selection*, not an observation; allocated VRAM says weights are resident,
not that the decode step ran on them; and plausible output is what a silent CPU
fallback also produces.

``None`` therefore means *unverified*, which
:func:`vllm_omni.edge.local.external.stage` already refuses with
``REFUSE_EP_PLACEMENT`` -- the same answer the NPU gets when it has not been
measured. Refusing our own engine on the rule we wrote for a vendor's is the
point: the rule is about evidence, not about whose code it is.

To make this route admissible, give it a *measured* device attribution -- a
worker that runs a marked op under a profiler and reports where it landed -- and
set this to that fraction. Do not set it to 1.0 because the code looks right.
"""


@dataclass(frozen=True)
class Route:
    """One way to reach a device, resolved or not."""

    name: str
    interpreter: str | None
    worker: str
    """Absolute path to the worker script, in this filesystem's spelling."""
    ep: str
    """The execution provider the worker should ask for."""
    is_windows: bool
    """Whether paths handed to this interpreter need Windows spelling."""
    available: bool
    reason: str = ""
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def repo_root() -> Path:
    """The checkout this package was imported from, when there is one.

    Used only to resolve the relative interpreter candidates above. An
    installed wheel has no checkout around it, in which case the relative
    routes simply do not resolve and say so.
    """
    override = os.environ.get("VLLM_OMNI_REPO_ROOT")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    # ``.venvs`` first, and in a separate pass: ``vllm-omni`` is a nested
    # checkout with its own ``.git``, so a single loop that accepts either
    # marker stops one level too deep and every relative candidate misses.
    for marker in (".venvs", ".git"):
        for parent in here.parents:
            if (parent / marker).is_dir():
                return parent
    return here.parents[-1]


def is_wsl() -> bool:
    """True when this process can launch native-Windows executables.

    The marker is interop, not the kernel string: a WSL2 kernel without
    ``/proc/sys/fs/binfmt_misc/WSLInterop`` cannot exec a ``.exe``, and that is
    the capability the caller actually needs.
    """
    if os.name == "nt":
        return False
    return Path("/proc/sys/fs/binfmt_misc/WSLInterop").exists() or Path(
        "/proc/sys/fs/binfmt_misc/WSLInterop-late"
    ).exists()


def to_worker_path(path: str | Path, *, is_windows: bool) -> str:
    """Spell a path the way the worker's interpreter will understand it.

    A native-Windows worker running a script that lives in WSL sees it as a
    ``\\\\wsl.localhost\\...`` UNC path. ``wslpath -w`` is the translation, and
    it is the reason a Windows process can load graphs and write profiles
    straight into the Linux tree with no copying.
    """
    text = str(path)
    if not is_windows or os.name == "nt":
        return text
    if not shutil.which("wslpath"):
        return text
    try:
        out = subprocess.run(
            ["wslpath", "-w", text], capture_output=True, text=True, timeout=20, check=True
        )
    except (subprocess.SubprocessError, OSError):
        return text
    return out.stdout.strip() or text


def _resolve_interpreter(route: str, root: Path) -> tuple[str | None, str]:
    env_key = f"{ENV_PREFIX}{route.upper().replace('-', '_')}"
    override = os.environ.get(env_key)
    if override:
        if Path(override).exists():
            return override, ""
        return None, f"{env_key}={override!r} does not exist"

    tried: list[str] = []
    for candidate in _DEFAULT_CANDIDATES.get(route, ()):
        path = Path(candidate) if Path(candidate).is_absolute() else root / candidate
        tried.append(str(path))
        if path.exists():
            return str(path), ""
    return None, f"no interpreter found; tried {tried} and {env_key} is unset"


def resolve(route: str, *, root: Path | None = None) -> Route:
    """Find the interpreter for one route, or say why it is not there."""
    if route not in _WORKER_FOR_ROUTE:
        raise ValueError(f"unknown route {route!r}; expected one of {sorted(_WORKER_FOR_ROUTE)}")
    root = root or repo_root()
    is_windows = route in (ROUTE_VITISAI, ROUTE_DML, ROUTE_WIN_CUDA)
    worker_key = f"{WORKER_ENV_PREFIX}{route.upper().replace('-', '_')}"
    worker = os.environ.get(worker_key) or str(
        Path(__file__).resolve().parent / _WORKER_FOR_ROUTE[route]
    )

    if is_windows and not is_wsl() and os.name != "nt":
        return Route(
            name=route,
            interpreter=None,
            worker=worker,
            ep=_EP_FOR_ROUTE[route],
            is_windows=True,
            available=False,
            reason=(
                "this route needs a native-Windows interpreter, and this "
                "process is neither on Windows nor in a WSL distro with "
                "interop enabled, so it cannot start one"
            ),
        )

    interpreter, reason = _resolve_interpreter(route, root)
    # An interpreter without a worker script is not a route. Checked here
    # rather than at start(), because ``available`` is what the device
    # enumeration turns into ``runnable`` -- reporting a device as runnable and
    # then failing to launch it is the same silent-substitution mistake this
    # module exists to avoid, one layer down.
    if interpreter is not None and not Path(worker).is_file():
        return Route(
            name=route,
            interpreter=interpreter,
            worker=worker,
            ep=_EP_FOR_ROUTE[route],
            is_windows=is_windows,
            available=False,
            reason=f"worker script not found: {worker}",
        )
    return Route(
        name=route,
        interpreter=interpreter,
        worker=worker,
        ep=_EP_FOR_ROUTE[route],
        is_windows=is_windows,
        available=interpreter is not None,
        reason=reason,
    )


def resolve_all(*, root: Path | None = None) -> list[Route]:
    """Every route, resolved or not, in a stable order."""
    root = root or repo_root()
    return [resolve(name, root=root) for name in (ROUTE_VITISAI, ROUTE_DML, ROUTE_TORCH_DML, ROUTE_CPU)]


def describe(routes: list[Route]) -> str:
    lines = []
    for route in routes:
        mark = "available" if route.available else "missing"
        lines.append(f"  [{mark:^9}] {route.name:<12} ep={route.ep:<8} {route.interpreter or ''}")
        if route.reason:
            lines.append(f"              {route.reason}")
    return "\n".join(lines) if lines else "  (none)"
