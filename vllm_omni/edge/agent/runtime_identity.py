# SPDX-License-Identifier: Apache-2.0
"""Identity of the Python runtime that actually executes an Agent route.

The installed distribution version is insufficient for editable checkouts and
for local changes made after installation. Hash the imported Omni, vLLM and
shared stage-contract Python sources, using relative names so moving a
checkout does not change the identity. A missing source tree fails closed at
qualification time.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import platform
from functools import lru_cache
from pathlib import Path
from types import ModuleType


_DEPENDENCIES = (
    "cryptography", "numpy", "nvidia-ml-py", "pillow", "playwright", "psutil", "pynvml",
    "pyside6", "torch", "vllm", "vllm-omni",
)


def _package_source_sha256(module: ModuleType) -> str:
    """Hash the code Python imports, including a source-checkout bridge.

    ``omni_stage_contracts`` in a source checkout has a small bridge
    ``__init__.py`` whose ``__path__`` points at the actual implementation in
    ``packages/``. Both locations matter. Package roots are labelled by import
    order, rather than absolute location, so relocating the checkout preserves
    the identity while changing any imported Python source invalidates it.
    """
    root_strings = getattr(module, "__path__", None)
    if not root_strings:
        raise RuntimeError(f"{module.__name__} has no imported package source tree")
    roots = list(dict.fromkeys(Path(root).resolve() for root in root_strings))
    entries: list[tuple[str, Path]] = []
    for index, root in enumerate(roots):
        if not root.is_dir():
            raise RuntimeError(f"{module.__name__} imported source root is unavailable")
        entries.extend(
            (f"root{index}/{path.relative_to(root).as_posix()}", path)
            for path in root.rglob("*.py") if path.is_file()
        )
    origin_string = getattr(module, "__file__", None)
    if origin_string:
        origin = Path(origin_string).resolve()
        if origin.suffix == ".pyc" and origin.with_suffix(".py").is_file():
            origin = origin.with_suffix(".py")
        if origin.suffix == ".py" and origin.is_file() and not any(
            origin.is_relative_to(root) for root in roots
        ):
            entries.append((f"bootstrap/{origin.name}", origin))
    if not entries:
        raise RuntimeError(f"{module.__name__} imported Python source tree is unavailable")
    digest = hashlib.sha256()
    for relative, path in sorted(entries):
        name = relative.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def imported_omni_source_sha256() -> str:
    import vllm_omni

    package = Path(vllm_omni.__file__).resolve().parent
    files = sorted(path for path in package.rglob("*.py") if path.is_file())
    if not files or not (package / "edge" / "agent" / "controller.py").is_file():
        raise RuntimeError("imported Omni Agent source tree is unavailable")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(package).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def imported_vllm_source_sha256() -> str:
    return _package_source_sha256(importlib.import_module("vllm"))


@lru_cache(maxsize=1)
def imported_stage_contract_source_sha256() -> str:
    return _package_source_sha256(importlib.import_module("omni_stage_contracts"))


def loaded_runtime_identity() -> dict[str, object]:
    versions: dict[str, str] = {}
    for name in _DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return {
        "python": platform.python_version(),
        "omni_source_sha256": imported_omni_source_sha256(),
        "vllm_source_sha256": imported_vllm_source_sha256(),
        "stage_contract_source_sha256": imported_stage_contract_source_sha256(),
        "dependencies": versions,
    }


def loaded_runtime_sha256() -> str:
    import json

    identity = loaded_runtime_identity()
    return hashlib.sha256(json.dumps(identity, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()
