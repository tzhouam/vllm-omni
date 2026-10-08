# SPDX-License-Identifier: Apache-2.0
"""Auditable artifacts and admission claims for complete-model stage backends.

These types describe storage and compute separately. They do not schedule MoE
experts, allocate weights, or infer placement from a mapped pointer. Backends
continue to own those operations. Sizes are bytes, including on Windows.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Any

from vllm_omni.engine.resource_ledger import Reservation, ResourceLedger

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GGUF_SPLIT = re.compile(r"(?P<prefix>.+)-(?P<index>[0-9]{5})-of-(?P<count>[0-9]{5})\.gguf\Z")


def _bytes(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer byte count")


def _digest(value: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError("SHA256 must contain 64 lowercase hexadecimal characters")


def _relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("artifact path must be a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("artifact path must stay within its root")
    reserved = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
    if any(part.endswith((".", " ")) or part.split(".")[0].casefold() in reserved
           or any(ord(char) < 32 for char in part) for part in path.parts):
        raise ValueError("artifact path has a Windows device or alias component")
    return path


def _local_path(root: Path, relative: str) -> Path:
    path = root.joinpath(*_relative_path(relative).parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"artifact resolves outside its root: {relative}")
    return path


def sha256_file(path: Path | str) -> str:
    """Hash streaming chunks rather than loading a multi-gigabyte shard."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactFile:
    path: str
    size_bytes: int
    sha256: str
    role: str = "weights"
    quantization: str | None = None
    layout: str | None = None

    def __post_init__(self) -> None:
        _relative_path(self.path)
        _bytes("size_bytes", self.size_bytes)
        _digest(self.sha256)
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("artifact file role is required")


@dataclass(frozen=True)
class ArtifactTransformation:
    """One declared transformation; author quality claims remain separate."""

    kind: str
    source_checkpoint: str
    source_revision: str
    recipe: str
    teacher_checkpoint: str | None = None
    teacher_revision: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"ptq", "qat", "qad", "distillation", "rl", "pruning", "repack", "conversion"}:
            raise ValueError(f"unknown artifact transformation: {self.kind}")
        if not all(isinstance(v, str) and v.strip() for v in (
            self.source_checkpoint, self.source_revision, self.recipe,
        )):
            raise ValueError("transformation source, revision and recipe are required")
        if bool(self.teacher_checkpoint) != bool(self.teacher_revision):
            raise ValueError("teacher checkpoint and revision must be recorded together")


@dataclass(frozen=True)
class ArtifactManifest:
    checkpoint: str
    revision: str
    license: str
    files: tuple[ArtifactFile, ...]
    lineage: tuple[ArtifactTransformation, ...] = ()
    schema: str = "omni-weight-artifacts-v1"
    hash_origin: str = "declared"

    def __post_init__(self) -> None:
        if self.schema != "omni-weight-artifacts-v1":
            raise ValueError("unsupported artifact manifest schema")
        if not all(isinstance(v, str) and v.strip() for v in (self.checkpoint, self.revision, self.license)):
            raise ValueError("checkpoint, pinned revision and license are required")
        if self.revision.lower() in {"main", "master", "latest"}:
            raise ValueError("artifact revision must be pinned, not a floating branch")
        if self.hash_origin not in {"declared", "local_observation"}:
            raise ValueError("hash origin must distinguish declared hashes from local observations")
        object.__setattr__(self, "files", tuple(self.files))
        object.__setattr__(self, "lineage", tuple(self.lineage))
        if not self.files or not all(isinstance(f, ArtifactFile) for f in self.files):
            raise ValueError("manifest needs typed artifact files")
        if not all(isinstance(item, ArtifactTransformation) for item in self.lineage):
            raise ValueError("manifest needs typed transformation records")
        paths = [f.path.casefold() for f in self.files]
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate artifact paths, including Windows case aliases")
        groups: dict[str, tuple[int, set[int]]] = {}
        for item in self.files:
            match = _GGUF_SPLIT.fullmatch(item.path)
            if match is None:
                continue
            count, index = int(match["count"]), int(match["index"])
            if count < 1 or not 1 <= index <= count:
                raise ValueError("invalid GGUF shard numbering")
            previous_count, indices = groups.setdefault(match["prefix"], (count, set()))
            if previous_count != count:
                raise ValueError("inconsistent GGUF shard counts")
            indices.add(index)
        for prefix, (count, indices) in groups.items():
            if indices != set(range(1, count + 1)):
                raise ValueError(f"incomplete GGUF shard group: {prefix}; expected {count} shards")

    @property
    def total_size_bytes(self) -> int:
        return sum(item.size_bytes for item in self.files)

    @property
    def weight_size_bytes(self) -> int:
        return sum(item.size_bytes for item in self.files if item.role == "weights")

    @property
    def manifest_sha256(self) -> str:
        encoded = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["files"] = [asdict(item) for item in sorted(self.files, key=lambda item: item.path)]
        value["lineage"] = [asdict(item) for item in self.lineage]
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactManifest:
        data = dict(value)
        data["files"] = tuple(ArtifactFile(**item) for item in data.get("files", ()))
        data["lineage"] = tuple(ArtifactTransformation(**item) for item in data.get("lineage", ()))
        return cls(**data)

    def verify(self, root: Path | str) -> tuple[Path, ...]:
        """Verify every declared shard and auxiliary file before loading.

        This proves agreement with the manifest, not that an author-published
        expected digest was authentic. ``hash_origin`` preserves that boundary.
        """
        root = Path(root).resolve()
        verified = []
        for item in self.files:
            path = _local_path(root, item.path)
            if not path.is_file():
                raise FileNotFoundError(f"artifact file missing: {item.path}")
            if path.stat().st_size != item.size_bytes:
                raise ValueError(f"artifact size mismatch: {item.path}")
            if sha256_file(path) != item.sha256:
                raise ValueError(f"artifact SHA256 mismatch: {item.path}")
            verified.append(path)
        return tuple(verified)


def manifest_from_gguf(
    first_shard: Path | str,
    *,
    checkpoint: str,
    revision: str,
    license: str,
    root: Path | str | None = None,
    auxiliary_files: Mapping[str, Sequence[Path | str] | Path | str] | None = None,
    lineage: Sequence[ArtifactTransformation] = (),
) -> ArtifactManifest:
    """Inventory the exact GGUF split set, not every quantization in its folder.

    For authoritative verification, deserialize a manifest containing expected
    upstream hashes instead. This helper records locally observed bytes only.
    """
    first = Path(first_shard).resolve()
    root = Path(root).resolve() if root is not None else first.parent
    match = _GGUF_SPLIT.fullmatch(first.name)
    if match:
        count = int(match["count"])
        if int(match["index"]) != 1 or count < 1:
            raise ValueError("GGUF inventory must start at shard 00001")
        weights = [first.with_name(f"{match['prefix']}-{i:05d}-of-{count:05d}.gguf")
                   for i in range(1, count + 1)]
    else:
        if first.suffix.lower() != ".gguf":
            raise ValueError("a GGUF weight file is required")
        weights = [first]
    entries = []

    def add(path: Path | str, role: str) -> None:
        path = Path(path)
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(root):
            raise ValueError("artifact resolves outside its root")
        if not path.is_file():
            raise FileNotFoundError(f"artifact file missing: {path.name}")
        entries.append(ArtifactFile(path.relative_to(root).as_posix(), path.stat().st_size, sha256_file(path), role))

    for path in weights:
        add(path, "weights")
    for role, paths in (auxiliary_files or {}).items():
        for path in (paths,) if isinstance(paths, (str, Path)) else paths:
            add(path, role)
    return ArtifactManifest(
        checkpoint, revision, license, tuple(entries), tuple(lineage), hash_origin="local_observation",
    )


@dataclass(frozen=True)
class WeightTierBudget:
    """Disjoint steady allocations plus absolute peaks, never virtual mmap size.

    ``cpu_mapped_weights_bytes`` bounds resident file pages (including file
    cache), not the entire file address span. ``pinned_bytes`` is a separate
    allocation, not a second count of another field. Loading/commit peaks are
    absolute route peaks, so they are combined using max rather than addition.
    SSD capacity is a route-addressable ceiling: free space plus already
    present route files, not raw free space alone.
    """

    gpu_weights_bytes: int = 0
    gpu_expert_cache_bytes: int = 0
    cpu_resident_weights_bytes: int = 0
    cpu_mapped_weights_bytes: int = 0
    cpu_expert_cache_bytes: int = 0
    pinned_bytes: int = 0
    host_kv_bytes: int = 0
    gpu_kv_bytes: int = 0
    host_workspace_bytes: int = 0
    gpu_workspace_bytes: int = 0
    host_transfer_bytes: int = 0
    gpu_transfer_bytes: int = 0
    host_headroom_bytes: int = 0
    gpu_headroom_bytes: int = 0
    host_loading_peak_bytes: int = 0
    gpu_loading_peak_bytes: int = 0
    windows_commit_peak_bytes: int = 0
    ssd_artifact_bytes: int = 0
    ssd_temporary_bytes: int = 0

    def __post_init__(self) -> None:
        for item in fields(self):
            _bytes(item.name, getattr(self, item.name))

    @property
    def host_steady_bytes(self) -> int:
        return sum((self.cpu_resident_weights_bytes, self.cpu_mapped_weights_bytes, self.cpu_expert_cache_bytes,
                    self.pinned_bytes, self.host_kv_bytes, self.host_workspace_bytes,
                    self.host_transfer_bytes, self.host_headroom_bytes))

    @property
    def gpu_steady_bytes(self) -> int:
        return sum((self.gpu_weights_bytes, self.gpu_expert_cache_bytes, self.gpu_kv_bytes,
                    self.gpu_workspace_bytes, self.gpu_transfer_bytes, self.gpu_headroom_bytes))

    def resource_demands(
        self, *, gpu_pool: str = "vram:0", include_wsl: bool = False, include_windows_commit: bool = False,
        shared_gpu_memory: bool = False,
    ) -> dict[str, int]:
        host = max(self.host_steady_bytes, self.host_loading_peak_bytes)
        gpu = max(self.gpu_steady_bytes, self.gpu_loading_peak_bytes)
        if shared_gpu_memory:
            host += gpu
        result = {"host_ram": host}
        if gpu and not shared_gpu_memory:
            if gpu_pool in {"host_ram", "wsl_ram", "windows_commit", "ssd"} or not gpu_pool:
                raise ValueError("GPU allocations require a separate physical VRAM pool")
            result[gpu_pool] = gpu
        if include_wsl:
            result["wsl_ram"] = host
        if include_windows_commit:
            # File-backed resident pages constrain RAM, not anonymous commit.
            result["windows_commit"] = max(
                self.host_steady_bytes - self.cpu_mapped_weights_bytes, self.windows_commit_peak_bytes,
            ) + (gpu if shared_gpu_memory else 0)
        if self.ssd_artifact_bytes or self.ssd_temporary_bytes:
            result["ssd"] = self.ssd_artifact_bytes + self.ssd_temporary_bytes
        return result

    def check_manifest(self, manifest: ArtifactManifest) -> None:
        if self.ssd_artifact_bytes < manifest.total_size_bytes:
            raise ValueError("SSD artifact claim is below the complete manifest size")


@dataclass(frozen=True)
class BackendCapabilities:
    """Unknown capabilities fail closed without being called unsupported."""

    cpu_block_offload: bool | None = None
    cpu_expert_offload: bool | None = None
    bounded_ssd_expert_cache: bool | None = None
    lookup_tables_on_demand: bool | None = None
    vision: bool | None = None
    mtp: bool | None = None

    def __post_init__(self) -> None:
        if any(getattr(self, f.name) is not None and type(getattr(self, f.name)) is not bool for f in fields(self)):
            raise ValueError("capabilities must be true, false or unknown")


@dataclass(frozen=True)
class WeightTierPlan:
    route_id: str
    artifact_manifest_sha256: str
    backend: str
    backend_revision: str
    budget: WeightTierBudget
    cpu_block_layers: int = 0
    cpu_expert_layers: int = 0
    ssd_experts: bool = False
    lookup_tables_on_demand: bool = False
    mtp: bool = False
    prefetch: bool = False
    batch_size: int = 1
    max_active_requests: int = 1

    def __post_init__(self) -> None:
        _digest(self.artifact_manifest_sha256)
        if not all(isinstance(v, str) and v.strip() for v in (self.route_id, self.backend, self.backend_revision)):
            raise ValueError("route and backend revision must be explicit")
        if not isinstance(self.budget, WeightTierBudget):
            raise ValueError("a typed weight-tier budget is required")
        for name in ("cpu_block_layers", "cpu_expert_layers"):
            _bytes(name, getattr(self, name))
        for name in ("ssd_experts", "lookup_tables_on_demand", "mtp", "prefetch"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if type(self.batch_size) is not int or self.batch_size != 1:
            raise ValueError("weight-tier qualification supports batch size 1 only")
        if type(self.max_active_requests) is not int or self.max_active_requests != 1:
            raise ValueError("weight-tier qualification supports one active request only")

    def check_capabilities(self, capabilities: BackendCapabilities) -> None:
        required = {
            "cpu_block_offload": bool(self.cpu_block_layers),
            "cpu_expert_offload": bool(self.cpu_expert_layers),
            "bounded_ssd_expert_cache": self.ssd_experts,
            "lookup_tables_on_demand": self.lookup_tables_on_demand,
            "mtp": self.mtp,
        }
        for name, needed in required.items():
            if needed and getattr(capabilities, name) is not True:
                raise ValueError(f"required backend capability is not verified: {name}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WeightTierPlan:
        data = dict(value)
        data["budget"] = WeightTierBudget(**data["budget"])
        return cls(**data)


@dataclass(frozen=True)
class ComputePlacement:
    component: str
    execution_units: tuple[str, ...]
    evidence: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution_units", tuple(self.execution_units))
        if not self.component or not self.execution_units or not self.evidence:
            raise ValueError("observed compute placement needs units and evidence")


@dataclass(frozen=True)
class StoragePlacement:
    component: str
    tier: str
    resident_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.component or self.tier not in {"gpu", "ram", "file_mapped_ram", "ssd"}:
            raise ValueError("observed storage needs a component and known tier")
        if self.resident_bytes is not None:
            _bytes("resident_bytes", self.resident_bytes)


@dataclass(frozen=True)
class PlacementReport:
    """Missing sensor readings stay None; logical reads are not physical I/O."""

    compute: tuple[ComputePlacement, ...] = ()
    storage: tuple[StoragePlacement, ...] = ()
    logical_read_bytes: int | None = None
    physical_ssd_read_bytes: int | None = None
    ssd_wait_seconds: float | None = None
    expert_cache_hit_ratio: float | None = None
    io_mode: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "compute", tuple(self.compute))
        object.__setattr__(self, "storage", tuple(self.storage))
        if not all(isinstance(item, ComputePlacement) for item in self.compute):
            raise ValueError("compute observations must be typed")
        if not all(isinstance(item, StoragePlacement) for item in self.storage):
            raise ValueError("storage observations must be typed")
        for name in ("logical_read_bytes", "physical_ssd_read_bytes"):
            if getattr(self, name) is not None:
                _bytes(name, getattr(self, name))
        for name in ("ssd_wait_seconds", "expert_cache_hit_ratio"):
            value = getattr(self, name)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be a finite nonnegative observation")
        if self.expert_cache_hit_ratio is not None and self.expert_cache_hit_ratio > 1:
            raise ValueError("cache hit ratio must be within [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reserve_tier_plan(
    ledger: ResourceLedger,
    owner: str,
    plan: WeightTierPlan,
    *,
    gpu_pool: str = "vram:0",
    include_wsl: bool = False,
    include_windows_commit: bool = False,
    shared_gpu_memory: bool = False,
) -> Reservation:
    """Charge the caller's shared ledger; never construct a second controller."""
    return ledger.reserve(owner, plan.budget.resource_demands(
        gpu_pool=gpu_pool, include_wsl=include_wsl, include_windows_commit=include_windows_commit,
        shared_gpu_memory=shared_gpu_memory,
    ))
