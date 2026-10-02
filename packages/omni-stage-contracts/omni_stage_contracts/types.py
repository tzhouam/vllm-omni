"""Versioned host-copy contracts shared by Python and native stage adapters."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 2
FEATURES_BY_VERSION = {
    1: frozenset({"host-copy", "request-epochs"}),
    2: frozenset({
        "host-copy", "request-epochs", "backend-state", "ordered-events",
        "credit-ack", "artifact-manifest-v2",
    }),
}
_ITEM_SIZES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "float16": 2,
    "float32": 4,
    "float64": 8,
}


def negotiate(version: int, required: list[str] | tuple[str, ...] = ()) -> None:
    if type(version) is not int or version not in FEATURES_BY_VERSION:
        raise ValueError(f"unsupported stage protocol version: {version!r}")
    if not isinstance(required, (list, tuple)) or any(not isinstance(feature, str) for feature in required):
        raise ValueError("required stage features must be a list of names")
    missing = set(required) - FEATURES_BY_VERSION[version]
    if missing:
        raise ValueError(f"unsupported required stage features: {sorted(missing)}")


@dataclass(frozen=True)
class BufferRef:
    object_id: str
    owner: str
    generation: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    memory_domain: str = "host-copy"
    layout: str = "C"
    ready: bool = True

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.object_id, self.owner, self.generation)
        ):
            raise ValueError("buffer identity/owner/generation are required")
        if self.memory_domain != "host-copy" or self.layout != "C" or not self.ready:
            raise ValueError("v1 only supports ready, contiguous copied host buffers")
        if self.dtype not in _ITEM_SIZES:
            raise ValueError(f"unsupported dtype: {self.dtype}")
        if len(self.shape) > 32 or any(type(x) is not int or x < 0 for x in self.shape):
            raise ValueError("invalid tensor dimensions")
        expected = math.prod(self.shape) * _ITEM_SIZES[self.dtype]
        if type(self.nbytes) is not int or self.nbytes != expected or self.nbytes > 2 << 30:
            raise ValueError("tensor shape/dtype/byte length mismatch or payload exceeds limit")


@dataclass(frozen=True)
class StateHandle:
    session_id: str
    backend: str
    artifact_id: str
    layout_version: int = 1
    epoch: int = 0
    replayable: bool = False
    migratable: bool = False
    backend_instance_id: str = ""
    worker_generation: str = ""
    state_id: str = ""

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.session_id, self.backend, self.artifact_id)
        ):
            raise ValueError("state session, backend and artifact are required")
        if any(
            not isinstance(value, str)
            for value in (self.backend_instance_id, self.worker_generation, self.state_id)
        ):
            raise ValueError("state identity fields must be strings")
        if type(self.layout_version) is not int or self.layout_version < 1:
            raise ValueError("state layout version must be positive")
        if type(self.epoch) is not int or self.epoch < 0:
            raise ValueError("state epoch must be a nonnegative integer")
        if type(self.replayable) is not bool or type(self.migratable) is not bool:
            raise ValueError("state replay/migration flags must be booleans")

    def accepts(self, other: StateHandle) -> bool:
        return all(
            getattr(self, k) == getattr(other, k)
            for k in (
                "session_id",
                "backend",
                "artifact_id",
                "layout_version",
                "epoch",
                "backend_instance_id",
                "worker_generation",
                "state_id",
                "replayable",
                "migratable",
            )
        )

    def require_bound(self, worker_generation: str, epoch: int) -> None:
        """Require an opaque, live worker-owned state for a continuation."""
        if not self.state_id or not self.backend_instance_id or not self.worker_generation:
            raise ValueError("persistent state needs an opaque ID and owning worker identity")
        if self.worker_generation != worker_generation or self.epoch != epoch:
            raise ValueError("state belongs to another worker generation or request epoch")

    def next_epoch(self) -> StateHandle:
        from dataclasses import replace

        return replace(self, epoch=self.epoch + 1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StageRequest:
    request_id: str
    stage_id: int
    epoch: int = 0
    worker_generation: str = ""
    operation: str = "run"
    inputs: tuple[BufferRef, ...] = ()
    state: StateHandle | None = None
    artifact_id: str = ""
    state_layout_version: int = 0
    session_id: str = ""

    def __post_init__(self):
        if any(
            not isinstance(value, str) or not value
            for value in (self.request_id, self.worker_generation, self.operation)
        ):
            raise ValueError("request identity, worker generation and operation are required")
        if not isinstance(self.artifact_id, str) or not isinstance(self.session_id, str) or type(self.state_layout_version) is not int:
            raise ValueError("request artifact, session and state layout types are invalid")
        if any(type(n) is not int or n < 0 for n in (self.stage_id, self.epoch)):
            raise ValueError("stage_id and epoch must be nonnegative integers")
        if self.operation not in {"run", "cancel", "release"}:
            raise ValueError("unsupported stage operation")
        if self.state is not None:
            self.state.require_bound(self.worker_generation, self.epoch)
            if not self.session_id or self.session_id != self.state.session_id:
                raise ValueError("state session differs from the stage request")
            if self.artifact_id != self.state.artifact_id or self.state_layout_version != self.state.layout_version:
                raise ValueError("state artifact or layout differs from the stage request")
        if self.operation == "release" and (self.state is None or self.inputs):
            raise ValueError("release requires a state handle and no input buffers")

    @classmethod
    def release(cls, request_id: str, stage_id: int, state: StateHandle) -> StageRequest:
        """Describe explicit backend release; the adapter must execute it."""
        return cls(
            request_id, stage_id, state.epoch, state.worker_generation, "release",
            state=state, artifact_id=state.artifact_id, state_layout_version=state.layout_version,
            session_id=state.session_id,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StageRequest:
        values = dict(data)
        values["inputs"] = tuple(BufferRef(**ref) for ref in values.get("inputs", ()))
        if values.get("state") is not None:
            values["state"] = StateHandle(**values["state"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StageEvent:
    request_id: str
    stage_id: int
    epoch: int
    seq: int
    kind: str
    worker_generation: str
    buffers: tuple[BufferRef, ...] = ()
    terminal: bool = False
    error: str | None = None
    state: StateHandle | None = None
    started_monotonic_ns: int = 0
    emitted_monotonic_ns: int = 0
    input_watermark: int = 0
    payload_nbytes: int = 0
    release_token: str = ""

    def __post_init__(self):
        if any(
            not isinstance(value, str) or not value
            for value in (self.request_id, self.worker_generation, self.kind)
        ):
            raise ValueError("event identity, worker generation and kind are required")
        if not isinstance(self.release_token, str):
            raise ValueError("event acknowledgement token must be a string")
        if any(type(n) is not int or n < 0 for n in (self.stage_id, self.epoch, self.seq)):
            raise ValueError("stage_id, epoch and seq must be nonnegative integers")
        if any(
            type(n) is not int or n < 0
            for n in (self.started_monotonic_ns, self.emitted_monotonic_ns, self.input_watermark, self.payload_nbytes)
        ):
            raise ValueError("event timing, watermark and byte count must be nonnegative integers")
        if bool(self.started_monotonic_ns) != bool(self.emitted_monotonic_ns):
            raise ValueError("event requires both start and emission timestamps")
        if self.emitted_monotonic_ns and self.emitted_monotonic_ns < self.started_monotonic_ns:
            raise ValueError("event emission precedes start")
        if self.state is not None:
            self.state.require_bound(self.worker_generation, self.epoch)
        if self.payload_nbytes and self.payload_nbytes < sum(ref.nbytes for ref in self.buffers):
            raise ValueError("event byte count is smaller than its buffers")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StageEvent:
        values = dict(data)
        values["buffers"] = tuple(BufferRef(**ref) for ref in values.get("buffers", ()))
        if values.get("state") is not None:
            values["state"] = StateHandle(**values["state"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeviceDescriptor:
    """Physical identity is independent from per-domain execution routes.

    Unknown package/power/bandwidth relationships are empty, never inferred
    from an integrated/discrete label. Pool IDs identify backing constraints.
    """

    device_id: str
    kind: str
    domain_id: str
    memory_pool_ids: tuple[str, ...]
    integration: str = "unknown"
    package_id: str | None = None
    bandwidth_group_ids: tuple[str, ...] = ()
    power_domain_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"cpu", "gpu", "npu"}:
            raise ValueError(f"unknown device kind: {self.kind}")
        if self.integration not in {"integrated", "discrete", "unknown"}:
            raise ValueError("unknown integration class")
        if not self.device_id or not self.domain_id or not self.memory_pool_ids:
            raise ValueError("device, domain and physical memory constraints are required")
        if len(set(self.memory_pool_ids)) != len(self.memory_pool_ids):
            raise ValueError("duplicate physical memory constraint")


@dataclass(frozen=True)
class ArtifactMetadata:
    """Provenance and validation references for a v2 executable artifact.

    Validation files are part of the hashed bundle. ``passed=False`` remains
    readable as failure evidence, but is never an approval to deploy it.
    """

    checkpoint_id: str
    checkpoint_revision: str
    precision: str
    runtime: str
    runtime_version: str
    target_abi: str
    adapter_version: str
    exporter_version: str
    compiler_version: str
    state_layout_version: int
    shape_bucket: str
    numerical_validation: dict[str, Any]
    task_validation: dict[str, Any]
    calibration_file: str | None = None
    numerical_report_passed: bool = field(default=False, init=False)
    task_report_passed: bool = field(default=False, init=False)
    reports_verified: bool = field(default=False, init=False)

    @classmethod
    def from_manifest(
        cls, metadata: dict[str, Any], files: dict[str, str], root: Path | None = None
    ) -> ArtifactMetadata:
        if not isinstance(metadata, dict) or not isinstance(metadata.get("artifact"), dict):
            raise ValueError("v2 artifact metadata is required")
        try:
            result = cls(**metadata["artifact"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid v2 artifact metadata: {exc}") from exc
        for name in (
            "checkpoint_id", "checkpoint_revision", "precision", "runtime",
            "runtime_version", "target_abi", "adapter_version",
            "exporter_version", "compiler_version", "shape_bucket",
        ):
            if not isinstance(getattr(result, name), str) or not getattr(result, name):
                raise ValueError(f"v2 artifact {name} is required")
        if type(result.state_layout_version) is not int or result.state_layout_version < 1:
            raise ValueError("v2 state layout version must be positive")
        for name in ("numerical_validation", "task_validation"):
            record = getattr(result, name)
            if (
                not isinstance(record, dict) or set(record) != {"file", "passed"}
                or not isinstance(record["file"], str) or record["file"] not in files
                or type(record["passed"]) is not bool
            ):
                raise ValueError(f"v2 {name} needs a hashed file and boolean result")
        if result.calibration_file is not None and result.calibration_file not in files:
            raise ValueError("v2 calibration record must be a hashed artifact payload")
        if root is not None:
            for kind in ("numerical", "task"):
                record = getattr(result, f"{kind}_validation")
                passed = _validation_report_passed(root / record["file"], files[record["file"]], kind, result)
                if passed != record["passed"]:
                    raise ValueError(f"v2 {kind} manifest result differs from verified report")
                object.__setattr__(result, f"{kind}_report_passed", passed)
            object.__setattr__(result, "reports_verified", True)
        return result

    @property
    def qualified(self) -> bool:
        return self.reports_verified and self.numerical_report_passed and self.task_report_passed


def _validation_report_passed(
    path: Path, expected_digest: str, kind: str, artifact: ArtifactMetadata
) -> bool:
    raw = path.read_bytes()
    if not raw or len(raw) > 1 << 20 or hashlib.sha256(raw).hexdigest() != expected_digest:
        raise ValueError(f"v2 {kind} validation report is empty, too large or changed")
    try:
        report = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"v2 {kind} validation report is not JSON") from exc
    if not isinstance(report, dict) or type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        raise ValueError(f"v2 {kind} validation report has an unsupported schema")
    expected = {
        "kind": kind,
        "checkpoint_id": artifact.checkpoint_id,
        "checkpoint_revision": artifact.checkpoint_revision,
        "precision": artifact.precision,
        "shape_bucket": artifact.shape_bucket,
        "target_abi": artifact.target_abi,
    }
    if any(report.get(name) != value for name, value in expected.items()):
        raise ValueError(f"v2 {kind} validation report is for a different artifact")
    checks = report.get("checks")
    if not isinstance(checks, list) or not checks or len(checks) > 1000:
        raise ValueError(f"v2 {kind} validation report needs measured checks")
    names: set[str] = set()
    passed = True
    for check in checks:
        if not isinstance(check, dict):
            raise ValueError(f"v2 {kind} validation check is invalid")
        name, observed, limit, comparison = (
            check.get("name"), check.get("observed"), check.get("limit"), check.get("comparison")
        )
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"v2 {kind} validation check name is missing or repeated")
        names.add(name)
        if any(type(value) is bool or not isinstance(value, Real) or not math.isfinite(value)
               for value in (observed, limit)):
            raise ValueError(f"v2 {kind} validation check needs finite observed and limit values")
        if comparison not in {"<=", ">=", "=="}:
            raise ValueError(f"v2 {kind} validation check comparison is invalid")
        passed &= (
            observed <= limit if comparison == "<=" else
            observed >= limit if comparison == ">=" else observed == limit
        )
    return passed


@dataclass(frozen=True)
class ArtifactManifest:
    """A complete artifact payload set; paths are relative to the manifest."""

    schema_version: int
    component: str
    files: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)
    _verified_metadata: ArtifactMetadata | None = field(default=None, init=False, repr=False, compare=False)

    def artifact_metadata(self) -> ArtifactMetadata | None:
        if self.schema_version == 1:
            return None
        if self.schema_version != 2:
            raise ValueError("unsupported artifact manifest schema")
        return self._verified_metadata or ArtifactMetadata.from_manifest(self.metadata, self.files)

    @classmethod
    def read(cls, path: str | Path) -> ArtifactManifest:
        path = Path(path)
        manifest = cls(**json.loads(path.read_text(encoding="utf-8")))
        if type(manifest.schema_version) is not int or manifest.schema_version not in (1, 2) or not manifest.files:
            raise ValueError("unsupported or empty artifact manifest")
        if not isinstance(manifest.component, str) or not manifest.component:
            raise ValueError("artifact component is required")
        root = path.parent.resolve()
        for name, digest in manifest.files.items():
            if not isinstance(name, str) or "\\" in name or ":" in name:
                raise ValueError("artifact payloads must use portable relative POSIX paths")
            target = (root / name).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"artifact path leaves bundle: {name}")
            if not target.is_file() or file_digest(target) != digest:
                raise ValueError(f"artifact payload missing or digest mismatch: {name}")
        if manifest.schema_version == 2:
            object.__setattr__(
                manifest, "_verified_metadata", ArtifactMetadata.from_manifest(manifest.metadata, manifest.files, root)
            )
        return manifest


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
