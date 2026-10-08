# SPDX-License-Identifier: Apache-2.0
"""Mobile candidate/evidence boundaries; CPU, GPU and NPU share physical RAM.

Catalog entries are pinned publisher artifacts, not runnable or qualified
routes. A concrete Android adapter still needs its runtime pin, artifact
verification and target-local measurements before engine/Agent admission.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from omni_stage_contracts import DeviceDescriptor
from vllm_omni.engine.weight_tiers import ArtifactManifest, WeightTierBudget


class MobileEvidenceKind(str, Enum):
    DESIGN = "design"
    HOST_CONFORMANCE = "host_conformance"
    AIHUB_COMPONENT = "aihub_component"
    AIHUB_CHAIN = "aihub_chain"
    DEVICE_LOCAL = "device_local"


@dataclass(frozen=True)
class MobileCandidate:
    candidate_id: str
    runtime_family: str
    artifact_manifest: ArtifactManifest
    candidate_compute_units: tuple[str, ...]
    candidate_modalities: tuple[str, ...]
    runtime_revision: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_compute_units", tuple(self.candidate_compute_units))
        object.__setattr__(self, "candidate_modalities", tuple(self.candidate_modalities))
        if not self.candidate_id or self.runtime_family not in {"llama.cpp", "litert-lm"}:
            raise ValueError("mobile candidate needs an explicit identity and backend family")
        if not self.candidate_compute_units or not set(self.candidate_compute_units) <= {"cpu", "gpu", "npu"}:
            raise ValueError("unknown mobile compute candidate")
        if not self.candidate_modalities or not set(self.candidate_modalities) <= {"text", "image", "audio"}:
            raise ValueError("unknown mobile modality candidate")

    @property
    def admission_ready(self) -> bool:
        # Source hashes alone cannot verify an Android adapter/runtime build.
        return False


@dataclass(frozen=True)
class MobileEvidence:
    """Evidence scope, never a substitute for the engine qualification gate."""

    candidate_id: str
    kind: MobileEvidenceKind
    raw_evidence: tuple[str, ...] = ()
    complete_request: bool = False
    task_quality_passed: bool = False
    observed_execution_units: tuple[str, ...] | None = None
    shared_ram_peak_bytes: int | None = None
    whole_request_seconds: float | None = None
    cancel_release_passed: bool | None = None
    sustained_seconds: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_evidence", tuple(self.raw_evidence))
        object.__setattr__(self, "kind", MobileEvidenceKind(self.kind))
        if self.observed_execution_units is not None:
            object.__setattr__(self, "observed_execution_units", tuple(self.observed_execution_units))
        if not self.candidate_id:
            raise ValueError("mobile evidence needs a candidate identity")
        if any(type(value) is not bool for value in (self.complete_request, self.task_quality_passed)):
            raise ValueError("mobile completion and quality observations must be booleans")
        if self.shared_ram_peak_bytes is not None and (
            type(self.shared_ram_peak_bytes) is not int or self.shared_ram_peak_bytes < 0
        ):
            raise ValueError("shared RAM peak must be a nonnegative byte observation")
        for value in (self.whole_request_seconds, self.sustained_seconds):
            if value is not None and (type(value) not in (float, int) or not math.isfinite(value) or value < 0):
                raise ValueError("mobile timing must be a finite nonnegative observation")
        if self.kind is not MobileEvidenceKind.DEVICE_LOCAL and any(value is not None for value in (
            self.shared_ram_peak_bytes, self.whole_request_seconds, self.cancel_release_passed, self.sustained_seconds,
        )):
            raise ValueError("hosted/component evidence cannot claim device-resident measurements")

    @property
    def hosted_functional_pass(self) -> bool:
        return (self.kind is MobileEvidenceKind.AIHUB_CHAIN and bool(self.raw_evidence)
                and self.complete_request and self.task_quality_passed)

    @property
    def ready_for_device_qualification(self) -> bool:
        return (
            self.kind is MobileEvidenceKind.DEVICE_LOCAL and bool(self.raw_evidence)
            and self.complete_request and self.task_quality_passed
            and bool(self.observed_execution_units) and self.shared_ram_peak_bytes is not None
            and self.whole_request_seconds is not None and self.cancel_release_passed is True
            and self.sustained_seconds is not None and self.sustained_seconds >= 1800
        )


def mobile_devices(domain_id: str, *, gpu: bool = True, npu: bool = True) -> tuple[DeviceDescriptor, ...]:
    """Describe physical backing; three execution devices do not add RAM pools."""
    kinds = ("cpu",) + (("gpu",) if gpu else ()) + (("npu",) if npu else ())
    return tuple(DeviceDescriptor(
        device_id=f"{domain_id}:{kind}", kind=kind, domain_id=domain_id,
        memory_pool_ids=("host_ram",), integration="integrated",
    ) for kind in kinds)


def mobile_resource_demands(budget: WeightTierBudget) -> dict[str, int]:
    return budget.resource_demands(shared_gpu_memory=True)


def load_mobile_candidates(path: Path | str) -> MappingProxyType[str, MobileCandidate]:
    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "omni-mobile-candidates-v1":
        raise ValueError("unsupported mobile candidate catalog")
    result = {}
    for raw in payload.get("candidates", ()):
        value = dict(raw)
        value["artifact_manifest"] = ArtifactManifest.from_dict(value["artifact_manifest"])
        candidate = MobileCandidate(**value)
        if candidate.candidate_id in result:
            raise ValueError("duplicate mobile candidate identity")
        result[candidate.candidate_id] = candidate
    if not result:
        raise ValueError("mobile candidate catalog is empty")
    return MappingProxyType(result)
