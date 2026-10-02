"""Source-checkout bridge; wheels use the canonical package directly."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[1] / "packages/omni-stage-contracts/omni_stage_contracts")]

from .stream import BoundedStageEventStream, StageStreamClosed
from .reference_controller import (
    ControlledStageRun,
    ControllerLimits,
    ReferenceStageController,
    StageBackend,
    StageCancelTimeout,
)
from .types import (
    PROTOCOL_VERSION,
    ArtifactManifest,
    ArtifactMetadata,
    BufferRef,
    DeviceDescriptor,
    StageEvent,
    StageRequest,
    StateHandle,
    negotiate,
)

__all__ = [
    "PROTOCOL_VERSION",
    "ArtifactManifest",
    "ArtifactMetadata",
    "BoundedStageEventStream",
    "BufferRef",
    "ControlledStageRun",
    "ControllerLimits",
    "DeviceDescriptor",
    "ReferenceStageController",
    "StageBackend",
    "StageCancelTimeout",
    "StageEvent",
    "StageRequest",
    "StateHandle",
    "StageStreamClosed",
    "negotiate",
]
