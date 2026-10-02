"""Portable Omni data contracts. Importing this module needs only the stdlib."""

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
