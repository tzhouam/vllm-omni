"""Compatible M0 contracts; new backend state uses types.StateHandle."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

STATE_LAYOUT_VERSION = 1


@dataclass(frozen=True)
class StateHandle:
    """A reference to session state the backend owns.

    The engine never reads the state. It reads this, to decide whether a
    request may continue a session or must start a new one.
    """

    session_id: str
    backend: str
    artifact_id: str
    """Ties the state to the checkpoint that produced it. The same shapes from
    a different quantization are not the same state."""
    layout_version: int = STATE_LAYOUT_VERSION
    epoch: int = 0
    """Incremented on every cancel. Events from a lower epoch are stale."""
    replayable: bool = True
    """A text session can be rebuilt by re-submitting its committed prompt, so
    recovery after a backend crash is possible at an output boundary."""
    migratable: bool = False
    """Never true in M0. Moving KV between backends needs a measured layout
    conversion for the specific pair; assuming it is the failure the proposal
    warns about."""
    created_unix: float = field(default_factory=time.time)

    def next_epoch(self) -> StateHandle:
        from dataclasses import replace

        return replace(self, epoch=self.epoch + 1)

    backend_instance_id: str = ""
    worker_generation: str = ""

    def accepts(self, other: StateHandle) -> bool:
        """Whether ``other`` may continue this session's state."""
        return (
            other.backend == self.backend
            and other.backend_instance_id == self.backend_instance_id
            and other.worker_generation == self.worker_generation
            and other.session_id == self.session_id
            and other.artifact_id == self.artifact_id
            and other.layout_version == self.layout_version
            and other.epoch == self.epoch
            and other.replayable == self.replayable
            and other.migratable == self.migratable
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChunkEvent:
    """One unit of output, with everything needed to order and retire it."""

    request_id: str
    stage_id: int
    seq: int
    """Monotonic within (request, epoch). Gaps mean loss; repeats mean a bug."""
    epoch: int
    kind: str
    """``token`` | ``done`` | ``error`` | ``cancelled``."""
    payload: Any
    """For ``token``: the incremental text and token ids. Never the whole
    output so far -- a cumulative payload makes a dropped chunk invisible."""
    started_unix: float
    emitted_unix: float
    final: bool = False
    error: str | None = None
    input_watermark: int = 0
    """How much of the input the producer had consumed when this was emitted.
    One number in M0 (prompt tokens); it is what a duplex turn will align on."""
    release_token: str = ""
    """Returned by the consumer to give credit back. Empty when the event
    carries no resource the producer is holding on its behalf."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
