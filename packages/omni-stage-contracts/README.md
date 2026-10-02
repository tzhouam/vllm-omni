# Omni stage contracts

Versioned, dependency-free data contracts for local Omni stage adapters.
The optional `wire` extra provides the NumPy copied-host tensor transport.
Importing the contracts never imports torch, vLLM or vLLM-Omni.

For adapters that implement persistent state, `StageRequest.state` and
`StageEvent.state` carry an opaque reference to backend-owned KV, recurrent,
or codec state. A continuation must identify the
same artifact, layout, backend instance, worker generation, state ID, and
epoch. `StageRequest.release(...)` describes explicit state disposal; a
backend must execute that operation before claiming its memory is free.

Contract protocol v2 adds `backend-state`, `ordered-events`, `credit-ack`, and
`artifact-manifest-v2` feature negotiation. Legacy v1 peers retain the
host-copy/request-epoch contract. The graph worker socket is still v1 and
rejects v2 frames until it implements the additional operations.

`BoundedStageEventStream` provides one-request event order, epoch fencing, and
byte **and** chunk credit. `submit` waits for credit; `receive` does not
release it. The consumer acknowledges `release_token` only after consuming a
payload. A single oversized event is refused instead of exceeding the bound.
Cancellation discards queued events but keeps already delivered payloads
charged until their consumer acknowledges them. A returned state must match
the request's session, artifact, layout, worker generation, and epoch.
Adapters may retain their existing output transport while adopting this
contract; declaring a state handle alone does not make an adapter stateful.
`external.graph.v1` remains a stateless, fixed-bucket, one-shot graph adapter.
The existing M0 local-text stream is a separate implementation and does not
yet use this portable ACK-on-consumption stream.

`ReferenceStageController` is a small host-testable control boundary for a
single active mobile/embedded request. It maps stage IDs to injected adapters,
checks that continuations use a live state on the same backend instance,
publishes ordered events through `BoundedStageEventStream`, and passes `cancel`
and `release` operations to the owning adapter. The next request cannot start
until the previous adapter has stopped and all event credits have been ACKed.
Cancellation fences queued events and retires state for continuation; the
caller must ACK any already delivered payloads and explicitly release the
state. If an adapter does not acknowledge cancellation or stop within the
configured timeout, `StageCancelTimeout` leaves its run and state reserved.
Release retries reuse the same operation ID; native adapters must treat that
ID idempotently if an acknowledgement is lost.
The controller does not load models, copy tensors, own KV/codec state, or claim
hardware quiescence merely because a Python task was cancelled. Native device
controllers can use `conformance/v2.json` and the reference tests as the
protocol target; this Python implementation is not device-resident latency or
mobile release evidence.

Manifest schema 1 remains readable for existing graph artifacts. Schema 2
requires checkpoint/runtime/adapter/export/compiler versions, target ABI,
precision, shape bucket, state
layout version, and hashed calibration (when used), numerical-validation, and
task-validation records. The numerical and task reports must each contain
matching artifact identity and at least one finite observed value checked
against a declared limit. The pass decision is recomputed from those checks;
an empty report or a manifest/report disagreement is rejected. A valid failed
validation remains readable as evidence. `external.graph.v1` refuses v2
deployment until worker ABI/runtime/layout/bucket compatibility is checked.

Build from this directory with `uv build`; both the wheel and source archive
contain the canonical package. The parent Omni distribution bundles the same
source for compatibility. Install matching releases when using both distributions.
