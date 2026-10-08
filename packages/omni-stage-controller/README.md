# Native Omni stage controller

This C++17 control layer follows the existing portable v2 stage semantics.
It is a host-tested library boundary for an Android JNI/NDK adapter, not an
Android model runtime, installed app, or device-resident qualification.

The controller bounds event count and bytes, orders events, binds opaque state
to checkpoint/artifact, session, layout, epoch and worker, and enforces one
active request. Payloads and their actual memory remain backend-owned. The
caller must keep them alive through the consumer ACK.

`try_emit` returns `backpressure` without consuming sequence or credit. The
adapter waits for consumer credit before retrying; it must not accumulate an
unbounded output queue. `receive` does not release credit. An ACK includes the
request identity so an old ACK cannot free a new request's buffer.

`cancel` fences later output and drops queued metadata, while delivered payloads
remain charged. The caller sends cancel to the real backend and calls `finish`
only after its quiescence ACK. Missing ACKs keep the request active. Backend
state disposal is a separate `request_release` / `confirm_release` pair; retry
uses the same release operation ID.

No scheduler, tensor allocator, kernels, tokenizer, sampling, model execution,
or device telemetry are implemented here. Concrete llama.cpp/LiteRT-LM JNI
adapters must marshal their own typed payloads and verify artifacts before
calling this boundary. NPU placement requires actual backend execution evidence.

The host test in `tests/edge/test_native_stage_controller.py` generates native
test data from the same `omni-stage-contracts/conformance/v2.json` used by Python.
It checks fixture acceptance/rejection, backpressure, cancellation, explicit
release, stale worker/state/ACK rejection, and a request that fails before its
terminal event. A host compiler pass does not substitute for Android NDK build
or phone-local correctness, memory, latency and thermal validation.
