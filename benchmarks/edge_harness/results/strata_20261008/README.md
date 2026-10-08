# Strata / shared-lease evidence — 2026-10-08

This directory records implementation verification and explicit blockers.
It contains **no successful Strata model request, no DeepSeek request, and no
new Agent route qualification**. All request protocols are batch=1 and one
active request.

| Record | What was actually tested | What it does not establish |
| --- | --- | --- |
| [protocol_audit.json](protocol_audit.json) | Five native Windows checks against the pinned upstream Strata Python service, using its mock engine through the production UTF-8 bootstrap: SSE, authentication, context refusal and completed-request metrics | No neural network, GPU kernel, actual expert cache or SSD execution |
| [shared_lease_native_smoke.json](shared_lease_native_smoke.json) | Existing Spark CPU llama.cpp route on native Windows: cancel a stream, verify worker exit and exact shared lease release, start a fresh worker and recover a request | No Strata model execution, general task quality or performance qualification; exact-answer check was false and is preserved |
| [local_readiness.json](local_readiness.json) | Native hardware/driver/runtime identity and real ResourceLedger refusals for 24/32/40 GiB expert-cache lower bounds, before allocation | Not a full-model memory estimate, a model-load failure, or permanent hardware unsupported status |
| [implementation_checks.json](implementation_checks.json) | Source-bound automated checks for the integration, portable controller, lifecycle, Agent and profiling contracts | Not model quality, native GPU/SSD operation, Android execution or release qualification |

The Spark raw request log remains in the existing private Agent results
directory; its hash and reviewed lifecycle facts are published here. Later
source changes make it historical for strict final-source matching. Full
request payloads and answers are not introduced into this public lifecycle
receipt.

The Windows Strata release archive was verified against its published digest:
`766373e74d1e25c9e87d7464b1e0c04857ec9fb4326a4d50c3a9d53ea84b26e4`.
Source checkout HEAD is `d5ea7133741e67743c0e886bb426c0ce8d69cf6c`.
This establishes selected bytes and source origin, not an independently
reproducible executable build or a complete Python-environment qualification.

At the readiness snapshot, free RAM was about 11.45 GiB, free Windows commit
about 2.16 GiB, and free GPU memory about 6.28 GiB. Availability is volatile;
every actual launch probes it again. The cache-only lower bounds already
exceed these resources. Complete Q4/ISTA weight sets are still downloading;
sparse file logical sizes are not completion evidence. All source files must
pass the pinned size and SHA-256 checks before packing or execution.

Reproduction commands, model pins, budget declarations and the full batch-1
protocol are in [STRATA.md](../../STRATA.md). Current milestones and the
architecture diagram are in [EDGE_ENGINE_STATUS.md](../../EDGE_ENGINE_STATUS.md).
