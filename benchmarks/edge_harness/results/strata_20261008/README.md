# Strata / shared-lease evidence — 2026-10-08

The newest [actual cancellation/recovery record](native_q4_cancel_recovery.json)
passed one text-stream lifecycle check: two chunks delivered, cancellation
and empty shared-ledger drain in 2.59 s, then a fresh worker/generation exact
short response in 11.85 s. Actual initial/recovery plans capture the same
explicit-cache controls. Full stage starts of 236.50 / 218.04 s include
rehashing and service startup. This is one functional lifecycle observation,
not stability or performance qualification, and does not fill missing plan
capture in earlier Agent records.

The earlier [explicit-cache Agent rerun](native_q4_bounded_cache_agent.json)
records two successful narrow tasks, including recall across controller
sessions. It binds new source/environment identities and observed native
INFO to explicitly labeled reconstructed cache limits. Full loaded-plan and
control hashes were not captured for this Agent run. Full responses took
266.39 / 21.51 s; the first includes hashing/loading. No p95, aggregate VRAM
hard-cap, physical SSD or release qualification follows.

The earlier [native Q4 reproduction](native_q4_reproduction.json) records three
complete Windows Omni Strata text requests, with exact English, Chinese
arithmetic and JSON-equality checks all passing. The complete-response samples
are 9.63 / 17.70 / 16.63 s; there is one sample per case, no p95 or performance
qualification. Four source shards, runtime, compatibility pack, raw receipts
and sampled telemetry are bound by hash. Loaded CPU+CUDA configuration and
routed decode expert execution are verified at their separate scopes;
physical SSD attribution and hard cache-budget qualification remain pending.
A separate actual experimental Agent smoke passed 4/4 narrow short-response,
same-session memory and structured controlled-loopback read tasks, with
controller close and process exit reviewed; it uses the same historical
auto-sized cache configuration and adds no default qualification.
No DeepSeek request or new default-route qualification is recorded.
All request protocols are batch=1 and one active request.

During these Q4 neural, Agent and cancellation/recovery smokes, ISTA Q2/IQ3
downloads and WSL
checksum work were active on the same SSD. Latencies describe functional
smokes under background disk load, not isolated performance tests.

The records and download/capacity descriptions below retain the earlier
implementation-readiness snapshot, before this successful Q4 run. Its first
conversion-manifest validation failure is also retained in the new receipt.

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
exceed these resources. At that earlier snapshot, complete Q4/ISTA weight sets were still downloading;
sparse file logical sizes were not completion evidence. The later Q4 run
verified all four source files and its prepared pack before execution. Q2/IQ3
weight verification and complete requests remain pending.

Reproduction commands, model pins, budget declarations and the full batch-1
protocol are in [STRATA.md](../../STRATA.md). Current milestones and the
architecture diagram are in [EDGE_ENGINE_STATUS.md](../../EDGE_ENGINE_STATUS.md).
