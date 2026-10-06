# Native Agent route qualification

The profiler records evidence but never promotes a model. `native_app` refuses
the old `qualification_file` list because its pass flags can be asserted without
raw requests. To enable a default route, a reviewer must provide an Ed25519
signed `qualification_bundles` entry and configure its public key in
`trusted_review_keys`. With no bundle, routes remain experimental only when an
explicit `experimental_bootstrap_route_id` is configured.

One bundle qualifies **one route × task class × exact hardware/software/power
condition**. The verifier rereads the complete-Agent JSONL, recomputes case
evaluation, event order, TTFT, p50/p95, and endurance, and requires batch=1,
one active request, separate warmups, at least 20 requests per short/medium/long
input, and 30 minutes of active sequential Agent work. The indexed source
configuration must still match the **entire live native behavior config**,
including limits, route set, and bootstrap policy. Only review references
(`qualification_bundles`, `trusted_review_keys`, and the legacy
`qualification_file`) may differ. The source config file hash, model, runtime,
checkpoint revision, projector, and lineage must also match. The profile must
record the imported Omni source SHA-256 and loaded Agent/dependency runtime
SHA-256, including the actual imported vLLM and shared stage-contract Python
sources; all must equal the currently loaded code. Installed distribution
metadata alone cannot prove which editable checkout executed. A file size,
stage profile, old profile without these digests, or short smoke cannot
satisfy this check.

The signed bundle has this shape (paths may be absolute or relative to the
bundle):

```json
{
  "schema": "omni-agent-qualification-review-v1",
  "identity": {
    "route_id": "...", "artifact_id": "...", "artifact_sha256": "...",
    "checkpoint_revision": "...", "backend": "...", "placement": "...",
    "suite_id": "...", "environment_fingerprint": "...",
    "power_condition": "AC", "profile_raw_sha256": "..."
  },
  "profile_summary": {"path": "summary.json", "sha256": "..."},
  "profile_index": {"path": "index.json", "sha256": "..."},
  "gates": {
    "memory_admission": {"path": "memory.review.json", "sha256": "..."},
    "cancel_recovery": {"path": "cancel.review.json", "sha256": "..."},
    "runtime_placement": {"path": "placement.review.json", "sha256": "..."},
    "checkpoint_lineage": {"path": "lineage.review.json", "sha256": "..."},
    "reference_quality": {"path": "quality.review.json", "sha256": "..."}
  },
  "review": {
    "key_id": "reviewer-1", "reviewer": "name", "reviewed_at": "ISO-8601",
    "signature_ed25519": "base64 signature"
  }
}
```

Each gate receipt uses `schema: omni-agent-independent-gate-v1`, its exact gate
name, the same `identity`, `outcome: pass`, and a `{path, sha256}` reference to a
separate raw JSON observation file. The raw file uses
`record_type: <gate>_evidence_v1`, the same `identity`, and gate-specific
`observations`. The verifier rejects naked pass flags:

- Memory: admitted demand and live ceiling per physical pool; measured pool
  baseline and peak, plus their exact difference as the route's incremental
  peak. The incremental peak must fit its reservation. Include loading and
  complete-request samples and an over-budget request with an explicit
  refusal reason. System-wide samples can include other processes, so the
  reviewer must inspect the raw series and sampling conditions.
- Cancellation: ordered events for a cancelled request, a later
  `state_released` event proving the tool request finished, worker process
  exited with an exit code, backend request state and Omni graph gate were
  released, and both stage and host ledgers are empty. Require a separate
  recovered request in a later epoch, with no post-cancellation answer output.
- Placement: actual placement, artifact hash, and a SHA-bound startup log with
  an unambiguous llama.cpp offload report and successful load. An override
  based CPU-expert route must show all three expert tensor overrides for each
  declared CPU layer in that log. Startup override selection alone does not
  verify final expert storage or compute placement;
  the current verifier refuses `Vulkan_Host+Vulkan0` release qualification
  from startup logs.
- Lineage: exact non-placeholder checkpoint revision and artifact hash,
  source repository, and license.
- Quality: a separate bilingual reference suite with per-case reference and
  answer hashes and passing results for this task class.

The reviewer signs the canonical UTF-8 bytes returned by
`benchmarks.edge_agent.evidence.bundle_signing_bytes(bundle)` with their
Ed25519 private key, then adds the base64 signature. The application config
includes the matching 32-byte raw public key encoded as base64:

```json
{
  "qualification_bundles": ["C:\\path\\to\\promotion.json"],
  "trusted_review_keys": {"reviewer-1": "base64-public-key"}
}
```

The signature and hashes protect the local review chain against accidental or
unreviewed edits. The reviewer remains responsible for whether observations
reflect the actual machine and whether the separate quality suite is broad
enough for the intended task. `native_app` rechecks live memory admission and
actual placement on every selected request; a reviewed route still cannot run
when resources or power condition differ.

The full Gemma fixed-memory protocol begun on 2026-10-05 loaded code before
the trusted-task browser URL policy, runtime digest capture, and strengthened
cancellation proof landed. Its [public aggregate entry](public_evidence/agent_native_smokes_20261005.json)
records a valid trace audit, 60/60 measured successes (20 per length), and
1,800.42 seconds of active endurance across 172 requests. Those measurements
remain useful historical evidence, but the run cannot qualify the current
release because it lacks the loaded-source/runtime digests and predates the
current code. A second current-source [public aggregate entry](public_evidence/agent_native_smokes_20261005.json)
has 60/60 measured fixed-memory successes, 158 sequential requests over
1,806.73 active seconds, a valid raw trace audit, and matching loaded-source
and runtime digests. It still does not qualify a default route: the independent
signed memory, cancellation, placement, lineage, and quality gates have not
been reviewed and attached, and other task classes need their own full profiles.
