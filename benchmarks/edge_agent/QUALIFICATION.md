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
input, and 30 minutes of active sequential Agent work. Every measured request
must pass the fixed-suite output, tool-safety, trace, and placement checks;
59/60 is an experiment, not a qualified default. Both the signed evidence
verifier and the router reject a partial fixed-suite result, even if other gate
flags are marked true. Among fully passing routes, whole-answer p95 breaks the
quality tie. The indexed source
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
  answer hashes and passing results for this task class. The current gate
  accepts only `comparison: exact_sha256` with equal, lowercase SHA-256
  hashes; semantic or graded quality needs its own reviewed comparator before
  it can qualify a route.

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

The four historical [Gemma fixed-memory protocols](public_evidence/agent_native_smokes_20261005.json)
each had 60/60 measured successes and 30 minutes of active sequential work
at their recorded code states. The latest of those historical runs,
`native_4bba0301e4d04f4f9af9dd45e1c6deea` has 60/60 measured
fixed-memory successes (20 per length), 611/611 sequential endurance
requests over 1,802.26 active seconds, and valid raw trace, measurement
protocol and fixed-suite case audits. Its imported source and runtime digests
matched at measurement time. Subsequent browser click/fill boundary and
approved-POST changes altered the imported Omni source, so this and the
earlier three completed runs are historical for final-code qualification.
The newer [current-source fixed-memory profile](public_evidence/agent_native_memory_current_20261006.json)
`native_17a071233dec43b892cade1e8ebbe512` passed **60/60 measured**
requests, 20 per length, and **564/564 sequential endurance** requests over
1,803.75 active seconds. Raw trace, fixed-suite case, and batch-one protocol
audits passed without errors. Its loaded Agent runtime digest is
`c15b43d95183c4cb55fb5352d112891187751f0eb203704c524d4659a80816e9`.
Whole-answer p95 was 2.58, 2.80, and 4.10 s for short, medium, and long
fixed-memory inputs. The large difference from prior memory timings has no
established cause or controlled paired comparison and must not be called a
route speedup.

A separate [profile-bound bilingual quality probe](public_evidence/agent_memory_quality_20261006.json)
passed **4/4** source/distractor, exact-answer, and deletion cases. Bound
[memory-admission and cancel/recovery diagnostics](public_evidence/agent_native_memory_diagnostics_20261006.json)
show one complete measured Agent turn, two preallocation over-ceiling refusals,
released ledgers, and an exact recovered answer after cancellation. Their
raw hashes and profile binding are public; full traces remain private. These
diagnostics have **not** been independently reviewed and signed as the five
required release gates, and the four-case suite cannot establish open-domain
memory quality. No route is a qualified default. The separate
later browser-text protocol below covers only its fixed local-page read task
and predates the approved HTTP POST path; it is also historical for
current-code release matching. Other classes require their own full profiles.

The separate Gemma artifact probe matches the pinned official GGUF and visual
projector bytes to their published LFS identities, commit and license metadata.
It explicitly does not verify base-model provenance and remains a private,
unsigned observation requiring human review. Sampled whole-system memory
increments and startup offload logs are diagnostic observations, not complete
memory-admission or per-operation placement proofs. The earlier unbound 4/4
bilingual memory observation predates the exact-content index change; the
newer bound 4/4 run above is still narrow and unsigned. An
earlier browser-vision attempt completed one warmup and one measured
browser screenshot, then stopped during setup of the next desktop case because
Windows could not verify the Edge window in the foreground. This is an
environmental blocker, not a measured desktop-model failure. All indexed
visual attempts predate the latest isolated-context network guard. The
post-hardening browser-text smoke `native_f97dc44f38024b2d82794e73b1b745de`
passed 6/6 measured requests with raw trace audit, but has only two requests
per length and no endurance. The following later fixed local-page
read run `native_9f12bd06a5074fdd97429953825be09b` completed **60/60
measured requests**, 20 per length after separate warmups, and **165/165
sequential endurance requests** over 1,808.30 active seconds. Its raw trace,
fixed-suite case, and batch-one protocol audits pass; imported-source and
runtime digests matched the reviewed code at measurement time. Subsequent
`browser_post` changes altered that source/runtime identity. Answer p95 was
9.46, 11.97, and 14.44 seconds for short, medium, and long inputs, so medium
and long missed the 10-second normal-answer target. This narrow fixed-page
read result does not qualify a default route or open-web browsing, browser
writes, desktop vision, or other task classes. Independent signed memory,
cancellation, placement, lineage, and bilingual quality gates remain open.

Every `screen_capture` now needs one-shot sensitive-read approval, even when
the trusted-task intent matcher admits the tool. URL and path tokens are masked
before intent checks; negated or interrogative wording can still be mistaken
for a request, so the matcher alone never grants pixel access. On native
Windows the challenge binds the visible foreground window's HWND, process ID,
title, and physical bounds. The desktop UI restores focus after review and
the backend rechecks that target before and after grabbing screen pixels in
those bounds. The exact `capture_scope` is
`visible_screen_pixels_within_foreground_window_bounds`; an injected backend
that cannot identify and bind the target is refused before approval or pixels.
Overlays and background visible through transparent regions can
be included; this is not off-screen window rendering. The Agent UI itself may
be the selected foreground target. A reliable target picker and a full visual
Agent task with reviewed capture approvals and reference output are still
needed; prior visual smokes cannot qualify this current path or any default
route.

The production managed browser refuses `browser_click` and `browser_fill`
before target inspection, approval, or page action. DOM form submission and
tool-initiated JavaScript write actions remain disabled; page scripts still run
during rendering, and this is not a hardened web sandbox. Separately,
`browser_post` can send one HTTP POST after exact UI review and approval. Its
frozen target binds a canonical same-origin URL, body bytes and SHA-256,
media type, current page, and HMAC fingerprint of applicable cookies; all are
rechecked before dispatch. The body is capped at 64 KiB, URL at 2,048 bytes,
and raw response excerpt at 4 KiB. Four exact media types are allowlisted;
redirects and automatic retries are disabled. Timeout or disconnection leaves
the server-side outcome unknown, so the Agent must not retry automatically.
The isolated HTTP client can differ from Edge in proxy, CA, or CSRF behavior.
Mock approval and local transport tests do not qualify a live browser-write
Agent task. It needs its own complete output and server-result evidence; a
fixed-memory or read-only browser qualification cannot extend to it.
