# 2026-10-02 batch-1 framework and evidence update

This update advances the Omni local framework while retaining the accepted
single-device architecture and the [rolling 60-cell review](../e2e_profiling_20260922/evidence/summary/README.md).
The review remains **25 scoped complete requests, seven synthetic policy
paths, and 28 without a complete workload; zero release-qualified cells**.
No existing component or CPU fallback is relabeled as NPU/model E2E.

## Framework progress

- Native and external stages can enter one atomic, declared resource ledger;
  native-only plans without new budgets retain their existing admission path.
  The local text planner can select a faster route only from same-checkpoint,
  qualified whole-request batch-1 evidence with a positive paired gain bound.
- The portable contract now describes opaque backend-owned state, versioned
  artifact validation, ordered/timestamped events and strict ACK-backed byte
  credit. The live local text stream also retains credit through delivery,
  while a host-only reference controller shares the v2 conformance fixture.
  The graph backend still executes **stateless** fixed-shape requests and
  rejects v2 deployment until worker compatibility is checked; a device-
  resident mobile controller and native persistent-KV integration remain
  M1/M2 work.
- New text/TTS profiling and summary checks require batch size 1, one active
  request, three lengths, at least 20 measurements per length and a separate
  30-minute sequential phase. Historical concurrency sweeps remain archived.
  The machine-readable qualification ledger marks all 60 cells open until the
  model-specific and device-local gates have evidence.

## New raw evidence and remaining gates

| Evidence | Depth | Finding and next gate |
| --- | --- | --- |
| [AI Hub target inventory](evidence/aihub_device_inventory/README.md) | D | All five exact hosted registrations exist; no exact usable-RAM SKU is exposed. Obtain that before a 27B device-specific capacity verdict. |
| [Spark fixed-cache free-running CPU check](evidence/spark_free_running/README.md) | numerical CPU reference | Own-token continuations match 128/128 across both cache boundaries, but K/V error still fails. Repair and compile a target-compatible state path before S25 E2E. |
| [Spark CUDA batch-1 profiler smoke](evidence/batch1_text_smoke/README.md) | diagnostic complete requests | Three one-sample length bands and recovery pass; 20-per-band, 30-minute, quality and power gates remain open. |
| [Spark CUDA batch-1 profile](evidence/batch1_text_profile20/README.md) | scoped complete-request timing | 20/20 requests per band and stream/recovery checks pass with raw NVML conditions; no 30-minute or new quality gate, so still unqualified. |
| [Spark CUDA batch-1 sustained profile](evidence/batch1_text_sustained/README.md) | 30-minute single-request timing protocol complete | 20/20 measured 128-token requests per length and 613 sequential medium requests over 1,801.720 s passed with ordered output. The WSL RTX Spark cell still needs reference quality, longer context and paired route comparisons. |
| [Qwen3-TTS CUDA batch-1 stream](evidence/batch1_tts_smoke/README.md) | one medium measured complete stream and ASR proxy | Seven chunks, 8.0 s audio, 117 ms first audio, exact-text warmup ASR; playback, 20-per-band, cancellation and 30-minute gates remain open. |
| [Qwen3-TTS CUDA batch-1 profile](evidence/batch1_tts_profile20/README.md) | scoped complete-request timing | 20/20 requests pass in each length band with finite PCM and zero simulated underruns; short/medium warmup ASR matches exactly. Real playback, broad quality and 30-minute gates remain open. |
| [Qwen3-TTS CUDA batch-1 sustained profile](evidence/batch1_tts_sustained/README.md) | 30-minute single-request timing protocol complete | 20/20 measured streams per length and 1,076 sequential medium streams over 1,801.557 s completed with finite PCM; one sustained request had a 9.534 ms simulated playback deficit. Three warmup WAVs passed pinned ASR intelligibility proxies (long audio in two segments). Real sound-device playback, broad speech quality and mobile placement remain open. |
| [InternVLA Place_Markpen reference-data probe](evidence/internvla_dataset_access/README.md) | named external data blocker | The matching archive exists but the current account is not approved; no real-observation/action task-quality gate can be completed with the available synthetic fixture. |
| [Mobile complete-artifact inventory](evidence/mobile_artifact_gap/README.md) | C and gap audit | All 25 hosted cells lack a complete target stage bundle/replay; existing component jobs and synthetic CPU handoff remain separate evidence. |

The five desktop source checkpoints needed for the model work are already
local. A 2026-10-02 `safetensors.safe_open` header/index audit found every
indexed shard and read its tensor directory: Spark 1.7B BF16 **2 shards /
226 tensors / 3,415,340,496 bytes**; Qwen3-TTS 0.6B CustomVoice **1 /
402 / 1,811,626,576** (HF cache snapshot `85e237c12c027371202489a0ec509ded67b5e4b5`);
Qwen3.8-27B FP8 **66 / 1606 / 30,866,866,928**; MiniCPM-o 4.5 BF16 **4 /
1414 / 18,743,737,228**; InternVLA Place_Markpen **1 / 1465 /
6,721,209,208**. This checks local presence and readable headers, not a
fresh full-byte hash or device-compiled artifact. Existing successful scoped
loads provide separate execution evidence. Additional quantized formats in
the model directory remain distinct artifacts with separate quality gates.

M0 remains accepted on its original workload and now has a separate 30-minute
batch-1 Spark CUDA timing pass; reference quality remains open. M1 has no device-resident S25
loop. M2 now has a desktop 30-minute timing pass with one simulated playback
deficit; it still needs real playable-audio, quality and mobile co-residency
validation. M3 has
real AMD components but no qualified beneficial default split. M4 has narrow
desktop paths but lacks mobile bundles and matching InternVLA task references.
Qwen3.8-27B long-context/image quality and no-NVIDIA performance remain a
separate experiment line. AI Hub model jobs can support hosted component or
host-replay evidence, not resident Omni memory, streaming or thermal claims.
