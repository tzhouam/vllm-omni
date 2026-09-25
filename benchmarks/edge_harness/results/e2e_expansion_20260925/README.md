# E2E expansion follow-up — 2026-09-25

The [Spark fixed-cache decode reference](evidence/spark_decode_boundary/README.md)
now runs the real 28-layer export with a bounded 511-entry sliding ring and
full-attention K/V buffers after Hugging Face CPU prefill. FP32 reference and
export agreed on 128/128 next-token choices across both 512 and 1024 context
transitions, while the BF16-reference versus FP32-export comparison exposed
a short-prefix token mismatch and up to 14.287% long-prefix logit error.
A later opt-in BF16 CPU reference recovered bitwise logits and new K/V across
128-step short, 512-crossing and 1024-crossing runs when full-attention inputs
used their filled lengths and sliding-ring reads preserved chronological
order. Fixed padded buffers still produced logit differences, reaching 9.857%
in the 500-token run. This validates a CPU state contract only; S25 still has
one-layer NPU component evidence and no resident full-generation path.

The [MiniCPM-o HX370 AMD NPU projection experiment](evidence/minicpmo_amd_npu_projection/README.md)
extracts a state-safe speech-token projection after the CPU speech head.
It passes one fixed-fixture NPU component numerical check, but the original
FP32 CPU projection is faster even before cross-OS handoff. This projection
remains component-only evidence. A separate [resampler KV cut](evidence/minicpmo_amd_npu_vision/README.md)
subsequently passed three-image BF16 suffix parity and three serial complete
image-to-text+speech requests through Omni with verified HX370 NPU execution.
The [60-cell matrix](../e2e_profiling_20260922/evidence/summary/README.md)
records that narrowly scoped experimental E2E pass; default support, paired
benefit, broader quality, cancellation and sustained power remain open.

The [InternVLA HX370 AMD NPU batch-one experiment](evidence/internvla_amd_npu_batch1/README.md)
isolated a six-frame VitisAI convolution failure to frames 1–5, then
recovered a numerical component pass with six batch-one calls. The measured
NPU boundary was replayed through the real CPU encoder suffix and policy
for synthetic numerical and action-sensitivity checks. This is component
evidence; the native full-policy NPU path remains NOT E2E.

The [MiniCPM-o WSL RTX CUDA shutdown retest](evidence/minicpmo_cuda_shutdown_fix/README.md)
repeated the real-weight image-to-text+speech path for one warmup plus 20
serial measured requests. All outputs passed and all three stage workers
retired without a forced kill or shared-memory warning after scheduler-owned
chunk transport cleanup. Small per-request terminal-marker segments still
remained until engine shutdown, so sustained-session cleanup is open. A
separate [WSL CPU BF16 image retest](evidence/minicpmo_cpu_shutdown_final/README.md)
completed one image-to-text+speech request and exited cleanly after giving
the CPU thinker 30 seconds to retire and budgeting the three serial stage
shutdowns together. The CPU run used substantial WSL RAM and swap. Neither
single-fixture result broadens model quality or speech qualification.

A later [MiniCPM-o WSL RTX sender-role retest](evidence/minicpmo_cuda_marker_fix/README.md)
corrects the preceding serial marker-lifetime finding. The model runner
already sent stage-0 output through the orchestrator, but its AR scheduler
also enqueued an empty shared-memory terminal chunk on an edge with no
sender connector. After aligning the scheduler with the existing sender-role
contract, another one-warmup/20-measured image run passed with clean
shutdown. Across 188 shared-memory samples, stage-0 segments were transient
and returned to zero during the run instead of accumulating per completed
request. Concurrent and sustained-session bounds remain unverified.
The [WSL CPU BF16 regression](evidence/minicpmo_cpu_marker_fix/README.md)
also passed one red-square image-to-text+speech request on the installed
vLLM 0.28 wheel, with clean shutdown and no stage-0 segment left after exit;
the run was heavily swapped and does not qualify sustained CPU behavior.

The [Qwen3-TTS HX370 rolling-KV NPU boundary](evidence/qwen_tts_stateful_npu_boundary/README.md)
exports the real eight-layer decoder state with source/ONNX CPU parity and
places the first transformer layer on the AMD NPU. Two early NPU-fed waveform
chunks pass a provisional 1% gate after the unchanged CPU suffix, but an
eleven-step generated-code rollout passes only 3/11 chunks as layer-0 KV error
accumulates. The complete eight-layer NPU graph has no inference evidence;
Qwen3-TTS on AMD NPU remains NOT E2E.
An eleven-step control that resets each NPU input cache to the CPU reference
still passes only 3/11 waveform chunks. This isolates the remaining quality
failure to the placed first-layer computation for this artifact, beyond its
rolling-state accumulation.
An input-projection CPU cut reduces NPU first-layer hidden error below 1%
across eleven steps, but its NPU-owned KV error grows beyond 2% and the
unchanged CPU waveform suffix passes only 10/11 chunks. The 22-frame joined
audio segment passes at 0.591% relative L2, while frame 109 misses the
per-chunk gate at 2.398%; exact CPU KV does not repair that outlier. There is
no live complete-request or transfer-inclusive benefit evidence.
