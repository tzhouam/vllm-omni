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
A tighter full-cache capacity left the physical ring at 9.878% worst logit
error; a chronological ring read reduced it to 0.511%, while an alternative
rolled-window controller reached 10.216% on the same 500-token fixture.
The tested fixed-shape alternatives remain unqualified for mobile generation.

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
The [combined-audio+image waveform follow-up](evidence/minicpmo_amd_npu_waveform/README.md)
retains CPU and CPU+NPU speech outputs. The earlier 128-token talker plan
truncated both routes. Raising its bounded cap to 256 produced two full
spoken responses per route with pinned Whisper tiny.en WER 0, exact thinker
text-token parity, and verified live VitisAI placement. Audio durations and
samples still differ; no whole-chain speedup or default NPU selection follows.
The [natural-image/human-speech MiniCPM-o follow-up](evidence/minicpmo_amd_npu_natural/README.md)
repaired the opt-in AMD NPU projection adapter's fixed 1,024-token image
shape by bounded tokenwise tiling. Two full Omni CPU and CPU+NPU requests
with astronaut/cat photos and LibriSpeech narration completed. The actual
NPU graph session handled four tiles for the three-crop astronaut and one
for the cat; each output met a same-input BF16 CPU projection gate below
1% relative L2. A 384-token talker plan allowed both NPU spoken responses
to finish in the pinned ASR proxy. Text and acoustic outputs still differ,
and the separate heavy-swap runs do not justify default NPU placement.
A [same-engine MiniCPM-o NPU abort/recovery probe](evidence/minicpmo_amd_npu_cancel/README.md)
aborted a cat audio+image request after its first text token with no late
audio, then completed a distinct astronaut audio+image request with finite
speech. The NPU worker executed one plus four KV-projection calls across
both requests and closed cleanly. Mid-NPU-call and post-PCM cancellation,
worker crashes, broad quality and whole-chain benefit remain open.

A [native Windows MiniCPM-o natural-photo follow-up](evidence/minicpmo_windows_natural_image/README.md)
completed four whole-session Omni GGUF image+spoken-question → text+speech
requests: astronaut and cat on CPU and Radeon 890M+CPU. Both routes described
the scene and emitted finite speech; pinned Whisper tiny.en WER versus the
displayed answer was 0–0.089. The first 20 s output-bound attempt refused a
long astronaut response as designed; explicit 60 s/4 MiB bounds admitted the
four passing requests. These are two images with synthetic spoken input and
one request per route/photo, so no broad quality or cross-device speedup
follows.

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

A separate [Qwen3-TTS exact-state CUDA power-limited run](evidence/qwen_tts_cuda_exact_sustained30/README.md)
exposed a 33 W RTX software cap (95 W default). All 20 measured and 32
completed sustained medium streams remained finite and ordered, but every
request missed its simulated playback schedule at RTF p50 1.418/1.435. The
planned 30-minute phase was stopped after 374 s of sustained requests. This
is a failed power-condition gate, not a full-duration profile or a paired
comparison with the earlier 20-request exact-state pass.
A [targeted exact-state abort/recovery run](evidence/qwen_tts_cuda_exact_reliability_33w_fixed/README.md)
then verified zero late PCM after cancellation, a complete fresh request,
an explicit error after deliberate vocoder-worker termination and no owned
process needing manual cleanup. The first two launch attempts failed because
the reliability harness selected FlashInfer sampling without a discoverable
CUDA compiler; matching the streaming profiler's sampler setting fixed the
launch. These lifecycle checks do not qualify playback under the 33 W cap.

The [joint InternVLA CPU+AMD NPU+Radeon recovery probe](evidence/internvla_joint_npu_radeon_recovery/README.md)
aborted a synthetic policy request after the worker started and delivered no
stale action. A fresh Omni stage loaded the same pinned placement and returned
the baseline action hash under a different worker generation. The blocking
worker is deliberately retired on in-flight abort, so the former stage cannot
accept another request. Real observation quality, paired benefit and
worker-crash recovery remain open.

A [public real-camera A2D follow-up](evidence/internvla_public_a2d_real_camera/README.md)
fixed the installed torchvision loader's missing `VideoReader` through a
PyAV fallback and prepared one pinned two-frame/three-camera A2D observation.
Seven complete Omni policy requests passed across WSL/native CPU, Radeon,
AMD NPU, joint NPU+Radeon and WSL/native RTX placements, with terminal
16-value decoded action buffers and cleared reservations. The source task is
laundry sorting, whereas the checkpoint is fine-tuned for Place_Markpen;
therefore this adds real-sensor execution evidence only, not task accuracy or
robot-control qualification. One request per route is not a latency profile.

The [native Windows CPU Qwen3-TTS recovery probe](evidence/qwen_tts_windows_cpu_recovery/README.md)
aborted a public Omni request after the BF16 CPU worker started and delivered
no stale audio. A fresh stage loaded the same checkpoint and returned the
same 109,440-frame PCM hash under a different worker generation. This extends
the existing complete-WAV profile only with fresh-stage recovery; playable
streaming and sustained real-time speech remain unqualified.

The [Spark HX370 CPU/NPU paired-order profile](evidence/spark_cpu_npu_paired_verified/README.md)
alternated CPU, CPU+AMD NPU, CPU+AMD NPU and CPU phases, with one warmup and
20 complete 64-token requests per phase. All 80 measured requests matched the
same BF16 token hash; the hybrid phases each verified a VitisAI graph node.
CPU phase p50s were 4.766/4.829 s, while hybrid p50s were 5.003/5.030 s.
The NPU split was slower by 4.2–5.0% on this fixture, so CPU remains the
default. Package/NPU power and sustained thermal behavior were not measured.

A [MiniCPM-o RTX+AMD NPU joint-route follow-up](evidence/minicpmo_cuda_amd_npu_joint/README.md)
passed complete cat-photo and synthetic red-square image-to-text+speech
requests. The BF16 thinker/talker/codec and vision suffix used RTX CUDA,
with 5 GiB of thinker weights offloaded to host RAM by the vLLM plan;
only the real resampler KV projection crossed to the HX370 NPU. Raw ORT
placement, bounded CUDA↔host transfers, checkpoint/graph hashes and retained
waveforms are audited. Both routes' generated speech matched their own text
under a pinned Whisper tiny.en proxy, but the joint and RTX-only texts and
output lengths differ. These two separate runs show no paired benefit or
default-placement qualification.

A [Spark BF16 RTX+AMD NPU joint-route profile](evidence/spark_cuda_amd_npu_joint_verified/README.md)
then completed one warmup and 20 serial 64-token requests on each of RTX-only
and RTX-decoder+HX370-NPU-head placements. All 42 requests returned the same
greedy token sequence. Raw VitisAI placement and CUDA top-64 BF16 re-ranking
were verified; the NPU worker peak fit its explicit shared-RAM reservation.
Complete-request p50 was 0.728 s RTX-only versus 1.438 s joint, so the
unsplit RTX route remains the default. This one prompt, one phase order and
uncontrolled power state do not qualify broad quality or performance.

An [independent Qwen3-TTS fixed-history control](evidence/qwen_tts_amd_npu_cut100_independent/README.md)
tested the earlier 72-history/two-frame Code2Wav artifact on a second
generated utterance. Before NPU execution, its CPU waveform differed from the
matching full-decoder segment by 1.3115% at frame 2 and 2.2341% at frame 23,
exceeding the provisional 1% chunk gate. This confirms that the fixed-history
prefix cannot be promoted to a continuous NPU stage without a source-faithful
rolling state contract. The independent NPU MLP component finding remains
valid, but Qwen3-TTS on AMD NPU is still NOT E2E.

A [full eight-layer rolling-state CPU follow-up](evidence/qwen_tts_full_state_npu_extended/README.md)
then ran eleven consecutive ONNX steps on each of two generated utterances.
The exported graph owned its rolling K/V, and the unchanged vocoder tail's
worst waveform relative L2 against the checkpoint's exact decoder was
3.34e-5 and 2.05e-6. This validates the source-equivalent CPU state contract
more broadly than the earlier two-step check. The same eight-layer graph
subsequently compiled on HX370 AMD NPU after 1,228.444 s and executed two
consecutive steps with verified VitisAI placement. Captured NPU hidden states
missed the 1% tensor gate at 1.891%/2.074%; their unchanged CPU vocoder tail
missed the 1% waveform gate at 1.924%/1.036%. Its two observed NPU calls were
also slower than corresponding CPU graph calls before handoff. This is
executed component evidence, not a complete TTS stream or qualified NPU stage.
An audit of the retained 16 K/V outputs first crosses the 1% tensor gate at
layer 2 on step one and layer 1 on step two; output-instrumented diagnosis is
tracked separately from the original executed graph.
The [output-instrumented full-state replay](evidence/qwen_tts_full_state_npu_checkpoints/README.md)
then retained exactly the same 17 original CPU and NPU outputs, two VitisAI
events and 26 CPU events. Its added checkpoints found the layer-0 hidden
output already 1.3449%/1.6589% from CPU on the two steps. This localizes an
observed boundary, not an operation, and leaves the NPU TTS cell NOT E2E.
