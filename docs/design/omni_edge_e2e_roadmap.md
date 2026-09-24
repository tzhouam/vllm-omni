# Local PC/mobile end-to-end support and profiling roadmap

This plan covers the [60 named device/model pairings](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md).
Each pairing must finish with either a qualified local pipeline or a documented
feasibility/implementation blocker. It does not promise that every model fits
every device. The [current profiling review](../../benchmarks/edge_harness/results/e2e_profiling_20260922/README.md)
records measured execution separately from release qualification.

The [2026-09-23 recovery run](../../benchmarks/edge_harness/results/e2e_recovery_20260923/README.md)
adds a functional CPU TTS stream with playback underruns, a scoped RTX
Qwen3.8 text/image pass, verified MiniCPM-o and InternVLA weights, a MiniCPM-o
three-stage text-to-speech functional pass after correcting request embeddings,
and an InternVLA real-weight synthetic action forward. Model-quality and device gates
remain open; the 2026-09-22 audit remains the historical baseline.

The [2026-09-24 Spark BF16 WSL CPU evidence](../../benchmarks/edge_harness/results/e2e_expansion_20260924/evidence/spark_bf16_wsl_cpu/README.md)
adds a complete pinned dense checkpoint to the fork-branch M0 text route. Twelve
128-token requests passed and exactly matched standalone vLLM greedy tokens;
one warmup plus 20 serial 64-token requests measured complete-request
p50/p95 **5.411/5.751 s** and TTFT **0.129/0.146 s** on HX370 WSL CPU.
This closes only the stated BF16 text workload, not M1 mobile generation.
For M2, a [two-frame S25 vocoder export](../../benchmarks/edge_harness/results/e2e_expansion_20260924/evidence/qwen_tts_vocoder_qualcomm/s25/qnn_short_chunk/README.md)
matched local eager/ONNX CPU on one fixed window, but its FP16 QNN DLC failed
NPU-requested inference at the hosted device-memory limit. The earlier
25-frame FP16 artifact failed the same memory gate. Different precision or
graph layout, and a complete device-local stream, remain separate work.
A [TFLite GPU-requested two-frame S25 component](../../benchmarks/edge_harness/results/e2e_expansion_20260924/evidence/qwen_tts_vocoder_qualcomm/s25/tflite_short_chunk/README.md)
returned finite audio at 2.325% relative L2 against local ONNX CPU on one
fixture. Its 100-sample component profile measured p50/p95 0.559/1.052 s
for 0.16 s audio, with 403 CPU and 335 GPU node rows and median component
RTF 3.49. It still needs listening quality and playable complete streaming
before M2 can advance.

The [Spark hosted Qualcomm component expansion](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/spark_s24_full_attention/README.md) advances M1 only to one real fixed-shape decoder attention layer. The same W8A16 QNN DLC ran on S24, Snapdragon X Elite CRD and SA8775P ADP NPUs with identical outputs on one pinned fixture; S25 had prior separate component evidence. On [RB3 Gen 2](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/spark_rb3_full_attention/README.md), the tested W8A16 graph failed QNN loading even after an exact-device compile. FP32 ONNX ran on CPU with near-source parity, while calibrated W8A8 ran on NPU but had 20.9% hidden-state relative L2 and remains numerically unqualified. These are component C results, not full M1: device-local 28-layer prefill/continuous decode, KV/ring, sampling, token quality, admission/cancellation, complete-request timing and sustained power/thermal gates remain open. AI Hub is the test facility, not a deployment dependency.

The [Qwen3-TTS Code2Wav Qualcomm follow-up](../../benchmarks/edge_harness/results/e2e_expansion_20260924/evidence/qwen_tts_vocoder_qualcomm/README.md) adds a retained same-fixture numerical check to M2. A fresh SA8775P GPU-requested output saturated 96.63% of samples and differs from original ONNX CPU by 11.832 relative L2, so that route fails the numerical gate despite profiling. The same binary's CPU-requested output on the exact SA device matched original ONNX CPU within 4.13e-6 relative L2; its 100-sample CPU-requested component profile passed placement (738/738 execution-detail rows on CPU) at nearest-rank p50/p95 5.384/5.534 s, still without a full stream. A FP32-preserving GPU-option control still saturated 84.89% of samples, so that option did not fix the tested route. RB3 TFLite GPU-requested inference returned an unsaturated but 15.73%-relative-L2 divergent waveform; its fresh 100-sample mixed CPU/GPU component p50/p95 was 3.099/3.133 s. The same binary requested on exact-RB3 CPU matched original ONNX CPU within 4.13e-6 relative L2 on the retained fixture; its 47-sample component profile measured p50/p95 12.727/13.424 s with all 738 execution-detail rows on CPU. That measured component latency is a major blocker for playable streaming on this route. A one-fixture-calibrated RB3 INT8 QNN vocoder returned a finite but 66.93%-relative-L2 divergent waveform (3.49 dB SNR) versus ONNX CPU, with 100 NPU component samples at p50/p95 1.971/1.992 s and all 728 execution-detail rows on NPU; it fails the numerical gate even on its calibration fixture. X Elite ONNX CPU inference matched the local ONNX Runtime output with 8.09e-7 relative L2; its 94-sample component profile was p50/p95 6.384/6.644 s with all execution-detail rows on CPU. Corrected DirectML inference and profile both failed during provider initialization; separate FP16 QNN DLC inference returned an unsaturated waveform at 1.138% relative L2 / 38.88 dB SNR versus local ONNX CPU, but its placement profile timed out during QNN graph preparation without samples. A separate SA8775P FP16 QNN DLC vocoder compile passed, but NPU-requested same-fixture inference failed after compilation without waveform or placement samples. Fresh S24 and S25 CPU-requested TFLite vocoder inferences matched ONNX CPU within 4.04e-6 relative L2 on the retained fixture. S25 measured 100 component samples at p50/p95 4.418/4.641 s with all 738 execution-detail rows on CPU; S24 measured 100 at 4.826/4.958 s with all 738 rows on CPU. Separate explicit INT8 and W8A16 QDQ ONNX exports checked on local ORT CPU failed the same-fixture numerical gate at 246.97% and 331.83% waveform relative L2, respectively. S25 CPU talker inference preserved token 80 and its 100-sample component profile measured p50/p95 30.161/31.301 ms with all 1,800 rows on CPU. M2 still requires a quality-passing talker/predictor/vocoder set, device-local co-residency and handoff, playable streaming and sustained memory/power validation.

The identical SA/RB3 TFLite vocoder binary also matched original ONNX CPU within 3.45e-6 relative L2 on a separate local HX370 x86-64 LiteRT 2.2.0 CPU control. Exact-SA and RB3 CPU-requested parity narrow the failures to their GPU-requested routes on the tested fixture. The FP32-preserving GPU option did not fix it, so per-node placement and delegate-operation isolation remain open.

A separate [28-layer talker decode-step attempt](../../benchmarks/edge_harness/results/e2e_expansion_20260924/evidence/qwen_tts_talker_qualcomm/README.md) retains a synthetic fixed-cache fixture and CPU reference. Historical FP16 QNN context-binary inference executed on S24, S25, SA8775P and X Elite, but all four changed the CPU top-1 speech token with 8.57–9.43% logits relative L2. RB3 rejected that floating-point-input artifact. Explicit X Elite ONNX CPU and SA8775P TFLite CPU fallback compiles passed. X Elite CPU-requested inference preserved CPU top-1 token 80 with logits relative L2 2.07e-5; its 100-sample decode-step profile measured p50/p95 191.414/309.428 ms with all 1,744 execution-detail rows on CPU. SA8775P TFLite CPU-requested talker inference also preserved token 80 with logits relative L2 3.33e-5; its 100-sample decode-step profile measured p50/p95 52.635/76.925 ms with all 1,800 execution-detail rows on CPU. An RB3 TFLite CPU-requested step preserved token 80 with logits relative L2 2.29e-5 and maximum new-cache relative L2 3.04e-5; its 100-sample profile measured p50/p95 138.022/139.642 ms with all 1,800 execution-detail rows on CPU. A new S24 TFLite CPU-requested talker step preserved token 80 with logits relative L2 3.33e-5 and maximum new-cache relative L2 3.41e-5; its 100-sample profile measured p50/p95 36.976/43.264 ms with all 1,800 execution-detail rows on CPU. These fixed-step CPU component results do not establish a stateful talker loop or full TTS stream.

For M4, a [MiniCPM-o 4.5 speech-head study](../../benchmarks/edge_harness/results/e2e_expansion_20260924/evidence/minicpmo_tts_qualcomm/README.md) recovered the historical 20-layer ONNX source and matching S24/S25 fixtures. Both FP16 context-binary outputs preserved the ONNX CPU top-1 speech token on one decode step, with logits relative L2 0.346%/0.360%. Their 100-sample component profiles had p50/p95 8.537/9.006 ms and 7.197/7.615 ms, with all 1,029 execution-detail rows on NPU for each. New FP16 QNN DLC compiles passed for SA8775P, X Elite and RB3. SA8775P same-fixture inference preserved CPU top-1 speech token 1867 with 0.3568% logits relative L2 and 1.225% maximum K/V relative L2; its 100-sample component profile measured p50/p95 11.901/13.038 ms with all 944 execution-detail rows on NPU. X Elite same-fixture inference also preserved token 1867 with 0.3170% logits relative L2; its 100-sample component profile measured p50/p95 9.905/11.037 ms with all 944 execution-detail rows on NPU. RB3 FP16 QNN inference failed at graph load despite compile success; a TFLite CPU fallback same-fixture inference preserved top-1 token 1867 with logits relative L2 1.26e-6 versus ONNX CPU; its 100-sample CPU-requested profile measured p50/p95 59.518/71.347 ms with all 1,008 execution-detail rows on CPU. This is component C numerical evidence only, not thinker/encoder/vocoder or complete multimodal support.

The [native Windows Qwen3-TTS sustained audit](../../benchmarks/edge_harness/results/e2e_expansion_20260924/evidence/qwen_tts_sustained_drift/README.md) identifies an additional M2 desktop blocker. The historical RTX 5090 Laptop complete-model stream changed regime around minute 26–27; all 17 requests in minutes 27–30 had simulated playback underruns and median RTF 1.331. The separate WSL run had no sustained underruns. A completed current-source 30-minute repeat with timestamped device-wide GPU telemetry did not reproduce the collapse: 119 requests in minutes 27–30 had median RTF 0.185 and three brief simulated underruns. The run had 69 simulated underrun requests within the full 30-minute windows. Different source/import conditions and uncontrolled ambient load prevent a causal explanation; underrun-free native Windows playback remains unqualified.

For M4, a [Galaxy S25 Cosmos encoder attempt](../../benchmarks/edge_harness/results/e2e_expansion_20260924/evidence/internvla_cosmos_qualcomm/README.md) repackages the real Place_Markpen FP32 ONNX export after retaining two CPU parity fixtures. FLOAT16 QNN DLC inference returned a latent within 0.0524% relative L2 of the source on one synthetic pattern. Injecting that output into the real policy on local CPU changed synthetic actions by 0.742% relative L2, without a task tolerance. S25 six-frame component profile returned 100 samples at p50/p95 49.665/51.382 ms, with all 121 execution-detail rows on NPU; the action policy did not run on S25. The same encoder compiled to FP16 QNN DLC for X Elite CRD; same-fixture NPU-requested inference returned a finite latent at 0.0608% relative L2 versus source, with its 100-sample six-frame profile at p50/p95 82.939/83.837 ms and all 121 execution-detail rows on NPU. An SA8775P FP16 QNN DLC inference returned a finite same-fixture latent at 0.0613% relative L2 versus source; its 100-sample six-frame profile measured p50/p95 116.021/116.327 ms with all 121 execution-detail rows on NPU. This is component/action-sensitivity evidence, not device-local M4 qualification. The checkpoint train config names a2d_real/a2d_pen_holder; an [authenticated access check](../../benchmarks/edge_harness/results/e2e_expansion_20260924/evidence/internvla_real_observation_access/README.md) returned HTTP 404. The official vLLM-Omni example names an InternData-A1 Place_Markpen archive, but that dataset returned HTTP 403 gated-access denial under the current account, so real-observation reference-action quality is still unmeasured.

The same pinned Cosmos encoder now has [Galaxy S24 FP16 QNN](../../benchmarks/edge_harness/results/e2e_expansion_20260924/evidence/internvla_cosmos_qualcomm/s24/audit_report.json) and [RB3 ONNX CPU](../../benchmarks/edge_harness/results/e2e_expansion_20260924/evidence/internvla_cosmos_qualcomm/rb3_cpu/audit_report.json) same-fixture numerical results at 0.0613% and 2.20e-6 relative L2 versus source. S24 measured 100 six-frame component samples at p50/p95 64.314/67.472 ms with all 121 execution-detail rows on NPU; RB3 measured 46 samples at 13.018/13.053 s with all 590 rows on CPU. Hosted-latent handoffs to the real policy on local CPU changed fixed synthetic actions by 0.631% and 0.594% relative L2, respectively. Neither is a complete device-local policy or a real-observation task-quality result.

The [WSL CPU expansion](../../benchmarks/edge_harness/results/e2e_expansion_20260923/README.md)
adds constrained three-stage MiniCPM-o text-to-speech requests and a CPU-only
InternVLA synthetic policy forward. A subsequent 20-request serial MiniCPM-o
profile measured p50 21.86 s/p95 22.97 s request wall time after one warmup, with
swap use under the 30.91 GiB WSL limit. It does not qualify speech quality,
concurrency, streaming, loading peak or sustained behavior. A separate native
Windows RTX 5090 Laptop MiniCPM-o three-stage text-to-speech request passed
with explicit 900/600 s startup limits after the default 300 s overall timeout
had expired. A later native Windows 20-request serial run measured p50 18.38 s
and p95 18.73 s, but sampled host available RAM fell to 0.78 GB and pagefile
use reached 11.17 GB. Native Windows InternVLA direct-policy runs separately
passed on CPU and RTX 5090 Laptop with synthetic inputs; AMD and mobile cells
do not inherit these passes.

A [native Windows Radeon 890M InternVLA Cosmos encoder probe](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/internvla_cosmos_radeon890m/README.md)
now establishes only component depth C: the real CI8x8 image encoder weights
and output were checked on DirectML, with one warmup and 20 measured patterned
256×256 encodes at 26.56/27.49 ms p50/p95 including transfer and readback,
versus 156.49/158.71 ms for the same CPU source. Relative L2 error was
4.73e-07. The separate older DirectML runtime cannot host the current Omni
policy directly. A follow-up fixed-six-frame FP32 export ran in the existing
Omni external DirectML worker while the real-weight BF16 action policy stayed
on WSL CPU. Twenty paired synthetic complete-policy requests after one warmup
gave hybrid p50/p95 2.952/3.322 s versus CPU 3.178/3.266 s, with 16/20 paired
calls faster but a worse hybrid tail. Action relative L2 difference was 0.9007%
without a task tolerance. This advances M4 to a synthetic CPU+iGPU policy
path, while real observations/reference actions, units/time, shared-RAM
admission, cancellation and sustained benefit remain open. It does not turn
the joint CPU+iGPU+NPU matrix cell into a pass.

A later [bounded Omni InternVLA whole-policy stage](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/internvla_omni_policy/README.md)
closes the admission and basic cancellation gaps for this **synthetic** path.
The strict-loaded CPU policy and optional FP32 Radeon Cosmos worker each ran
one warmup and 20 serial patterned requests with terminal action events.
Omni CPU complete-request p50/p95 was 3.175/3.219 s; CPU+Radeon was
3.120/3.197 s in a separate run. A 16 GiB shared-RAM reservation, 8 GiB
pre-launch refusal, a public hybrid `AsyncOmni.generate` request, and
in-flight hybrid abort with no stale output/remaining reservation passed.
The hybrid action difference from CPU was 2.305% relative L2 on this fixture,
without a task tolerance. A first hybrid shutdown quarantined its reservation;
the corrected process-tree retirement and reruns ended at zero. Real
observation/reference-action quality, physical units/order/step time,
post-cancel restart, loading peak and sustained benefit remain open.

The [native Windows CPU InternVLA Omni stage](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/internvla_omni_windows_cpu/README.md)
also ran one warmup plus 20 serial synthetic complete-policy requests under
a 16 GiB shared-RAM ceiling: p50/p95 4.878/4.902 s, with 20/20 identical
finite actions. The public action entrypoint, 8 GiB pre-load refusal and
in-flight abort passed. Native-vs-WSL CPU action relative L2 difference on
the same fixture was 2.1795%, without a task tolerance. This deepens the
native Windows CPU cell, not any robot-task or mobile qualification.

The [native Windows CPU+Radeon Omni stage](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/internvla_omni_windows_hybrid/README.md)
then ran the same fixture with a pinned FP32 six-frame Cosmos graph in the
native DirectML worker and the BF16 policy on CPU. Twenty of twenty requests
returned identical finite actions; complete-request p50/p95 was
4.575/4.725 s versus the separate native CPU run's 4.878/4.902 s. Its
public action output, 8 GiB pre-load refusal and in-flight abort passed
under a 16 GiB shared-RAM reservation. The selected graph output device was
Radeon 890M, but per-operator placement was not measured. Action relative
L2 difference was 2.4391% without a task tolerance; the hybrid startup was
4.72 s longer. Real task quality and sustained benefit remain open.

The [native Windows Radeon MiniCPM-o repeated profile](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/minicpmo_omni_radeon_profile20/README.md)
passed 20 serial combined synthetic audio+image to text+speech requests after
one warmup, but complete-request p50/p95 was 87.27/91.45 s and a separate
cold rerun was 88.01 s. The earlier single cold Omni request was 18.27 s;
all 21 successful new worker logs show Windows prefetch-memory warnings and
roughly 25–27 s model loads rather than 6.07 s. The placement remains scoped
functionally, while repeated latency and its cause remain unqualified. The
next M4a performance experiment must capture matched CPU/Radeon host-memory,
pagefile, shared-GPU-memory, file-I/O and power traces, not infer a speedup
from the earlier single sample.

The WSL RTX MiniCPM-o image-to-text+speech path now passes for one 448×448
synthetic red-square image under an explicit 0.61 thinker GPU budget. The
previous 0.58 budget failed the 2,048-token KV admission gate; the successful
run still force-killed stage 0 during shutdown and reported a shared-memory
cleanup warning. A separate 2 s, 440 Hz synthetic tone now traverses audio
input to text+speech and yielded a plausible whistling description, also with
a shared-memory cleanup warning. Native Windows RTX also completed these two
separate synthetic image and tone inputs through all three stages, returning
the same respective text and thinker token IDs as WSL with nonzero WAVs.
Broader image/audio quality, video, repeated multimodal latency and clean
lifecycle remain open.

WSL CPU MiniCPM-o now also completes one synthetic red-square image and one
440 Hz tone through thinker, talker and vocoder under separate one-item plans.
Both return plausible text and nonzero WAVs, but CPU wording/token IDs differ
from GPU, swap is heavily used, and stage 0 requires forced shutdown with
shared-memory cleanup warnings. These single requests do not inherit the
20-request text latency profile or establish real image/audio quality.

A separate [pinned GGUF C++ route](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/minicpmo_cpp_cpu/README.md)
completed one **combined** synthetic image+audio to text+speech request on WSL
CPU. Its [native Windows CPU follow-up](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/minicpmo_cpp_windows_cpu/README.md)
completed the same input path with a corrected bounded Token2Wav wait and an
x86-64-v3 build: 50 verified WAV chunks/49.16 s audio in one 137.34 s cold
process, with 13.10 GB maximum sampled RSS. The initial native attempt
[exited 0 with incomplete speech](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/minicpmo_cpp_windows_cpu_attempt1/README.md),
and is retained as a failure. Both complete C++ responses were generic rather
than input descriptions; the fixture supplied no explicit description task,
so quality remains unqualified. These earlier standalone runs did not establish
an Omni stage, admission, cancellation, streaming or M4 video/real-observation gate.

The same GGUF set now also completes [native Windows Radeon 890M+CPU combined
input](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/minicpmo_cpp_radeon890m/README.md).
The runtime logged all 37 language layers and vision on the Radeon Vulkan
device, TTS weights with zero GPU layers, and CPU Token2Wav. A separate
[spoken visual question](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/minicpmo_cpp_radeon890m_prompted/README.md)
about the red square returned the correct sentence and 2.56 s speech in one
18.63 s cold process; input and output ASR word error rates were both zero.
The [same prompted CPU run](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/minicpmo_cpp_windows_cpu_prompted/README.md)
returned identical text and eight language token IDs in 24.66 s, though speech
PCM differed. This advances the M4 desktop feasibility evidence for a
CPU+iGPU hybrid; neither the one-case quality check nor the standalone C++
control path establishes streaming, real-input quality, mobile artifacts or
sustained profiling.

The [Omni MiniCPM-o GGUF follow-up](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/minicpmo_omni_cpp/README.md)
now binds that complete audio+image → text+speech request as one graph stage on
native Windows CPU and Radeon 890M+CPU. The same pinned weights and spoken
red-square fixture produced the correct answer and a complete WAV on both;
one cold complete request took 24.34 s on CPU and 18.27 s on Radeon+CPU.
Artifact hashes, actual CPU/Vulkan placement, 24 GiB shared-RAM reservation,
terminal event/acknowledgement and in-flight cancellation with zero remaining
ledger reservation were checked. An 8 GiB demand was refused. This closes the
scoped desktop GGUF Omni stage-binding task, not M4 acceptance: the C++ worker
does not stream playable chunks, post-cancel restart is untested, and the
single synthetic case does not establish broad multimodal quality, loading
peak, concurrency, video, mobile artifacts or sustained power behavior.

The same [GGUF Omni whole-session stage on WSL CPU](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/minicpmo_omni_wsl_cpu/README.md)
now passes a combined spoken visual question to text+speech on CPU-only C++
execution. One warmup plus 20 serial complete requests all returned the correct
red-square answer and complete speech, with nearest-rank wall p50/p95
22.67/24.83 s. The first measured WAV independently transcribed exactly with
pinned Whisper tiny.en; the 20 speech durations varied, so this is one scoped
intelligibility check rather than a quality or alignment qualification. A
24 GiB WSL shared-RAM reservation, 8 GiB refusal and in-flight cancellation
with zero remaining ledger passed. This closes the WSL CPU GGUF Omni stage
binding for the named input, while real input suites, video, playable streaming,
loading peak, concurrency and sustained behavior remain M4 work.

Native Windows HX370 CPU now also completes two Qwen3-TTS CustomVoice
text-to-WAV requests through Qwen's standalone PyTorch wrapper using the pinned
0.6B checkpoint. A separate 1-warmup/20-measured serial profile of one short
prompt took p50/p95 13.94/14.68 s to generate 4.56 s of audio. Whisper tiny.en
ASR gave word error rates 0.20 and 0.00 for the two WAVs, an intelligibility
proxy rather than a quality pass. The [raw record](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/qwen_tts_native_cpu/README.md)
documents the isolated Transformers 4.57.3 dependency and short process-memory
trace. The subsequent [Omni CPU whole-session stage](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/qwen_tts_omni_cpu/README.md)
matched both standalone PCM hashes exactly, returned a public `AsyncOmni`
audio event, and completed one warmup plus 20 serial measured requests at
p50/p95 15.32/15.60 s for 4.56 s of audio. It reserves 10 GiB host RAM,
rejects an insufficient 4 GiB demand before worker launch, and drains an
in-flight cancellation without stale output or remaining reservation. This
closes the scoped CPU stage-binding task. Playable streaming, post-cancel
restart, loading-peak admission, broad speech quality and sustained
power/thermal tests remain M2 work.

A separate native Windows CPU+Radeon 890M Qwen3-TTS route now completes two
named text-to-WAV requests in standalone CrispASR using a Q8_0 talker, F16
codec, Vulkan talker/codec and CPU FP32 code predictor. Whisper tiny.en returned
both reference sentences exactly; one warmup and 20 serial resident-server
requests measured p50/p95 3.310/3.345 s for 2.56 s of audio. The
[placement and failed-route record](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/qwen_tts_radeon890m_joint/README.md)
shows that default Vulkan silently becomes CPU for this backend, all-Vulkan
generation ran away, and DirectML did not pass quality. The explicit hybrid
route remains outside Omni and slower than real time. Its CPU+iGPU evidence
does not advance the NPU joint-execution row or mobile M2 gate.

## Architecture and prerequisites

Keep Omni's PipelineConfig, StageRuntime, StageClient, orchestration and admission
as the control plane. Keep vLLM's model execution, caches and internal batching.
Other runtimes implement stage backends. Model adapters own preprocessing,
sampling, modality alignment, state semantics and output interpretation.

The portable boundary is request/events, tensor or buffer contracts, opaque
state handles, device capabilities, artifact manifests and execution plans.
Persistent weights, KV/ring/recurrent state and workspaces stay with the backend.
Prefer one backend per autoregressive session; split coarse stages only after
measuring a complete-chain benefit.

Android may use a small native controller derived from Omni's stage, session and
streaming semantics, constrained by the same contract tests. A full Python/vLLM
service port is not a prerequisite. AI Hub remains a validation facility rather
than a deployment dependency.

Before advancing individual cells:

1. Pin installed runtime builds and checkpoint revisions. Repair the CPU
   vLLM/Omni mismatch and native Windows CPU attention-operator gap independently.
2. The complete MiniCPM-o 4.5 and InternVLA-A1 source checkpoints are now
   SHA-256 verified. Record export, quantization, calibration, compiler, shape
   buckets, state layout and quality evidence before target-specific claims.
3. Integrate native and external stages into the shared admission ledger.
   CPU/iGPU/NPU share physical RAM; Windows availability and WSL quota are
   separate constraints on that RAM. Budget loading peaks, weights, state,
   activations, workspace, transfers and safety margin. Discrete VRAM is separate.
4. Require bounded queues, consumer acknowledgement, cancellation propagation,
   sequence/epoch fencing, state retirement and actual placement reporting.
   No silent precision/model/context changes or CPU fallback labelled as NPU.
5. Record exact SKU/RAM, OS, drivers, installed packages, power state and device
   access. A local mobile application needs full on-device execution access;
   AI Hub component jobs alone do not establish that access or an E2E pipeline.

Implementation ownership follows the existing source boundaries:
[portable contracts](../../packages/omni-stage-contracts/omni_stage_contracts),
[host routes and process lifetime](../../vllm_omni/host),
[StageClient](../../vllm_omni/engine/stage_client.py) and
[backend adapters](../../vllm_omni/engine/backends),
[StageRuntime](../../vllm_omni/engine/stage_runtime.py), and the
[resource ledger](../../vllm_omni/engine/resource_ledger.py).
The current `GraphStageClient` represents one bounded, non-preemptible graph
call. Persistent mobile sessions require an explicitly negotiated backend
capability and lifecycle implementation; a state-handle type alone is insufficient.

## Work packages and completion gates

| Order | Work package | Implementation | Gate before wider rollout |
|---|---|---|---|
| Foundation | Runtime, contracts and accounting | Close the prerequisites above; reuse the existing qualification runners | Reproducible launch or explicit pre-load refusal, correct state ownership and placement, auditable memory budget |
| M1 | S25 Spark | Device-local Omni-compatible controller; embedding, prefill, continuous decode, ring/full KV, head and sampler; resident state | Reference checks, at least 128 output tokens, 512-window/1024-bucket transitions, cancellation, resident memory and sustained profile |
| M2 | Qwen3-TTS | Resolve desktop playback and shutdown findings and CPU compatibility; then mobile talker/predictor plus GPU vocoder | Complete PCM and tail, history/chunk correctness, interruption/recovery, quality checks, no post-startup underruns and RTF below 1 in the declared workload |
| M3 | AMD stages | Connect compatible real encoders or other useful coarse stages through current Omni workers; start from the existing 890M vision evidence | Actual node/device placement, numerical and task quality, complete downstream output, shared-memory and handoff costs; reject unhelpful splits |
| M4a | MiniCPM-o 4.5 | Three-stage desktop text+WAV produced coherent text and nonzero WAV on RTX 5090 Laptop under WSL and native Windows, and WSL CPU, using separate constrained plans. One synthetic red-square image and one synthetic tone each passed through the three-stage WSL RTX, native Windows RTX and WSL CPU paths to text and nonzero WAV. RTX text/token IDs matched across OS environments; CPU wording/token IDs differed while remaining plausible. The WSL RTX image plan needed a thinker GPU budget increase from 0.58 to 0.61 for KV admission. WSL multimodal shutdown cleanup remains open, including forced stage-0 termination on CPU. The CPU plan needed vLLM 0.28/0.29 processor compatibility and used swap; 20 serial **text** CPU requests measured p50 21.86 s/p95 22.97 s. Native Windows needed longer startup limits; 20 serial text requests measured p50 18.38 s/p95 18.73 s with host RAM/pagefile pressure. Add video encoder and persistent-state reference before mobile artifacts | Speech intelligibility/alignment, loading-peak admission, concurrency, broad image/audio-understanding suites, video, shutdown lifecycle, then combined streaming and interruption |
| M4b | InternVLA-A1 | Strict-load synthetic policy forwards pass on WSL and native Windows CPU/RTX. Bounded Omni whole-policy stages pass on WSL CPU, WSL CPU+Radeon Cosmos, native Windows CPU and native Windows CPU+Radeon Cosmos with explicit shared-RAM admission, terminal action metadata and in-flight cancellation; their public CPU/hybrid entrypoints pass where measured. This is not robot-task qualification. | Reference action agreement on real observations, physical units/order/step time, observation age and deadlines; post-cancel restart, loading peak, concurrency, power and thermal behavior. |
| Independent | Qwen3.8-27B | Public Omni pipeline binding; text, then image/video and longer contexts; same-model CPU/iGPU artifact and explicit offload evaluation | Modality-specific quality, actual loading/runtime memory and measured performance; each precision/context receives its own qualification |

The initial M0 desktop text acceptance remains scoped to its tested checkpoint,
precision and workload. New concurrency, state-boundary or quality evidence is
required before broadening that acceptance. Current batch-dependent greedy
outputs need reference analysis; timing alone cannot identify their cause.

The 2026-09-23 [Spark GGUF Omni follow-up](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/spark_omni_llamacpp/README.md) adds a separate native Windows CPU/Radeon 890M whole-session option. Its pinned 1.7B Q4_K_M artifact passed two named complete text requests and 20 serial inventory requests per device through `StageRuntime`/`StagePool`; public `AsyncOmni` short requests passed on both devices. It verifies CPU or Vulkan1 placement, explicit 4 GiB host reservation, terminal event/acknowledgement and cancellation drain. This does not change the vLLM M0 acceptance or establish M1 mobile generation. Its next gates are incremental output, post-cancel restart, broader token/quality parity, loading/device-memory peaks, concurrency and sustained power/thermal behavior.

The [Qwen3-TTS Omni hybrid follow-up](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/qwen_tts_omni_hybrid/README.md) adds one complete-request stage for the native Windows HX370 CPU+Radeon 890M route. A pinned Q8_0 talker and F16 codec produced two named PCM outputs exactly matching the prior standalone backend; one warmup plus 20 serial requests measured 3.298/3.336 s p50/p95 for 2.56 s of audio. Public `AsyncOmni` audio, cancellation drain and explicit 8 GiB host admission passed. This is progress toward M2 integration, not M2 acceptance: RTF remains above 1, and playable chunk streaming, post-cancel restart, loading/iGPU peaks, broader voice quality and 30-minute thermal behavior remain open. No NPU participated.

The [WSL CPU Qwen3-TTS Omni follow-up](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/qwen_tts_omni_wsl_cpu/README.md)
adds the same bounded whole-session CPU stage on Ubuntu/WSL with a pinned
worker-only dependency overlay. One warmup plus 20 serial complete requests
produced identical 3.44 s WAVs at nearest-rank wall p50/p95 8.27/8.47 s,
excluding worker startup. The public audio path, one-output Whisper proxy
(WER 0), 10 GiB admission, 4 GiB refusal and cancellation drain passed. WSL
PCM differs from the same-checkpoint native Windows result; its numerical
cause remains open. The existing vLLM 0.28 short-stream path still has playback
underruns. These complete-WAV findings advance M2 desktop coverage but do not
establish playable streaming, RTF below one, broad quality or sustained use.

VLA qualification here ends at action data. Physical robot control, collision
avoidance and actuator safety belong to the external controller.

## Hardware routes

Every row below covers all five model families through the work packages above.
Specific artifact/backend feasibility is a gate, not assumed from the route.

| Execution configuration | Route to qualify | Main additional gate |
|---|---|---|
| CPU only, WSL | Compatible vLLM CPU; a model-specific external CPU backend where required | ISA, complete model operations, artifact format and total RAM |
| CPU only, native Windows | Native CPU worker implementing the same contracts | Executable attention/state operations and matching model adapter; CUDA success does not cover this row |
| CPU + NVIDIA, WSL | Complete-model reference using vLLM/Omni CUDA | Full modalities and quality, stable power conditions, loading and sustained memory |
| CPU + NVIDIA, native Windows | Native installation of the same declared pipeline | IPC/process lifetime, filesystem/startup scope, streaming and shutdown validated separately |
| CPU + Radeon 890M | Compatible whole-model backend or CPU model plus coarse iGPU stages | Full downstream model result and transfers, not only an encoder result |
| CPU + AMD NPU | CPU model plus compiled NPU stages | Accepted artifact, actual NPU placement, quality and net E2E contribution |
| CPU+iGPU+NPU, with/without NVIDIA | Select stage placements after establishing individual routes | Co-resident memory/power; identical-workload single-backend versus split-serial versus split-overlapped comparison |
| Snapdragon X Elite PC | Windows ARM controller and target-specific CPU/GPU/QNN stages | Full application access, Windows ARM artifacts and RAM; Android binaries are not interchangeable |
| Galaxy S25 | Spark first, TTS second; memory-qualified MiniCPM-o/VLA afterward | Complete device-local state/streaming loop and sustained measurements |
| Galaxy S24 | Rebuild and revalidate the selected mobile pipeline | Independent SoC/runtime/quality qualification; S25 results do not transfer |
| SA8775P ADP | Target-local controller and compatible compiled stages | Complete application access, parity, memory and sustained co-residency |
| RB3 Gen 2 / QCS6490 | Supported integer NPU artifacts or an explicitly chosen CPU/GPU route | Resolve rejected artifact constraints, then full-model quality and performance |

Expand the joint-execution row into separate deployments with and without
NVIDIA when implementing it. Likewise, checkpoint size, precision, RAM SKU and
OS variants require separate execution plans and results beneath each overview
cell; a pass for one variant does not qualify the entire model or device family.

For Qwen27, nominal four-bit language weights alone are about 12.6 GiB;
the actual artifact, remaining weights, state, workspace and OS require more.
Gate small-memory targets before export/integration investment. Record an
artifact-specific capacity rejection rather than substituting a smaller model.
Treat unenumerated mobile/embedded SKUs as new qualification rows.

## Profiling protocol for every executable cell

Use [the profiling harness](../../benchmarks/edge_harness/PROFILING.md) and keep
raw samples, outputs, configuration, logs and traces. Record the actual installed
runtime as well as the source revision. Run competing workloads serially on
the same physical machine.

| Dimension | Required measurements |
|---|---|
| Startup | Compilation, process/import/preparation, constructor/load, first output and warm restart separately; explicit cache conditions |
| Workload | Short/medium/long inputs, concurrency 1 then 2/4 where admitted, at least 20 measured requests per group; separate warmup and state-boundary cases |
| Text | TTFT, delivery/token intervals, decode rate, complete request time; effective prefill separated from isolated device compute |
| Speech | First playable PCM, RTF, chunk arrival gaps, playback deficit count/duration, tail/flush and interruption latency |
| Multimodal/VLA | Preprocessing, encoders, downstream completion, modality alignment, observation-to-action age and deadline misses |
| Memory | Loading/runtime peaks, state growth with context/concurrency, workspace and transfers; physical RAM, WSL quota and VRAM kept distinct |
| Placement | Actual executed devices and fallback; copies, conversions, synchronization, stage queues and handoff time |
| Sustained behavior | At least 30 minutes; power mode, clocks, temperature, memory and latency trends; energy only within available sensor scope |
| Reliability | Slow consumers, cancellation, repeated sessions, memory admission and owned-worker failure; no stale/duplicate output and verified retirement |
| Instrumentation | Separate matched before/traced/after diagnostics; profiler RPC/export costs; retain uninstrumented timing baseline |

Report p50/p95 with sample counts and measurement boundaries. Preserve failures
and retries. Component sums are not E2E measurements, and whole-GPU power is
not CPU/NPU or whole-device energy. Low-clock/power-limited measurements remain
valid for those observed conditions; they do not establish optimized performance.

For heterogeneous candidates, hold inputs, artifact quality, context, concurrency
and power conditions fixed. Keep a split only if the full chain improves latency,
capacity or energy after transfers, synchronization, initialization and contention.

## Per-cell completion and repair record

Progress each cell through artifact readiness, correctness, complete local
execution, profiling and qualification. Functional and performance qualification
are separate fields. A final record contains:

- Exact device/OS and artifact/backend identities; declared modalities and limits.
- Reproducible command and explicit execution plan, including actual placement.
- Quality, state/streaming and memory-admission outcomes with raw evidence.
- Performance distributions, sustained behavior and sensor limitations.
- On failure: failing layer, configuration, observed reason, evidence and the
  smallest next repair/experiment. Distinguish missing prerequisites from an
  attempted run that failed, a rejected artifact and an unqualified result.

Text latency and VLA deadlines require workload-specific product targets.
Do not invent universal thresholds. A completed timing protocol alone is not
a release pass. The current 60-cell matrix and its profiling follow-up are the
starting backlog; subsequent repairs should replace only the affected cells
after repeating their complete gates.
