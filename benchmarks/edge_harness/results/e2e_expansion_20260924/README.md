# 2026-09-24 E2E and component expansion

A separate [complete Spark 1.7B BF16 WSL CPU run](evidence/spark_bf16_wsl_cpu/README.md)
now uses both verified upstream weight shards in the fork-branch Omni local text
route. Twelve 128-token requests passed cancellation and exact greedy-token
parity against standalone vLLM. One warmup plus 20 serial 64-token complete
requests measured nearest-rank p50/p95 wall time **5.411/5.751 s** and
TTFT **0.129/0.146 s** on HX370 WSL CPU. This qualifies the stated text
workload only. A separate neighboring-checkout profile is retained in the
evidence directory with its own provenance. A new [S25 two-frame Code2Wav candidate](evidence/qwen_tts_vocoder_qualcomm/s25/qnn_short_chunk/README.md)
matched eager and ONNX CPU on one fixture, but its FP16 QNN DLC failed
NPU-requested inference at the Workbench device-memory limit, just as the
25-frame artifact did. Neither result is a complete mobile TTS stream.
A [separate TFLite GPU-requested two-frame inference](evidence/qwen_tts_vocoder_qualcomm/s25/tflite_short_chunk/README.md)
returned a finite unsaturated waveform at 2.325% relative L2 / 32.67 dB SNR
against local ONNX CPU on the same fixture. Its 100-sample component p50/p95
was **0.559/1.052 s** for 0.16 s audio, with 403 CPU and 335 GPU placement
rows and a median component RTF of 3.49. Listening quality and playable
complete-stream behavior remain open.

An additional [Spark BF16 restart probe](evidence/spark_bf16_wsl_cpu/restart_branch_report.json)
cancelled after eight token events, rejected the retired session handle and
verified that a fresh session produced the same 128 greedy token IDs with no
late events. A [native Windows Spark GGUF restart probe](evidence/spark_omni_windows_restart/README.md)
confirmed server task start before cancellation on both HX370 CPU and Radeon
890M Vulkan1, then loaded the same artifact in a fresh Omni stage, returned
Paris and cleared the ledger without stale output. This is fresh-stage
recovery, not same-session token continuation.

A separate [native Windows RTX Spark GGUF Omni run](evidence/spark_omni_windows_rtx/README.md)
used the pinned 1.7B Q4_K_M artifact on RTX Vulkan0 with 29/29 layers
offloaded. Two named requests and 20/20 serial inventory requests passed at
nearest-rank wall p50/p95 **0.294/0.302 s**, along with public AsyncOmni text,
context refusal and fresh-stage recovery after server-started cancellation.
This is a different artifact/runtime from the earlier 4B BF16 CUDA profile.

The [native Windows Qwen3.8-27B NVFP4 startup attempt](evidence/qwen38_omni_windows_rtx/README.md)
loaded all three shards under vLLM 0.29 but failed in Marlin post-load scale
permutation with an unsupported PTX toolchain error. Driver 610.71 reports
CUDA 13.3 while the installed Windows vLLM wheel was built with CUDA 13.4.
No native image/text inference completed; the WSL pass remains separate.

A [native Windows Qwen3-TTS hybrid restart probe](evidence/qwen_tts_omni_windows_restart/README.md)
also confirmed backend prefill before cancellation. The worker and host-RAM
reservation drained; a fresh Omni stage with the same pinned Radeon Vulkan0
talker/codec and CPU predictor returned the identical 24 kHz PCM under a new
generation. This adds one recovery case to the earlier 20-request profile,
without establishing playable streaming or a new latency distribution.

For S25, the historical [25-frame TFLite GPU-requested vocoder](evidence/qwen_tts_vocoder_qualcomm/s25/full_tflite_gpu_profile/README.md)
has a newly measured 100-sample p50/p95 **0.839/1.157 s** for 2 s of audio,
with mixed CPU/GPU placement and component median RTF **0.420**. Its 4.24%
same-fixture waveform relative L2, unknown exact checkpoint revision and
missing full-stream gates prevent a mobile TTS support claim.
The pinned CustomVoice eager 25-frame decoder now matches the historical
ONNX CPU waveform at 1.89e-6 relative L2 on that fixture, without proving
the source export's revision.
A [fresh 25-frame export](evidence/qwen_tts_vocoder_qualcomm/s25/full_pinned_tflite/README.md)
from that pinned snapshot matches local eager CPU at 1.99e-6 relative L2 and
has compiled to S25 TFLite. Its GPU-requested inference returned an
unsaturated waveform at 4.239% relative L2 versus local ONNX CPU, bitwise
identical to the historical GPU output on this fixture. Its own placement
profile measured 100 component samples at p50/p95 **0.671/1.454 s** for
2 s audio (median RTF **0.335**), with 403 CPU and 335 GPU rows. The
historical profile is a separate run, and neither establishes full streaming.
A [generated-code window](evidence/qwen_tts_vocoder_qualcomm/s25/real_code_window/README.md)
from one pinned CustomVoice utterance passed local FP32 window/full-decoder
continuity; the pinned S25 GPU-requested target returned an unsaturated
waveform at **1.153% relative L2 / 38.76 dB SNR** versus local ONNX CPU.
No listening-quality tolerance or complete device-local stream passed.

The same pinned [two-frame graph on Galaxy S24](evidence/qwen_tts_vocoder_qualcomm/s24/tflite_short_chunk/README.md)
matched local ONNX CPU at 5.75e-6 relative L2 when requested on CPU, while
its GPU-requested waveform was fully saturated at 24.218 relative L2. The
100-sample CPU component p50/p95 was 3.433/3.873 s for 0.16 s audio (RTF
21.46); the numerically failed mixed CPU/GPU route measured 0.601/0.895 s.
A FP32-preserving GPU control returned the same saturated waveform bitwise.
A
[Snapdragon X Elite FP16 QNN short graph](evidence/qwen_tts_vocoder_qualcomm/xelite/qnn_short_chunk/README.md)
compiled and returned an unsaturated NPU-requested waveform at 0.930%
relative L2 versus local ONNX CPU on one fixture. Its 100-sample profile
attributed all 728 rows to NPU, but p50/p95 was **3.662/3.676 s** for
0.16 s audio (median component RTF **22.89**). These are component-only
results and the short NPU vocoder is too slow for playable streaming.

The [Qwen3-TTS vocoder study](evidence/qwen_tts_vocoder_qualcomm/README.md) reuses a retained fixed-shape Code2Wav export and historical Workbench input to test numerical behavior on SA8775P and other Qualcomm settings. The fresh SA8775P GPU-requested inference produces a saturated waveform, while the same binary's CPU-requested output on the exact SA device matches ONNX CPU within 4.13e-6 relative L2. Its 100-sample CPU-requested component profile had p50/p95 5.384/5.534 s with all 738 execution-detail rows on CPU. A FP32-preserving GPU-option control still saturated 84.89% of samples, so that option did not repair the tested path. RB3 GPU-requested execution produces an unsaturated but 15.73%-divergent waveform; its fresh 100-sample mixed CPU/GPU component p50/p95 was 3.099/3.133 s, not a usable TTS timing claim. On the exact RB3 device, the same binary requested on CPU matched original ONNX CPU within 4.13e-6 relative L2 on the retained fixture; its 47-sample component profile measured p50/p95 12.727/13.424 s with all 738 execution-detail rows on CPU. This is a major streaming-latency blocker. A single-fixture-calibrated RB3 INT8 QNN vocoder returned a finite but 66.93%-relative-L2 divergent waveform (3.49 dB SNR) versus ONNX CPU; its 100-sample component profile was p50/p95 1.971/1.992 s with all 728 execution-detail rows on NPU; these timings describe numerically unqualified audio. X Elite ONNX CPU output matches the local ONNX Runtime reference on the retained fixture; its 94-sample component p50/p95 was 6.384/6.644 s with all execution-detail rows on CPU. Corrected X Elite DirectML inference and profile both failed during provider initialization; separate FP16 QNN DLC inference returned an unsaturated waveform at 1.138% relative L2 / 38.88 dB SNR against original ONNX CPU; its placement profile timed out during QNN graph preparation without samples. A separate SA8775P FP16 QNN vocoder compile passed, but same-fixture NPU-requested inference failed after compilation without a waveform or placement samples. Fresh S24 and S25 CPU-requested TFLite vocoder inferences matched ONNX CPU within 4.04e-6 relative L2 on the retained fixture. S25 measured 100 component samples at p50/p95 4.418/4.641 s with all 738 execution-detail rows on CPU; S24 measured 100 at 4.826/4.958 s with all 738 rows on CPU. Separate explicit INT8 and W8A16 QDQ ONNX exports checked on local ORT CPU failed the same-fixture numerical gate at 246.97% and 331.83% waveform relative L2, respectively. S25 CPU talker inference preserved token 80 and its 100-sample component profile measured p50/p95 30.161/31.301 ms with all 1,800 rows on CPU. No complete on-device TTS stream or model-quality qualification is claimed.

A local HX370 x86-64 LiteRT CPU control of the identical SA/RB3 TFLite binary matched the original ONNX CPU waveform within 3.45e-6 relative L2. The SA saturation is therefore not universal to the converted artifact; the exact SA CPU control passed numerical and placement checks, while the faulty GPU delegate operation remains unidentified.

The [talker decode-step attempt](evidence/qwen_tts_talker_qualcomm/README.md) pins a separate 28-layer source and synthetic cache fixture. Historical FP16 QNN context-binary inference executed on S24, S25, SA8775P and X Elite, but all four changed the CPU top-1 speech token with 8.57–9.43% logits relative L2. These artifacts are numerically unqualified on the fixture. RB3 rejected the floating-point-input artifact. Explicit X Elite ONNX CPU and SA8775P TFLite CPU fallback compiles passed. X Elite CPU-requested inference preserved CPU top-1 token 80 with logits relative L2 2.07e-5; its 100-sample decode-step profile measured p50/p95 191.414/309.428 ms, all 1,744 execution-detail rows on CPU. SA8775P TFLite CPU-requested talker inference also preserved token 80 with logits relative L2 3.33e-5; its 100-sample decode-step profile measured p50/p95 52.635/76.925 ms, all 1,800 execution-detail rows on CPU. A new RB3 TFLite CPU-requested step preserved token 80 with logits relative L2 2.29e-5 and maximum new-cache relative L2 3.04e-5; its 100-sample profile measured p50/p95 138.022/139.642 ms with all 1,800 execution-detail rows on CPU. A new S24 TFLite CPU-requested talker step preserved token 80 with logits relative L2 3.33e-5 and maximum new-cache relative L2 3.41e-5; its 100-sample profile measured p50/p95 36.976/43.264 ms with all 1,800 execution-detail rows on CPU. No complete talker loop or TTS stream passed.

The [MiniCPM-o speech-head study](evidence/minicpmo_tts_qualcomm/README.md) recovered the real 20-layer ONNX source and a matching retained fixture. S24 and S25 FP16 context-binary outputs preserved the ONNX CPU top-1 speech token with 0.346%/0.360% logits relative L2 on one step. Their 100-sample component profiles had p50/p95 8.537/9.006 ms and 7.197/7.615 ms, with all 1,029 execution-detail rows on NPU for each. FP16 QNN DLC compiles passed for SA8775P, X Elite and RB3. SA8775P NPU-requested inference preserved CPU top-1 token 1867 with 0.3568% logits relative L2; its 100-sample profile measured p50/p95 11.901/13.038 ms, all 944 execution-detail rows on NPU. X Elite same-fixture inference also preserved token 1867 with 0.3170% logits relative L2; its 100-sample profile measured p50/p95 9.905/11.037 ms, all 944 execution-detail rows on NPU. RB3 FP16 QNN inference failed at graph load despite compile success; an explicit TFLite CPU fallback same-fixture inference preserved top-1 token 1867 with logits relative L2 1.26e-6 versus ONNX CPU; its 100-sample CPU-requested profile measured p50/p95 59.518/71.347 ms with all 1,008 execution-detail rows on CPU. The complete multimodal model remains unqualified on those devices.

The [native Windows Qwen3-TTS sustained analysis](evidence/qwen_tts_sustained_drift/README.md) finds a sharp playback throughput collapse around minute 26–27 in the historical 30-minute RTX 5090 Laptop stream. The last 17 serial requests all had simulated underruns with median RTF 1.331; the separate WSL run did not show that behavior. The revised local profiler recorded a completed current-source 30-minute repeat with device-wide NVML telemetry. The final 119 requests had median RTF 0.185 and three brief simulated underruns, so the prior collapse did not recur. The historical evidence and uncontrolled repeat conditions cannot identify its cause; underrun-free native Windows playback remains unqualified.

The [InternVLA Cosmos S25 attempt](evidence/internvla_cosmos_qualcomm/README.md) repackages the real policy's image encoder from its parity-tested ONNX export. FLOAT16 QNN DLC inference returned a latent within 0.0524% relative L2 of the source on one pattern. Injecting it into the real policy on local CPU changed synthetic actions by 0.742% relative L2, without a task tolerance. S25 six-frame component placement profile returned 100 samples at p50/p95 49.665/51.382 ms, with all 121 execution-detail rows on NPU. The same encoder compiled to FP16 QNN DLC for X Elite CRD; same-fixture NPU-requested inference returned a finite latent at 0.0608% relative L2 / 64.32 dB SNR versus the source, with its 100-sample six-frame profile at p50/p95 82.939/83.837 ms and all 121 execution-detail rows on NPU. An SA8775P FP16 QNN DLC inference returned a finite same-fixture latent at 0.0613% relative L2 / 64.25 dB SNR versus source; its 100-sample six-frame profile measured p50/p95 116.021/116.327 ms with all 121 execution-detail rows on NPU. Device-local action-policy support remains unqualified. The checkpoint-named Place_Markpen dataset returned HTTP 404, and the official vLLM-Omni example archive in InternData-A1 returned HTTP 403 gated-access denial under the current authenticated account; [real-observation access evidence](evidence/internvla_real_observation_access/README.md) records the blocker without treating synthetic actions as task validation.

New exact-device [Galaxy S24 QNN](evidence/internvla_cosmos_qualcomm/s24/audit_report.json) and [RB3 ONNX CPU](evidence/internvla_cosmos_qualcomm/rb3_cpu/audit_report.json) Cosmos encoder inferences returned finite same-fixture latents at 0.0613% and 2.20e-6 relative L2, respectively. The S24 profile measured 100 six-frame samples at p50/p95 64.314/67.472 ms with all 121 execution-detail rows on NPU. The RB3 profile measured 46 six-frame samples at p50/p95 13.018/13.053 s with all 590 execution-detail rows on CPU. Injecting the two hosted latents into the real policy on local CPU changed fixed synthetic actions by 0.631% and 0.594% relative L2, respectively. Neither result runs the action policy on its device or establishes a real-observation task tolerance.
