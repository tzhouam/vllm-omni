# Qwen3.8-27B vision merger on HX370 AMD NPU: single Gemms pass, joint quantization fails

**Depth C, not P.** On 2026-09-27, the real Qwen3.8-27B-FP8 vision checkpoint's merger was exported at 448×448 image geometry. The complete vision tower's earlier A16W8 exports placed **0/3246** and **0/2290** nodes on this NPU; this smaller component tests the next bounded partition, not the whole tower or a complete text+image request. The merger input is `[784,1152]`, and its output is `[196,5120]`.

The HX370 host used Windows 11 build 26200, AMD NPU driver **32.0.203.329**, Windows `onnxruntime` **1.30.0**, and VitisAI EP **1.8.63**. The model weights are the local `Qwen3.8-27B-FP8` checkpoint; [the numerical records](component_numeric_gemm_perchannel.json) contain its `config.json` and `outside.safetensors` SHA-256, the exporter/runtime versions, image SHA-256s, calibration input digest, graph hashes, and shapes. The Omni fork started at `9883eedbcf8da91ff1d51538c1c5baf6e94a8343`. NPU work ran from the WSL Omni process through its native-Windows worker. No power limit was set; power/thermal state and concurrency were not controlled. The two calibration images were cat and astronaut photos in the repo; a red-square image was held out.

| Candidate | Same-checkpoint output relative L2: cat / astronaut / held-out red | VitisAI placement | Complete-request result |
|---|---:|---:|---|
| FP32 source ONNX vs Torch | ~0.000034% / 0.000032% / 0.000041% | not an NPU graph | no |
| A16W8 all ops, per-tensor weights | 2.162% / 2.144% / 2.416% on ORT CPU | 1/8 nodes | numerical failure |
| A16W8 Gemm-only, per-tensor weights | 2.104% / 2.094% / 2.305% on ORT CPU | not re-profiled here | numerical failure |
| A16W8 Gemm-only, per-channel weights | 0.229% / 0.221% / 0.933% on ORT CPU | 1/4 nodes | **NPU output numerical failure** |
| A16W8 first Gemm only, per-channel weights | 0.111% / 0.108% / 0.906% on ORT CPU | 1/6 nodes | component numerical pass; not E2E |
| A16W8 second Gemm only, per-channel weights | 0.199% / 0.193% / 0.233% on ORT CPU | 1/7 nodes | component numerical pass; not E2E |

The per-channel candidate reduced quantization error, but its **actual NPU output** on the held-out image differed by **46.360% relative L2** from the same quantized graph on WSL ORT CPU; the per-tensor all-op candidate differed by **46.309%**. This is not a silent CPU fallback: the retained [per-channel ORT node trace](qwen38_vision_merger_448__2026-09-27_04-24-58_279.json) assigned one fused node to `vitisai` and three to CPU; the [all-op trace](qwen38_vision_merger_448__2026-09-27_04-21-17_707.json) assigned one to `vitisai` and seven to CPU. A native-Windows CPU control on ORT 1.30.0 differed from WSL ORT CPU by only **0.0215% relative L2** on the same per-channel input, while native-Windows CPU versus NPU differed by **46.360%**. All outputs were finite. The archived [held-out activation](heldout_input.npz), [WSL CPU output](wsl_cpu_output_perchannel.npz), [native CPU output](native_cpu_output_perchannel.npz), [NPU output](npu_output_perchannel.npz), and [saved-output comparison](saved_output_comparison.json) permit direct audit. The graph binaries are omitted because they are reproducible from the checkpoint and code; their hashes are in the numerical JSON.

The follow-up quantized just one named Gemm at a time. With `node_linear` (first Gemm) quantized, the held-out NPU output differed from quantized CPU by **0.00158% relative L2**, and from the same-checkpoint FP32 source by **0.906%**; the retained [numeric result](npu_numeric_gemm_fc1_perchannel.json) and [raw node trace](qwen38_vision_merger_fc1_448__2026-09-27_04-39-16_947.json) show one VitisAI node. With `node_linear_1` (second Gemm) quantized, the corresponding errors were **0.0103%** and **0.233%**, with one VitisAI node in its [numeric result](npu_numeric_gemm_fc2_perchannel.json) and [raw trace](qwen38_vision_merger_fc2_448__2026-09-27_04-39-57_159.json). The [first](component_numeric_gemm_fc1_perchannel.json) and [second](component_numeric_gemm_fc2_perchannel.json) CPU quantization records use the same real-image calibration and held-out input. Thus neither Gemm is intrinsically rejected or numerically corrupt when isolated. The failure is specific to this **jointly quantized two-Gemm graph / EP partition**, not yet pinned to an internal compiler operation.

| Isolated 448×448 merger graph | 20 native-Windows ORT CPU calls, p50 | 10 VitisAI worker calls, p50 | 10 WSL↔Windows round trips, p50 | NPU load peak |
|---|---:|---:|---:|---:|
| First Gemm only A16W8 | 28.822 ms | 26.393 ms | 32.952 ms | 1,143 MiB |
| Second Gemm only A16W8 | 30.071 ms | 25.675 ms | 31.826 ms | 1,111 MiB |

These are separate runs, not a controlled paired speedup test. Both isolated paths have a longer observed round trip than native CPU and duplicate a large graph across the process boundary, so neither earns default placement under the whole-chain benefit rule. [First NPU profile](npu_profile_fc1.json), [second NPU profile](npu_profile_fc2.json), [first CPU profile](native_cpu_profile_fc1.json), and [second CPU profile](native_cpu_profile_fc2.json) retain individual timings. An initial first-Gemm launch with a **1 GiB worker-peak hint** was correctly refused after load: its **1,143 MiB** measured peak exceeded the **1,126 MiB** graph-plus-runtime reservation. The successful rerun reserved a **1.5 GiB** peak hint and admitted **2,970 MiB** of shared host RAM. This is a measured memory-gate outcome, not a numerical failure.

A further attempt inserted explicit FP64 casts around GELU to block fusion while quantizing both Gemms. The exact double GELU first failed to initialize on ORT CPU because this build lacks the required double `Erf(13)` kernel. An explicitly recorded tanh-GELU variant then passed the source comparison on the held-out input at **0.0127%** relative L2 and its quantized CPU comparison at **0.934%**. VitisAI nevertheless took **0/32 nodes**, after **72.4 s** session creation; the external stage refused it rather than reporting a CPU fallback as NPU execution. [Numerical record](component_numeric_gemm_perchannel_cpu_gelu.json), [refusal with plan](npu_refusal_cpu_gelu_barrier.json), and [raw ORT trace](qwen38_vision_merger_fp64_gelu_barrier_448__2026-09-27_04-47-19_232.json) preserve this failed candidate. It is not a repair.

For context only, ten unpaired NPU calls measured per-channel worker-time median **21.812 ms** and WSL↔Windows round-trip median **29.203 ms**, with **12.0 s** NPU session creation and **842 MiB** worker peak RSS. Twenty native-Windows ORT CPU calls on the same quantized graph measured **31.391 ms** median after three warmups. These are component timings on one shape and input, not whole-request speedup evidence; the numerical failure prohibits the split regardless. The external stage reserved **2406 MiB** shared host RAM and verified its load peak before execution. See [NPU profile](npu_profile_gemm_perchannel.json), [native CPU profile](native_cpu_profile_perchannel.json), and [NPU numerical comparison](npu_numeric_gemm_perchannel.json). The corresponding per-tensor [profile](npu_profile.json), [numerical comparison](npu_numeric_all.json), and [CPU control](native_cpu_profile.json) are retained.

Reproduce after installing the same checkpoint and NPU worker runtime:

```bash
PYTHONPATH=. python benchmarks/edge_harness/experiments/probe_qwen38_vision_merger_npu.py \
  --model /home/zhout/project/edge_infer/models/Qwen3.8-27B-FP8 \
  --out /tmp/qwen38_merger_repro --quantize-ops gemm --per-channel \
  --images \
  benchmarks/edge_harness/results/e2e_expansion_20260925/evidence/minicpmo_amd_npu_natural/inputs/chelsea_cat.png \
  benchmarks/edge_harness/results/e2e_expansion_20260925/evidence/minicpmo_amd_npu_natural/inputs/astronaut.png \
  benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/minicpmo_wsl_image/red_square.png
PYTHONPATH=. python benchmarks/edge_harness/experiments/compare_external_onnx_component.py \
  --graph /tmp/qwen38_merger_repro/merger_a16w8_gemm_perchannel.onnx \
  --source-graph /tmp/qwen38_merger_repro/merger_fp32.onnx \
  --inputs /tmp/qwen38_merger_repro/heldout_input.npz \
  --profile-dir /tmp/qwen38_merger_repro \
  --out /tmp/qwen38_merger_repro/npu_numeric.json
```

For the isolated artifacts, rerun the exporter with `--gemm-node fc1` or `--gemm-node fc2` and use the corresponding `merger_a16w8_gemm_fc1_perchannel.onnx` or `merger_a16w8_gemm_fc2_perchannel.onnx` path in the comparison command, with `--worker-peak-rss-hint-bytes 1610612736`. To reproduce the rejected barrier, use a fresh output directory with `--cpu-gelu-boundary`; its graph is `merger_a16w8_gemm_perchannel_cpu_gelu.onnx`. The external-stage CLI must report a placement refusal rather than treating its CPU output as NPU execution.

The next experiment should use a supported FP32 CPU operation as a partition barrier, or change QDQ boundaries without an unsupported double cast; compare each intermediate with same-version native-Windows CPU, then test any repaired two-Gemm artifact on held-out natural images and through complete text+image generation. Only a numerically valid component with measured whole-chain benefit may be wired into the default route. This result does **not** show that AMD NPU can never run a Qwen vision component; it rejects the tested joint artifacts and validates the two isolated components only at depth C.
