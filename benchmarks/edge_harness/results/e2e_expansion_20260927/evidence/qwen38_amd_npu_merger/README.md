# Qwen3.8-27B vision merger on HX370 AMD NPU: placement passes, numerical gate fails

**Depth C, not P.** On 2026-09-27, the real Qwen3.8-27B-FP8 vision checkpoint's merger was exported at 448×448 image geometry. The complete vision tower's earlier A16W8 exports placed **0/3246** and **0/2290** nodes on this NPU; this smaller component tests the next bounded partition, not the whole tower or a complete text+image request. The merger input is `[784,1152]`, and its output is `[196,5120]`.

The HX370 host used Windows 11 build 26200, AMD NPU driver **32.0.203.329**, Windows `onnxruntime` **1.30.0**, and VitisAI EP **1.8.63**. The model weights are the local `Qwen3.8-27B-FP8` checkpoint; [the numerical records](component_numeric_gemm_perchannel.json) contain its `config.json` and `outside.safetensors` SHA-256, the exporter/runtime versions, image SHA-256s, calibration input digest, graph hashes, and shapes. The Omni fork started at `9883eedbcf8da91ff1d51538c1c5baf6e94a8343`. NPU work ran from the WSL Omni process through its native-Windows worker. No power limit was set; power/thermal state and concurrency were not controlled. The two calibration images were cat and astronaut photos in the repo; a red-square image was held out.

| Candidate | Same-checkpoint output relative L2: cat / astronaut / held-out red | VitisAI placement | Complete-request result |
|---|---:|---:|---|
| FP32 source ONNX vs Torch | ~0.000034% / 0.000032% / 0.000041% | not an NPU graph | no |
| A16W8 all ops, per-tensor weights | 2.162% / 2.144% / 2.416% on ORT CPU | 1/8 nodes | numerical failure |
| A16W8 Gemm-only, per-tensor weights | 2.104% / 2.094% / 2.305% on ORT CPU | not re-profiled here | numerical failure |
| A16W8 Gemm-only, per-channel weights | 0.229% / 0.221% / 0.933% on ORT CPU | 1/4 nodes | **NPU output numerical failure** |

The per-channel candidate reduced quantization error, but its **actual NPU output** on the held-out image differed by **46.360% relative L2** from the same quantized graph on WSL ORT CPU; the per-tensor all-op candidate differed by **46.309%**. This is not a silent CPU fallback: the retained [per-channel ORT node trace](qwen38_vision_merger_448__2026-09-27_04-24-58_279.json) assigned one fused node to `vitisai` and three to CPU; the [all-op trace](qwen38_vision_merger_448__2026-09-27_04-21-17_707.json) assigned one to `vitisai` and seven to CPU. A native-Windows CPU control on ORT 1.30.0 differed from WSL ORT CPU by only **0.0215% relative L2** on the same per-channel input, while native-Windows CPU versus NPU differed by **46.360%**. All outputs were finite. The archived [held-out activation](heldout_input.npz), [WSL CPU output](wsl_cpu_output_perchannel.npz), [native CPU output](native_cpu_output_perchannel.npz), [NPU output](npu_output_perchannel.npz), and [saved-output comparison](saved_output_comparison.json) permit direct audit. The graph binaries are omitted because they are reproducible from the checkpoint and code; their hashes are in the numerical JSON.

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

The next experiment should bisect the two Gemm subgraphs and compare each NPU intermediate against same-version native-Windows CPU to identify the corrupting partition, then re-export/validate an A16W8 candidate on several held-out natural images. Only a numerically valid component with measured whole-chain benefit may be wired into a Qwen complete-request route. This result does **not** show that AMD NPU can never run a Qwen vision component; it rejects these exact artifacts and EP/driver combination.
