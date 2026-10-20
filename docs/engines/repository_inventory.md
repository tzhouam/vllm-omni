# Repository inventory

Verified on 2026-09-08 (UTC+8 machine clock) with `git rev-parse`, `git status --porcelain`, `git remote -v`, and `git submodule status`. No checkout was modified. The analysis root is `/data/zhoutaichang/embedding_infer/`; `/home/zhoutaichang/embedding_infer` is the same directory (same inode), so both paths refer to one set of checkouts.

## Expected vs actual

| Engine | Local path | Remote | Expected branch / HEAD | Actual branch | Actual HEAD | Working tree | Last commit (author date) |
|---|---|---|---|---|---|---|---|
| vLLM | [`../vllm`](../vllm) | `https://github.com/vllm-project/vllm.git` | main / 51da0ca66 | `main` | `51da0ca66c8065619c79e35dff97aa99aeaf5644` | clean | 2026-09-07 15:15 UTC, "[CI/Build] Unskip ColQwen3 multimodal pooling tests on Transformers v5 (#55588)" |
| vLLM-Omni | [`../vllm-omni`](../vllm-omni) | `https://github.com/vllm-project/vllm-omni.git` | main / 7be014bc | `main` | `7be014bce6374f06c95b703763bdbac4c6198f31` | clean | 2026-09-07 14:50 UTC, "[Bugfix][NPU] Fix MiniMax-H3 INT8 quantization dispatch (#6876)" |
| llama.cpp | [`../llama.cpp`](../llama.cpp) | `https://github.com/ggml-org/llama.cpp.git` | master / e71b80510 | `master` | `e71b80510c848c00175924ecf3c40333ccae8eb5` | clean | 2026-09-07 16:28 +0200, Revert "CUDA: size routed MoE MMQ N-tiles ... (#24546)" (#28551) |
| ncnn | [`../ncnn`](../ncnn) | `https://github.com/Tencent/ncnn.git` | master / 3b7bdba7 | `master` | `3b7bdba7fc8aea8fd46779533eee027df77c639d` | clean | 2026-09-07 19:42 +0800, "Fix SIGFPE with missing or incomplete CPU topology (#6967)" |
| MNN | [`../MNN`](../MNN) | `https://github.com/alibaba/MNN.git` | master / bef71b97 | `master` | `bef71b9756a2c77549eddbe33eb97290e3b16602` | clean | 2026-09-04 17:39 +0800, "[Vulkan:Perf] Optimize INT4 cooperative matrix path" |
| ExecuTorch | [`../executorch`](../executorch) | `https://github.com/pytorch/executorch.git` | main / cfc6ecd516 | `main` | `cfc6ecd5166bec70c5fde2bad836fed26018fe52` | clean | 2026-09-07 08:47 -0700, "Qualcomm AI Engine Direct - Enable QNN Windows ARM64 build in CI (#22438)" |

**Result: all six checkouts match the expected branch and commit exactly, and all six working trees are clean.** No discrepancies in branch or HEAD.

## Discrepancies and caveats (not corrected, reported only)

1. **Submodules are not initialized** in the checkouts that declare them (`git submodule status` prints a leading `-`):
   - ncnn: `glslang`, `python/pybind11`. Consequence: a Vulkan-enabled build of ncnn from this tree would need `git submodule update --init` (not run). CPU-only analysis is unaffected.
   - executorch: 21 submodules including `backends/xnnpack/third-party/XNNPACK`, `backends/vulkan/third-party/*`, `backends/mlx/third-party/mlx`, `extension/llm/tokenizers`, `third-party/{flatbuffers,flatcc,googletest,gflags,json,ao,ios-cmake}`, `kernels/optimized/third-party/eigen`, `shim`, `backends/cadence/utils/FACTO`. Consequence: the tokenizer implementation used by ExecuTorch's LLM runner (`extension/llm/tokenizers`) is **not present on disk**, so statements about it in [engines/executorch.md](engines/executorch.md) are limited to how it is called, not how it is implemented.
   - vLLM, vLLM-Omni, llama.cpp, MNN declare no submodules at these commits (empty `git submodule status`).
2. **Installed Python packages differ from the analyzed checkouts.** The machine's Python 3.12 has `vllm 0.28.0` installed as a wheel in `~/.local/lib/python3.12/site-packages`, and `vllm-omni 0.28.0rc2.dev22+gbe335a86f` installed editable from `/home/zhoutaichang/feature/vllm-omni-main` (commit `be335a86f`), **not** from `../vllm-omni` (`7be014bc`). Any runtime experiment therefore exercises the installed versions, not the analyzed commits, unless a separate environment is created (which would be a dependency installation and was not done without permission). See [experiments.md](experiments.md).
3. **Host is x86_64, not ARM.** Native ARM CPU, Apple Silicon, and mobile GPU/NPU measurements cannot be taken on this machine; only x86 CPU and NVIDIA L20X data-center GPU runs are possible here. Edge numbers in this study are therefore either repo-reported (cited) or marked as unknown.

## Tree size snapshot

Rough sizes to calibrate reading effort (from `du -sh`, excluding `.git`); see per-engine reports for module maps.

| Engine | Primary language(s) | Notable top-level dirs |
|---|---|---|
| vLLM | Python, CUDA/C++ (`csrc/`) | `vllm/v1`, `vllm/model_executor`, `vllm/platforms`, `vllm/multimodal`, `vllm/entrypoints`, `csrc/`, `docs/` |
| vLLM-Omni | Python | `vllm_omni/{engine,core,worker,diffusion,model_executor,entrypoints,distributed,config,host_weight_runtime}`, `examples/`, `docs/`, `recipes/` |
| llama.cpp | C/C++ | `ggml/`, `src/`, `include/`, `tools/` (server, mtmd, tts), `examples/`, `gguf-py/`, `convert_hf_to_gguf.py` |
| ncnn | C++ (+GLSL shaders) | `src/` (`layer/{arm,x86,vulkan,...}`), `tools/` (pnnx, quantize), `python/`, `examples/`, `docs/` |
| MNN | C++ (+Metal/OpenCL/Vulkan/CUDA), Python | `source/{core,backend,geometry,shape}`, `express/`, `schema/default`, `tools/converter`, `transformers/{llm,diffusion}`, `apps/`, `pymnn/` |
| ExecuTorch | Python (AOT), C++ (runtime) | `exir/`, `runtime/`, `backends/`, `kernels/`, `extension/`, `examples/`, `devtools/`, `docs/source` |

## Per-repo agent guidance honored during analysis

- `vllm/AGENTS.md` (via `CLAUDE.md`): no system `pip`; use `uv`. No packages were installed for this study.
- `llama.cpp/AGENTS.md`: fully autonomous agents must not contribute; this study is read-only analysis and proposes no PRs.
- `MNN/AGENTS.md`: `schema/private/` and `source/internal/` are restricted and were not read or referenced.
- `executorch/AGENTS.md`: points to `.claude/` reference docs and `.wiki/`; used as documentation sources.

## Related material outside the six checkouts

- GPU scheduling guide: `/home/zhoutaichang/feature/GPU调度工具快速上手.pdf` (read; procedures summarized in [experiments.md](experiments.md)).
- Other vLLM-Omni worktrees exist under `/home/zhoutaichang/feature/` (`vllm-omni-main`, `vllm-omni-abot`, `vllm-omni-pr6463`, ...). They were **not** analyzed; only `../vllm-omni` at `7be014bc` is in scope.
