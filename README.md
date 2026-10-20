# edge-infer

Running multimodal models on **edge CPUs** — a mini-PC or a gateway box with
tens of gigabytes of RAM and no GPU — instead of on servers.

Built as a fork of [vLLM-Omni](https://github.com/vllm-project/vllm-omni),
because it is the only engine with a multi-stage multimodal pipeline,
continuous batching and an OpenAI-compatible server; the alternatives were
evaluated and the reasoning is in [docs/engines/](docs/engines/).

The work turned out to be less about kernels than about three things nobody
profiles: how much memory is resident while the model *loads*, how much is
resident that nothing ever reads, and whether the number you just measured came
from the code path you thought it did. The largest single win here — 843 MB —
was a setting that had been marked "no effect, do not retry" because the code
path under test discarded it.

This repository holds the engine changes, the measurement harness, the raw
results, the baselines they are compared against, and the written analyses.
Upstream's own README is preserved at [docs/README.vllm-omni.md](docs/README.vllm-omni.md).

## What's here

| area | where |
|---|---|
| 4-bit CPU quantization | `vllm_omni/model_executor/layers/quantization/cpu_int4.py` — W4A16 on PyTorch's tinygemm kernel, prefill reading the kernel's own weight layout instead of a duplicate |
| Memory | RoPE tables sized to the servable context; weights repacked in blocks; a KV budget that is actually honoured on CPU |
| CPU platform + vLLM patches | `vllm_omni/platforms/cpu/`, `vllm_omni/patch.py` — vLLM is monkey-patched at import; its *source* is unmodified and no vLLM is redistributed. See [docs/edge/environment.md](docs/edge/environment.md) |
| Device routing | `vllm_omni/edge/weight_path.py` picks the weight format per device class, from measurements rather than assumption |
| Device placement | `vllm_omni/edge/placement.py` decides which parameters leave the GPU when the checkpoint is bigger than the VRAM — cold ones first, because `embed_tokens` and `lm_head` are the same size and the opposite decision |
| Measurement harness | `benchmarks/edge_harness/` — the grid runner, benchmarks, memory ledger, fidelity metric and agent suite every number here came from |
| Raw results | `benchmarks/edge_harness/results/` — 100+ artifacts backing the published figures, each with provenance |
| Baselines | `baselines/` — llama.cpp runs and the comparison scripts |
| Engine analyses | `docs/engines/` — llama.cpp, ExecuTorch, MNN, ncnn, vLLM, vLLM-Omni |
| Study documents | `docs/analysis/` — the full reports these summaries distil |

## Headline results

Spark-X2.5-1.7B, Xeon Platinum 8480C, 24 pinned node-local cores, W4A16 g128.
Every figure is a median of at least three interleaved passes against a control
arm, on a shared host where end-to-end throughput swings ~30% run to run.

**Memory** — each measured as its own A/B with the alternative reachable by a flag:

| lever | saving | cost |
|---|---|---|
| KV budget 1 GiB → 192 MiB | **843 MB** | fewer cached tokens |
| RoPE table capped to `max_model_len` | **618 MB** | none |
| One packed weight layout instead of two | **615 MB** | ~1.6× prefill |
| Blocked weight repack | **489 MB** of load-time peak | none |

These overlap and **must not be added** — see [docs/edge/results.md](docs/edge/results.md).

**Speed**, same machine:

| path | decode | prefill (pp512) |
|---|---|---|
| W4A8 (AMX `int4_scaled_mm_cpu`) | 74.4 tok/s | — |
| W4A16 (tinygemm) | 71.6 tok/s | 2190 tok/s |
| `cpu_gemm_wna16` fallback | 32.3 tok/s | — |
| llama.cpp Q4_K_M (reference) | 90.0 tok/s | 1121 tok/s |

Decode is llama.cpp's; prefill is ours by ~2×. Neither engine is
bandwidth-bound — both sit at 33–39% of the measured 222 GB/s ceiling — so the
decode gap is the execution model, not the kernels. See
[docs/edge/comparisons.md](docs/edge/comparisons.md).

## Why the numbers here are worth reading

This project has retracted five conclusions, every one of them a measurement
error rather than a coding error. The harness exists because of them:

- a "3.1 GB saving" that was `VmRSS` counting the mmap'd checkpoint;
- W4A8 timings taken on a checkpoint whose output head was silently left in bf16;
- a memory benchmark reading `/proc/self/status` while the model ran in a *worker process*;
- "KV sizing saves only 188 MB" — measured through a code path that discarded the setting, so both arms ran the same cache;
- a 1.9× prefill cost quoted from the one pass of a grid whose other passes had crashed.

[docs/edge/methodology.md](docs/edge/methodology.md) is the resulting discipline,
and it is the most transferable thing in this repository.

## Quick start

```bash
# CPU build of vLLM, then this fork
pip install vllm==0.28.0+cpu
VLLM_TARGET_DEVICE=cpu pip install -e .

# a right-sized edge deployment
VLLM_CPU_KVCACHE_SPACE=1 python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='<your-cpu_int4-checkpoint>', quantization='cpu_int4',
          dtype='bfloat16', max_model_len=2048, max_num_seqs=1,
          kv_cache_memory_bytes=192 << 20)   # the single largest memory lever
print(llm.generate(['Hello'], SamplingParams(max_tokens=32))[0].outputs[0].text)
"
```

## Documentation

| | |
|---|---|
| [docs/edge/results.md](docs/edge/results.md) | every measured figure, with spread and baseline |
| [docs/edge/methodology.md](docs/edge/methodology.md) | how to measure on a shared machine without fooling yourself — the five retractions and the rules they left |
| [docs/edge/cpu-4bit.md](docs/edge/cpu-4bit.md) | the two 4-bit paths, the packed layout as measured, building a checkpoint |
| [docs/edge/comparisons.md](docs/edge/comparisons.md) | against llama.cpp and the other engines considered |
| [docs/edge/environment.md](docs/edge/environment.md) | pins, machine, and why there is no vLLM patch series |
| [docs/edge/local-text-mode.md](docs/edge/local-text-mode.md) | the single-stage local text engine: device capabilities, memory admission, explicit refusals, cancellation |
| [docs/edge/external-stage-backends.md](docs/edge/external-stage-backends.md) | placing a stage on the Radeon 890M or the AMD NPU: out-of-process workers, proven placement, the shared-RAM ledger |
| [docs/edge/harness.md](docs/edge/harness.md) | running a comparison |
| [docs/edge/laptop-heterogeneous.md](docs/edge/laptop-heterogeneous.md) | Qwen3.8-27B and Spark-X2.5-4B on a Ryzen AI laptop — what the CPU, NPU and GPU are each worth when decode is bandwidth-bound |
| [ROADMAP.md](ROADMAP.md) | what is next, with expected gains and how each would be verified — and what was rejected |

## Licence

Apache-2.0, inherited from vLLM-Omni. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
