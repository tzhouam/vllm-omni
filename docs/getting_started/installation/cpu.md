# CPU

vLLM-Omni can run **autoregressive and generation stages** (for example the
Qwen3-TTS talker → code2wav pipeline) on vLLM's CPU backend. Diffusion stages
are not supported on CPU.

## Requirements

- x86-64 with AVX-512 (bf16 needs `avx512_bf16` or AMX) or aarch64 with NEON
  (`fp16`/`bf16` extensions recommended). See the vLLM CPU installation page
  for the supported ISA matrix.
- A vLLM **CPU** build (`+cpu` wheel). The CUDA wheel does not contain the
  CPU kernels.
- Python 3.10–3.13.

## Install

```bash
uv venv --python 3.12 && source .venv/bin/activate
# vLLM CPU wheel (x86-64 example; see vLLM docs for aarch64 and per-commit wheels)
uv pip install --torch-backend cpu \
  https://github.com/vllm-project/vllm/releases/download/v0.28.0/vllm-0.28.0+cpu-cp38-abi3-manylinux_2_34_x86_64.whl
# vLLM-Omni without re-resolving torch, then its CPU requirements
VLLM_TARGET_DEVICE=cpu uv pip install --torch-backend cpu --no-deps -e .
VLLM_TARGET_DEVICE=cpu uv pip install --torch-backend cpu -r requirements/cpu.txt
```

## Run

```bash
export VLLM_TARGET_DEVICE=cpu                    # selects the CPU OmniPlatform
export LD_PRELOAD=$(python -c "import glob,sys; print(glob.glob(sys.prefix+'/lib/libiomp5.so')[0])")
# one utterance, Qwen3-TTS 0.6B, CPU overlay from vllm_omni/deploy/qwen3_tts.yaml (platforms: cpu:)
python examples/offline_inference/text_to_speech/qwen3_tts/end2end.py --query-type CustomVoice
# or let the hardware probe size everything (threads, KV budget, dtype, chunking):
python benchmarks/tts/stream_latency_bench.py --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --query-type CustomVoice --deploy-profile auto
```

What the CPU platform changes (`vllm_omni/platforms/cpu/platform.py`):

| Item | CPU behaviour |
|---|---|
| Workers | `CPUARWorker` / `CPUGenerationWorker` compose vLLM's `CPUWorker` with the Omni runners |
| Scheduling | `async_scheduling=False` (forced by vLLM's `CpuPlatform`), no CUDA graphs |
| Memory | `gpu_memory_utilization` is a NUMA-node memory fraction; prefer `kv_cache_memory_bytes` |
| Threads | per-stage `VLLM_CPU_OMP_THREADS_BIND` ranges (`platforms: cpu:` overlay or `deploy_profile="auto"`) |
| Compile | code-predictor `torch.compile` off by default; `VLLM_OMNI_CPU_INDUCTOR=1` enables it |
| dtype | bf16 with `avx512_bf16`/AMX or ARM `bf16`, otherwise `float32` (`deploy_profile="auto"`) |
| Sleep mode | not supported (returns an error ACK) |

## Tests

```bash
VLLM_TARGET_DEVICE=cpu pytest tests/platforms/test_cpu_platform.py tests/worker/test_cpu_ar_model_runner.py -m "core_model and cpu"
```
