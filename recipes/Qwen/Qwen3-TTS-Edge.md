# Qwen3-TTS on edge-class hardware (edge/omni-edge branch)

This recipe covers the edge deploy profile, the CPU platform, hardware
auto-adaptation, codec-token streaming with on-device decoding, and the
profiling tools added on the `edge/omni-edge` branch. Numbers quoted here were
measured on one NVIDIA L20X (143 GB) and a 2x56-core Xeon 8480C host on 2026-09-09; the
JSON files are committed under `benchmarks/tts/edge_results/`
(`python benchmarks/tts/edge_results/summarize.py` renders them).

## 1. Edge profile (one GPU, small batch, ramped chunks)

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --omni --deploy-profile edge --trust-remote-code
# offline
python - <<'PY'
from vllm_omni import Omni
omni = Omni(model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", deploy_profile="edge")
PY
```

`vllm_omni/deploy/edge/qwen3_tts.yaml` inherits the default deploy YAML and sets
`codec_chunk_ramp: [2, 4, 8, 16, 25]`, `decode_batch_max_size: 1`,
`max_num_seqs: 4`, both stages on one device with 0.25/0.15 memory fractions,
and `orchestrator: parallel_stage_init: true` (both stages initialise
concurrently; 94.6 s vs 152 s warm init for 1.7B on L20X).

| model | deploy | TTFA p50 | playback start p50 | stall at TTFA | RTF |
|---|---|---|---|---|---|
| 1.7B | default | 32 ms | 116 ms | 85 ms | 0.10 |
| 1.7B | edge | 44 ms | 44 ms | 0 ms | 0.10 |
| 0.6B | default | 28 ms | 95 ms | 67 ms | 0.10 |
| 0.6B | edge | 41 ms | 41 ms | 0 ms | 0.10 |

Playback-start latency = max over chunks of (arrival time − audio already
buffered); the ramp removes the initial 80 ms gap at the cost of ~12 ms TTFA
(first chunk is 2 frames instead of 1).

## 2. Hardware auto-adaptation (`deploy_profile="auto"`)

`vllm_omni/edge/hardware_probe.py` classifies the host (jetson / arm64_cpu /
x86_cpu / cuda_discrete / npu_phone), `adapt.derive_overrides` derives dtype,
absolute KV budgets (never allowing recompute preemption:
`max_num_seqs x max_model_len` talker tokens must fit), thread binding, eager
mode and adaptive-chunk parameters, and `deploy/edge/hardware/<class>.yaml`
overlays pin per-class values. `VLLM_OMNI_HW_PROFILE=<json>` overrides the probe
(used by `benchmarks/tts/hw_emulation.py` to emulate other classes on one host).

## 3. CPU platform

Install the `+cpu` vLLM wheel (see `docs/getting_started/installation/cpu.md`),
then:

```bash
VLLM_TARGET_DEVICE=cpu python examples/offline_inference/text_to_speech/qwen3_tts/end2end.py \
    --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --query-type CustomVoice --streaming
```

The `platforms: cpu:` overlay in `vllm_omni/deploy/qwen3_tts.yaml` sizes the two
stages and binds disjoint OpenMP thread ranges; `benchmarks/tts/cpu_thread_sweep.py`
measures 8/16/32 threads. On the 8480C host: 8 threads TTFA 426 ms, RTF 2.8;
12+4 threads RTF 1.7 (bf16/AMX). The CPU path is usable for offline synthesis,
not for real-time streaming on this class of CPU.

## 4. Codec-token streaming + on-device decoder

Run the talker-only deployment and decode on the client (CPU or an exported
decoder):

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --omni --trust-remote-code \
    --deploy-config vllm_omni/deploy/qwen3_tts_talker_only.yaml
python examples/online_serving/text_to_speech/qwen3_tts/edge_decoder_client.py \
    --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --text "..." --out out.wav
```

See `docs/serving/codec_token_streaming.md` for the wire format (binary frames
with a `<IHB` header, 16 int16 codebooks per 80 ms frame). For phone/NPU
targets export the windowed decoder (`python -m vllm_omni.edge.decoder_export`)
and pass `--decoder-backend executorch --pte-manifest <dir>/manifest.json`.

## 5. Profiling tools

- `benchmarks/tts/stream_latency_bench.py` — TTFA / playback-start / RTF / memory JSON; `--step-stats` adds per-step overhead attribution (scheduler, engine step, chunk hop, orchestrator dispatch) as a share of the 80 ms frame budget.
- `benchmarks/tts/init_profile.py` — init-phase timeline per configuration (`VLLM_OMNI_INIT_TIMELINE`).
- `benchmarks/tts/hw_emulation.py`, `benchmarks/tts/cpu_thread_sweep.py`, `benchmarks/tts/codec_stream_e2e.py`.
