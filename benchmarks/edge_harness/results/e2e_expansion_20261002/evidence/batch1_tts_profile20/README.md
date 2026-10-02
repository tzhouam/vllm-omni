# Qwen3-TTS desktop batch-1 profile: 20 requests per length

This is **scoped complete-stream timing** through the two-stage local Omni
Qwen3-TTS-12Hz-0.6B-CustomVoice path, not release qualification. The run used
the local checkpoint at HF revision
`85e237c12c027371202489a0ec509ded67b5e4b5`, WSL2 Linux, the RTX 5090
Laptop GPU, PyTorch 2.13.0+cu130, vLLM 0.28.0 and Omni source on
`codex/omni-edge-abstraction`. [Report](report.json),
[request samples](requests.jsonl),
[whole-device 1 Hz NVML records](gpu_telemetry.jsonl), and
[warmup float32 PCM chunks](audio_chunks/) are retained. The report records
the exact settings, loaded package paths and worker memory samples. The
startup observation was 39.32 seconds with existing disk/JIT caches;
initialization is excluded from request timing.

All 60 measured requests used **batch size 1, concurrency 1**, with a
separate warmup for each input length. Every request finished with finite
PCM. Nearest-rank percentiles are calculated from the 20 raw samples in each
band; streamed seconds vary because generation is stochastic.

| Input | Complete wall p50 / p95 | First audio p50 / p95 | Streamed audio p50 / p95 | Total RTF p50 / p95 | Simulated underrun requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short | 0.662 / 0.731 s | 105.4 / 136.9 ms | 2.96 / 3.36 s | 0.222 / 0.239 | 0/20 |
| Medium | 1.654 / 1.787 s | 106.5 / 135.6 ms | 8.00 / 8.56 s | 0.204 / 0.215 | 0/20 |
| Long | 6.147 / 6.606 s | 107.2 / 131.8 ms | 29.84 / 33.20 s | 0.200 / 0.215 | 0/20 |

All 230 sampled whole-device NVML records span 18.3–123.7 W GPU-board
power (median 82.7 W) and 53–71 °C. They do not attribute power to this
process or capture subsecond peaks; no controlled power-mode comparison was
made. The 0.25-second memory sampler saw up to 17,805,852,672 bytes of
whole-device GPU memory in use and 7,696,060,416 bytes process-tree RSS;
those counters have different accounting and must not be added.

The three archived warmup outputs are [short](warmup-short-1-1.wav),
[medium](warmup-medium-1-22.wav), and [long](warmup-long-1-43.wav).
Their WAVs are PCM16 conversions of the archived float32 chunks, verified
against each chunk's SHA-256 in the request records. A separately pinned
local Whisper tiny.en checkpoint exactly transcribed the short and medium
warmups: [short ASR proxy](asr_short_proxy.json),
[medium ASR proxy](asr_medium_proxy.json), each word error rate 0.0. The
32.32-second long warmup was not assessed with that single-window proxy.
ASR agreement does not measure speaker identity or perceptual audio quality.

Reproduce the profile from the repository root:

```bash
PYTHONPATH=$PWD:$PWD/packages/omni-stage-contracts \
  /home/zhout/project/edge_infer/.venvs/omni-cuda/bin/python \
  benchmarks/edge_harness/profile_local_tts.py \
  --model /home/zhout/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5 \
  --out /tmp/omni_batch1_tts_profile20_20261002 \
  --batch-size 1 --concurrency 1 --repeats 20 \
  --sustained-seconds 0 --gpu-telemetry-interval-s 1
```

The 30-minute sequential phase was intentionally not run in this profile.
Real sound-device playback and tail delivery, wider listening/speaker
references, verified late-event exclusion after cancellation and mobile
serial-versus-overlapped execution remain open. The slow-consumer diagnostic
experienced an arrival-based underrun. The abort acknowledgement was 7.03 ms
and a fresh short request completed, but that does not prove the stricter
late-event gate.
