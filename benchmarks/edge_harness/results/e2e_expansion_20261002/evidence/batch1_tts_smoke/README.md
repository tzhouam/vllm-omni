# Qwen3-TTS desktop batch-1 streaming smoke, 2026-10-02

This is a **diagnostic complete-stream run**, not a qualified latency or
playback result. It used the local Qwen3-TTS-12Hz-0.6B-CustomVoice checkpoint
at Hugging Face revision `85e237c12c027371202489a0ec509ded67b5e4b5`,
Omni source on the `codex/omni-edge-abstraction` branch, vLLM 0.28.0,
PyTorch 2.13.0+cu130, WSL2 Linux, and the RTX 5090 Laptop GPU. The
[report](report.json) records loaded packages, settings, memory sampling,
and the full invocation. [Request records](requests.jsonl) and
[1 Hz whole-device GPU telemetry](gpu_telemetry.jsonl) are raw evidence.

The run enforced batch size 1 and one active request. It completed one
medium warmup, one medium measurement, a slow-consumer probe, and a short
request after abort. The one measured medium request produced seven finite
24 kHz PCM chunks and 8.0 seconds of streamed audio in 1.649 seconds wall
time, with first audio at 117.2 ms. The arrival-based simulated player had
no underrun on that request; the slow-consumer probe had one. The abort
acknowledgement took 2.87 ms, but late-event exclusion was not verified.
Device-wide NVML usage and process memory are separate counters and must
not be summed; the report's sampled peaks may miss short transients.

The archived [warmup PCM chunks](audio_chunks/) were concatenated in index
order to [this 24 kHz WAV](warmup-medium-1-1.wav). A separately pinned local
Whisper tiny.en checkpoint transcribed the complete 8.32-second warmup
audio as the exact input text: [ASR proxy](asr_proxy.json), word error rate
0.0. This is an intelligibility proxy only; it does not establish voice
identity, perceptual quality, or robustness across prompts. The WAV is
PCM16 converted from the original float32 chunks; the latter remain the
unmodified output evidence.

Reproduce the profiler from the repository root:

```bash
PYTHONPATH=$PWD:$PWD/packages/omni-stage-contracts \
  /home/zhout/project/edge_infer/.venvs/omni-cuda/bin/python \
  benchmarks/edge_harness/profile_local_tts.py \
  --model /home/zhout/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5 \
  --out /tmp/omni_batch1_tts_medium_smoke_20261002 \
  --batch-size 1 --concurrency 1 --length-band medium \
  --repeats 1 --sustained-seconds 0 --gpu-telemetry-interval-s 1
```

The remaining desktop gates are 20 measured requests **for each of three
lengths**, real-device audio playback and quality references, a verified
late-event/cancellation boundary, and a separate 30-minute sequential
single-request run. Mobile serial and overlapped NPU+GPU runs remain
dependent on a quality-passing complete mobile artifact bundle and
device-shell access.
