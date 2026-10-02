# Qwen3-TTS desktop batch-1 sustained profile (2026-10-02)

**Depth: complete local text-to-PCM stream with a completed timing protocol, not
release qualification.** The two-stage Omni Qwen3-TTS-12Hz-0.6B-CustomVoice
route completed three input lengths with 20 measured requests each, followed
by **1,076 sequential medium requests over 1,801.557 seconds**. Every request
used batch size 1 and one active request. The read-only [analysis](analysis.json)
reports `timing_protocol_complete=true`, no protocol violations, and zero
unfinished, nonfinite-audio or missing-metric requests. The complete
[report](report.json), [raw per-request chunk timelines](requests.jsonl),
[1 Hz whole-device telemetry](gpu_telemetry.jsonl),
[warmup float32 PCM chunks](audio_chunks/) and
[contemporaneous Windows power snapshot](power_snapshot_during.json) are
retained. SHA-256 checks matched all 35 copied raw files to the completed run.

The real BF16 checkpoint is the Hugging Face snapshot
`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` revision
`85e237c12c027371202489a0ec509ded67b5e4b5` (all 402 tensors in its
single `model.safetensors` header report BF16). The host is an AMD Ryzen AI 9
HX 370 laptop with an NVIDIA GeForce RTX 5090 Laptop GPU under WSL2 Linux
6.18.33.2. The active stack was PyTorch 2.13.0+cu130, vLLM 0.28.0 and Omni
`0.29.0rc2.dev71+g00510ce38.d20260922` from source checkout
`def6ebd00485dd587bbeb6a975ec053dd89d9988-dirty`; the report retains
loaded package paths. The GPU driver reported 610.71 at archive time.
The benchmark used English CustomVoice speaker Vivian, 19/34/92 estimated
input token positions for short/medium/long text, 24 kHz PCM, existing
disk/JIT caches, and the recorded sampling parameters (talker temperature
0.9, seed unset). Startup was 39.129 s and is excluded from per-request
latency. Generated audio length varies across stochastic requests.

Nearest-rank p50/p95 over the **20 measured** requests in each band, excluding
warmups and the sustained phase:

| Input | Complete wall p50 / p95 | First audio p50 / p95 | Total RTF p50 / p95 | Simulated underrun requests |
| --- | ---: | ---: | ---: | ---: |
| Short | 0.660 / 0.735 s | 106.1 / 126.7 ms | 0.220 / 0.232 | 0/20 |
| Medium | 1.677 / 1.759 s | 108.1 / 133.4 ms | 0.204 / 0.216 | 0/20 |
| Long | 6.104 / 6.976 s | 104.5 / 148.2 ms | 0.200 / 0.220 | 0/20 |

The **1,076 sustained medium requests** had complete-wall p50/p95
1.661/1.921 s, first-audio p50/p95 108.0/142.8 ms and total RTF p50/p95
0.208/0.230. The first 100 versus last 100 sustained requests had complete-
wall p50/p95 1.678/1.859 versus 1.660/1.922 s, and first-audio p50/p95
107.2/136.5 versus 107.4/144.9 ms. These are two observed cohorts, not a
controlled thermal comparison. One sustained request,
`sustained-medium-1-438`, had a **9.534 ms arrival-simulated playback
underrun** when starting at first audio; the other 1,075 did not. The
observed minimum extra startup buffer needed to avoid that deficit was
9.534 ms for that request (0 ms at sustained p95). A deliberately slow
consumer after the sustained phase separately had a 40.825 ms simulated
deficit. Terminal audio is counted once in the complete-stream totals;
warmup terminal chunks are archived separately with their SHA-256 values in
the request records.

All warmup PCM chunks, including terminal chunks, were hash-checked and
assembled once into [short](warmup-short.wav), [medium](warmup-medium.wav)
and [long](warmup-long.wav) float32 WAVs; the [validation manifest](warmup_audio_validation.json)
records 24 kHz sample counts and WAV hashes. Pinned local Whisper tiny.en
transcribed the [short](asr_short_proxy.json) and [medium](asr_medium_proxy.json)
warmups with WER 0. A single full-clip Whisper pass on the 32.32 s long WAV
overtranscribed the repeated prompt (WER 0.5), while two non-overlapping
16-second passes together matched all 72 reference words (WER 0); see the
[long diagnostic](asr_long_segment_diagnostic.json). The long prompt itself
contains four repetitions of the opening sentence. These are intelligibility
proxies on three stochastic warmups, not perceptual or speaker-quality checks.

The Windows power snapshot taken during the run recorded the **Performance**
scheme, battery status code 2 and 100% charge. Across 1,792 sustained-window
1 Hz samples, whole-GPU board power ranged **33.15–128.65 W**, GPU temperature
**64–72 °C**, SM clock **1,462–2,197 MHz**, and reported throttle-reason
bitmask was 4. The 0.25 s memory sampler observed up to **18,057,818,112
bytes** of whole-device GPU memory in use and **7,748,214,784 bytes** of
process-tree RSS. These counters cannot be added; samples may miss shorter
peaks and cannot attribute power or GPU memory to this process. The driver
query did not expose a power-limit value. No alternate power condition was
paired with this run. Lightweight host-side tests and evidence reads occurred
during the sustained phase, so these samples are not an isolated-system or
paired route comparison.

An abort was acknowledged in 31.0 ms and a fresh short request completed,
but late-event exclusion after abort was not verified. Finite PCM and an
arrival-based simulated player do **not** establish sound-device playback,
speaker identity, listening quality, reference-output agreement or mobile
NPU+GPU co-residency. The separately archived short/medium ASR proxies from
the [20-per-band profile](../batch1_tts_profile20/README.md) concern that
earlier run's warmups; the three checks above concern this run. This route
remains scoped and unqualified for release.

Reproduce from the repository root with a fresh output directory:

```bash
PYTHONPATH=$PWD:$PWD/packages/omni-stage-contracts \
  /home/zhout/project/edge_infer/.venvs/omni-cuda/bin/python \
  benchmarks/edge_harness/profile_local_tts.py \
  --model /home/zhout/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5 \
  --out /tmp/omni_batch1_tts_sustained_new \
  --batch-size 1 --concurrency 1 --repeats 20 \
  --sustained-seconds 1800 --gpu-telemetry-interval-s 1

PYTHONPATH=benchmarks/edge_harness \
  /home/zhout/project/edge_infer/.venvs/omni-cuda/bin/python \
  benchmarks/edge_harness/analyze_tts_sustained.py \
  /tmp/omni_batch1_tts_sustained_new > /tmp/omni_batch1_tts_sustained_new_analysis.json
```
