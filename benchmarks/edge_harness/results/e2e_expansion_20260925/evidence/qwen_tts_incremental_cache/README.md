# Qwen3-TTS incremental x-vector decoder cache

**Scope:** pinned Qwen3-TTS 0.6B CustomVoice revision `85e237c12c027371202489a0ec509ded67b5e4b5`, the retained CPU-generated 117-frame/16-code utterance, and its FP32 Code2Wav decoder on HX370 WSL. This verifies generated-code → waveform state and one-pass chunk timing. It does not run the talker, an Omni complete request, playback, NPU or GPU.

The earlier [fixed 95-history export](../qwen_tts_rolling_history/README.md) loses full-decode agreement as old frames are discarded. The [new probe](../../../../experiments/probe_qwen_tts_incremental_codes.py) first measured the existing eager incremental decoder against a full FP32 decode. At its source 12-frame downstream context, the joined waveform differs **1.5163%** with 59 two-frame chunks and **0.9640%** with five up-to-25-frame chunks. The 25-frame schedule first diverges at frame 75, just past the decoder's 72-frame attention window. Increasing only the downstream audio context to 25, 50, 95 or 117 frames leaves the 25-frame joined error at **0.9640%** while increasing one-pass per-chunk CPU work. This isolates the observed mismatch from that downstream context on this fixture.

An append-only per-layer sliding KV cache plus two quantized convolution-history frames and 12 transformer-output frames was then checked against the same full decode. Its waveform differed about **1.01×10⁻⁶ relative L2** with 25-frame chunks and **1.68×10⁻⁶** with two-frame chunks. The integrated `decode_xvec_exact` path in the existing decoder and its batched eager fallback reproduced those values on the same retained codes. At frame 117 its transformer cache reported 117 frames seen but physically retained only **71 key frames** in the first layer; the downstream hidden tail held **12 frames**. This initially qualified the source decoder's CPU x-vector async-chunk path. A later [RTX 5090 Laptop check](../qwen_tts_cuda_incremental/README.md) validated the same exact state on CUDA and enabled it through the segmented wrapper's stateful eager fallback. ICL prefix mode and stateless CUDA graph execution keep their existing behavior.

The [report](report.json) lists every chunk's shape boundary, waveform error, one-pass wall time and retained-state lengths. The [saved waveforms](waveforms.npz) and SHA-256 in that report permit direct comparison. The integrated CPU path's post-first-chunk wall p50/p95 was **0.756/0.813 s** for four measured up-to-25-frame chunks, and **0.259/0.285 s** for 58 measured up-to-two-frame chunks. Those are single-order decoder calls on one generated stream, excluding model load and the first chunk; they are **not** a warmed 20-request or complete-text-to-audio profile. PyTorch was 2.13.0+cpu with four CPU threads; the local vLLM 0.28 wheel still differs from this Omni 0.29 source checkout. The generated codes and decoder weight SHA-256 values are pinned in the report and [source request](../../../e2e_expansion_20260924/evidence/qwen_tts_vocoder_qualcomm/s25/real_code_window/README.md).

The implementation is enabled for new CPU x-vector states in `Qwen3TTSCode2Wav` when `async_chunk` is on; its scheduler request ID continues to own cleanup. It leaves CUDA graph and ICL states unchanged and routes exact CPU states through the eager fallback instead of the old grouped truncated-window path. The targeted 72-frame test plus the existing decoder/model suites passed **68 tests**. This does not make the HX370 AMD NPU component a continuous stream: its exported graph has no matching KV-state contract yet. To qualify that route, export and validate a stateful NPU boundary, execute a complete admitted Omni TTS request with live handoff, then assess listening quality, cancellation, memory, latency, playback and sustained power.

Reproduce from the repository root:

```bash
repo=/home/zhout/project/edge_infer/vllm-omni-edge
model=/home/zhout/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5
source=benchmarks/edge_harness/results/e2e_expansion_20260924/evidence/qwen_tts_vocoder_qualcomm/s25/real_code_window
evidence=benchmarks/edge_harness/results/e2e_expansion_20260925/evidence/qwen_tts_incremental_cache
PYTHONPATH="$repo" /home/zhout/project/edge_infer/.venvs/omni-cpu/bin/python \
  benchmarks/edge_harness/experiments/probe_qwen_tts_incremental_codes.py \
  --model "$model" --source-report "$source/report.json" \
  --codes "$source/codes.npz" --threads 4 \
  --report "$evidence/report.json" --output "$evidence/waveforms.npz"
```
