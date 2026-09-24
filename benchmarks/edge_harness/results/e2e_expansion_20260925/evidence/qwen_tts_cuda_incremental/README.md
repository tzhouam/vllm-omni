# Qwen3-TTS CUDA x-vector streaming state on RTX 5090 Laptop

**Scope:** the pinned Qwen3-TTS 0.6B CustomVoice decoder revision `85e237c12c027371202489a0ec509ded67b5e4b5`, one retained CPU-generated 117-frame/16-code utterance, and its FP32 Code2Wav decoder on WSL2 HX370 + RTX 5090 Laptop. This is generated-code → waveform component C evidence. It does not run the talker, an Omni complete request, a captured CUDA graph, playback or sustained load.

The [raw report](report.json) compares each chunk directly with a single full FP32 CUDA decode of the same codes and weights. The previous eager sliding-window replay differs **0.9640%** joined waveform relative L2 for five up-to-25-frame chunks and **46.2251%** for 59 up-to-two-frame chunks. The latter has a 0.6155 maximum absolute sample error, so it cannot be dismissed as only a near-silence relative denominator. The mismatch grows after the 72-frame attention window. The exact per-layer sliding KV path differs **1.18×10⁻⁶** and **1.46×10⁻⁶** respectively, with 71 physical first-layer key frames at the 117-frame end. A separate call through the segmented CUDA wrapper's stateful eager dispatcher reproduced the 25-frame exact result. No CUDA graph was captured in that check.

The source now enables exact state for new x-vector async-chunk requests on CUDA as well as CPU. The segmented wrapper routes those requests to its eager exact decoder while stateless captures and ICL retain their prior paths. On this one run, post-first-chunk CUDA wall p50/p95 was **42.9/58.8 ms** for the four measured up-to-25-frame exact chunks and **17.3/24.3 ms** for 58 up-to-two-frame exact chunks. The corresponding legacy values were 50.2/146.3 ms and 34.9/44.5 ms. These are one-pass decoder-only calls with one request, no explicit warmup, uncontrolled laptop power and no transfer or talker time; they are not a paired sustained performance claim. Peak PyTorch CUDA allocation includes decoder weights and was 628.4 MB/546.8 MB for the exact 25-/two-frame schedules.

The run used WSL2 kernel `6.18.33.2`, NVIDIA driver `610.71`, PyTorch `2.13.0+cu130`, CUDA 13.0 runtime, FP32 decoder weights/activations, TF32 disabled and four CPU threads. The report pins checkpoint and code-file SHA-256 values, every chunk error and time, device memory and execution conditions. The [CPU cache experiment](../qwen_tts_incremental_cache/README.md) contains the same generated stream and independently reached full-decode parity. The focused decoder and Code2Wav suites pass 70 tests after this CUDA routing change, including two simultaneous request states crossing the attention window. A subsequent [short live Omni request run](../qwen_tts_cuda_exact_live/README.md) exercised the route across a 92-frame medium utterance. M2 still needs audio quality, late-cancellation, sound-device playback, sustained memory and power evidence; the Qualcomm and AMD NPU paths remain component-only.

Reproduce from the repository root:

```bash
repo=/home/zhout/project/edge_infer/vllm-omni-edge
model=/home/zhout/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5
source=benchmarks/edge_harness/results/e2e_expansion_20260924/evidence/qwen_tts_vocoder_qualcomm/s25/real_code_window
evidence=benchmarks/edge_harness/results/e2e_expansion_20260925/evidence/qwen_tts_cuda_incremental
PYTHONPATH="$repo" /home/zhout/project/edge_infer/.venvs/omni-cuda-029/bin/python \
  benchmarks/edge_harness/experiments/probe_qwen_tts_cuda_incremental.py \
  --model "$model" --source-report "$source/report.json" \
  --codes "$source/codes.npz" --threads 4 --report "$evidence/report.json"
```
