# Qwen3-TTS rolling Code2Wav history audit

**Scope:** CPU numerical check on one generated 117-frame Qwen3-TTS 0.6B CustomVoice utterance, revision `85e237c12c027371202489a0ec509ded67b5e4b5`. This tests whether the previously NPU-executed fixed 95-history/two-new-frame Code2Wav shape preserves the full FP32 decoder output as the window advances. It does not run the NPU, a live talker-to-vocoder stream, or a complete-request performance profile.

The [source request](../../../e2e_expansion_20260924/evidence/qwen_tts_vocoder_qualcomm/s25/real_code_window/README.md) pins the prompt, speaker, seed, generated codes and weights. The [probe](../../../../experiments/probe_qwen_tts_rolling_history.py) verified their SHA-256 values and decoded all 117 code frames with the pinned FP32 decoder. Quantizing all frames reproduced the retained first-97-frame fixture **exactly**. Each two-frame window used the same generated code stream and was compared with the matching samples of the full 117-frame FP32 waveform. The [report](report.json) has per-window shapes, errors and one-off CPU forward times; the [waveforms](waveforms.npz) retain each two-frame reference and output. The one-off times are not a latency profile.

| New-frame start | 72-history error | 88-history error | Fixed 95-history error | All-prior-history control |
|---:|---:|---:|---:|---:|
| 95 | 3.5123% | 0.9133% | 0.0000246% | 0.0000246% |
| 97 | 2.6420% | 0.7556% | 0.5310% | 0.0000303% |
| 103 | 2.6988% | 1.3297% | 0.7093% | 0.0000536% |
| 109 | 12.7636% | 3.5196% | **2.0214%** | 0.0001487% |
| 115 | 7.1431% | 1.6155% | 0.9065% | 0% |

The absolute frame-109 reference RMS is 0.01642 and the fixed-95 waveform RMS error is 0.000332, so its 2.02% relative error is not solely a near-silence denominator artifact. Frame 115 is much quieter (reference RMS 0.0000303); interpret its relative percentage with that level in mind. These are numerical differences; no listening tolerance has been established.

The earlier [95-history NPU-prefix/CPU-suffix waveform pass](../qwen_tts_chunk_history/README.md) remains valid for its frame-95 input. It does **not** extend to a rolling stream: the fixed-size window drops older frames at frame 97, and on this one utterance it misses a 1% relative-L2 gate at frame 109. Retaining all prior frames recovers the full decode throughout, but grows the graph input and has no NPU execution or bounded-memory proof. A subsequent [CPU decoder-state experiment](../qwen_tts_incremental_cache/README.md) validated an append-only sliding KV path on these 117 generated frames; the existing NPU export does not implement that state contract. The next device step is to validate a matching stateful NPU boundary, then measure a warmed live NPU+CPU loop against the same complete CPU request, including memory, cancellation, output ordering and power. Full device-local TTS remains unqualified.

Reproduce from the repository root with the installed CPU environment:

```bash
repo=/home/zhout/project/edge_infer/vllm-omni-edge
model=/home/zhout/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5
source=benchmarks/edge_harness/results/e2e_expansion_20260924/evidence/qwen_tts_vocoder_qualcomm/s25/real_code_window
evidence=benchmarks/edge_harness/results/e2e_expansion_20260925/evidence/qwen_tts_rolling_history
PYTHONPATH="$repo" /home/zhout/project/edge_infer/.venvs/omni-cpu/bin/python \
  benchmarks/edge_harness/experiments/probe_qwen_tts_rolling_history.py \
  --model "$model" --source-report "$source/report.json" \
  --codes "$source/codes.npz" --first97-fixture "$source/fixture.npz" \
  --threads 4 --report "$evidence/report.json" --output "$evidence/waveforms.npz"
```

The run used HX370 WSL, PyTorch 2.13.0+cpu, 4 CPU threads and the pinned FP32 decoder weights. The source generation was BF16 on CPU; this audit begins from its saved integer codes. No GPU, NPU or hosted device participated in the rolling check.
