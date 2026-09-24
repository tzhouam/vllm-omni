# Qwen3-TTS generated-code chunk history on HX370

**Scope:** pinned Qwen3-TTS 0.6B CustomVoice revision `85e237c12c027371202489a0ec509ded67b5e4b5`, one retained CPU-generated 117-frame code stream and its first 25-frame Code2Wav window. These are local CPU numerical experiments on HX370 WSL, not full TTS streaming or listening quality. The [source request and files](../../../e2e_expansion_20260924/evidence/qwen_tts_vocoder_qualcomm/s25/real_code_window/README.md) fix the prompt, speaker, seed, weights and quantized 97-frame input.

The [CPU history sweep](report.json) compared the two audio frames at generated-code offset 23 with the matching segment of the pinned 25-frame ONNX CPU output. Each run used the same FP32 decoder weights and quantized code frames; only the amount of prior context changed. The retained [waveforms](waveforms.npz) and report show:

| History frames | Input shape | Relative waveform L2 vs 25-frame CPU segment |
|---:|---|---:|
| 72 | `[1,512,74]` | 3.5124% |
| 80 | `[1,512,82]` | 2.0814% |
| 88 | `[1,512,90]` | 0.9132% |
| 95 | `[1,512,97]` | 0.0001556% |

Thus the previously observed 3.51% later-window difference is explained by the shortened history on this fixture. The sweep uses one eager forward per history length; its timings are not a latency profile. The [95-history/two-new-frame ONNX export](long_context_export/report.json) has source graph SHA-256 `235da9f3bc3828255442b7e21215e6935963f564849b1fbcc942550cf351e721` and matches eager at 1.513e-6 relative L2. Its eager waveform also matches the long 25-frame CPU segment at 1.513e-6 relative L2. The export is an external 405 MB model artifact at `/home/zhout/project/edge_infer/models/qwen3tts_code2wav_c2_ctx95_20260925.onnx`; it is not added to Git. The [input](long_context_export/fixture.npz), [eager output](long_context_export/eager_output.npz) and [ORT output](long_context_export/ort_output.npz) are retained here.

On HX370 native Windows 11 build 26200 with NPU driver `32.0.203.329`, ORT 1.30.0 and AMD VitisAI EP 1.8.63.0 ([environment capture](long_context_export/environment.json)), the matching [`val_340` prefix extraction](long_context_export/cut105_extract.json) used node ordinal 105 (the 72-history export used ordinal 100). The [live probe](long_context_export/npu_probe.json) created one NPU partition in **348.460 s** and ran one cold prefix call in **7.487 ms**. Its [profile](long_context_export/npu_profile_2026-09-25_02-07-49_953.json) attributes one node to `vitisai`, with [paired CPU/NPU tensors](long_context_export/paired_outputs.npz) retained. Global boundary error was 0.3906% relative L2, dominated by float32-max attention-mask values as in the earlier short-graph experiment. The ordinal-100 [extraction control](long_context_export/cut100_extract.json) ended at a different tensor and was not used for this handoff.

The [unchanged CPU suffix replay](long_context_export/waveform_handoff.json) reconstructs the 95-history source graph waveform bitwise from its CPU boundary. The measured NPU boundary plus that suffix differs **0.01212% relative L2 / 78.33 dB SNR** from the corresponding short-graph CPU waveform. A separate [long-reference audit](long_context_export/long_reference_audit.json) compares the saved [waveforms](long_context_export/waveforms.npz) directly with the matching 25-frame CPU decode: the CPU short graph differs **0.0000397%** and NPU-prefix/CPU-suffix output differs **0.01212%** relative L2. Replacing the measured boundary with zeros changes the short-graph waveform **1.998%** relative L2, so the boundary is causally relevant on this fixture. No listening tolerance has been established.

After correctness calls, 20 alternating CPU-only repeats measured full 95-history graph **1.981/2.053 s** p50/p95, CPU suffix with CPU boundary **2.005/2.041 s**, and CPU suffix with captured NPU boundary **1.985/2.042 s** for 2.0 s of audio. These repeats **did not rerun the NPU**. The CPU suffix by itself occupies about the whole audio duration, and adding NPU compilation, transfers, memory and power has no demonstrated complete-request benefit. This remains a component C numerical result, not an Omni TTS request or a device-local stream.

Reproduce the CPU checks from the repository root with `/home/zhout/project/edge_infer/.venvs/omni-cpu/bin/python` and `PYTHONPATH=/home/zhout/project/edge_infer/vllm-omni-edge`. Run [`probe_qwen_tts_real_code_chunk_history.py`](../../../../experiments/probe_qwen_tts_real_code_chunk_history.py) with `--model` set to the pinned Hugging Face snapshot, `--real-report`, `--fixture` and `--long-reference` set to the preceding source-request files, `--start-frame 23`, `--threads 4`, and `--report`/`--output` set to this directory. Run [`export_qwen_tts_vocoder_short_chunk.py`](../../../../experiments/export_qwen_tts_vocoder_short_chunk.py) on the same snapshot, fixture and long reference with `--context-frames 95 --chunk-frames 2 --start-frame 23`; the retained export report fixes all output hashes and exact arguments. The WSL export environment used PyTorch 2.13.0+cpu, ONNX 1.22.0 and ORT 1.29.0. The exporter requested opset 17 but retained opset 18 after its converter failed; the report records the actual opsets.

For example, the CPU history sweep from the repository root is:

```bash
repo=/home/zhout/project/edge_infer/vllm-omni-edge
model=/home/zhout/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5
source=benchmarks/edge_harness/results/e2e_expansion_20260924/evidence/qwen_tts_vocoder_qualcomm/s25/real_code_window
evidence=benchmarks/edge_harness/results/e2e_expansion_20260925/evidence/qwen_tts_chunk_history
PYTHONPATH="$repo" /home/zhout/project/edge_infer/.venvs/omni-cpu/bin/python \
  benchmarks/edge_harness/experiments/probe_qwen_tts_real_code_chunk_history.py \
  --model "$model" --real-report "$source/report.json" \
  --fixture "$source/fixture.npz" --long-reference "$source/ort_output.npz" \
  --start-frame 23 --threads 4 --report "$evidence/report.json" \
  --output "$evidence/waveforms.npz"
```

The NPU side can be reproduced with [`bisect_qwen_tts_vitisai_graph.py`](../../../../experiments/bisect_qwen_tts_vitisai_graph.py) by extracting cut 105 of the pinned 95-history ONNX graph, then probing that candidate against `long_context_export/fixture.npz` with the installed VitisAI EP. The resulting paired tensor and report feed [`probe_qwen_tts_npu_prefix_waveform.py`](../../../../experiments/probe_qwen_tts_npu_prefix_waveform.py), which extracts and checks the unchanged CPU suffix; [`audit_qwen_tts_long_history_waveform.py`](../../../../experiments/audit_qwen_tts_long_history_waveform.py) compares its saved waveform against the long decode. Each script rejects changed source, fixture or captured-output hashes. The suffix ONNX artifact stays outside Git at `/home/zhout/project/edge_infer/models/qwen3tts_code2wav_cut105_ctx95_suffix_20260925.onnx`, SHA-256 `d761767c508d785919c75e0f0095c25c5add80fcfb9ae0b56a7ed4ba341ef175`.

Long-history agreement on one CPU-generated utterance does not establish a bounded rolling-state implementation or arbitrary-prompt quality. The subsequent [rolling-history audit](../qwen_tts_rolling_history/README.md) found 2.0214% waveform relative L2 at later frame start 109 for the same fixed 95-history shape versus the full decoder, while an all-prior-history control was near exact. That result narrows this NPU component pass to its measured frame-95 window. A later device route must preserve the same effective history or validate an explicit state/cache equivalent across chunks, then pass complete-request audio quality, live handoff timing, memory admission, cancellation and sustained power gates.
