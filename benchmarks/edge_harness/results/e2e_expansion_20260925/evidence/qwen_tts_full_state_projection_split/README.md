# Qwen3-TTS full-state input-projection split

This 2026-09-26 experiment tests an exact-weight cut after the first input
projection of the real eight-layer, rolling-K/V Code2Wav pre-transformer.
The [source graph and prior full-NPU result](../qwen_tts_full_state_npu_extended/README.md)
showed a waveform numerical failure. The [layer-hidden checkpoints](../qwen_tts_full_state_npu_checkpoints/README.md)
located a hidden-output failure at layer 0. This split is a numerical
diagnostic; a per-step CPU/NPU handoff is not an accepted deployment plan
without a complete-request benefit.

The [preparation script](../../../../experiments/prepare_qwen_tts_full_state_projection_split.py)
pins the original graph SHA-256
`2a74a49e99ffeddd4918cade906aea0a497bc9651279475eaaa583cd386b09fd`
and two generated-utterance fixture hashes. It extracts a CPU input-projection
prefix and an eight-layer suffix that accepts projected `[1,2,512]` hidden
states, positions and 16 K/V tensors. The large ONNX artifacts and split
fixtures remain outside Git; their hashes are in [preparation.json](preparation.json).
On ONNX Runtime CPU, prefix plus suffix reproduced **all 17 original graph
outputs exactly** at each of **11 consecutive two-frame steps on both
utterances**, with the suffix owning its next-step K/V state. This validates
the cut itself, not NPU numerics or complete speech generation.

The [native Windows probe](probe_first.json) used the [captured environment](environment.json):
HX370 CPU, NPU driver 32.0.203.329, ONNX Runtime 1.30.0 and VitisAI EP
1.8.63.0. Session creation took **1,296.262 s**. The [raw ORT trace](profile_first_2026-09-26_00-13-16_002.json)
records **11 VitisAI node events and 143 CPU node events** for 11
consecutive two-frame steps with NPU-owned K/V state. Every output was
finite. The NPU calls took 15.291–25.280 ms, compared with 4.410–6.825 ms
for the same CPU suffix calls in that process. These are sequential
component calls, including a cold NPU call, without transfers, prefix time,
power control or a warmed paired profile; they do not show a full-request
benefit.

The pinned [CPU/NPU output capture](full_state_outputs.npz) and
[all-layer K/V audit](all_layer_kv_first.json) show a maximum hidden-state
relative L2 of **1.855%** and maximum K/V relative L2 of **2.914%** versus
the same suffix on CPU. The first K/V output above the provisional 1% gate
appears at layer 2 on frame 95, layer 1 on frame 97 and layer 0 from frame
99 onward. These are numerical observations on one generated utterance,
not an operation-level diagnosis.

The [waveform replay](waveform_first.json) fed the captured NPU hidden
states through the unchanged source decoder's CPU vocoder tail. The source
prefill state matched the fixture exactly; the CPU control matched the
original graph. Only **1/11** two-frame chunks met the provisional 1%
waveform relative-L2 gate. Frame 109 differed **7.553%**, and the joined
22-frame segment differed **1.809%**. The first replay invocation loaded
an older installed Omni wheel without `decode_xvec_exact` and stopped
before numerical comparison; the successful replay explicitly loaded the
checked-out source package. The raw NPU capture and probe were unchanged.

This cut removes the input projection from NPU execution but does not
produce numerically qualified audio or a measured whole-chain benefit. The
second generated utterance has CPU split parity but was not run on NPU after
the first failed its gates. This candidate is not integrated into Omni.
The HX370 AMD NPU Qwen3-TTS matrix cell remains **NOT E2E**. A new NPU
boundary would need to pass both utterances, full audio behavior and a
transfer-inclusive paired benefit gate before becoming a stage plan.

To regenerate the split from the repo root with the native Windows ONNX
environment, use the pinned source and retained fixture paths:

```powershell
python benchmarks/edge_harness/experiments/prepare_qwen_tts_full_state_projection_split.py --source ../models/qwen3tts_stateful_transformer_step_20260925.onnx --prefix ../models/qwen3tts_stateful_full8_projection_prefix_20260926.onnx --suffix ../models/qwen3tts_stateful_full8_projection_suffix_20260926.onnx --fixture benchmarks/edge_harness/results/e2e_expansion_20260925/evidence/qwen_tts_stateful_npu_boundary/step95_rollout11.npz --fixture benchmarks/edge_harness/results/e2e_expansion_20260925/evidence/qwen_tts_stateful_npu_boundary/independent_codes/replay_fixture.npz --split-fixture ../models/qwen3tts_full8_projection_split_first_20260926.npz --split-fixture ../models/qwen3tts_full8_projection_split_independent_20260926.npz --report benchmarks/edge_harness/results/e2e_expansion_20260925/evidence/qwen_tts_full_state_projection_split/preparation.json
```

The native probe uses `probe_qwen_tts_stateful_npu.py` with the suffix and
first split fixture hashes in `preparation.json`, `--steps 11` and
`--npu-state-source self` (the default). The replay uses
`replay_qwen_tts_full_state_npu_waveform.py` with the retained `codes.npz`,
captured NPZ and report hashes. Set `PYTHONPATH` to this checkout when
running the replay, so it loads the source decoder with
`decode_xvec_exact` rather than the older installed wheel.
