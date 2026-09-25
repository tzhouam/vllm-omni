# Qwen3-TTS eight-layer rolling-state export follow-up

The exact-weight Qwen3-TTS 0.6B CustomVoice pre-transformer ONNX step is
pinned to revision `85e237c12c027371202489a0ec509ded67b5e4b5`, graph
SHA-256 `2a74a49e99ffeddd4918cade906aea0a497bc9651279475eaaa583cd386b09fd`.
It accepts two new convolution frames and all eight layers' explicit
71-frame sliding K/V state. This is a stateful **component boundary**, not a
text-to-speech request. The [original state export](../qwen_tts_stateful_npu_boundary/README.md)
passed two CPU steps; this follow-up checks eleven consecutive steps on each
of two independently generated utterances before interpreting NPU results.

| CPU-generated code stream | ONNX CPU steps | Worst hidden relative L2 | Worst K/V relative L2 | Worst waveform relative L2 vs checkpoint exact decoder | Raw report |
|---|---:|---:|---:|---:|---|
| First, 117 frames | 11 | 6.76e-7 | 3.61e-7 | 3.34e-5 | [first_cpu_state11.json](first_cpu_state11.json) |
| Independent seed-91, 144 frames | 11 | 7.24e-7 | 3.68e-7 | 2.05e-6 | [independent_cpu_state11.json](independent_cpu_state11.json) |

The [CPU verifier](../../../../experiments/probe_qwen_tts_full_state_cpu_fixture.py)
loads the real checkpoint, pins generated-code, weight, ONNX and fixture
hashes, rolls the exported graph's own K/V over eleven steps, and compares
each output with the checkpoint's own pre-transformer and exact waveform
decoder. Runs used HX370 WSL2 Ubuntu, PyTorch 2.13.0+cpu, ONNX Runtime 1.29.0,
four CPU threads and FP32 decoder weights. One-off per-step times in the JSON
are not a warmed performance profile. These passes validate the CPU state
contract on two inputs, not a device artifact or listening quality.

The [full-state waveform replay tool](../../../../experiments/replay_qwen_tts_full_state_npu_waveform.py)
checks a native provider report, captured tensors and model hashes before
feeding measured NPU hidden outputs into the unchanged CPU vocoder tail. A
CPU-as-candidate self-check passed two waveform chunks within 1.21e-6
relative L2 of the exact decoder; its synthetic
placement report and tensors remain outside Git and are **not NPU evidence**.
The [extended native probe](probe.json) did compile the full eight-layer graph
on Windows 11 build 26200 with HX370 NPU driver 32.0.203.329, ONNX Runtime
1.30.0 and VitisAI EP 1.8.63.0 (EP DLL SHA-256
`c37699dfe12128b4c8c491b4e951a7b985cad695d487151ebdcc7657ffb10991`).
Session creation took **1,228.444 s**. The [raw ORT trace](profile_2026-09-25_23-13-37_446.json)
has **two VitisAI node events** and **26 CPU node events** across two
consecutive two-frame steps. Individual calls took 26.956/15.328 ms versus
10.081/5.983 ms for the corresponding CPU graph calls in the same native
process. These are one cold and one subsequent call, **not** a warmed latency
distribution, transfer-inclusive comparison or complete-request profile.

The NPU graph returned finite hidden and rolling-K/V tensors, but its hidden
relative L2 versus the same ONNX CPU graph was **1.891%/2.074%**, already
above the provisional 1% tensor gate. The probe's `max_state_relative_l2`
field came from the pre-fix script and checks only layer 0; the subsequent
[waveform replay](waveform_replay.json) audits **all 16 K/V outputs** against
the checkpoint. Their maximum relative L2 is **1.040%/1.319%**. The retained
[paired CPU/NPU tensors](full_state_outputs.npz) are pinned by SHA-256 in both
reports.

An [all-layer K/V audit](all_layer_kv_audit.json) independently checks the
retained paired capture, report hash, exact model hash, NPU placement and
NPU-owned rolling state. The first output above the provisional 1% tensor
gate is layer 2 K at frame 95 (**1.0404%**) and layer 1 K at frame 97
(**1.1961%**); layer 2 K reaches **1.3185%** at frame 97. This locates the
earliest observed failing output, not the operation that introduced the error.

Feeding the measured NPU hidden states into the unchanged source decoder's
CPU vocoder tail produced finite chunks at frames 95 and 97, but they differed
by **1.924%/1.036% waveform relative L2** from exact CPU decode; **0/2**
met the provisional 1% chunk gate. The CPU-capture control matched the
checkpoint within 5.35e-7 relative L2. The joined four-frame segment was
1.872% relative L2. This is an offline downstream replay, not a live Omni
text-to-audio request or listening-quality result. The full-state NPU graph
therefore moves from an inconclusive compile cap to **executed but numerically
unqualified component evidence**. Its observed calls and long build also give
no whole-chain performance benefit on this fixture. The HX370 AMD NPU TTS
matrix cell remains **NOT E2E**.

A separate [eight-layer hidden-output diagnostic](../qwen_tts_full_state_npu_checkpoints/README.md)
subsequently executed two more NPU steps with the same 17 original CPU and
NPU outputs and provider event counts. Its extra checkpoints show the hidden
output already over 1% at layer 0 on both steps. This narrows the observed
boundary but does not attribute an operation or fix the waveform error. The
next experiment should test a source-faithful numerical correction on both
utterances before another live-stage attempt. The corrected probe now checks
every K/V output; the raw native report above is kept unchanged to preserve
what the running version actually computed.
