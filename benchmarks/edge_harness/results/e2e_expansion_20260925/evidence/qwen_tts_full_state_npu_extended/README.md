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
An extended eight-layer native VitisAI compile is in progress locally. Its
result will be added with the raw provider trace if the session returns, or a
timed cap record if it does not. Only completed inference with VitisAI node
events and captured outputs can advance this device cell; a compiler warning,
detection, or CPU reference alone cannot.
