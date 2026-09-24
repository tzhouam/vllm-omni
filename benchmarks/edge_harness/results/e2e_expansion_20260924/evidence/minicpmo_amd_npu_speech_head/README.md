# MiniCPM-o speech-head step on HX370 AMD NPU (2026-09-24)

**Disposition: real component executed on NPU; numerical gate failed;
NOT E2E.** This is one fixed-cache, real-weight 20-layer speech-head
decode step, not the thinker, multimodal encoders, vocoder, streaming
speech, or a complete MiniCPM-o request.

The retained 776,341,465-byte ONNX artifact (SHA-256
`78cf64804f11ad269ee4180da61588ccc5eefe2942773227345499756e4cc229`)
was recovered for the earlier [Qualcomm component study](../minicpmo_tts_qualcomm/README.md).
Its exact checkpoint revision is not attested by that historical export;
the file hash pins this experiment. The model and 44-input fixture remain
outside Git under `/home/zhout/project/edge_infer/models/`.
The fixture has a synthetic hidden activation and 20 K/V cache pairs at
length 256. The graph emits 6,562 logits and 40 new-cache tensors.

On HX370 native Windows 11 build 26200, AMD NPU driver
`32.0.203.329`, ONNX Runtime `1.30.0`, and AMD VitisAI EP
`1.8.63.0`, the [CPU control](cpu_control.json) reproduced all
41 retained ORT CPU outputs **bitwise**. The
[NPU attempt](npu_probe.json) then placed the complete step in one
VitisAI node event with no CPU node events. It preserved top speech token
**1867**, but logits differed **2.172% relative L2** and the worst
new-cache tensor differed **4.353% relative L2** from that CPU control.
Both exceed the probe's provisional 1% component gate. The source ONNX
is FP32; [AMD documents](https://ryzenai.docs.amd.com/projects/WinML/en/latest/model_support.html)
automatic BF16 conversion when compiling float models for this NPU.
The profile does not independently reveal internal precision, so this
is a possible cause, not a proven explanation.

Session creation took **420.34 s** and one cold inference took
**39.22 ms**. These are not warmup-adjusted or repeated latencies,
and power was not measured. The exact model, fixture and reference hashes,
driver, raw log and [placement profile](npu_profile_2026-09-24_20-20-16_148.json)
are pinned in [audit_report.json](audit_report.json). The mixed-encoding
[native log](npu_probe.log) is retained. The
[probe](../../../../experiments/probe_minicpmo_amd_npu_speech_head.py)
checks every input/output name, CPU parity, actual node provider,
top token, logits and cache numerics.

Next, test a same-checkpoint export or calibrated A16W8 artifact with
representative hidden/cache values and a task-derived tolerance. A
passing one-step component would still need continuous speech-head state,
talker and vocoder handoff, cancellation, admission, memory, quality and
whole MiniCPM-o behavior through Omni before this device/model cell
could be qualified.
