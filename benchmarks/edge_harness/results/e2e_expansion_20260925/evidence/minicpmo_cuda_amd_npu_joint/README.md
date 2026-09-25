# MiniCPM-o joint RTX 5090 Laptop + HX370 AMD NPU image requests

**Disposition: two scoped complete Omni requests passed on the joint route;
default joint placement remains unqualified.** The real MiniCPM-o 4.5 BF16
checkpoint ran its thinker, talker, codec, vision transformer and resampler
suffix through WSL RTX CUDA execution, with 5 GiB of thinker weights offloaded
to host RAM by the unchanged vLLM plan. An opt-in experiment moved only the
thinker's pinned A16W8 resampler KV projection through Omni `ExternalStage` to the native
Windows HX370 AMD NPU and returned the BF16 tensor to CUDA. The matching
RTX-only route used the same config except for that opt-in hook. Both routes
completed a [CC0 Chelsea cat photo](../minicpmo_amd_npu_natural/inputs/chelsea_cat.png)
and a [synthetic red square](../../../e2e_expansion_20260923/evidence/minicpmo_wsl_image/red_square.png),
each image → text + finite 24 kHz speech in one three-stage session.

| Same image | RTX + AMD NPU text/audio | RTX-only text/audio | Complete-request walls |
|---|---|---|---:|
| Cat photo | Full cat description; 13.88 s speech | Similar full description; 13.72 s speech | 40.04 / 39.05 s |
| Red square | Correct short description; 3.08 s speech | More detailed description; 8.36 s speech | 8.94 / 17.20 s |

The [audited comparison](audit.json) verifies matching image hashes, retained
float32 [waveforms](joint_384_audio/request_00.wav), nonzero finite samples,
and the complete [joint](joint_384_report.json) and [RTX-only](cuda_384_report.json)
reports. Token sequences and waveform lengths differ on both inputs; the
red-square wall difference mainly compares different output lengths and is
**not a speedup**. Pinned Whisper tiny.en transcribed each route's speech
exactly against that route's own generated text on these two clips (WER 0 in
[joint](joint_384_asr.json) and [RTX-only](cuda_384_asr.json) audits). This
is a small text/speech-alignment proxy, not gold image quality, human listening
or speaker-similarity qualification.

The [event log](joint_384_events.jsonl) records CUDA input and output tensors,
one 1,014-row cat projection and one 1,024-row red-square projection. Their
same-device BF16 reference projection relative-L2 errors were 0.643% and
0.592%. The [raw ORT trace](kv_profile/resampler_kv_projection_32x32__2026-09-25_22-25-13_715.json)
and placement report verify one VitisAI and two CPU graph nodes in the worker;
the same open session served both live requests. Complete-request NPU round
trips were 145/184 ms, and explicit CUDA↔host copies added 3.0/2.6 ms. The
worker peak RSS was 394,641,408 bytes under its 1,932,735,284-byte external
stage budget. vLLM's *startup memory profile* first sent a synthetic
`[10,1056,1152]` bucket, requiring eleven 1,024-row NPU tiles. The first
attempt correctly [refused](kv_attempt1_events.jsonl) that bucket at the
original 4,096-row bound and never reached a request. The opt-in CUDA path
now has an explicit 16,384-row bound, while the CPU path retains 4,096; the
[tile tests](../../../../../../tests/edge/test_minicpmo_kv_tiles.py) cover both.

The as-run graph SHA-256 is
`330bbbd0d18caaf6903aa41f836dbeaef57643a8afbaba333df68e3c1e722aeb`;
an [exact copy](../minicpmo_amd_npu_waveform/resampler_kv_a16w8.onnx)
is retained, while the as-run config points to the Windows-side copy.
Checkpoint shard and run hashes are in the audit. WSL Ubuntu 26.04 used vLLM 0.29.0,
PyTorch 2.13.0+cu130 and RTX 5090 Laptop CUDA; native Windows 11 build
26200 used ORT 1.30.0, VitisAI EP 1.8.63.0 and AMD NPU driver
32.0.203.329. Joint/RTX-only startup took 106.08/96.87 s in separate,
unpaired runs. There is no whole-host loading peak, controlled GPU power,
paired warm latency, cancellation/restart or sustained stream result.

Reproduce with the pinned checkpoint and
[`profile_minicpmo_image_suite.py`](../../../../experiments/profile_minicpmo_image_suite.py),
using the two above images and the [joint](cuda_npu_image_384.yaml) or
[RTX-only](cuda_image_384.yaml) config with `--archive-audio-dir`. The opt-in
hook lives in
[`minicpmo_kv_patch.py`](../../../../experiments/minicpmo_kv_site/minicpmo_kv_patch.py).
Use `VLLM_TARGET_DEVICE=cuda`, `CUDA_VISIBLE_DEVICES=0`,
`VLLM_USE_FLASHINFER_SAMPLER=0`, `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`
and the pinned `omni-cuda-029` environment. Re-run
[`audit_minicpmo_cuda_npu_joint.py`](../../../../experiments/audit_minicpmo_cuda_npu_joint.py)
with this directory and the checkpoint to verify the reports, plan,
waveforms, NPU trace, request coverage and numerical gate.

Next: compare gold image-task outputs and listen to varied generated speech;
then run paired warmed equal-output requests, explicit shared-RAM/VRAM
loading-peak admission, concurrent requests, cancellation/recovery and
sustained power/thermal profiling. The split stays benchmark-only because
the whole-chain benefit and quality gates are not met.
