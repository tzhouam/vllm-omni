# Five hosted mobile/embedded targets: complete-artifact gap

The [rolling matrix](../../../e2e_profiling_20260922/evidence/summary/README.md)
contains 25 Qualcomm-target × model cells. Existing Workbench jobs provide
device, backend and real-checkpoint **component** evidence for four models,
but no cell has a complete target-stage set plus host input-to-output replay.
No hosted run measures resident Omni latency, shared peak RAM, continuous
state, playback or thermal behavior. The five exact registrations are
[listed separately](../aihub_device_inventory/README.md); none discloses an
exact usable-RAM capacity.

| Model | Current target component | First missing complete-chain gate |
| --- | --- | --- |
| Spark-X2.5 | One attention layer, with quality-dependent QNN/CPU variants; see [S24](../../../e2e_expansion_20260923/evidence/spark_s24_full_attention/README.md) and [RB3](../../../e2e_expansion_20260923/evidence/spark_rb3_full_attention/README.md) | Repair the fixed-cache numerical gate, then compile/validate 28-layer prefill and decode, embedding/head, sampling and KV/ring state for each target. |
| Qwen3-TTS | Talker and vocoder component exports; [talker](../../../e2e_expansion_20260924/evidence/qwen_tts_talker_qualcomm/README.md) and [vocoder](../../../e2e_expansion_20260924/evidence/qwen_tts_vocoder_qualcomm/README.md) | Quality-passing talker/predictor/vocoder bundle with persistent history and a complete streaming replay; several present accelerated components fail numeric or runtime gates. |
| Qwen3.8-27B | No qualified target bundle | Obtain exact target usable RAM, then prove an exact-checkpoint capacity refusal or validate language, vision and KV artifacts. File size alone does not prove fit. |
| MiniCPM-o | Speech-head step on several targets; [component evidence](../../../e2e_expansion_20260924/evidence/minicpmo_tts_qualcomm/README.md) | Thinker, vision/audio encoders, vocoder and stateful generation with end-to-end quality. |
| InternVLA | Cosmos encoder; [component evidence](../../../e2e_expansion_20260924/evidence/internvla_cosmos_qualcomm/README.md) | Target policy/action path, full handoff and [matching real observation/action references](../internvla_dataset_access/README.md). |

S25, S24 and RB3 Cosmos latents were injected into the real policy **on
local CPU** using a synthetic fixture. That measures a partial numerical
handoff, not a hosted target policy, physical action tolerance or task
success. The available Qualcomm API can run hosted model inference jobs but
does not expose an ADB/device shell to this workspace; `adb devices -l`
reported no attached device. The next useful Workbench job is a complete
target-compatible stage bundle, after component quality passes. A sum of
its stage times would remain an estimate until a resident controller can
run a whole request.
