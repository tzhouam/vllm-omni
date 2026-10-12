# Qwen3-TTS: real-input predictor component on hosted Qualcomm targets

**Depth: C, not E2E.** A batch-1, single-request CustomVoice utterance was
generated from `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` revision
`85e237c12c027371202489a0ec509ded67b5e4b5` on local CPU. Its first
code-predictor input is a real talker hidden state plus generated-code
embeddings, not a random shape fixture. The pinned 5-layer predictor body was
exported to ONNX and locally matched eager output at relative L2 3.34e-6.
The source graph SHA-256 is
`c0eb88559bbcafdea66de164f8d17ec0100407e5aa8f84bbf29628634c0b97c2`.
The real-input NPZ SHA-256 is
`97b2366cca628e6293e2d6edf5a6cd792f28878fb7a6c1e1f9c3161893a5a16b`.
The 301 MB source and compiled target downloads are retained outside Git at
`/home/zhout/project/edge_infer/models/qwen3tts_predictor_real_20261002/`;
[fixture_report.json](fixture_report.json) pins the source and fixture;
per-target job records pin each compiled target SHA-256. Only the
stage output snapshots, job records, and placement profiles are retained here;
the input NPZ remains alongside the large source graph.

| Exact target | Runtime, requested unit | Compile job | Real-input inference job | Outcome |
| --- | --- | --- | --- | --- |
| Samsung Galaxy S25 | TFLite, CPU | [j57ewl9qp](https://workbench.aihub.qualcomm.com/jobs/j57ewl9qp/) | [jp4yod3qp](https://workbench.aihub.qualcomm.com/jobs/jp4yod3qp/) | SUCCESS; relative L2 5.91e-6; [profile](s25/profile.json) all 367 detail rows CPU |
| Samsung Galaxy S24 | TFLite, CPU | [jpyowjxr5](https://workbench.aihub.qualcomm.com/jobs/jpyowjxr5/) | [jglyz2xe5](https://workbench.aihub.qualcomm.com/jobs/jglyz2xe5/) | SUCCESS; relative L2 5.91e-6; [profile](s24/profile.json) all 367 detail rows CPU |
| SA8775P ADP | TFLite, CPU | [jp0mq2j2g](https://workbench.aihub.qualcomm.com/jobs/jp0mq2j2g/) | [jp3z319x5](https://workbench.aihub.qualcomm.com/jobs/jp3z319x5/) | SUCCESS; relative L2 5.91e-6; [profile](sa8775p/profile.json) all 367 detail rows CPU |
| Dragonwing RB3 Gen 2 Vision Kit | TFLite, CPU | [jp8e9mxzp](https://workbench.aihub.qualcomm.com/jobs/jp8e9mxzp/) | [jgol0nr4g](https://workbench.aihub.qualcomm.com/jobs/jgol0nr4g/) | SUCCESS; relative L2 5.91e-6; [profile](rb3/profile.json) all 367 detail rows CPU |
| Snapdragon X Elite CRD | ONNX, CPU | [jgk2nq4yg](https://workbench.aihub.qualcomm.com/jobs/jgk2nq4yg/) | [jgjrm277p](https://workbench.aihub.qualcomm.com/jobs/jgjrm277p/) | SUCCESS; relative L2 4.05e-6; [profile](xelite/profile.json) all 357 detail rows CPU |

The four TFLite target IDs are S25 `mng2xo7om`, S24 `mq26go6ln`, SA8775P
`mmrg89gwq`, and RB3 `mq9y2oyrn`. Their downloaded target binaries are
byte-identical, SHA-256
`666b0649034e9fcffa4a45f4d852ff5c4a687ad76926123a0300182c768ab7c0`;
their device-output NPZ SHA-256 is
`334b892937fd911a1584d147790fefe6741e50ece891a26547cd4fdd2e2ba182`.
X Elite target `mq807j03n` is a separate ONNX package with SHA-256
`e435606ea13594344c0fecfe10edbe61bb8586760553cd3975c14b7667202b19`;
its output NPZ SHA-256 is
`f4d7b9ba00f31b39833717522273e351390ca03a55cff0c5390bf999bb55dc0d`.
Distinct compile/inference job IDs above still identify each exact device;
identical target bytes do not imply identical placement or performance.

The 15 pinned code heads applied offline to the hosted hidden tensor preserved
all **15/15 top-1 codes** versus the local predictor-body reference on all
five targets. The locally generated request sampled the subtalker
(`subtalker_dosample=true`), so its sampled code sequence matches reference
top-1 in only 8/15 groups; that is not a device mismatch. Per-target
[`code_audit.json`](s25/code_audit.json) records both comparisons.

The first missing neural stage for a hosted **complete TTS request** is
revision-pinned prompt/text embedding and talker prefill with its full KV
state. The retained target talker artifact only covers one fixed-cache decode
step. The hosted predictor body here is teacher-forced, whereas a complete
frame requires recursively sampling all 15 remaining codebooks; target
quantizer, vocoder history, and tail-frame handoff are also not connected.
The real utterance's talker, embeddings, and vocoder ran locally to make this
fixture. This evidence therefore cannot promote any TTS mobile matrix cell to
AI Hub E2E or claim device-local playback/latency.

Reproduce the fixture with
[`prepare_qwen_tts_predictor_real_fixture.py`](../../../../experiments/prepare_qwen_tts_predictor_real_fixture.py)
and run a target with
[`run_qwen_tts_predictor_real_aihub.py`](../../../../experiments/run_qwen_tts_predictor_real_aihub.py)
using `qai-hub==0.56.0`. The latter is resumable by job ID and sets
`profile=False` for its functional inference. Separate placement profiles
measure a component only; their timings are not summed into whole-request
latency. These jobs were submitted before the replay runner added
`--max_profiler_iterations 1`; `profile=False` alone disables per-layer
profiling but AI Hub still performs its default repeated inference timing.
