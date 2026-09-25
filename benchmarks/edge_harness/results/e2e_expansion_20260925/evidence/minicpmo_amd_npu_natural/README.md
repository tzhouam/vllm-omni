# MiniCPM-o natural image + recorded speech on HX370 CPU and CPU+AMD NPU

**Disposition (2026-09-25): two scoped complete Omni requests pass on both
routes, including real photos and human narration. The AMD NPU cut remains
opt-in and is not release-qualified or a measured speedup.** The BF16
MiniCPM-o 4.5 thinker, talker and codec stay on WSL CPU. Only the thinker's
real resampler KV projection uses the native-Windows VitisAI worker. This
extends the earlier [synthetic-input, retained-waveform experiment](../minicpmo_amd_npu_waveform/README.md).

The [input manifest](inputs/input_manifest.json) pins two `scikit-image`
photos: the public-domain NASA Eileen Collins [astronaut](inputs/astronaut.png)
and Stefan van der Walt's CC0 [Chelsea cat](inputs/chelsea_cat.png). The
human [LibriSpeech excerpt](inputs/libri1_first_6s.wav) is the first six
seconds of the `librosa` `libri1` recording, Garth Comira reading Marion
Bryce's *The Ashiel Mystery*, CC-BY-4.0. The manifest records the original
OGG SHA-256, excerpt SHA-256, image hashes and package versions. These are
two familiar photos and one narration, not a representative quality set.
An independent pinned Whisper tiny.en pass heard the opening ship description;
there is no gold transcript for the six-second excerpt in this evidence.

The initial fixed-bucket NPU adapter refused the astronaut's actual
`[3,1035,1152]` BF16 tensor and the request failed; its
[event record](audio_image_long_kv_events.jsonl) preserves the refusal. The
graph accepts `[1,1024,1152]` and is a tokenwise MatMul, so the opt-in
adapter now packs at most 4,096 logical token rows into 1,024-row graph
calls, pads the last tile, and restores crop/token order. It still refuses
wrong precision/device, oversize input, missing NPU placement, invalid output
or numerical-gate failure. A first tiled launch then failed vLLM's weight
accounting because the already-loaded CPU parity module was registered as a
new parameter; [that failed startup log](npu_driver.log) and
[event](natural_tiled_kv_events.jsonl) remain. Retaining the reference only
as an unregistered validation object fixed the loader. The
[tile tests](../../../../../../tests/edge/test_minicpmo_kv_tiles.py)
passed 2/2, including the three-crop shape and bounded rejection.

The 256-token [CPU](cpu_report.json) and [NPU](npu_report.json) plans then
completed both audio+image → text+speech requests. The NPU run's
[placement events](natural_tiled2_kv_events.jsonl) show one VitisAI and two
CPU graph nodes, four graph calls for the astronaut (3,105 rows) and one
for the cat (1,014 rows), and a clean close. Its same-input BF16 CPU
projection check measured 0.647%/0.640% relative L2. The thinker token
sequences differed 0/2 exactly, but both routes described the narrated
white-and-scarlet ship and the visible astronaut/space shuttle or green-eyed
cat. Pinned Whisper tiny.en transcribed the generated speech against each
route's **own text**, with [CPU](cpu_asr.json) WER 6.9%/11.1% and
[NPU](npu_asr.json) WER 16.7%/10.7%. The NPU astronaut WAV ended before the
last phrase under this cap. The [256-token comparison](comparison.json)
retains output and memory details; acoustic samples and durations differed.

The matching [384-token CPU](cpu_audio_image_384.yaml) and
[CPU+NPU](cpu_npu_audio_image_384.yaml) plans kept the checkpoint, inputs,
512-token talker context and placement unchanged. The NPU plan also enforces
a provisional **1% same-input projection relative-L2 limit** before the
resampler suffix. The [CPU](cpu_384_report.json) and [NPU](npu_384_report.json)
each completed 2/2 requests; the [audited comparison](comparison_384.json)
verified all five calls through the one-VitisAI-node graph session, two complete
request groups and clean worker closure. The projection errors remained
0.647%/0.640%. Text tokens still differed 0/2 exactly, while the key
spoken and visual facts remained present. The 384-token generated WAVs are
retained for [CPU](cpu_384_audio/) and [NPU](npu_384_audio/). Pinned Whisper
transcribed the full responses, including the previously missing astronaut
ending; WER against each route's own text was [CPU](cpu_384_asr.json)
6.9%/11.1% and [NPU](npu_384_asr.json) 13.3%/10.7%. Compound words and
the ship's proper name account for part of those proxy errors. These ASR
checks do not establish listening quality, speaker similarity or gold
input-grounded transcription accuracy.

On the HX370, WSL Ubuntu used PyTorch 2.13.0+cpu, vLLM 0.28.0 and Omni
0.29 source; native Windows 11 build 26200 used AMD NPU driver
32.0.203.329, ORT 1.30.0 and VitisAI EP 1.8.63.0. The 384-token CPU/NPU
startups were 103.37/109.56 s. Separate complete-request walls were
76.29/80.00 s for astronaut and 30.91/31.94 s for cat. The NPU graph
round trips totaled 170/164 ms; its worker peak RSS was 394,928,128 bytes.
Sampled NPU WSL RAM use reached 32.13 GB and swap 6.38 GB, versus CPU
31.98 GB and 6.02 GB. Its [raw events](natural_tiled384_kv_events.jsonl),
[ORT placement trace](natural_tiled384_kv_profile/) and
[driver log](npu_384_driver.log) pin actual execution. The as-run ONNX graph
SHA-256 is `330bbbd0d18caaf6903aa41f836dbeaef57643a8afbaba333df68e3c1e722aeb`;
an exact [copy](../minicpmo_amd_npu_waveform/resampler_kv_a16w8.onnx) is
retained nearby. The checkpoint revision is
`503e754207c94da6bb26850b4469f367c9ea3582`; exact report/config/input
hashes are in the JSON evidence. Loading peaks, shared Windows RAM pressure,
power and thermals were not paired or controlled, so no benefit or default
placement is inferred from these walls.

Reproduce from the repository root by running
[`prepare_minicpmo_natural_inputs.py`](../../../../experiments/prepare_minicpmo_natural_inputs.py)
with `--output-dir` pointing here, then
[`profile_minicpmo_image_suite.py`](../../../../experiments/profile_minicpmo_image_suite.py)
with the pinned model directory, two `--image` paths, the input `--audio`,
`--archive-audio-dir`, and either 384-token config. Use
`VLLM_TARGET_DEVICE=cpu`, `VLLM_CPU_KVCACHE_SPACE=1`, `OMP_NUM_THREADS=8`,
`MKL_NUM_THREADS=8`, `CUDA_VISIBLE_DEVICES=`, and
`VLLM_ENABLE_V1_MULTIPROCESSING=0` in the pinned `omni-cpu` environment.
The as-run NPU config points to the original local ONNX copy; set its graph
path to the retained artifact on another machine. Convert `.npy` to 24 kHz
float WAV with `soundfile`, then run
[`audit_minicpmo_image_suite.py`](../../../../experiments/audit_minicpmo_image_suite.py)
on both reports and the NPU event log, plus
[`audit_minicpmo_speech_asr.py`](../../../../experiments/audit_minicpmo_speech_asr.py)
on each report. The latter verifies WAV/sample identity before transcription.

Next: use more varied natural and recorded inputs with human listening and
task references; explain CPU/NPU text and acoustic differences; measure
paired warmed whole requests and actual shared-RAM loading peaks; exercise
cancellation/restart and sustained power/thermal behavior. Keep the NPU
split opt-in until whole-chain quality and benefit justify selection.
