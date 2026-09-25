# MiniCPM-o HX370 CPU+AMD NPU speech-content follow-up

This is a scoped complete Omni audio+image → text+speech experiment on the
HX370. The BF16 thinker, talker, and codec remain on WSL CPU. Only the image
resampler KV projection is routed to the native-Windows AMD NPU worker. The
worker reported one VitisAI graph node and two CPU nodes, two live calls, and
clean closure in both NPU sessions. This placement is opt-in, not a default
selection or an all-NPU claim.

The checkpoint is MiniCPM-o 4.5 revision
`503e754207c94da6bb26850b4469f367c9ea3582` (weight index SHA-256
`e578de05a95804bb15237a6fd7c236414e0160cd76751da82c9f7f0d134596e7`).
The retained [A16W8 projection ONNX](resampler_kv_a16w8.onnx) SHA-256 is
`330bbbd0d18caaf6903aa41f836dbeaef57643a8afbaba333df68e3c1e722aeb`.
Input WAV and synthetic-image hashes are recorded in the reports; these are
the same fixtures as the [earlier combined-input experiment](../minicpmo_amd_npu_audio_image/README.md).
The source checkout was `3dd4614504420b1a0eb884b1bbe04f300e69fb1d` plus
the waveform-capture harness changes in this evidence commit. WSL Ubuntu used
PyTorch 2.13.0+cpu, vLLM 0.28.0 and Omni 0.29 source. Native Windows 11 build
26200 used AMD NPU driver 32.0.203.329, ORT 1.30.0 and VitisAI EP 1.8.63.0.
The running vLLM/Omni version mismatch remains an explicit tested boundary.

The [128-token CPU](cpu_report.json) and [NPU](npu_report.json) plans each
completed two serial requests, with 2/2 exact thinker text-token sequences and
124,800 finite nonzero audio samples per request. Archived float32 waveforms
showed **no waveform parity**: NPU versus CPU relative L2 was 1.300/1.291.
A [same-image CPU repeat](cpu_repeat_report.json) returned identical text
tokens; its first red-square waveform was bitwise equal to the first CPU
session, and its second in-session waveform differed by 0.00978 relative L2.
Thus the observed CPU repeat variation does not explain the CPU–NPU waveform
gap. Pinned Whisper tiny.en revision
`87c7102498dcde7456f24cfd30239ca606ed9063` transcribed only the opening
part of each 5.2 s WAV: WER versus the full text was 0.286/0.500 for CPU and
0.214/0.400 for NPU. These are ASR-content proxies, not perceptual-quality
scores. [Short-plan audit](comparison.json), [CPU ASR](cpu_asr_00.json),
[NPU ASR](npu_asr_00.json), [worker events](audio_image_kv_events.jsonl), and
the corresponding `_01` ASR reports preserve the observations.

The short plan capped the talker at 128 tokens. The otherwise matching
[CPU](cpu_audio_image_long.yaml) and [CPU+NPU](cpu_npu_audio_image_long.yaml)
plans raise that cap to 256, within the existing 512-token talker context.
Both [long CPU](cpu_long_report.json) and [long NPU](npu_long_report.json) runs
completed the red-square and blue-circle/green-triangle requests. The
[long-plan audit](comparison_long.json) verifies 2/2 exact thinker text-token
sequences, two live VitisAI calls, one NPU plus two CPU graph nodes, and clean
worker closure. CPU speech durations were 6.96/9.16 s; NPU durations were
6.76/8.52 s, so the waveform shapes still differ. The pinned ASR proxy
transcribed **the full expected response with WER 0 on all four WAVs**:
[CPU red](cpu_long_asr_00.json), [CPU blue](cpu_long_asr_01.json),
[NPU red](npu_long_asr_00.json), [NPU blue](npu_long_asr_01.json).
This fixes the observed speech-content truncation on these fixtures; it does
not prove acoustic parity, speaker similarity, natural-scene quality, or
correctness at longer outputs.

The long CPU/NPU startups were 102.48/102.06 s. Complete-request walls were
65.24/61.62 s for red and 25.70/24.89 s for blue, respectively. NPU graph
round trips were 75.3/177.3 ms, and its worker peak RSS was 383,320,064 bytes.
Sampled WSL RAM use reached 32.09 GB and swap 6.04 GB on the NPU run; the
separate CPU control reached 31.99 GB and 5.17 GB. Run order, caches, memory
pressure and power were not paired or controlled, so these two observations
do not establish speedup or admission safety. The [long worker events](audio_image_long_kv_events.jsonl)
and [ORT placement trace](audio_image_long_kv_profile/) retain the device
evidence. The generation limit, single concurrency, generated input audio,
synthetic images, and lack of listening/thermal measurements bound the result.

Reproduce the reports from the repository root with
`profile_minicpmo_image_suite.py`, the pinned model directory, two `--image`
arguments, the pinned `--audio`, `--archive-audio-dir`, and the corresponding
config above. Use `VLLM_TARGET_DEVICE=cpu`, `VLLM_CPU_KVCACHE_SPACE=1`,
`OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`, `CUDA_VISIBLE_DEVICES=`, and
`VLLM_ENABLE_V1_MULTIPROCESSING=0` in the pinned `omni-cpu` environment.
The as-run NPU config points to the original local copy of the ONNX file;
set `VLLM_OMNI_MINICPMO_KV_GRAPH` in a reproduced config to the retained
artifact's path when running on another machine.
Convert the saved float32 `.npy` files to 24 kHz float WAV with `soundfile`
for `probe_tts_asr.py`; run `audit_minicpmo_image_suite.py` on each CPU/NPU
report pair and its event log. The archived `.npy` files preserve exact
generated samples, while the WAVs provide ASR/listening inputs. Absolute
local paths in reports are as-run provenance; the files are adjacent here.

Next: inspect speech-token and codec state to explain the acoustic divergence,
then evaluate representative human speech/natural images with listening and
task-quality gates, paired warmed whole-request benefit, shared-RAM loading
peaks, cancellation/restart, and sustained power/thermal behavior. Keep the
NPU cut opt-in until those checks justify it.
