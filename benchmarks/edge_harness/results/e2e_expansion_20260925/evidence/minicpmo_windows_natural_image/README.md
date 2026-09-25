# MiniCPM-o native Windows natural-photo and spoken-question follow-up

Four **complete** MiniCPM-o 4.5 image+audio → text+speech requests passed through
the bounded Omni `StageRuntime` whole-session GGUF stage: one astronaut photo
and one cat photo on each of HX370 CPU and Radeon 890M+CPU. The language model
and vision encoder were on CPU for the CPU route; the Radeon route offloaded
37/37 language layers and vision to Vulkan0, with Token2Wav on CPU. This
advances the previously synthetic native Windows cells to two natural photos
with a spoken image question. It does not qualify broad visual or speech
quality, video, streaming playback, concurrent requests or a speedup.

The [input manifest](inputs/input_manifest.json) hashes the 448×448 JPEGs
derived from the previously retained public-domain NASA Eileen Collins
[astronaut photo](../minicpmo_amd_npu_natural/inputs/astronaut.png) and CC0
[Chelsea cat photo](../minicpmo_amd_npu_natural/inputs/chelsea_cat.png). Windows
SAPI `Microsoft Huihui Desktop` spoke “What is in the image?”; the resulting
mono audio was resampled to 16 kHz PCM. Pinned Whisper tiny.en transcribed the
[actual input](inputs/question_asr.json) exactly. Thus the input is spoken,
but it is synthesized speech rather than human speech.

| Route and photo | Complete-request wall (s) | Text content | Output speech (s) | Whisper WER vs own text |
|---|---:|---|---:|---:|
| [CPU astronaut](cpu_astronaut_report.json) | 68.501 | woman in orange astronaut suit, helmet, shuttle and flag | 21.28 | 0.000 |
| [CPU cat](cpu_cat_report.json) | 63.883 | tabby cat with green eyes | 19.44 | 0.039 |
| [Radeon+CPU astronaut](radeon_astronaut_report.json) | 106.293 | orange space suit, helmet, shuttle and flag | 39.24 | 0.089 |
| [Radeon+CPU cat](radeon_cat_report.json) | 60.598 | tabby cat with green eyes | 20.76 | 0.000 |

The [audit](audit.json) checks the four report/input/model hashes, completed
terminal events, 24 kHz mono PCM frame counts and digests, expected image
descriptions, zero post-shutdown reservations and the saved worker placement
logs. It independently transcribes each retained WAV with pinned
`openai/whisper-tiny.en` revision
`87c7102498dcde7456f24cfd30239ca606ed9063`; the 39.24 s output is
transcribed in ≤25 s windows. The Radeon astronaut ASR appends a short phrase
after the displayed text, so its 0.089 WER is a proxy warning, not a listening
quality pass. The [WAVs](cpu_astronaut.wav) and all four worker/driver logs are
retained beside the reports.

The first [CPU astronaut attempt](cpu_astronaut_20s_refusal.json) generated a
relevant answer, but its long speech exceeded the previous 20 s admitted
output bound and was **correctly refused**. The passing runs explicitly used
60 s/4 MiB output limits; they did not relax the backend's hard maximum of
120 s/64 MiB. All four used a 21 GiB shared-RAM demand, 2048-token context,
96 new-token limit, one request and no warmup. Native NTFS held the same ten
GGUF Q4_K_M/F16 artifacts at revision
`db25077c33951fe163b42986fba0132e279872a2`; each run verified the
artifact hashes in its report. The CPU and Radeon C++ executables have SHA-256
`29247187ddf3b13591cd14cc070e5e25c6bb7547296acc5e8a74e2945811d725`
and `66a512386aafeb708f49a7394a9054d94ec777c362395c8917a3d198aef1b523`.
The controller ran on Windows 11 build 26200 with installed vLLM-Omni 0.29;
the Radeon driver was `32.0.22018.6001` (queried after the runs).
the C++ worker executed the model. Output lengths, image contents and cache
states vary across these independent runs, and power/cache conditions were
not paired, so the wall times are not a controlled CPU/Radeon comparison.

To reproduce a request, run
[`probe_omni_minicpmo_cpp.py`](../../../../experiments/probe_omni_minicpmo_cpp.py)
from the repo root in the Windows Omni environment with the pinned model
directory, matching worker executable and CLI SHA, the prior
[`full_report.json`](../../../e2e_expansion_20260923/evidence/minicpmo_cpp_radeon890m_prompted/full_report.json)
as `--artifact-report`, and these options:

```text
--audio-wav <this evidence dir>/inputs/question_16k.wav
--image-jpeg <this evidence dir>/inputs/astronaut_448.jpg
--placement cpu                 # or radeon-hybrid with its matching executable
--reserve-gib 21 --capacity-gib 21 --warmups 0 --repeats 1
--expected-text-any astronaut --expected-text-any shuttle --expected-text-any space
--max-output-audio-s 60 --max-wav-bytes 4194304
```

The probe also requires `--model-dir`, `--cli-bin`, `--cli-sha256`,
`--reference-wav`, `--work-root`, `--log-root`, `--output-report` and
`--output-wav`. Use `--expected-text-any cat` for the cat. The actual reports
pin media, weights and worker SHA values and should be checked before
comparing a rerun. The independent audit command is:

```text
PYTHONPATH=. python benchmarks/edge_harness/experiments/audit_minicpmo_cpp_natural.py \
  --evidence-dir benchmarks/edge_harness/results/e2e_expansion_20260925/evidence/minicpmo_windows_natural_image \
  --asr-model-dir /home/zhout/project/edge_infer/models/whisper-tiny.en-87c7102 \
  --asr-revision 87c7102498dcde7456f24cfd30239ca606ed9063 \
  --output benchmarks/edge_harness/results/e2e_expansion_20260925/evidence/minicpmo_windows_natural_image/audit.json
```

Next gates are additional independent photos and human-recorded questions,
text/speech alignment and listening assessment, simultaneous and repeated
requests, cancel/restart under long speech, loading peak and sustained
memory/power. The AMD NPU and mobile cells are unaffected by this evidence.
