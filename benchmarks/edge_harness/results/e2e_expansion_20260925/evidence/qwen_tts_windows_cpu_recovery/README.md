# Native Windows CPU Qwen3-TTS fresh-stage recovery

**Disposition (2026-09-25): scoped text-to-complete-WAV recovery passed.**
On a Ryzen AI 9 HX 370 laptop running Windows 11 build 26200, the public
`AsyncOmni.generate` route loaded the pinned Qwen3-TTS 0.6B CustomVoice
checkpoint in its isolated BF16/SDPA CPU worker. It returned a 24 kHz,
109,440-frame WAV for the named sentence. A second request reached the
worker-side `request-start` marker before Omni aborted it. No audio from that
request reached the client. A newly initialized Omni stage returned the
same PCM SHA-256 under a different worker generation. The route and artifact
pins are those of the [20-request Windows CPU profile](../../../e2e_expansion_20260923/evidence/qwen_tts_omni_cpu/README.md).

The [raw report](report.json) records initial and fresh-stage startup at
26.033/12.166 s, the separate complete requests at 15.570/15.636 s, and
abort acknowledgement at 0.263 s. Both WAVs have PCM SHA-256
`c8a49b5c73efd642a1eb04be973d6aefdc8ed9cfd18b34ff67690b8464906bed`,
matching the prior standalone and Omni profile. The [first worker log](worker.log)
contains one completed WAV followed by the abort-request start marker with
no completion line. The [fresh worker log](worker_restarted.log) contains
the new worker's completed WAV. Both worker properties report CPU placement,
PyTorch 2.13.0+cu130 on CPU, Transformers 4.57.3, `qwen-tts` 0.1.1,
BF16/SDPA and 64 maximum new tokens. The [driver log](driver.log) records
two stage shutdowns. The [independent audit](audit.json) verifies those
properties, audio-free abort output, exact PCM, terminal events, and
distinct generations; it also hashes the raw evidence files.

Reproduce with
[`probe_omni_qwen_tts_cpu_entrypoint.py`](../../../../experiments/probe_omni_qwen_tts_cpu_entrypoint.py)
using the pinned arguments and PowerShell environment in the
[Windows CPU profile](../../../e2e_expansion_20260923/evidence/qwen_tts_omni_cpu/README.md),
plus `--abort-recovery` and fresh `--server-log`/`--output-report` paths.
Set `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8` and
`VLLM_ENABLE_V1_MULTIPROCESSING=0`. Re-run
[`audit_qwen_tts_cpu_restart.py`](../../../../experiments/audit_qwen_tts_cpu_restart.py)
against the report, two worker logs and driver log. The model stays on CPU;
the installed RTX and Radeon are not selected.

This is a fresh-stage restart after one genuinely started in-flight abort,
not resumed generation in the retired stage. It does not qualify playable
incremental speech, broad voice quality, concurrency, loading peak, or
sustained power and thermal behavior. The earlier 20-request latency profile
remains the performance evidence; these two requests are not a new p50/p95.
