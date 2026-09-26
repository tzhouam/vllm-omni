# Qwen3-TTS CPU fallback on an HX370 PC with AMD NPU

**Disposition: scoped complete-request E2E on the CPU fallback; AMD NPU
acceleration remains unqualified.** On 2026-09-27 the native Windows 11
build-26200 host detected its healthy `VEN_1022&DEV_17F0` NPU through PnP.
The opt-in [probe](../../../../experiments/probe_omni_qwen_tts_cpu.py) selected
Omni's existing `external.qwen_tts.cpu.v1` whole-session stage. The
[execution plan](report.json) and loaded worker both report CPU BF16/SDPA;
the worker masks CUDA. The [audit](audit.json) checks NPU presence and CPU
placement separately, so this result does **not** count as NPU execution.

The pinned Qwen3-TTS 0.6B CustomVoice revision is
`85e237c12c027371202489a0ec509ded67b5e4b5`, with talker SHA-256
`bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb`
and speech-tokenizer SHA-256
`836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258`.
The isolated worker used Python 3.12, PyTorch 2.13.0+cu130 on CPU,
Transformers 4.57.3, Qwen TTS 0.1.1, eight threads, Ryan/English, seed 42,
and 64 maximum new tokens. Its complete output is 24 kHz mono PCM16.

| Check | Evidence |
|---|---|
| Named requests | Both complete WAVs matched the previously pinned standalone PCM SHA-256 values exactly. “Hello from the local computer.” returned 109,440 frames; “The blue car is parked beside the library.” returned 82,560. |
| Serial repeats | One warmup excluded; 3/3 measured requests returned identical first-sentence PCM. Nearest-rank complete-request wall p50/p95 was 14.824/14.970 s. With only three samples, p95 is the maximum, not a sustained estimate. Startup was 25.893 s separately. |
| Memory | A 10 GiB host-RAM reservation admitted the stage. The sampled maximum process-tree private memory was 5,401,538,560 bytes; this 0.25 s sampling does not prove the load peak. The ledger returned to zero after shutdown. |
| Cancellation | A separate [abort run](abort_report.json) completed a known request, then aborted an in-flight one. It returned no stale output, retired the worker, and released the host-RAM reservation. |

The archived [profile](report.json), [abort report](abort_report.json),
[WAV](output.wav), [abort-run WAV](abort_output.wav), and worker
[profile](worker.log)/[abort](abort_worker.log) logs retain raw results. The
[auditor](../../../../experiments/audit_qwen_tts_cpu_fallback_npu_host.py)
verifies the hashes, WAV format, measured request count, actual CPU placement,
NPU detection, budget, and cancellation lifecycle. The earlier
[20-request native CPU profile](../../../e2e_expansion_20260923/evidence/qwen_tts_omni_cpu/README.md)
is a separate run on the same laptop; its timing is not substituted for this
three-sample fallback profile.

This fallback is appropriate under the accepted whole-session-first plan:
the separate [two-utterance AMD NPU MLP cut](../qwen_tts_stateful_npu_boundary/independent_codes/README.md)
passed 11/11 offline waveform chunks per utterance but measured 0.548 ms NPU
versus 0.045 ms CPU p50 per isolated MLP call before transfer. A full eight-layer
NPU suffix also missed waveform gates. Neither is justified as a live stage
today. The CPU fallback is complete-WAV, not playable streaming; broader
speech quality, loading peak, paired performance, concurrency, and sustained
power/thermal behavior remain open. NPU acceleration needs a coarser cut that
passes numerical and whole-chain benefit gates.

To reproduce, use the pinned interpreter/model/overlay arguments from the
[native CPU profile](../../../e2e_expansion_20260923/evidence/qwen_tts_omni_cpu/README.md),
then add `--require-amd-npu-present`, `--warmups 1`, `--repeats 3` for the
profile and `--warmups 0`, `--repeats 1`, `--abort-check` for cancellation.
The flag refuses on non-Windows hosts or when the HX370 NPU is not detected
healthy; it does not request NPU execution.
