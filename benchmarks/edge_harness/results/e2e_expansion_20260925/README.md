# E2E expansion follow-up — 2026-09-25

The [MiniCPM-o HX370 AMD NPU projection experiment](evidence/minicpmo_amd_npu_projection/README.md)
extracts a state-safe speech-token projection after the CPU speech head.
It passes one fixed-fixture NPU component numerical check, but the original
FP32 CPU projection is faster even before cross-OS handoff. The device/model
pair remains NOT E2E in the [60-cell matrix](../e2e_profiling_20260922/evidence/summary/README.md).

The [InternVLA HX370 AMD NPU batch-one experiment](evidence/internvla_amd_npu_batch1/README.md)
isolated a six-frame VitisAI convolution failure to frames 1–5, then
recovered a numerical component pass with six batch-one calls. The measured
NPU boundary was replayed through the real CPU encoder suffix and policy
for synthetic numerical and action-sensitivity checks. This is component
evidence; the native full-policy NPU path remains NOT E2E.
