# E2E expansion follow-up — 2026-09-25

The [MiniCPM-o HX370 AMD NPU projection experiment](evidence/minicpmo_amd_npu_projection/README.md)
extracts a state-safe speech-token projection after the CPU speech head.
It passes one fixed-fixture NPU component numerical check, but the original
FP32 CPU projection is faster even before cross-OS handoff. The device/model
pair remains NOT E2E in the [60-cell matrix](../e2e_profiling_20260922/evidence/summary/README.md).
