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

The [MiniCPM-o WSL RTX CUDA shutdown retest](evidence/minicpmo_cuda_shutdown_fix/README.md)
repeated the real-weight image-to-text+speech path for one warmup plus 20
serial measured requests. All outputs passed and all three stage workers
retired without a forced kill or shared-memory warning after scheduler-owned
chunk transport cleanup. Small per-request terminal-marker segments still
remained until engine shutdown, so sustained-session cleanup is open. A
separate [WSL CPU BF16 image retest](evidence/minicpmo_cpu_shutdown_final/README.md)
completed one image-to-text+speech request and exited cleanly after giving
the CPU thinker 30 seconds to retire and budgeting the three serial stage
shutdowns together. The CPU run used substantial WSL RAM and swap. Neither
single-fixture result broadens model quality or speech qualification.

A later [MiniCPM-o WSL RTX sender-role retest](evidence/minicpmo_cuda_marker_fix/README.md)
corrects the preceding serial marker-lifetime finding. The model runner
already sent stage-0 output through the orchestrator, but its AR scheduler
also enqueued an empty shared-memory terminal chunk on an edge with no
sender connector. After aligning the scheduler with the existing sender-role
contract, another one-warmup/20-measured image run passed with clean
shutdown. Across 188 shared-memory samples, stage-0 segments were transient
and returned to zero during the run instead of accumulating per completed
request. Concurrent and sustained-session bounds remain unverified.
The [WSL CPU BF16 regression](evidence/minicpmo_cpu_marker_fix/README.md)
also passed one red-square image-to-text+speech request on the installed
vLLM 0.28 wheel, with clean shutdown and no stage-0 segment left after exit;
the run was heavily swapped and does not qualify sustained CPU behavior.
