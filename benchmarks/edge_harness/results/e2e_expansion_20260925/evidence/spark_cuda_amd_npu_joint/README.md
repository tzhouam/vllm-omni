# Spark CUDA+AMD NPU first harness attempt

The first paired-phase harness completed 21/21 complete RTX-only and 21/21
RTX+NPU requests. Both phases generated the same pinned 64-token sequence;
the NPU worker retained a VitisAI node and all CUDA top-64 refinements. The
[raw report](report.json) nevertheless has `status=failed` because the
initial harness required exactly 1,344 NPU calls (one per generated token)
and rejected **two additional initialization calls** recorded by the
[worker](npu_worker.json). The [driver log](driver.log) and
[ORT trace](npu_raw_profile.json) are retained. This was an accounting
failure, not evidence that the model or accelerator could not execute.

The [corrected independent run](../spark_cuda_amd_npu_joint_verified/README.md)
repeated both phases with bounded extra-call accounting and a separate raw
trace. Use that record for the matrix status and measured comparison.
