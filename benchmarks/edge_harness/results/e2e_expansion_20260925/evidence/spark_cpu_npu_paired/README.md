# Spark paired benchmark first-attempt profile-path refusal

The first CPU/NPU/NPU/CPU attempt completed its [first CPU phase](01_cpu_report.json)
at 20/20 exact-reference requests, p50 4.755 s. The following NPU stage was
**refused before any measured request**: the [driver log](02_npu_driver.log)
shows `execution_provider_declined_graph` with 0/0 profiled nodes. The
[failed report](02_npu_report.json) and [paired-run record](paired_report.json)
are retained.

The new harness had placed `profile_dir` inside the WSL checkout. The
existing external-worker path translator only maps `/mnt/c` into native
Windows; the ORT worker therefore produced no verifiable node trace. This
was a probe-path failure, not evidence that the A16W8 graph cannot run on
the NPU. The harness was corrected to use the pinned Windows-accessible
profile root and rerun from a fresh output directory. The
[corrected four-phase profile](../spark_cpu_npu_paired_verified/README.md)
passed with one VitisAI node per NPU warmup.
