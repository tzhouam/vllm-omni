# Spark native Windows restart after in-flight cancellation

The pinned Spark-X2.5-1.7B Q4_K_M GGUF and llama.cpp b11124 binary from the
[native Windows Omni text profile](../../../e2e_expansion_20260923/evidence/spark_omni_llamacpp/README.md)
were reused without changing model or precision. The model SHA-256 is
`902bde2522394954ac17821b3e5fd0df02defbc6944f122253f2580acf0503f4`;
the server SHA-256 is
`9ffc5919acb4cb43c7be5f3053b59014cce70a87d3a801fcad49c4f459984f52`.
The host was Windows 11 build 26200 on Ryzen AI 9 HX 370, with Radeon 890M
driver 32.0.22018.6001 and the Performance power scheme. Native Python used
vLLM 0.29.0 and this Omni checkout. The RTX 5090 Laptop GPU was present but
not selected in the Radeon route. The current OS, CPU, GPU driver and power
scheme are retained in the [Windows hardware record](hardware.json).

The [probe](../../../../experiments/probe_omni_llamacpp_spark.py) ran one
complete `Paris` request, one 120-item inventory request (`1511`), and one
serial inventory measurement before submitting a second inventory request
for cancellation. It waited until the owned llama.cpp server log recorded a
fourth task start, checked no terminal output had arrived, then cancelled.
The first StageRuntime released its 4 GiB host-RAM reservation and terminated
the worker. A fresh StageRuntime loaded the same hashed artifact and returned
`Paris` with a different worker generation, no late output and no remaining
reservation or quarantine. The [auditor](../../../../experiments/audit_spark_windows_restart.py)
binds the [CPU](cpu_v4_report.json) and [Radeon](radeon_v4_report.json) reports
to their separate [CPU first](cpu_v4_server.log),
[CPU restart](cpu_v4_server_restart.log),
[Radeon first](radeon_v4_server.log) and
[Radeon restart](radeon_v4_server_restart.log) server logs. Both routes
[passed](audit_report.json); CPU offloaded 0/29 layers with CPU model buffers,
while Radeon 890M Vulkan1 offloaded 29/29 layers before and after restart.
This is **one fresh-stage restart per placement**, not automatic same-session
state migration or incremental token streaming.

The initial [CPU](cpu_report.json) and [Radeon](radeon_report.json) probes,
and the [second Radeon run](radeon_v2_report.json), cancelled after admission
but before the server had started the abort task; they were insufficient for
the in-flight claim. A [third CPU attempt](cpu_v3_report.json) waited for a
server task start but used a short counting prompt that finished before
cancellation, and failed explicitly. Those raw failures are retained. The
final inventory-triggered runs are separate from the earlier 20-request
latency profiles; they do not establish a new p50/p95, broader text quality,
concurrency, load peak, power or thermal behavior.
