# Local edge backends in Omni

The first backend is `external.graph.v1`: one complete, stateless graph operation
per request, with a fixed input bucket and copied host tensors. A second backend,
`external.llamacpp.text.v1`, owns one pinned GGUF whole-model server process and
returns complete text requests. `external.crisp.tts.v1` owns a pinned
CustomVoice GGUF talker, codec and local hybrid CPU+Radeon process, returning
complete PCM audio. `external.qwen_tts.cpu.v1` owns the original BF16 CustomVoice
checkpoint in an isolated native Windows CPU process and also returns complete
PCM audio. All use the normal `Omni` / `AsyncOmni` APIs,
`PipelineConfig`, `StageRuntime`, `StagePool` and orchestrator. Native vLLM stages
retain their model execution, KV cache, sampling and batching. No accelerator is
mandatory. The llama.cpp backend currently uses the complete-request graph
control path for admission and acknowledgement; it is not an ONNX graph and does
not yet support incremental token events or restart after active cancellation.
The CrispASR route likewise lacks playable incremental PCM and post-cancel
restart; its current placement gate is specific to the measured HX370 Radeon
890M shared-RAM configuration. The Qwen CPU route likewise returns a terminal
WAV only; its worker pins an isolated Transformers 4.57.3 overlay and verifies
CPU placement and weight hashes before serving.

## Boundaries

| Owner | Responsibility |
|---|---|
| `omni_stage_contracts` | Version/features, buffer descriptors, request/event identities, opaque state, physical device descriptors, artifact manifests. Root import uses only the standard library. |
| `vllm_omni.host` | Explicit interpreter/OS routes, process bootstrap, process identity and bounded tree retirement. |
| `vllm_omni.engine.backends` | Adapt backend operations to `StageClientBase`; observe placement; fence epochs and worker generations. |
| `StageRuntime` | Resolve all replicas, reserve the whole plan before loading, create clients, unwind failures. |
| `ResourceLedger` | Atomically charge every affected memory constraint; keep uncertain allocations quarantined. |
| `StagePool` / orchestrator | Existing stage dispatch, handoff, polling, metrics, cancellation and request cleanup. |
| Model adapter | Tokenization, media preparation, stage transforms, sampling semantics, artifact export and numerical/task validation. |

Native Windows device-init locks use the host temporary directory, independent
of a local or UNC checkout. This coordinates same-user native processes. It does
not establish a cross-user or Windows/WSL global lock authority; cross-domain
workers in one graph plan are coordinated by that controller's reservations.

Native Windows SHM transport also uses a host-local directory for locks and
acknowledgements. A producer retains each named mapping until its single consumer
copies the payload and acknowledges its generation, or the producer cancels/closes
it. A length header handles Windows page-rounded mapping sizes. Retained mappings
are limited to 64 MiB and 256 buffers per producer process; exceeding either limit
fails the write explicitly. This host transport fix does not enable cross-OS or
accelerator zero-copy. POSIX keeps its existing unlink-based lifetime.

`edge.local.session` and `edge.local.external.protocol` retain compatibility imports.
The old direct `ExternalStage` API remains available; new pipelines should use
the runtime-owned graph backend. New state handles default to non-replayable and
non-migratable. Legacy M0 defaults are explicitly preserved in `contracts.legacy`.

## Hardware composition

The [E2E support roadmap](omni_edge_e2e_roadmap.md) maps every configuration to
implementation work packages, model-specific acceptance gates and a common
profiling protocol, including explicit artifact/capacity rejection outcomes.

The [end-to-end profiling follow-up](../../benchmarks/edge_harness/results/e2e_profiling_20260922/README.md)
checks the same 60 pairs and profiles five executable desktop paths with 20
requests per length/concurrency group and 30-minute sustained runs. It retains
explicit blockers for the other 55 cells. Performance completion is separate
from release qualification: TTS playback/sustained real-time failures, quality,
memory-accounting and unavailable-device gaps remain documented repair items.

### Mobile and PC × model support matrix

This section preserves the 2026-09-22 audit. For the latest per-cell disposition and 2026-09-23 follow-up evidence, use the [current 60-cell E2E matrix](../../benchmarks/edge_harness/results/e2e_profiling_20260922/evidence/summary/README.md). The HX370 AMD NPU now has a placement- and quality-verified **Spark 1.7B output-head component** through Omni, while complete Spark generation on that route remains unverified; see the [component record](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/spark_amd_npu_output_head/README.md). Native Windows CPU and Radeon 890M completed [scoped whole-model Spark text requests](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/spark_radeon890m_llamacpp/README.md) with the same Q4_K_M GGUF in standalone llama.cpp, then [complete requests through a bounded Omni stage](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/spark_omni_llamacpp/README.md). The Omni route verifies pinned artifacts, actual placement, one-slot memory admission, terminal events and cancellation drain. Incremental streaming, restart after cancellation, memory peaks and broader quality remain open. These results advance the M3 hardware/backend evidence without claiming a joint multi-accelerator plan.

The [Qwen3-TTS CPU+Radeon whole-session path](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/qwen_tts_omni_hybrid/README.md) also completed two named text-to-audio inputs and 20 serial Omni requests with exact PCM parity to its standalone record. It verifies pinned executable and weights, logged hybrid placement, one-slot shared-RAM admission, terminal audio and cancellation drain. Playable streaming, post-cancel restart, loading/iGPU peaks, sustained use and broader speech quality remain open; RTF still exceeds 1. No joint CPU+iGPU+NPU claim follows from this result.

The [native Windows CPU Qwen3-TTS Omni path](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/qwen_tts_omni_cpu/README.md) completed two named text-to-WAV inputs and 20 serial Omni requests with exact PCM parity to the standalone PyTorch record. It verifies pinned weights, CPU-only placement, a 10 GiB host reservation, a 4 GiB pre-load refusal, terminal audio and cancellation drain. Request-wall p50/p95 were 15.32/15.60 s for 4.56 s of audio. Playable streaming, post-cancel restart, loading peak, broad voice quality and sustained use remain open. This CPU route does not qualify AMD or mobile execution.

**All 60 named pairings were checked on 2026-09-22** through actual runs,
raw-record review or artifact/runtime preflight. This is not 60 successful
executions. Each cell links to its evidence, finding and next required step.

PASS: specified complete workload passed. PARTIAL: functional output with
acceptance gaps. COMPONENT: model component only. PROFILE_ONLY: component
timing without quality qualification. FAILED: attempted startup/execution
failed. REJECTED: specific artifact refused. BLOCKED: prerequisite missing.

| Device / execution configuration | Spark-X2.5 | Qwen3-TTS 0.6B CustomVoice | Qwen3.8-27B | MiniCPM-o 4.5 | InternVLA-A1 |
|---|---|---|---|---|---|
| PC HX370: CPU / WSL | [PASS](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_cpu_wsl-spark) | [FAILED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_cpu_wsl-tts) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_cpu_wsl-qwen27) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_cpu_wsl-minicpm) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_cpu_wsl-vla) |
| PC HX370: CPU / native Windows | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_cpu_windows-spark) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_cpu_windows-tts) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_cpu_windows-qwen27) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_cpu_windows-minicpm) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_cpu_windows-vla) |
| PC HX370 + RTX 5090 Laptop / WSL | [PASS](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_cuda_wsl-spark) | [PARTIAL](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_cuda_wsl-tts) | [FAILED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_cuda_wsl-qwen27) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_cuda_wsl-minicpm) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_cuda_wsl-vla) |
| PC HX370 + RTX 5090 Laptop / Windows | [PASS](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_cuda_windows-spark) | [PARTIAL](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_cuda_windows-tts) | [FAILED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_cuda_windows-qwen27) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_cuda_windows-minicpm) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_cuda_windows-vla) |
| PC HX370 + Radeon 890M / Windows worker | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_igpu-spark) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_igpu-tts) | [COMPONENT](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_igpu-qwen27) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_igpu-minicpm) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_igpu-vla) |
| PC HX370 + AMD NPU / Windows worker | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_npu-spark) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_npu-tts) | [REJECTED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_npu-qwen27) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_npu-minicpm) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_npu-vla) |
| PC CPU+iGPU+NPU +/- discrete GPU, joint model execution | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_joint-spark) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_joint-tts) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_joint-qwen27) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_joint-minicpm) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_joint-vla) |
| PC Snapdragon X Elite CRD / AI Hub | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_xelite-spark) | [PROFILE_ONLY](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_xelite-tts) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_xelite-qwen27) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_xelite-minicpm) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#pc_xelite-vla) |
| Mobile Galaxy S25 / Snapdragon 8 Elite for Galaxy | [COMPONENT](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#mobile_s25-spark) | [COMPONENT](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#mobile_s25-tts) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#mobile_s25-qwen27) | [COMPONENT](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#mobile_s25-minicpm) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#mobile_s25-vla) |
| Mobile Galaxy S24 / Snapdragon 8 Gen 3 | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#mobile_s24-spark) | [COMPONENT](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#mobile_s24-tts) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#mobile_s24-qwen27) | [COMPONENT](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#mobile_s24-minicpm) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#mobile_s24-vla) |
| Embedded SA8775P ADP / AI Hub | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#embedded_sa8775p-spark) | [PROFILE_ONLY](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#embedded_sa8775p-tts) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#embedded_sa8775p-qwen27) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#embedded_sa8775p-minicpm) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#embedded_sa8775p-vla) |
| Embedded RB3 Gen 2 / QCS6490 / AI Hub | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#embedded_rb3-spark) | [PROFILE_ONLY](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#embedded_rb3-tts) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#embedded_rb3-qwen27) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#embedded_rb3-minicpm) | [BLOCKED](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md#embedded_rb3-vla) |

Counts: **3 PASS, 2 PARTIAL, 6 COMPONENT, 3 PROFILE_ONLY, 3 FAILED,
1 REJECTED, 42 BLOCKED**. The PC CPU Spark pass uses the existing vLLM 0.28
CPU wheel with the current Omni branch; CPU TTS fails a 3-versus-4-field
sampling ABI mismatch in that environment. Current Windows/WSL Omni Qwen
27B startup fails pipeline resolution; earlier bare-vLLM text evidence is
not a pass for this integration. Full MiniCPM-o/InternVLA checkpoints are
absent from the inventoried model roots. Native Windows CPU attention
kernels are absent from the installed extension.

S25 Spark, TTS predictor and MiniCPM-o speech-head components have fresh Hub
inference evidence. S24 and the other Hub targets retain explicitly marked
historical component/profile evidence. No direct ADB device is attached.

See the [full audit](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md) and its hashed raw records. Other hardware SKUs
remain outside this finite matrix; no universal device support is implied.

### Detailed PC model and backend pairings

The tested PC is a Ryzen AI 9 HX370 laptop with Radeon 890M, AMD NPU and
RTX 5090 Laptop GPU. Selecting CPU-only or a subset of those accelerators is
an execution configuration on that PC, not evidence from a different PC SKU.
Model support is specific to the following pairings:

| Model / artifact | PC execution configuration | OS / backend | Qualification |
|---|---|---|---|
| Spark-X2.5-4B BF16 | CPU + RTX 5090 Laptop GPU | WSL / vLLM CUDA | Current branch: M0 text/state/cancellation workload passed. |
| Spark-X2.5-4B BF16 | CPU + RTX 5090 Laptop GPU | Native Windows / vLLM CUDA | Current branch: M0 text/state/cancellation workload passed. |
| Spark-X2.5-1.7B INT8 | CPU execution; accelerators unused | WSL / vLLM CPU | Fresh 12-prompt acceptance passed on this branch with the existing vLLM 0.28 CPU wheel; historical token parity is separate. |
| Spark-X2.5-1.7B INT8 | CPU execution; no NVIDIA dependency | Native Windows | Model path not yet qualified; CPU Add graph success is not text-model support. |
| Qwen3-TTS-12Hz-0.6B CustomVoice | CPU + RTX 5090 Laptop GPU; both stages on GPU | WSL / vLLM CUDA | Current branch: real-weight audio streaming completes; playback stalls and remaining M2 gates stay open. |
| Qwen3-TTS-12Hz-0.6B CustomVoice | CPU + RTX 5090 Laptop GPU; both stages on GPU | Native Windows / vLLM CUDA | Current branch: real-weight audio streaming completes; playback stalls and remaining M2 gates stay open. |
| Qwen3-TTS-12Hz-0.6B CustomVoice, Q8_0 GGUF talker + F16 codec | CPU + Radeon 890M iGPU; code predictor on CPU FP32, talker/codec on Radeon | Native Windows / standalone CrispASR Vulkan | Two named text-to-WAV requests and a 20-request serial resident profile passed; [placement and quality evidence](../../benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/qwen_tts_radeon890m_joint/README.md). Still outside Omni, slower than real time and unqualified for streaming or sustained use. |
| Qwen3.8-27B vision tower, FP32 ONNX | CPU + Radeon 890M iGPU | Windows / ORT DirectML | Historical component validation only; not the complete 27B model or a newly qualified graph-stage vision-to-language pipeline. |
| Qwen3.8-27B vision tower, tested A16W8 exports | CPU + AMD NPU | Windows / ORT VitisAI | Tested whole-tower artifacts were rejected by NPU partitioning. This does not rule out other exports or smaller components. |
| Qwen3.8-27B complete multimodal pipeline | CPU/iGPU/NPU, with optional discrete GPU | Target-specific | Fresh public Omni startup fails pipeline resolution on Windows/WSL; earlier bare-vLLM short-text loading is separate and not multimodal validation. |
| MiniCPM-o | CPU/iGPU/NPU, with optional discrete GPU | Target-specific | Full model × PC combinations remain unqualified for this release. |
| InternVLA | CPU/iGPU/NPU, with optional discrete GPU | Target-specific | Full model × PC combinations remain unqualified for this release. |

ORT Add and the small AMD NPU graph are backend probes, so they are deliberately
not listed as supported application models. Likewise, a Galaxy S25 Spark attention
component run in AI Hub is a separate **model component × phone** pairing, not a
PC pairing or proof of complete mobile Spark execution. Other model × PC × backend
combinations require their own artifact and workload qualification.

### Hardware layouts represented by the abstraction

The abstraction supports these single-machine compositions. This table separates
topology/admission support from successful execution of a particular model; it
does not promise that every listed accelerator can run every model.

| Composition | Resource accounting | Current execution evidence |
|---|---|---|
| CPU only | Host RAM; WSL quota when applicable | ORT CPU graphs through public Omni on Windows and WSL. Native Windows CPU text remains unqualified. |
| CPU + integrated GPU | Shared host RAM | Radeon 890M DirectML graph through Omni. This extraction's probe is not a complete vision/language pipeline. |
| CPU + integrated NPU | Shared host RAM | AMD NPU VitisAI graph through Omni; permitted CPU partitions are explicit in the placement report. |
| CPU + integrated GPU + integrated NPU | Shared RAM plus optional observed package/power/bandwidth relationships | Topology and admission contracts tested; individual laptop routes execute. Simultaneous full-model overlap is not qualified. |
| CPU + integrated GPU + integrated NPU + discrete GPU | Shared RAM and separate discrete VRAM | Current Ryzen AI 9 HX370 / Radeon 890M / AMD NPU / RTX 5090 Laptop machine. Spark CUDA text and native two-stage TTS run on Windows and WSL; using all accelerators together is not qualified. |
| CPU + discrete GPU, with integrated devices unused or absent | Host RAM plus discrete VRAM | Native CUDA text/TTS execution uses this subset. NVIDIA is optional in the abstraction. |

Mobile/embedded SoCs fit the CPU + integrated GPU + NPU composition, but their
local controller and model paths still require device-specific qualification.
Galaxy S25 AI Hub inference/profiling passed for one Spark attention component;
this is component evidence, not full local mobile generation or streaming.
AI Hub is a test facility, never a deployment dependency. Intel, Apple and other
vendor combinations are not certified by the AMD/NVIDIA/Qualcomm results.

`DeviceDescriptor` separates physical device identity, execution domain and
integration. A CPU, integrated GPU and NPU can all reference `machine:ram`.
An additional discrete GPU references its own `machine:vram:<physical-id>`.
A WSL allocation can consume both `machine:ram` and `wsl:quota` constraints;
those are two limits on the same allocation, not two amounts of physical RAM.
Use the same physical GPU ID when Windows and WSL expose the same adapter.

Package IDs, bandwidth groups and power domains are optional observed relationships.
Missing values mean unknown. Fitting memory does not authorize overlap on a shared
package: selection and overlap still require model quality and complete workload
measurements. Synthetic topology tests cover CPU-only and all four integrated /
discrete combinations; they are not hardware certification.

## Deploy a graph

Register a `PipelineConfig` with `StageExecutionType.GRAPH`, `model_stage="graph"`
and an appropriate final output type. Set `async_chunk=False`. A graph following
another stage must have a model-owned `custom_process_input_func` that returns
`{"tensors": {name: numpy_array}}`. Names, dtypes and shapes must match the
manifest's placement examples. Graph stages do not tokenize prompts or apply
model-specific transformations.

Graph-containing v1 pipelines must be a linear chain with exactly one terminal
output at the last stage. Fan-out, joins and multiple final outputs are refused
before loading; those require reference-counted consumer leases.

Each stage in a graph-containing plan declares the same capacities and its own
demands, including native stages:

```yaml
pipeline: my_registered_pipeline
async_chunk: false
stages:
  - stage_id: 0
    backend:
      name: external.graph.v1
      manifest: /absolute/path/to/bundle/manifest.json
      route:
        name: local-cpu
        interpreter: /absolute/path/to/worker/python
        worker: /absolute/path/to/worker_ort.py
        ep: cpu
        os_domain: posix
      max_io_bytes: 8388608
      min_fraction_on_target: 1.0
      allowed_providers: [CPUExecutionProvider]
    resource_budget:
      capacities: {machine:ram: 2147483648, wsl:quota: 1073741824}
      demands: {machine:ram: 536870912, wsl:quota: 536870912}
```

Quote pool keys or use block mappings if your YAML producer treats colon-containing
keys specially. On native Windows use Windows paths and `os_domain: windows`.
From WSL use controller-visible paths; host services translate worker paths. Named
legacy routes still honor `VLLM_OMNI_EXTERNAL_PYTHON_<ROUTE>`, but explicit routes
have no developer-machine defaults.

Capacities are **pre-load controller ceilings**, below observed free host memory /
guest limits with safety margin. They are not continuously sampled available RAM.
Demands must include loading peaks, weights, state, activations, workspace,
transport copies, retained output and headroom. The ledger does not claim to be
an OS memory limiter or to attribute shared vendor allocations from RSS.
Driver/cache memory outside the worker remains part of the deployment estimate.

Budgeted native vLLM GPU stages also require `native_physical_vram_bytes`, the
operator-reported **total** VRAM of the exact device named by `native_pool` (for example,
`vram:0`). The controller ceiling may be lower than that total because other
consumers or a safety margin use the card. Native admission charges at least
`ceil(native_physical_vram_bytes * gpu_memory_utilization)` plus the graph
capture reserve; using the controller ceiling in that formula would undercount
vLLM's possible allocation. The stage refuses startup if this total is absent
or smaller than the controller ceiling. This field is operator-supplied and is
not independently probed by the budget checker. Native peak allocations are
still declared claims, not measured bounds, until StageClient exposes load and
steady-state memory telemetry.

The graph worker environment needs NumPy, ONNX Runtime and ONNX (for checking
external weight references against the manifest). Vendor providers need their
own qualified environment. The controller need not install their runtimes.

## Artifact and placement gate

```json
{
  "schema_version": 1,
  "component": "model-owned-component-name",
  "files": {"graph.onnx": "<sha256>", "weights.bin": "<sha256>", "inputs.npz": "<sha256>"},
  "metadata": {
    "graph_file": "graph.onnx", "example_inputs_file": "inputs.npz",
    "checkpoint_revision": "<revision>", "precision": "<precision>",
    "layout": "<layout>", "exporter": "<version>",
    "calibration": "<record or not applicable>", "validation": "<record>"
  }
}
```

All payload paths are relative to the manifest. Hashes are checked before startup;
ONNX external weight references must also belong to that verified set. Keep bundles
immutable during execution. Checkpoint, exporter, precision and validation metadata
are model-owned: supplying a manifest does not establish model quality.

Warmup records actual provider node assignments. Unverified placement, absent
providers and undeclared CPU partitions refuse startup. The default target fraction
is 1.0. A deliberately partitioned NPU graph can lower it and explicitly allow CPU,
but reports always retain the real split. `execution_plan` records the manifest
identity, worker generation, route, reservations and placement. It labels this B
(backend execution), not complete-model P.

## Flow, cancellation and ownership

* Graph pipelines admit one request before the engine ingress queue. Extra work
  receives `ResourceUnavailable`; await consumption or cancellation and retry.
  The current API does not queue a batch of graph requests.
* Each graph client has one in-flight call and one retained result. Delivery does
  not free the slot. Public synchronous/asynchronous generators acknowledge when
  iteration advances or closes. Direct engine consumers call
  `OmniRequestOutput.release_stage_buffers()` in `finally`.
* After acknowledgement, retained result arrays belong to the application. Keeping
  arbitrarily many consumed results is application memory, outside engine admission.
* Intermediate graph handoffs copy their next-stage inputs before acknowledgement.
  Buffer descriptors carry owner, generation, exact byte size, dtype and shape.
  Framing rejects unknown required features and malformed lengths before payload allocation.
* Cancellation fences the request epoch immediately. An active non-preemptible graph
  call retires its worker. It does not replay work or restart transparently; recreate
  the pipeline to use that backend again. Completed calls can cancel without killing
  an otherwise reusable worker.
* Windows workers announce PID plus process creation time. Cross-OS termination
  verifies that identity and retires the native process tree before stopping the WSL
  launcher. An absent/unverified tree keeps its reservation quarantined. Python
  call references stay charged until the call unwinds.

Native APIs and their admission logic remain compatible. This change adds a shared
explicit reservation transaction for graph-containing plans; it does not yet replace
every native memory observation/ledger or qualify mixed-model fan-out and streaming.

## Packaging, validation and scope

Build `packages/omni-stage-contracts` with `uv build`. Its wheel imports without
torch/vLLM; the optional `wire` extra installs NumPy. The main Omni wheel bundles
the same source. Python/native implementers can use `conformance/v1.json` from
the source distribution. A native mobile implementation has not yet run these fixtures.

`examples/edge/run_graph_stage.py` drives the public API, writes a deployment profile,
records the resolved plan and raw request samples, and saves outputs for numerical
comparison. It supports explicit CPU, DirectML and VitisAI worker routes.

The implementation has tests for real runtime startup, two-stage handoff on both
orchestrator modes, public APIs, ingress/consumer backpressure, malformed framing,
placement refusal, shared-pool overcommit, stale acknowledgements, cancellation,
worker failure and artifact membership. Native Spark acceptance and small actual
Windows/WSL graph probes are recorded separately in the workspace experiment report.

Validation snapshot (2026-09-22, base `00510ce38bbb96ef2e4ce462618d06ab142fae7c`):

* WSL focused regression: 415 passed, 1 skipped, 20 deselected. Windows focused
  regression: 78 passed, 2 skipped; the controller lacks ONNX for those two tests,
  while actual graph probes use their separate worker environments.
* Spark 4B BF16 CUDA M0: 11 tests passed on each host. These runs check the
  specified text/state/cancellation workload, not every model or device.
* CPU and DirectML Add graph outputs are exact. AMD NPU versus the same A16W8
  CPU graph has maximum absolute error `4.458427429199219e-05`, normalized
  maximum `3.111236521857185e-05` against a `0.002` bound. This is backend
  validation, not original-checkpoint model-quality validation.
* Real Qwen3-TTS 0.6B two-stage streaming completes on both hosts with checkpoint
  revision `85e237c12c027371202489a0ec509ded67b5e4b5`. One measured utterance per
  host still has playback stalls; these diagnostic samples do not satisfy the
  full streaming, interruption, quality or sustained-thermal acceptance gates.
* The built Omni wheel runs 20 public-API CPU graph requests outside the checkout
  with exact outputs and zero retained reservation after shutdown. The standalone
  contracts source archive imports in a dependency-free environment.
* Fresh Galaxy S25 [inference](https://workbench.aihub.qualcomm.com/jobs/jp0mvkj6g/)
  and [NPU profiling](https://workbench.aihub.qualcomm.com/jobs/jpyo7nx75/) passed
  for the existing W8A16 attention component. Three outputs match the same
  artifact/input reference exactly; the `full1024` artifact label denotes one
  attention component's cache bucket, not the complete Spark model.

Raw records, machine/package inventories, failed attempts and reproduction commands
are preserved in the surrounding edge-infer workspace under
`analysis/experiments/omni_abstraction_20260922/`. They are not bundled as runtime
dependencies. Tests of reduced-device configurations are not measurements on a
different physical hardware SKU.

Still outside the completed v1 scope: arbitrary device/model certification, mobile
C++ control integration, persistent external AR sessions, shared-device zero-copy,
state migration/replay, mixed-model streaming/fan-out qualification, joint thermal /
bandwidth policy, native Windows CPU text qualification, and the full M1–M4 device
matrix. These remain explicit rollout gates, not implied supported features.
