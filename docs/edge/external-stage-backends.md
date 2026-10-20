# External stage backends: the Radeon 890M and the AMD NPU

`vllm_omni.edge.local.external` lets the local engine place one stage on a device
vLLM does not drive, prove it actually ran there, and charge its memory to the
same ledger as everything else. Status: the plumbing is done and measured
(M3.0); no model component runs on either device yet (M3.1–M3.4).

## Why these devices need another process

Neither can be opened from the engine's interpreter, and for different reasons.

The **XDNA2 NPU** is an MCDM device. WSL2's GPU-PV forwards WDDM display
adapters only, so `/dev/accel` does not exist inside WSL and no amount of
installing there changes it; VitisAI is a Windows DLL besides.

The **Radeon 890M** *is* reachable from WSL, over DirectML through `/dev/dxg` —
but `torch-directml` pins torch 2.4.1 and Omni runs 2.13.

They cannot even share a Windows venv: `onnxruntime-directml` is pinned at
1.24.4, and the VitisAI EP needs onnxruntime ≥ 1.25.

So: one protocol, four routes, each in its own interpreter.

| route | interpreter | artifact | status on this laptop |
|---|---|---|---|
| `ort-vitisai` | native Windows, plain ORT ≥ 1.25 | `onnx:a16w8` | working |
| `torch-dml` | WSL `.venvs/dml` | `pt:fp16` / `pt:fp32` | working |
| `ort-dml` | native Windows, onnxruntime-directml | `onnx:fp16` / `onnx:fp32` | venv not installed |
| `ort-cpu` | anything with ORT | any ONNX | reference arm |

`python -m vllm_omni.edge.local devices` prints which resolved; a route that did
not resolve says what is missing rather than disappearing.

## Placement is verified, never assumed

A VitisAI session lists the EP in `get_providers()` **whether or not it claimed
a single node**, and a graph it declined runs entirely on the CPU and returns
bit-identical numbers. Identical output cannot confirm NPU execution, and a
small difference cannot confirm it either. The only evidence is ORT's per-node
provider assignment, which exists only after something has run — so `open()`
always does one profiled warm-up run and the planner refuses on the node split.

An *unmeasured* placement (`fraction_on_target is None`) is refused too. It is
not rounded to "fine".

On the torch-directml route there is no partitioner, so the evidence is coarser
and equivalent: the device the output tensors came back from. A module that
fell back to the CPU returns CPU tensors.

## Memory

The 890M and the NPU report `memory_bytes == 0` on purpose. All three devices
allocate out of the CPU's RAM, so repeating the pool size on each row would let
a caller add three copies of the same memory together. Capacity comes from the
host row — the same number the CPU stage is being budgeted against.

The worker is a different process, on Windows a different OS, so the engine's
psutil sampler cannot see it. It reports its own resident set, and that is what
closes the budget-versus-measurement loop.

## Refusal codes

| code | means |
|---|---|
| `no_exported_graph` | the device runs graphs and there is no export for this stage at this precision |
| `execution_provider_declined_graph` | the session ran and left the graph on the CPU |
| `numeric_gate_failed` | it ran on the device and disagreed with the reference |
| `device_unreachable_from_this_process` | the device works; no interpreter here drives it |
| `budget_exceeds_device` | the stage does not fit the shared host pool |

All of them exit 2 from the CLI. A refusal is a result.

## Using it

```bash
python -m vllm_omni.edge.local external \
  --graph out/code2wav.onnx --format onnx:fp16 --component code2wav \
  --device gpu_integrated:amd --runs 20 --json record.json
```

`--device` requires that device, so "does the NPU run this?" is answered with a
refusal rather than quietly with the iGPU. `--prefer` falls through instead.
`--min-placement` sets how much of the graph the target EP must take.

## First real model component: Qwen3.8-27B's vision tower

`vllm_omni.edge.local.artifacts.vision_tower` exports it at a fixed image size.
It is the component that best fits the transport constraint — 27 blocks run once
per image, 4.8 MB in and 4.0 MB out — and it needs no quantization story for the
iGPU, because the FP8 checkpoint keeps all 333 visual tensors in bf16.

The export wraps `transformers`' `Qwen3_5VisionModel` (whose parameters match the
checkpoint name for name, 333/333) and only fixes its shapes; the wrapper's
output is **bitwise identical** to the model's own `forward`. At a fixed size the
bilinear indices, position ids and `cu_seqlens` are constants, so folding them
into buffers removes every integer `Gather` from the graph.

| | 890M (DirectML, fp32) | AMD NPU (A16W8) |
|---|---|---|
| placement | **85 of 113 nodes** | **0 of 3246** — declined |
| latency | **227.0 ms** device, 237.2 round trip, 11.4 transport | — |
| vs torch CPU (1363 ms) | **6.0×**, or 5.7× including transport | — |
| vs RTX 5090 (95.9 ms) | 2.4× slower | — |
| parity vs torch CPU fp32 | 9.10e-06 normalized, cosine 0.99999994 | — |

Moving the tower off the discrete GPU frees **1862 MiB** of its VRAM, which
matters for a 21–29 GB model on a 23.89 GiB card.

The NPU refusal is about **this artifact**, not the model or the precision: the
same A16W8 scheme applied to the merger subgraph alone *was* accepted (1 of 8
nodes), and cutting requantization from 3058 to 984 Q/DQ nodes did not change the
whole-graph answer. The next step is a bounded bisect of what the partitioner
will take, not a broader claim.

One number worth keeping straight when comparing devices: the 890M's fp32 output
is **closer** to the CPU reference (9.10e-06) than the RTX 5090's is (5.65e-04),
because CUDA runs matmul in TF32 by default. That is a statement about what each
device actually computed, not about which is more accurate.

## Measured, 2026-09-15, this laptop

Transport, WSL ↔ native Windows: **11 µs** half round trip, **330 MB/s**.

| | NPU (A16W8 probe graph) | 890M (100.7 M-param torch.export graph) |
|---|---|---|
| device compute | 0.855 ms | 9.749 ms |
| same graph on CPU | 6.270 ms | 49.661 ms |
| transport | **2.797 ms** (cross-OS) | 0.523 ms (same OS) |
| placement | vitisai 1 / CPU 2 nodes | outputs on `privateuseone` |
| numerics vs CPU | 3.03e-05 normalized | — |

**The transport column is the design constraint.** On the NPU it consumes more
than half of a 5.4 ms compute saving, because that graph moves 1 MiB in and
1 MiB out. An external stage has to be compute-heavy and I/O-light: a vision
tower that runs once per image qualifies, a decode step that runs once per token
does not. This is the measured basis for the proposal's "coarse stages only".

The 890M's 4.8× includes transport, and deserves a caveat: the iGPU hit
1.32 TFLOP/s, matching its recorded 1.23 fp32 rate, while the CPU path reached
only 0.26 TFLOP/s against a 1.54 peak. It is as much a slow CPU path at that
shape as a fast iGPU. Separately measured this session: the 890M does
**2.85 TFLOP/s in fp16**, 2.3× the fp32-only figure the earlier notes carried.

## Exporting a graph these backends will accept

Run `sanitize_onnx` on every dynamo export. It is not a nicety.

torch's dynamo exporter emits `Reshape` with `allowzero=1`. onnxruntime's
DirectML provider rejects that at **session initialisation** with a bare
`E_INVALIDARG` out of `MLOperatorAuthorImpl.cpp` — and onnxruntime then does not
raise. It prints `Falling back to ['CPUExecutionProvider'] and retrying` and
hands back a working CPU session. Measured here: the 27B vision tower placed
**0 of 1171 nodes** on the iGPU and computed correct features entirely on the
CPU. With 58 `allowzero` attributes cleared, the same graph placed **85 of 113**.

`vllm_omni.edge.qnn_export.sanitize_onnx` was written for Qualcomm's converter
and fixes exactly this, along with `IsNaN`/`Where` guard pairs and rank-0 `axes`
initializers — all generic dynamo warts rather than vendor-specific ones. It
preserves external data, so a multi-gigabyte export stays a pair of files.

`vllm_omni.edge.local.artifacts.vision_tower.export_onnx` always calls it.

Two independent guards catch the fallback if a graph slips through: the worker
compares `session.get_providers()` against what was requested and reports the
swap by name, and the node-assignment gate refuses on the placement itself.

## Known limits

- **A16W8 is the only door on XDNA2.** A8W8 — what `quantize_static` produces by
  default — and fp32 are declined with no error and every node left on the CPU.
- **Dequantize nodes stay on the CPU** by the partitioner's own account, so a
  graph that is mostly requantization will not benefit.
- **TorchScript cannot be placed on DirectML** (torch 2.4.1 / torch-directml
  0.2.5): `map_location` raises `Cannot access storage of OpaqueTensorImpl`, and
  load-then-`.to(device)` trips `assert isinstance(param, Parameter)` inside
  `Module._apply`. Use `torch.export` `.pt2`, which is what
  `vllm_omni.edge.decoder_export` already emits — and do not call `.eval()` on
  the loaded module, which 2.4.1 does not support.
- Small graphs cannot amortise dispatch; the NPU's advantage grows with GEMM
  size (2.6× at 256×1024×1024, 5.3× at 1024×2048×2048).

Raw records: [`analysis/experiments/m3_amd_devices_20260915/`](../../../analysis/experiments/m3_amd_devices_20260915/README.md)
