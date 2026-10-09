# Strata native I/O observation patch

`strata_omni_io_v1.patch` applies to Strata upstream revision
`d5ea7133741e67743c0e886bb426c0ce8d69cf6c`. Its SHA-256 is
`31b98e5e02e21a6289d0713097036334a59aef2d15ad9e951ded80f59f4bfdcc`.
It changes seven files to expose existing expert/PLE counters and completed
Windows direct-read counters. It adds no kernel, cache policy or scheduler.
Upstream Strata is MIT-licensed; its [copyright and permission notice](LICENSE.strata)
is retained alongside the derivative patch.
Preserve the patch bytes, including single-space blank context lines. Local
Git whitespace attributes and a narrow trailing-whitespace hook exclusion
protect this hash-bound data file from automatic rewriting.

The native protocol emits three cumulative `OMNI_IO_V1` snapshots for each
serial request: `request_start`, `prefill_end_decode_start`, `request_end`.
The original `DONE` line remains unchanged. A host adapter must bind these
snapshots to the exact owned process generation and request before deriving
interval differences. Cancellation, missing phases and I/O errors leave
observation incomplete. Independently sampled counters are not one atomic
transaction; do not impose cross-counter conservation identities.

Logical useful bytes, aligned OS direct-transfer bytes and physical disk
traffic are different measures. This patch sets physical SSD bytes to null.
Loading is outside the request snapshots. PLE keep-alive transfers remain
separate, mapped fallback is explicit, and summed reader time is not wall
latency. The patch does not enforce aggregate RAM/VRAM limits or qualify the
three-tier memory system.

Build from an isolated Strata checkout and the exact ggml dependency:

```text
git clone https://github.com/Niko1221/Strata.git strata-observed
git -C strata-observed checkout --detach d5ea7133741e67743c0e886bb426c0ce8d69cf6c
git -C strata-observed apply /absolute/path/strata_omni_io_v1.patch
git clone https://github.com/ggml-org/llama.cpp.git llama-pinned
git -C llama-pinned checkout --detach 3cf03257f219afbe7334045ff7c6a06ac68c627d
```

In a Visual Studio 2022 x64 developer environment with CUDA 13.4, CMake and
Ninja available, the measured local build used:

```text
cmake -S strata-observed -B strata-observed-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_PORTABLE=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=OFF -DSTRATA_BUILD_TESTS=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CUDA_RUNTIME_LIBRARY=Static -DSTRATA_GGML_DIR=/absolute/path/llama-pinned
cmake --build strata-observed-build --target strata --parallel 2
```

The 2026-10-08 native Windows build succeeded using NVCC 13.4.59, MSVC
19.44.35229, Windows SDK 10.0.26100.0, CMake 3.31.6 and Ninja 1.12.1.
The executable SHA-256 is
`5bd6bd3e7c5f473d5b504f92840331177d2651106d408004457d3d661efa803b`;
the private build-receipt SHA-256 is
`47047533c9c35dcb5c11189fefc76ccd5f4dd22e08cc43662ca83847d14f108d`.
The ggml tree is `d255198f04f9b8349f1dff24501d513f83cbfeda`.
CLI `--help` passed; this build record alone contains no neural validation.
The [reviewed build summary](../results/strata_20261008/native_io_build.json)
records all seven source hashes, compiler/log identities, native dependencies
and the separate 2,335-file runtime manifest.
The subsequent [separate neural run](../results/strata_20261008/native_q4_observed_io.json)
passed three Q4 text requests plus cancellation/fresh-worker recovery. It
records actual selected modules and bound I/O phases, while leaving physical
SSD traffic and three-tier memory qualification unverified.

Static CUDA runtime does not make cuBLAS static. Pin the deployed CUDA DLLs
and verify actual loaded dependencies in the separate runtime bundle. A private
Windows supervisor can remove the current directory from its children's
standard DLL search with an empty-string
[SetDllDirectory](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setdlldirectoryw)
call; this does not replace observing the actual loaded modules. Bind
the binary, patch, dependency/source hashes, build logs and configuration,
Python environment, bootstrap and adapter to a new route identity. The
original CUDA 13.0 release remains separate; its earlier neural results and
latencies do not qualify this CUDA 13.4 executable. See
[current integration status](../EDGE_ENGINE_STATUS.md) for actual runs.

## Combined CPU/CUDA execution observation

Two additional byte-pinned patches extend the same I/O build with scoped
execution counters. Apply them in order after `strata_omni_io_v1.patch`:

```text
git -C strata-observed apply /absolute/path/strata_omni_execution_v1.patch
git -C strata-observed apply /absolute/path/strata_omni_execution_boundary_v2.patch
```

The incremental patch SHA-256 is
`884ce375cfab94756834f1e97211bfbef6471674e50e24c0bfcc03a9eca96358`;
the boundary correction SHA-256 is
`4689571b0c15873dba9e232baf3834db925c1cfb6d9f5929ef70f95c773ab583`.
Use the same exact base/dependency and CMake options above. Preserve all patch
bytes; the local attributes disable patch line-ending conversion, and their
single-space context lines are protected by the three-file trailing-whitespace
exclusion.

The [original wire specification](strata_omni_execution_schema_v1.md) is
retained verbatim, SHA-256
`d7ff209a7533f85e182e2be9954805c1e08d505e253a7e116c405542ea729cfc`.
Its source-only wording describes the document's original evidence scope.
The [separate build and assembly record](../results/strata_20261008/strata_execution_build_assembly_status.json)
tracks later actual work without rewriting that document.

The patch instruments existing CPU expert methods, CPU row-partition phases,
CUDA graph replay and completion at existing successful stream fences, plus
ordinary owned expert-cache allocation/free accounting. It adds no scheduler
or inference kernel. Counter families overlap and must not be summed as
disjoint work. Coverage stays partial: whole-model placement and physical SSD
bytes remain unknown, and memory observations do not enforce aggregate caps.
Raw native records carry no trustworthy process or Omni request ownership;
the Stage must establish actual process birth, GPU, loaded modules, adapter
identity and ordered request binding before consuming them.

The 2026-10-09 native configure and compile/link commands both returned zero;
the resulting EXE SHA-256 is
`39adcb4afc80d6fdf6d9c577231258ecfac9b3c798fa1e3409b49d0e310fc177`.
The original build wrapper then failed by treating a Ninja phony dependency
as a physical file. That failed receipt is preserved; a separate closure
supplement archives actual target membership and physical inputs. A compiled
standalone fixture is only an ABI reference, pending an owned engine frame.

The complete separate runtime assembled successfully, but its first actual
static verification refused the base repository's small experimental projection
GGUF. The linked record preserves that specific failure and the next check.
No model request, Agent result, default route or release qualification is
granted to this combined runtime by compilation or assembly.
