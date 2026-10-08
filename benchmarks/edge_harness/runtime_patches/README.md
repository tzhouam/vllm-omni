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
