# Measured results

Spark-X2.5-1.7B unless stated. Xeon Platinum 8480C, 24 pinned physical cores,
memory bound to the same NUMA node, W4A16 g128, `max_model_len` 2048.
Each row is a median over at least three interleaved passes against a control
arm running in the same grid.

## Memory

Every lever below was measured as its own A/B, with the alternative reachable by
a flag so both arms run identical code.

| lever | before | after | saving | cost |
|---|---|---|---|---|
| KV budget 1 GiB → 192 MiB | 3264 MB | 2421 MB | **843 MB** | fewer cached tokens |
| RoPE table → `max_model_len` | 3891 MB | 3273 MB | **618 MB** | none |
| One weight layout, not two | 4591 MB | 3976 MB | **615 MB** | ~1.63× prefill |
| Blocked repack (peak) | 4502 MB | 4013 MB | **489 MB** peak | none |

**These do not add.** The single-layout change removed `to_split_halves`, which
was doing its own whole-matrix expansion, so part of the blocked-repack saving
was already gone by the time it was measured. The report each was taken from
states its baseline; quote them individually.

### Where the memory actually is

From `memory_ledger.py`, which counts unique tensor storages by `data_ptr` (so
tied weights are counted once) and checks residency with `mincore(2)`:

| | MB | note |
|---|---|---|
| KV cache | 1024 | fully resident — 1024.1 of 1023.8 allocated |
| int4 packed weights | 686 | |
| RoPE cos/sin tables | 640 | **before the cap** — sized for 1 048 576 positions |
| embedding / head | 408 | tied; includes a 128 MB row-major copy |
| imports (`import vllm_omni`) | 501 | |

The 640 MB RoPE entry is the clearest single finding in the project. Spark
advertises a 1 048 576-token context and the model passed
`max_position_embeddings` straight to `get_rope`, so vLLM built
`torch.arange(1048576)` cos/sin tables — two of them, because the sliding and
full layers use different theta and partial-rotary factors. At `max_model_len`
2048 they need 1.25 MB between them. It had been sitting inside an
"unexplained ~1.6 GB" remainder.

### Peak vs settled

`release_after_load()` runs only once loading and warm-up return, so a device
that cannot hold the *peak* never reaches the release. Sampling `RssAnon` on a
timer through the load, rather than reading the endpoint:

| | peak | settled | transient |
|---|---|---|---|
| whole-matrix repack | 4502 MB | 3905 MB | 550 MB |
| blocked repack | 4013 MB | 3892 MB | 105 MB |

`unpack_nibbles` expands 4-bit codes to int32 — eight times the packed form —
because the kernel's packer wants int32. Whole-matrix that is 104 MiB for
`gate_up` and **1024 MiB for the tied embedding**, against 24 MiB of packed
weights for a whole layer. Chunking is also *faster* (79 ms vs 208 ms for the
embedding) because the working set stays in cache.

## Speed

| path | decode | prefill (pp512) | spread |
|---|---|---|---|
| W4A8, AMX `int4_scaled_mm_cpu` | 74.4 tok/s | — | 1.1% |
| W4A16, tinygemm | 71.6 tok/s | 2190 tok/s | 9.7% |
| W4A16, one layout | 71.6 tok/s | 1344 tok/s | 18.3% |
| W4A16, `prefill_dequant: false` | 73.0 tok/s | 453 tok/s | 0.3% |
| `cpu_gemm_wna16` fallback | 32.3 tok/s | — | 1.5% |
| llama.cpp Q4_K_M | 90.0 tok/s | 1121 tok/s | — |

`cpu_gemm_wna16` is the kernel vLLM selects when `use_w4a8` is off. It exists —
`weight_path.py` used to call tinygemm "the only option" without AMX, which was
false — but at 2.2× slower than tinygemm the routing was right anyway. Caveat:
`_get_isa_hint` returns `"amx"` whenever AMX is present regardless of the W4A8
flag, so this measured its *best* path on this host, not the `"vec"` path a
genuinely AMX-less machine takes.

## Bandwidth headroom

A node-local streaming read on the same 24 cores sustains **222 GB/s**.

| | GB/s | of ceiling |
|---|---|---|
| W4A16 GEMM only | 132 | 60% |
| W4A8 GEMM only | 153 | 69% |
| W4A16 whole step | 73 | 33% |
| llama.cpp whole step | 86 | 39% |

Neither engine is bandwidth-bound. An earlier draft asserted the GEMM ran at
"85% of what this socket delivers" and concluded the remainder was framework
overhead; at 60% of the measured ceiling, kernel and scheduling efficiency are
still on the table. See [ROADMAP.md](../../ROADMAP.md) item I10.

## Fidelity

Against each build's own bf16 output over 12 greedy prompts:

| build | exact reproductions | bits/weight |
|---|---|---|
| llama.cpp Q4_K_M | 4/12 | 4.5 (groups of 32, mixed types) |
| W4A16 | 2/12 | 4.25 |
| W4A8 | 1/12 | 4.25 |

**Not a matched-budget comparison** — Q4_K_M stores more. And greedy divergence
saturates: it discards everything after the first divergent token, so one
unlucky token scores the same as a build that is wrong everywhere.
`fidelity_kl.py` replaces it with top-1 agreement and KL over a corpus.

On the Android agent harness, every build navigates a multi-screen task
correctly and only some notice they have arrived: W4A8 turned Wi-Fi on and then
tapped the switch off again. **Navigation survives quantization; recognising
completion does not.** That is one run per cell — enough to show a failure
reproduces, not to rank success rates.
