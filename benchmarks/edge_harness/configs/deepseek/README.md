# DeepSeek V4.1 Q2 baseline readiness

`v41_q2_candidate.json` pins all seven Q2_K shards and their publisher LFS
hashes, plus the dedicated runtime and recipe commits, audited on 2026-10-08.
The pinned files total **264,515,279,456 bytes**. This exact metadata total
supersedes older recipe-size arithmetic for this revision; it is not a RAM
requirement or a measurement of a running process.

Sources:

- [Pinned model metadata](https://huggingface.co/api/models/vcruz305/DeepSeek-V4.1-Flash-GGUF/revision/c4a085541cb53f67ee5e57b63d255e80cef286e7?blobs=true).
- [Dedicated runtime commit](https://github.com/vcruz305/llama.cpp/tree/5210c7c5ed61dddaee6ed476623abf4b63093d16).
- [Pinned recipe](https://github.com/vcruz305/DeepSeek-V4.1-Flash-GGUF-DGX-Spark-recipe/blob/c3c548489771b964bc8b4c507509441276fd2bf9/README.md).

The recipe reports real Q2 CPU-mmap text generation on DGX Spark, but leaves
the two-level candidate mask unimplemented and does not establish corpus/task
quality or vision/MTP support. Its CUDA instructions discuss forced mmap and
also pinned-memory failures; the Laptop path requires direct verification.
That report cannot substitute for Windows or current Laptop execution.

Current local checks found neither these seven shards nor a build of the
dedicated runtime in the checked workspace. The existing `llama.cpp` checkout
is upstream commit `e71b80510c848c00175924ecf3c40333ccae8eb5`.

Required integration steps:

1. Fetch/verify the complete pinned shard set; build the pinned dedicated
   CPU runtime separately and record build flags and executable hashes.
2. Add an explicitly experimental CPU mmap route to the existing llama.cpp
   StageClient. Its mmap residency, logical reads and physical I/O are unknown
   until measured; do not reserve the full file as resident RAM or call demand
   paging a controlled expert cache.
3. Enable the runtime's explicit load mode and Engram CPU placement controls
   through validated backend options, without permitting arbitrary untracked
   command-line flags. Compare complete requests against checkpoint references.
4. Before calling a route a bounded SSD cache, integrate/probe the selected
   backend's actual GPU/RAM expert cache and I/O controls. Enforce admission
   budgets, observe physical reads, and test cancellation during read and DMA.
5. Verify Windows native separately. Long-context/task quality, omitted
   candidate-mask semantics and later vision artifacts remain specific gates.

The static manifest and shared tier-budget interfaces are implemented. No
DeepSeek full request, capacity rejection, or benchmark pass is fabricated by
this metadata. A model file larger than RAM remains an experimental execution
candidate; lack of its local runtime/weights is recorded as such.
