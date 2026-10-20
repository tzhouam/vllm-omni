# How to measure on a shared machine without fooling yourself

Five conclusions in this project were retracted. None was a coding error; every
one was a measurement error that looked like a result. This page is what was
left after each of them, and it is the part of this repository most likely to be
useful somewhere else.

## The five

**1. `VmRSS` counts pages you do not own.** A "3.1 GB saving" was the page cache
holding the mmap'd checkpoint. `VmRSS` moved 1–2 GB between *identical* runs;
`RssAnon` repeats to 0.2%. Measure allocation, not residency-of-everything.

**2. The build under test was not the build you think.** Every W4A8 timing was
taken on a checkpoint with `tie_word_embeddings: true` and no `lm_head` tensors,
so the output projection ran in bf16 and cost 2.97 ms/step by itself — more than
the entire 4-bit saving being measured. Artifacts now carry provenance: model
path, quantization, thread set, build SHAs. Two saved profiles in this tree were
indistinguishable from each other before that.

**3. You measured the wrong process.** The memory benchmark read
`/proc/self/status`, which is the whole story only when the model is in-process.
vLLM's CPU default promotes the executor to `mp`, putting the weights in a
worker: the benchmark reported 796 MB resident for a model whose weights alone
are 1.4 GB. Memory is summed over the process tree now.

**4. The knob was not connected.** "KV cache sizing saves only 188 MB, the pages
are never touched" sat on a *closed, do not retry* list. The comparison ran
through `CpuPlatform.check_and_update_config`, which overwrote
`kv_cache_memory_bytes` from an environment variable unconditionally — so both
arms ran the same cache. With the setting honoured the saving is **843 MB**, and
`mincore(2)` shows the cache fully in core rather than untouched. *A lever tested
through a path that silently ignores it measures nothing, and "no effect" is the
most dangerous result to accept without first checking the knob was connected.*

**5. One surviving pass is not a result.** A 1.9× prefill cost was quoted from
the single pass of a grid whose other passes had crashed — a number the harness
had already labelled `single pass — not a result`. With every pass completing it
is 1.63×, and only good to ±0.2× at that.

## What the harness enforces

- **Rotate arm order every pass.** Interleaving alone leaves arm 0 permanently
  first. `run_grid` rotates; a test pins it, because the previous test asserted
  the *un-rotated* order and that is how the bug survived.
- **Know which direction is better.** `best` is the maximum for a rate and the
  minimum for memory or latency. Taking the maximum for everything published the
  *worst* memory pass as the headline.
- **Refuse to rank inside the noise.** When the top two arms differ by less than
  their own spread, the summary prints `NOT A RANKING` and points at the per-op
  profiler. Per-op self time reproduces to ~5% here; end-to-end throughput swings
  ~30%, which is why throughput is explicitly ungateable in the regression check.
- **Guard the cores, not the host.** Load average is a whole-host number and this
  machine has 224 cores, so it says nothing about the 24 a run is pinned to. The
  pre-flight samples per-core utilisation over two windows, takes the minimum
  (so a blip is not load), and refuses on either >5% aggregate stolen capacity or
  any core sustained above 50% — one slow core gates every OpenMP barrier, so it
  cannot be averaged away.
- **Record failures.** Three passes of one grid died with
  `KeyError: 'weight_halves'`; the harness recorded the exit status and stderr
  instead of dropping the runs, which is the only reason the bug was found.
- **Stamp provenance.** Build SHAs, host, thread affinity, environment. A number
  whose origin was not recorded cannot be retracted later, only doubted.

## Two traps specific to this stack

**Compile-cache collisions.** vLLM keys its AOT compile cache on the *config*, so
two arms differing only by an environment variable share one cached artifact. If
that variable changes the traced graph, later passes load the wrong one. Give
each arm its own `VLLM_CACHE_ROOT`, and prefer expressing a graph-affecting
choice as a config field — `ModelConfig.compute_hash` hashes those.

**Allocated is not resident.** An untouched buffer costs address space. Use
`mincore(2)` before attributing it to memory: `memory_ledger.py` does, and that
is what proved the KV cache was real rather than reserved.

## The rule that generalises

Attribute, do not subtract. The "unexplained ~1.6 GB" in an earlier report was a
remainder computed from two uncertain terms, and it hid a 640 MB RoPE table in
plain sight. Counting unique tensor storages by `data_ptr`, naming the largest,
and checking residency per page turned "unexplained" into a one-line fix.
