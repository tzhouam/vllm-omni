# Environment, pins and reproducing a number

Every figure in this repository was produced with the versions below. A result
whose environment was not recorded cannot be retracted later, only doubted, so
the harness stamps these into each artifact automatically — see the
`provenance` block in any file under `benchmarks/edge_harness/results/`.

## Do we patch vLLM? Yes — at runtime. Is vLLM *modified*? No.

Those are two different questions and this document previously blurred them.

**We do patch vLLM.** `vllm_omni/patch.py` has nine `_patch_*` functions that
perform ten direct attribute replacements on vLLM classes and one on a torch
function. That is monkey-patching by any honest reading:

```python
CpuPlatform.check_and_update_config = classmethod(check_and_update_config)
CPUWorker.load_model = load_model
torch.compiler.load_compiled_function = load_compiled_function
```

**vLLM's source is not modified.** No file in the vLLM tree or in
`site-packages/vllm` is edited; the checkout is clean upstream (0 commits ahead,
0 uncommitted) and the installed wheel is stock `0.28.0+cpu`. There is no patch
series to apply and no vLLM fork to maintain — `pip install vllm==0.28.0+cpu`
gives you exactly upstream, and this package changes its behaviour at import.

### The three CPU monkey-patches

| patch | what it changes | why it is a patch and not a hook |
|---|---|---|
| `_patch_cpu_aot_compile_cache_load` | vLLM's CPU path fails to load its own AOT artifact (`finalize_loading` missing on a plain function), treats it as a miss, and recompiles on every start | replaces `torch.compiler.load_compiled_function`; torch offers no hook |
| `_patch_cpu_explicit_kv_cache_memory` | `CpuPlatform.check_and_update_config` overwrites `kv_cache_memory_bytes` from `VLLM_CPU_KVCACHE_SPACE` unconditionally, so a per-stage budget had no effect — this is what made "KV sizing is not a memory lever" look true | wraps a classmethod on a platform class |
| `_patch_cpu_release_memory_after_load` | returns freed allocator arenas to the OS after weight load | wraps `CPUWorker.load_model` and `compile_or_warm_up_model` |

They are guarded with `getattr`, made idempotent by a marker attribute, and
gated on `current_platform.is_cpu()`. They are **not** covered by any stability
contract: each is keyed to a specific method on a specific vLLM class, and an
upstream rename or behaviour change can break them or silently turn them into
no-op wrappers. Two of the three exist to work around upstream bugs and should
disappear when those are fixed.

### The genuine extension points

These are registries vLLM publishes, and they are stable across upgrades in a
way the patches above are not:

| mechanism | used for |
|---|---|
| `register_quantization_config` | the `cpu_int4` quantization method |
| `ir.ops.*.register_impl` | CPU providers for `rms_norm` / `fused_add_rms_norm` (off by default — fused is slower in situ) |
| platform plugins | the CPU `OmniPlatform` |
| `direct_register_custom_op` | `cpu_int4_linear`, `spark_head_gate` |

### What this means for you

Pin the vLLM version (below). Do not expect the monkey-patches to survive an
arbitrary upgrade — re-run `benchmarks/edge_harness/` after one, because the
failure mode of a stale patch is a silently missing optimisation, not a crash.
An earlier version of this page claimed these changes "survive a wheel upgrade";
that is true of the registrations and overstated for the patches.

## Pins

| component | pin |
|---|---|
| vLLM (installed) | `0.28.0+cpu` wheel |
| vLLM (source read for this work) | `51da0ca66c8065619c79e35dff97aa99aeaf5644` (`v0.28.1rc0`) |
| llama.cpp (baseline) | `e71b80510c848c00175924ecf3c40333ccae8eb5` (2026-09-07) |
| Python | 3.12 |
| torch | CPU build, as pulled by the vLLM CPU wheel |

```bash
pip install --torch-backend cpu \
  https://github.com/vllm-project/vllm/releases/download/v0.28.0/vllm-0.28.0+cpu-cp38-abi3-manylinux_2_34_x86_64.whl
VLLM_TARGET_DEVICE=cpu pip install -e .          # this fork
```

## Machine

Dual-socket Xeon Platinum 8480C (Sapphire Rapids, 56 cores/socket, AMX-BF16 and
AMX-INT8), 2 TB RAM, NUMA nodes 0 = CPUs 0-55/112-167 and 1 = 56-111/168-223.

It is **shared**, which shapes everything: end-to-end throughput swings ~30% run
to run, per-op self time reproduces to ~5%, `RssAnon` to 0.2%. Runs are pinned
with `numactl --physcpubind=<24 cores> --membind=<node>` and the harness refuses
to start when those cores are already busy.

## Reproducing a published number

```bash
# memory: any of the four levers, as its own A/B
python benchmarks/edge_harness/bench_harness.py --spec <arms>.json \
    --passes 3 --metric rss_anon_tree_mb --cores 56-79 --out summary.json

# where each number came from
python -c "import json,sys; print(json.load(open(sys.argv[1]))['provenance'])" \
    benchmarks/edge_harness/results/<file>.json
```

Two settings change what a memory or profile number *means*, and both have
produced a wrong published conclusion by being left at their default:
`--context-len` (without it the KV cache is allocated and never touched) and
`--prompt-len` (the default of 1 profiles an empty attention cache).

## Retracted artifacts

Files under `benchmarks/edge_harness/results/` carrying `"retracted": true` are
kept deliberately rather than deleted, with the reason and the superseding file
named. `profile_w4a8_g128_p{1,2}.json` are the clearest case: both show
`_C::onednn_mm` at ~2.97 ms/step, the signature of a checkpoint whose output
head was left in bf16, and every W4A8 number derived from them is void.
