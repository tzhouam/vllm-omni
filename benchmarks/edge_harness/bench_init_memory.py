"""Init time and resident memory, which decide whether a model can be served
from an edge box at all.

Three quantities, because the plan's items are defined in terms of them and
the earlier version of this file measured none of them:

* **ready_s -- process start to serving.** The old ``init_s`` started its timer
  *after* ``import vllm_omni``, so the 14.4 s of import cost the plan wants to
  cut was invisible to the benchmark that was supposed to prove the cut. The
  clock now starts at process creation, read from ``/proc/self/stat``, so
  interpreter start-up and imports are inside it. ``init_s`` is still reported,
  unchanged in meaning, so older rows stay comparable.
* **peak_rss_anon_mb -- the high-water mark, sampled.** What an edge device has
  to survive is the *peak*, not the steady state: this model loads at 7338 MB
  and settles at 3510 MB only after the allocator is asked for its arenas back,
  and a device that cannot hold the peak never reaches the release. There is no
  kernel high-water mark for RssAnon (``VmHWM`` and ``ru_maxrss`` both track
  VmRSS, which counts the mmap'd checkpoint and moved 1-2 GB between identical
  runs), so a sampler thread takes it.
* **rss_anon_at_context_mb -- memory once the cache is actually populated.**
  The old default generated 8 tokens, so the KV cache was allocated and never
  touched; "KV sizing saves only 188 MB" was measured there. ``--context-len``
  fills it.

And one correction that changes what the older rows mean. Everything here used
to read ``/proc/self/status``, i.e. **this process only**. That is the right
number only when the model is in-process, which happens when
``VLLM_ENABLE_V1_MULTIPROCESSING=0`` keeps the executor at ``uni``
(``vllm/platforms/cpu.py:273`` promotes it to ``mp`` otherwise). Every earlier
run set that variable, so the published figures are in-process figures. Run the
same benchmark at vLLM's defaults and the weights land in a worker process the
benchmark never looked at: 796 MB "resident" for a model whose weights alone
are 1.4 GB. Resident memory is now summed over the **process tree**, which is
what an edge device actually has to hold, and both numbers are reported so the
older rows remain comparable.

Every row carries provenance, so it can be invalidated later instead of merely
doubted.
"""
import time

_T_IMPORT = time.time()

import argparse, json, os, resource, sys, threading  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_harness import provenance  # noqa: E402


def _rss(field: str, pid: str | int = "self") -> float | None:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith(field):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        return None
    return None


def _descendants(root: int) -> list[int]:
    """Every process under ``root``. vLLM's default CPU path runs the model in
    a worker process, so a self-only reading misses the weights entirely."""
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            ppid = int(fields[1])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(entry))
    out, stack = [], [root]
    while stack:
        pid = stack.pop()
        for child in children.get(pid, []):
            out.append(child)
            stack.append(child)
    return out


def _tree_anon_mb(pids: list[int]) -> float:
    return round(sum(_rss("RssAnon:", p) or 0.0 for p in pids), 1)


def _process_start_epoch() -> float | None:
    """When this process was created, so imports are inside the measurement."""
    try:
        with open("/proc/self/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        starttime_ticks = int(fields[19])          # /proc/self/stat field 22
        hz = os.sysconf("SC_CLK_TCK")
        with open("/proc/stat") as f:
            for line in f:
                if line.startswith("btime"):
                    return int(line.split()[1]) + starttime_ticks / hz
    except Exception:
        return None
    return None


class AnonSampler(threading.Thread):
    """Peak RssAnon has no kernel counter, so poll for it -- across the whole
    process tree, because the model may not be in this process."""

    def __init__(self, interval: float = 0.02, rescan_every: float = 0.5):
        super().__init__(daemon=True)
        self.interval, self.rescan_every = interval, rescan_every
        self.peak = self.peak_self = 0.0
        self.marks: list[dict] = []
        self.pids = [os.getpid()]
        self.max_procs = 1
        # not `_stop`: threading.Thread already defines a private _stop().
        self._done = threading.Event()

    def _refresh(self) -> None:
        me = os.getpid()
        self.pids = [me] + _descendants(me)
        self.max_procs = max(self.max_procs, len(self.pids))

    def _sample(self) -> tuple[float, float]:
        mine = _rss("RssAnon:") or 0.0
        tree = _tree_anon_mb(self.pids)
        self.peak_self = max(self.peak_self, mine)
        self.peak = max(self.peak, tree)
        return mine, tree

    def run(self) -> None:
        last_scan = 0.0
        while not self._done.is_set():
            now = time.monotonic()
            if now - last_scan >= self.rescan_every:
                self._refresh()
                last_scan = now
            self._sample()
            self._done.wait(self.interval)

    def mark(self, name: str) -> float | None:
        """Record a phase boundary with the peak reached so far."""
        self._refresh()
        mine, tree = self._sample()
        self.marks.append({"phase": name, "anon_mb": mine,
                           "tree_anon_mb": tree, "n_procs": len(self.pids),
                           "peak_tree_so_far_mb": round(self.peak, 1)})
        return mine

    def stop(self) -> None:
        self._done.set()
        self.join(timeout=2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--kv-cache-memory-bytes", type=int, default=None,
                    help="exact KV budget; VLLM_CPU_KVCACHE_SPACE is integer GiB "
                         "so it cannot express an edge-sized cache")
    ap.add_argument("--max-num-batched-tokens", type=int, default=None,
                    help="vLLM sizes its warm-up forward from this; an edge "
                         "deployment serving one short stream does not need "
                         "the server default")
    ap.add_argument("--trim", action="store_true",
                    help="return freed allocator arenas to the OS after init")
    ap.add_argument("--generate", type=int, default=8,
                    help="tokens to generate, to prove the engine really works")
    ap.add_argument("--context-len", type=int, default=0,
                    help="after init, run a request with this many prompt "
                         "tokens so the KV cache is populated, and report "
                         "resident memory there. 0 skips it -- and a memory "
                         "number taken without it says nothing about cache "
                         "sizing.")
    ap.add_argument("--sample-interval", type=float, default=0.02)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    proc_start = _process_start_epoch()
    sampler = AnonSampler(a.sample_interval)
    sampler.start()

    anon_start = sampler.mark("before_import")
    import vllm_omni  # noqa: F401
    from vllm import LLM, SamplingParams

    anon_after_import = sampler.mark("after_import")
    import_s = time.time() - _T_IMPORT

    t0 = time.perf_counter()
    extra = {}
    if a.kv_cache_memory_bytes:
        extra["kv_cache_memory_bytes"] = a.kv_cache_memory_bytes
    if a.max_num_batched_tokens:
        extra["max_num_batched_tokens"] = a.max_num_batched_tokens
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len,
              max_num_seqs=1, quantization=a.quantization,
              enable_prefix_caching=False, trust_remote_code=False,
              disable_log_stats=True, **extra)
    init_s = time.perf_counter() - t0
    ready_s = (time.time() - proc_start) if proc_start else None

    anon_after_engine = sampler.mark("after_engine")
    peak_during_load = round(sampler.peak, 1)

    trimmed = None
    if a.trim:
        # The GPTQ repack unpacks 4-bit weights to int32 (4 bytes per weight)
        # and throws the intermediates away, but a caching allocator keeps the
        # arenas, so they stay resident. Ask for them back.
        import ctypes

        released = []
        for lib, sym in ((None, "malloc_trim"),
                         ("libtcmalloc_minimal.so.4",
                          "MallocExtension_ReleaseFreeMemory")):
            try:
                handle = ctypes.CDLL(lib) if lib else ctypes.CDLL("libc.so.6")
                fn = getattr(handle, sym)
                fn(0) if sym == "malloc_trim" else fn()
                released.append(sym)
            except Exception:
                pass
        trimmed = released
    anon_after_trim = sampler.mark("after_trim")

    out = llm.generate([{"prompt_token_ids": [10]}],
                       SamplingParams(temperature=0.0, max_tokens=a.generate,
                                      ignore_eos=True), use_tqdm=False)
    n_out = len(out[0].outputs[0].token_ids)
    sampler.mark("after_short_generate")

    anon_at_context = context_tokens = None
    if a.context_len:
        # Fill the cache. Without this the cache is allocated and never
        # written, so resident memory says nothing about how it was sized.
        context_tokens = min(a.context_len, a.max_model_len - 16)
        llm.generate([{"prompt_token_ids": [10] * context_tokens}],
                     SamplingParams(temperature=0.0, max_tokens=16,
                                    ignore_eos=True), use_tqdm=False)
        anon_at_context = sampler.mark("after_context_generate")

    sampler.stop()

    rec = {
        "tag": a.tag, "model": a.model,
        "ready_s": round(ready_s, 2) if ready_s else None,
        "import_s": round(import_s, 2),
        "init_s": round(init_s, 2),
        # VmRSS counts the mmap'd checkpoint's file pages, which come and go
        # with system page-cache pressure: identical configurations measured
        # 1.2-2.0 GB apart. RssAnon is the allocated memory -- weights copied
        # into tensors, the KV cache, activations -- and is what a memory
        # budget is actually about.
        # self-only, for comparability with rows taken before the tree fix
        "rss_anon_mb": _rss("RssAnon:"), "rss_file_mb": _rss("RssFile:"),
        # what the device actually has to hold
        "rss_anon_tree_mb": _tree_anon_mb([os.getpid()] + _descendants(os.getpid())),
        "peak_rss_anon_mb": round(sampler.peak, 1),
        "peak_rss_anon_self_mb": round(sampler.peak_self, 1),
        "peak_rss_anon_during_load_mb": peak_during_load,
        "n_procs_max": sampler.max_procs,
        "anon_before_import_mb": anon_start,
        "anon_after_import_mb": anon_after_import,
        "anon_after_engine_mb": anon_after_engine,
        "anon_after_trim_mb": anon_after_trim,
        "rss_anon_at_context_mb": anon_at_context,
        "context_tokens": context_tokens,
        "phase_marks": sampler.marks,
        "trim_calls": trimmed,
        "rss_mb": _rss("VmRSS:"), "rss_peak_mb": _rss("VmHWM:"),
        "maxrss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "kvcache_space_gib": os.environ.get("VLLM_CPU_KVCACHE_SPACE", ""),
        "kv_cache_memory_bytes": a.kv_cache_memory_bytes,
        "max_num_batched_tokens": a.max_num_batched_tokens,
        "cache_root": os.environ.get("VLLM_CACHE_ROOT", "(default)"),
        "aot_shim": os.environ.get("VLLM_OMNI_AOT_CACHE_SHIM", "1"),
        "max_model_len": a.max_model_len,
        "tokens_generated": n_out,
        "provenance": provenance({
            "quantization": a.quantization,
            "sample_interval_s": a.sample_interval,
            # uni vs mp decides whether the model is in this process at all
            "v1_multiprocessing": os.environ.get(
                "VLLM_ENABLE_V1_MULTIPROCESSING", "1"),
            "memory_scope": "process tree",
        }),
    }
    print(json.dumps(rec, indent=2))
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
