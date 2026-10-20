"""Where the resident memory actually is, by attribution rather than subtraction.

The report's breakdown was arithmetic: weights, imports, and "unexplained
~1.6 GB" left over after subtracting the first two from the total. Three
problems with that. It assigned a bf16 embedding to a checkpoint whose config
says ``quantize_embedding: true``; it left the prefill duplicate out of the
weight line entirely; and a remainder computed from two wrong terms is not a
measurement of anything.

So this counts instead:

* **Unique tensor storages.** Walking ``named_parameters`` double-counts tied
  weights -- Spark ties its embedding to its head, and the quantized path ties
  five tensors at once -- so storages are keyed by ``data_ptr`` and counted
  once, and the tied ones are reported separately rather than silently
  dropped.
* **The KV cache**, found by attribute rather than assumed from a formula --
  and split into *allocated* and *resident*. A cache that has been allocated
  but never written costs address space, not memory, so counting its size as
  attributed memory would hide the remainder rather than explain it.
  Residency comes from ``mincore(2)``, which reports per-page whether a
  mapping is actually in core.
* **Every mapping in /proc/self/smaps**, bucketed into heap, anonymous, file
  and the rest. ``smaps_rollup`` cannot do this: it is an aggregate, which is
  the whole thing this is trying to break apart.

What it cannot do is attribute native allocations that no longer belong to a
live tensor -- freed allocator arenas, inductor's generated code, transient
buffers the caching allocator is holding. Those land in "unattributed anon",
and the point of the exercise is to see how big that really is once the
attributable part is counted properly.
"""

from __future__ import annotations

import argparse, collections, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_harness import provenance  # noqa: E402


def resident_mb(ptr: int, nbytes: int) -> float:
    """How much of a mapping is actually in core, via mincore(2).

    Allocated is not resident: an untouched KV cache is address space. Without
    this the ledger would "explain" memory the process never used.
    """
    import ctypes

    if nbytes <= 0:
        return 0.0
    page = os.sysconf("SC_PAGE_SIZE")
    start = ptr - (ptr % page)                 # mincore needs a page boundary
    length = nbytes + (ptr - start)
    pages = (length + page - 1) // page
    vec = (ctypes.c_ubyte * pages)()
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.mincore(ctypes.c_void_p(start), ctypes.c_size_t(length), vec) != 0:
        return -1.0                            # unknown, rather than a guess
    return round(sum(v & 1 for v in vec) * page / 2**20, 1)


def _rss(field: str) -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field):
                return round(int(line.split()[1]) / 1024, 1)
    return 0.0


def smaps_breakdown() -> dict[str, float]:
    """Anonymous resident bytes per mapping kind. Needs smaps, not rollup."""
    buckets: collections.Counter = collections.Counter()
    name = "[anon]"
    try:
        with open("/proc/self/smaps") as f:
            for line in f:
                if not line[:1].isspace() and "-" in line.split()[0]:
                    parts = line.split()
                    path = parts[5] if len(parts) > 5 else "[anon]"
                    if path.startswith("["):
                        name = path
                    elif path.endswith(".so") or ".so." in path:
                        name = "shared libs"
                    elif path:
                        name = "file-backed"
                    else:
                        name = "[anon]"
                elif line.startswith("Anonymous:"):
                    buckets[name] += int(line.split()[1])
    except OSError:
        return {}
    return {k: round(v / 1024, 1) for k, v in buckets.most_common()}


def storage_ledger(model) -> dict:
    """Unique tensor storages, by role, counting each byte once."""
    import torch

    seen: dict[int, int] = {}
    biggest: list = []
    by_role: collections.Counter = collections.Counter()
    counts: collections.Counter = collections.Counter()
    tied_bytes = 0

    def role_of(name: str) -> str:
        low = name.lower()
        if "embed" in low or "lm_head" in low:
            return "embedding / head"
        for key in ("int4pack", "halves", "scales_and_zeros", "group_scale",
                    "group_zero", "weight_scale", "weight_zero", "weight_packed"):
            if key in low:
                return f"quant:{key}"
        if "weight" in low:
            return "other weights"
        return "other"

    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
            continue
        try:
            ptr = tensor.untyped_storage().data_ptr()
            nbytes = tensor.untyped_storage().nbytes()
        except Exception:
            ptr, nbytes = tensor.data_ptr(), tensor.numel() * tensor.element_size()
        if ptr in seen:
            tied_bytes += nbytes          # tied / shared: counted once already
            continue
        seen[ptr] = nbytes
        by_role[role_of(name)] += nbytes
        counts[role_of(name)] += 1
        if nbytes >= 16 * 2**20:
            biggest.append((name, nbytes, resident_mb(ptr, nbytes),
                            tuple(tensor.shape), tensor.dtype))

    biggest.sort(key=lambda t: -t[1])
    return {
        "unique_storage_mb": round(sum(seen.values()) / 2**20, 1),
        "tied_duplicate_mb": round(tied_bytes / 2**20, 1),
        "by_role_mb": {k: round(v / 2**20, 1) for k, v in by_role.most_common()},
        "tensors_by_role": dict(counts),
        # "other" is where an unclassified tensor hides, so name the largest
        "largest_tensors": [
            {"name": n, "mb": round(b / 2**20, 1), "resident_mb": r,
             "shape": list(sh), "dtype": str(dt)}
            for n, b, r, sh, dt in biggest[:15]
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--tag", default="ledger")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    anon_before = _rss("RssAnon:")
    import vllm_omni  # noqa: F401
    import torch
    from vllm import LLM

    anon_after_import = _rss("RssAnon:")
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len,
              max_num_seqs=1, quantization=a.quantization,
              enable_prefix_caching=False, trust_remote_code=False,
              disable_log_stats=True)

    # The model only exists in this process when the executor stayed uniproc;
    # say so rather than reporting a ledger of nothing.
    try:
        engine = llm.llm_engine
        runner = engine.engine_core.engine_core.model_executor.driver_worker.model_runner
        model = runner.model
    except AttributeError as exc:
        raise SystemExit(
            f"cannot reach the model in this process ({exc}). Set "
            f"VLLM_ENABLE_V1_MULTIPROCESSING=0 so the executor stays uniproc; "
            f"otherwise the weights are in a worker and there is nothing here "
            f"to attribute."
        )

    ledger = storage_ledger(model)

    kv_mb = kv_resident = 0.0
    try:
        caches = runner.kv_caches
        seen = set()
        for c in caches:
            if isinstance(c, torch.Tensor) and c.data_ptr() not in seen:
                seen.add(c.data_ptr())
                nbytes = c.numel() * c.element_size()
                kv_mb += nbytes / 2**20
                r = resident_mb(c.data_ptr(), nbytes)
                kv_resident += max(r, 0.0)
    except Exception:
        kv_mb = kv_resident = -1.0

    anon_total = _rss("RssAnon:")
    # Only resident KV counts: allocated-but-untouched cache is address space.
    attributed = (ledger["unique_storage_mb"] + max(kv_resident, 0.0)
                  + anon_after_import)
    rec = {
        "tag": a.tag, "model": a.model, "quantization": a.quantization,
        "anon_before_import_mb": anon_before,
        "anon_after_import_mb": anon_after_import,
        "anon_total_mb": anon_total,
        "kv_cache_allocated_mb": round(kv_mb, 1),
        "kv_cache_resident_mb": round(kv_resident, 1),
        **ledger,
        "attributed_mb": round(attributed, 1),
        "unattributed_anon_mb": round(anon_total - attributed, 1),
        "smaps_anon_mb": smaps_breakdown(),
        "provenance": provenance({"scope": "in-process ledger"}),
    }
    print(json.dumps({k: v for k, v in rec.items() if k != "provenance"}, indent=2))
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=2)


if __name__ == "__main__":
    main()
