# Baselines

Measurements of other engines and of unquantized references, kept so the
comparisons in [docs/edge/comparisons.md](../docs/edge/comparisons.md) can be
checked rather than taken on trust.

## `llama_cpp/`

llama.cpp is the reference for CPU inference of a quantized model. Pinned
commit and build flags are in
[docs/edge/environment.md](../docs/edge/environment.md); no llama.cpp source is
redistributed here, only output produced with it.

| file | what |
|---|---|
| `llamacpp_q4km_t{4,8,16,32}.json` | Q4_K_M across thread counts |
| `llamacpp_grid_t*.json`, `llamacpp_node1_t*.json` | thread and NUMA-node sweeps |
| `ab_llamacpp_*.json` | interleaved A/B against vLLM on the same cores |
| `il_llamacpp_p{1,2,3}.json` | the interleaved passes behind the headline decode figure |
| `cmp_llamacpp.py` | the comparison driver |

Both engines are pinned with `taskset` to the same physical cores and run
strictly one at a time; vLLM's prefix caching is disabled, because `llama-bench`
has no such cache and leaving it on would not be a comparison.

## Quality references

`quality_*.json` and `quality_cmp_*.json` hold the greedy-divergence comparison:
each build against its **own** bf16 output over 12 prompts, plus the
cross-engine roll-up. Read the caveats in
[docs/edge/comparisons.md](../docs/edge/comparisons.md) before quoting these —
the bit budgets are not matched, and the metric saturates.
