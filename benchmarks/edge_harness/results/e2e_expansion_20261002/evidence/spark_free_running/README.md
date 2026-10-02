# Spark BF16 fixed-cache free-running check (2026-10-02)

**Depth: CPU numerical reference; not an S25 or hosted AI Hub execution.**
The existing fixed-cache probe was extended with `--free-running`: the source
model and the 28-layer exported decode step each feed back their own selected
token and retain their own K/V state. This checks whether the prior
teacher-forced 128-token top-1 agreement survives actual greedy continuation.
It does not establish a compiled static-shape artifact's numerical fidelity.

| Prefix and fixed full-cache capacity | Greedy token agreement | Worst aligned logit relative L2 | Worst aligned new-K/V relative L2 |
| --- | ---: | ---: | ---: |
| [500 → 628, crosses 512](cross_512_bf16_static128.json) | 128/128 | 0.5114% | 1.3387% |
| [1000 → 1128, crosses 1024](cross_1024_bf16_static128.json) | 128/128 | 4.1967% | 257.4521% |

Both runs used the same Spark-X2.5-1.7B BF16 checkpoint, eight HX370 WSL CPU
threads, Transformers 5.14.1, PyTorch 2.13.0+cpu, batch size one and one
active sequence. The source supplied prefill and token embeddings; the export
consumed its **own committed cache** after each step. The numerical errors
match the earlier teacher-forced boundary study because no token diverged on
these two synthetic prompts. In particular, the 1024-boundary state error is
far above the 1% component gate. Sequence agreement on two prompts cannot
qualify state fidelity, general language quality, or mobile generation.
Neither latency nor power was measured.

Reproduce from the repository root with the pinned local checkpoint:

```bash
export PYTHONPATH=$PWD
PY=/home/zhout/project/edge_infer/.venvs/omni-cpu/bin/python
MODEL=/home/zhout/project/edge_infer/models/Spark-X2.5-1.7B-BF16
for case in '500 628' '1000 1128'; do
  set -- $case
  "$PY" benchmarks/edge_harness/experiments/probe_spark_decode_boundary.py \
    --model "$MODEL" --prefill-tokens "$1" --context-capacity "$2" \
    --decode-steps 128 --reference-dtype bfloat16 \
    --export-arithmetic hf_bf16_reference --ordered-sliding --free-running \
    --report "/tmp/spark_free_running_${1}.json"
done
```

Next: establish a target-compatible static attention normalization/cache
layout that passes the new-K/V and output-quality gates, then compile and
measure the resident 28-layer loop on the target. The matrix coverage count
is unchanged.
