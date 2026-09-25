# Spark-X2.5 fixed-cache decode reference across 512/1024

**Depth: CPU numerical reference, not S25 device execution or complete mobile
generation.** The existing `SparkDecodeStep` contains all 28 real decoder
layers. A new `SparkDecodeCache` owns fixed-size per-layer K/V buffers: 21
sliding layers use 511 previous entries in a ring, seven full layers use a
bounded context buffer. It constructs masks and absolute rotary positions,
validates a complete graph result before committing K/V, rejects a repeated
position, and erases state on retirement. This is a PyTorch reference contract
for a future device controller, not a replacement for Omni scheduling or a
QNN runtime.

The [probe](../../../../experiments/probe_spark_decode_boundary.py) loaded the
complete `XHToken/Spark-X2.5-1.7B` checkpoint at revision
`14d6e83c13c7add2b62a7c39b2131f4ed1cddcf8` (the prior
[weight manifest](../../../e2e_expansion_20260924/evidence/spark_bf16_wsl_cpu/weight_manifest.json)
pins both shards). The checkpoint index SHA-256 is
`cc2b212985f5d0469bf926903e4bd7ee81856687baa0c6c6b55ff200d4cdc63f`;
`config.json` is `16aa5d3619d0d893e7157f2499675430844eb7b383c80ecc1b3e8e8928a75d52`.
On an AMD Ryzen AI 9 HX 370 under WSL2 Ubuntu, PyTorch 2.13.0+cpu and
Transformers 5.14.1, the Hugging Face model prefills a synthetic chat-token
prefix. The reference and export then receive the same subsequent token at
each step, while **the exported step consumes its own committed cache**.
The probe uses an explicit compatibility shim for this checkpoint's
Transformers 4.57 mask API; it does not patch model files or alter weights.
All runs used eight CPU threads. No device acceleration, power state or
thermal condition was measured. The three final long-prefix JSON reports
retain token-level results, source hashes, prompt-ID hashes, cache shapes and
software/platform versions. The earlier short-prefix report retains its
token-level mismatch but predates the provenance fields and final validation
checks.

| Reference versus FP32 export | Prompt tokens | Decode steps | Next-token matches | Worst logits relative L2 |
|---|---:|---:|---:|---:|
| [FP32, crosses 512](cross_512_fp32_128.json) | 500 | 128 | 128/128 | 0.001114% |
| [FP32, crosses 1024](cross_1024_fp32_128.json) | 1000 | 128 | 128/128 | 0.011495% |
| [BF16, crosses 1024](cross_1024_bf16_128.json) | 1000 | 128 | 128/128 | 14.287% |
| [BF16, short prefix](short_128.json) | 28 | 128 | 127/128 | 0.968% |

The 1000-token FP32 and BF16 reports have identical prompt-ID hashes.
The BF16 comparison tests the source checkpoint's usual arithmetic against
the current FP32 export; its large worst-case logit difference and the
short-prefix token mismatch at absolute position 73 **block a BF16 fidelity
claim for that FP32 export**, even though the long synthetic prompt's top
token agreed throughout.
The FP32 control isolates the cache/graph semantics from that mixed-precision
difference. No broad language-quality tolerance has been set.

## BF16 arithmetic and static-shape follow-up

An opt-in `hf_bf16_reference` mode now reproduces the checkpoint's FP32
residuals, BF16 projections/attention/MLP/head, FP32 softmax, and expanded
grouped K/V heads. It is **rejected by `export_onnx`** because no mobile BF16
artifact for this mode has been compiled or qualified. The same real weights
and Hugging Face CPU prefill were used in these follow-up runs:

| CPU decode boundary | Prompt tokens | Next-token matches | Worst logits relative L2 | New K/V parity |
|---|---:|---:|---:|---|
| [BF16 reference, filled sliding/full inputs](short_bf16_compact_128.json) | 28 | 128/128 | 0, bitwise | all 128 steps bitwise |
| [BF16 reference, ordered ring + filled full inputs](cross_512_bf16_faithful_128.json) | 500 | 128/128 | 0, bitwise | all 128 steps bitwise |
| [BF16 reference, ordered ring + filled full inputs](cross_1024_bf16_faithful_128.json) | 1000 | 128/128 | 0, bitwise | all 128 steps bitwise |
| [BF16 reference, fixed padded buffers](short_bf16_static_final_128.json) | 28 | 128/128 | 0.562% | first difference at position 41 |
| [BF16 reference, fixed padded buffers](cross_512_bf16_static_final_128.json) | 500 | 128/128 | 9.857% | first difference at position 512 |

The ordered-ring/filled-full 500-token case crosses the 512 sliding-window
boundary; the 1000-token case crosses the 1024 full-attention length. Both
compare every generated next-token logit vector and newly produced K/V tensor
against the source while the export consumes its **own** committed state.
The three bitwise runs use variable filled lengths for the full-attention
input; the short run also uses a variable filled sliding length. They are
CPU mathematical references, **not fixed-shape QNN executions**.

The retained diagnostic records isolate two shape-sensitive effects. Reading
the ring in [physical order](cross_512_bf16_static_final_128.json) changes
BF16 attention reduction order when it wraps; a chronological read postpones
the first difference and reduces the prior worst error in a
[separate run](cross_512_bf16_ordered_128.json). On the short prefix,
[compacting only full-attention inputs](short_bf16_compact_full_20.json)
restored bitwise parity for 20 steps, while
[compacting only sliding inputs](short_bf16_compact_sliding_20.json) did not.
The later expanded-head BF16 reference removed a separate 1025-key matrix
shape difference seen in the [earlier 1024-crossing diagnostic](cross_1024_bf16_ordered_fullcompact_128.json).
Exact filled-length parity therefore
does not authorize reusing an oversized padded static graph at every length.
The next artifact must specify shape buckets and attention/KV ordering, then
pass token, logit and K/V gates on its **actual compiled backend**.

Reproduce from the repository root with the local checkpoint and CPU venv:

```bash
export PYTHONPATH=/home/zhout/project/edge_infer/vllm-omni-edge
PY=/home/zhout/project/edge_infer/.venvs/omni-cpu/bin/python
MODEL=/home/zhout/project/edge_infer/models/Spark-X2.5-1.7B-BF16
$PY benchmarks/edge_harness/experiments/probe_spark_decode_boundary.py \
  --model "$MODEL" --prefill-tokens 500 --decode-steps 128 \
  --reference-dtype float32 --report /tmp/spark_cross_512_fp32.json
$PY benchmarks/edge_harness/experiments/probe_spark_decode_boundary.py \
  --model "$MODEL" --prefill-tokens 1000 --decode-steps 128 \
  --reference-dtype float32 --report /tmp/spark_cross_1024_fp32.json
$PY benchmarks/edge_harness/experiments/probe_spark_decode_boundary.py \
  --model "$MODEL" --prefill-tokens 1000 --decode-steps 128 \
  --reference-dtype bfloat16 --report /tmp/spark_cross_1024_bf16.json
$PY benchmarks/edge_harness/experiments/probe_spark_decode_boundary.py \
  --model "$MODEL" --prefill-tokens 500 --decode-steps 128 \
  --reference-dtype bfloat16 --export-arithmetic hf_bf16_reference \
  --compact-inputs --compact-layer-type full --ordered-sliding \
  --report /tmp/spark_cross_512_bf16_reference.json
$PY benchmarks/edge_harness/experiments/probe_spark_decode_boundary.py \
  --model "$MODEL" --prefill-tokens 1000 --decode-steps 128 \
  --reference-dtype bfloat16 --export-arithmetic hf_bf16_reference \
  --compact-inputs --compact-layer-type full --ordered-sliding \
  --report /tmp/spark_cross_1024_bf16_reference.json
$PY -m pytest tests/edge/test_spark_export.py -q
```

The focused suite passed 20/20 tests, including fixed-capacity masks,
atomic shape validation, ring replacement, stale-commit rejection and state
retirement. The current probe still obtains prefill and token embeddings
from Hugging Face on WSL CPU. The next M1 gate is a resident S25 prefill and
28-layer decode backend with the same state contract, an accepted precision
and token-quality gate, bounded admission/cancellation, and measured whole
request latency, memory and sustained power. A hosted one-layer NPU job is
separate [component evidence](../../../e2e_expansion_20260923/evidence/spark_s24_full_attention/README.md).
