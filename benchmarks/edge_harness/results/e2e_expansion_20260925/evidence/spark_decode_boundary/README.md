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

## Fixed-shape cache ordering experiment

The 500-token prompt was repeated for 128 teacher-forced decode steps with a
628-entry full-attention buffer, exactly enough for this rollout. All four
runs used the same checkpoint, prompt IDs, BF16 source arithmetic and eight
HX370 CPU threads. Each row matched 128/128 next-token choices, but their
logit and newly written K/V differences separate the cache choices:

| Sliding-cache boundary | Full capacity | Worst logits relative L2 | First new-K/V difference |
|---|---:|---:|---:|
| [Physical ring, 1024-entry baseline](cross_512_bf16_static_final_128.json) | 1024 | 9.857% | position 512 |
| [Physical ring, tight capacity](spark_cross_512_bf16_static628_128.json) | 628 | 9.878% | position 512 |
| [Chronological ring read, tight capacity](spark_cross_512_bf16_ordered_static628_128.json) | 628 | 0.511% | position 562 |
| [Chronological roll-cache graph, tight capacity](spark_cross_512_bf16_roll_static628_final_128.json) | 628 | 10.216% | position 500 |

Shrinking the full cache alone did not fix the physical ring's BF16 reduction
order. Reordering its read to chronological order greatly reduced the worst
error, but that diagnostic would move or gather all 511 sliding K/V entries
per layer each token; no mobile transfer or kernel cost has been measured.
The exporter already represents a rolled sliding window. Its CPU cache
controller now seeds and commits that full-window output, but this real-weight
roll run failed the logit fidelity gate from the first padded step. Its
10.216% outlier at position 534 rules out treating roll as a drop-in fix.
The tight-capacity ring reports retain the prior probe/export source hashes;
the roll report records the revised state controller and probe hashes. No row
is an on-device or compiled QNN result.

## Full-cache bucket growth and new-K/V audit

The CPU reference cache can now grow its seven full-attention K/V buffers at
an admitted bucket boundary without discarding committed state. The
[bucket audit](bucket_audit.json), produced by the
[comparison script](../../../../experiments/audit_spark_cache_buckets.py),
compares fixed-capacity and 16-slot bucket runs using identical checkpoint,
prompt IDs, 128 teacher-forced input tokens, BF16 arithmetic and chronological
sliding-ring reads. All four runs used the HX370 WSL CPU, not an S25 backend.
The 500-token prompt crosses 512; the 1000-token prompt crosses 1024.

| CPU run | Full-attention capacity | Top-1 matches | Worst logits relative L2 | Worst new-K/V relative L2 | First new-K/V error >1% |
|---|---:|---:|---:|---:|---:|
| [500, fixed](cross_512_bf16_ordered_static628_kv_128.json) | 628 | 128/128 | 0.5114% | 1.3387% | position 562 |
| [500, bucket 16](cross_512_bf16_ordered_bucket16_kv_128.json) | 512→640, eight growths | 128/128 | 0.5114% | 1.3387% | position 562 |
| [1000, fixed](cross_1024_bf16_ordered_static1128_kv_128.json) | 1128 | 128/128 | 4.1967% | 257.4521% | position 1063 |
| [1000, bucket 16](cross_1024_bf16_ordered_bucket16_kv_128.json) | 1008→1136, eight growths | 128/128 | 4.1967% | 257.4521% | position 1063 |

For the 500-token fixture, bucketing changed neither per-step logit nor
new-K/V error. For the 1000-token fixture, it changed the per-step logit error
at 16 positions and new-K/V error at one position, but neither worst error nor
the first state-gate failure. The worst bucketed value tensor at position 1094
(layer 15) had reference L2 norm 8.699, candidate norm 22.203 and maximum
absolute error 3.079; its 257.4521% relative error is not caused by a
near-zero reference norm. Exact filled-length inputs previously matched the
source bitwise, so these failures are specific to the tested padded static
attention layouts. Top-1 agreement on this teacher-forced fixture does not
clear the state gate or establish free-running generation quality.

The next artifact experiment should test a target-compatible filled-length
attention normalization and cache ordering on its actual compiled backend. No
full-device Spark E2E or mobile performance claim follows from these CPU runs.

## First BF16 divergence and failed static-shape alternatives

The [position-562 attention-input trace](cross_512_bf16_static628_trace_attention_inputs562_63.json)
used the same 500-token prompt, 628-slot full cache, chronological sliding
ring, BF16 source arithmetic and eight HX370 WSL CPU threads. Layers 0–10
returned bitwise-equal hidden states. Layer 11, a full-attention layer, is the
first different hidden state (0.1808% relative L2). Its valid Q×K score
entries and repeated value inputs were **bitwise equal** between source and
export. The padded export's BF16 softmax probabilities first differed by
0.00509% relative L2; the ensuing probability×value result differed by
0.00774%. Downstream layers amplified this into the 1.2693% new-K/V error at
position 562. This identifies the first observed numerical boundary, not a
universal root cause for other prompts or devices. The earlier
[hidden-only trace](cross_512_bf16_static628_trace562_63.json) and
[matmul-output trace](cross_512_bf16_static628_trace_attention562_63.json)
are retained as intermediate records.

Several CPU-only alternatives were measured on the same real checkpoint and
prompt. Their different step counts are shown explicitly; each still matched
the source next-token choice at every tested step. The original fixed-buffer
run was bitwise through its first eight steps and first failed the 1% state
gate at position 562.

| Padded 628-slot CPU diagnostic | Decode steps | Worst logits relative L2 | Worst new-K/V relative L2 | First >1% state position |
|---|---:|---:|---:|---:|
| [Original BF16, first 64 steps](cross_512_bf16_ordered_static628_kv_128.json) | 64 of 128 | 0.3738% | 1.2693% | 562 |
| [FP32 score matmul](cross_512_bf16_static628_fp32_score_8.json) | 8 | 0.5056% | 20.7464% | 500 |
| [FP32 value matmul](cross_512_bf16_static628_fp32_value_64.json) | 64 | 4.2858% | 34.2471% | 500 |
| [Right-aligned full cache](cross_512_bf16_static628_rightfull_64.json) | 64 | 2.2862% | 34.7933% | 501 |
| [FP64 softmax, cast back to BF16](cross_512_bf16_static628_softmax_fp64_128.json) | 128 | 0.8542% | 19.7564% | 514 |

The input-repacking and precision experiments are diagnostic controls, not
compiled or device-executable fixes. Promoting attention arithmetic changed
the trained BF16 computation and often introduced an error before the original
position-562 boundary. The [eight-step FP64 control](cross_512_bf16_static628_softmax_fp64_8.json)
was bitwise equal, illustrating why a short prefix does not qualify the
128-step state path. A target-compatible design must either preserve the
source's effective filled-length softmax behavior or establish an accepted
task-level tolerance for a different normalization. The present reference
still lacks a qualified static-shape numerical path across 512/1024 and has
no S25 complete-request measurement.

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
$PY benchmarks/edge_harness/experiments/probe_spark_decode_boundary.py \
  --model "$MODEL" --prefill-tokens 500 --decode-steps 128 \
  --reference-dtype bfloat16 --export-arithmetic hf_bf16_reference \
  --cache-layout roll --context-capacity 628 \
  --report /tmp/spark_cross_512_bf16_roll_static628.json
$PY benchmarks/edge_harness/experiments/probe_spark_decode_boundary.py \
  --model "$MODEL" --prefill-tokens 1000 --decode-steps 128 \
  --reference-dtype bfloat16 --export-arithmetic hf_bf16_reference \
  --ordered-sliding --full-bucket-width 16 \
  --report /tmp/spark_cross_1024_bf16_bucket16.json
$PY -m pytest tests/edge/test_spark_export.py -q
```

The focused suite passed 22/22 tests, including fixed-capacity masks,
atomic shape validation, ring replacement, stale-commit rejection, state
retirement, bucket growth and chronological roll-cache state. The current probe still obtains
prefill and token embeddings
from Hugging Face on WSL CPU. The next M1 gate is a resident S25 prefill and
28-layer decode backend with the same state contract, an accepted precision
and token-quality gate, bounded admission/cancellation, and measured whole
request latency, memory and sustained power. A hosted one-layer NPU job is
separate [component evidence](../../../e2e_expansion_20260923/evidence/spark_s24_full_attention/README.md).
