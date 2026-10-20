# The 4-bit CPU path

Two 4-bit paths exist on an x86 CPU and they are not interchangeable. Which one
a device should use is a property of the device, and `vllm_omni/edge/weight_path.py`
picks it from a hardware probe.

## W4A8 — `_C::int4_scaled_mm_cpu`

Needs AMX tiles. Wants GPTQ-packed weights and quantizes activations to int8.
Fastest measured here (74.4 tok/s) and least faithful (1/12 greedy prompts
reproduced exactly).

One trap: a GPTQ checkpoint **must** quantize the output head. Spark ties its
embedding to its head, so the head has to be untied and emitted separately; if
it is not, that one projection runs in bf16 through `onednn_mm` and costs
2.97 ms/step by itself — more than the entire 4-bit saving. Every W4A8 number in
this project's first round was invalid for exactly this reason.

The symmetric grid is **our exporter's choice, not the kernel's limit**:
`_process_gptq_weights_w4a8` uses the checkpoint's zero points whenever
`config.zero_points` is set and synthesises the constant 8 only otherwise. An
asymmetric export is a config change and is the cheapest untried fidelity
experiment.

## W4A16 — PyTorch tinygemm

No AMX requirement, activations stay bf16, more faithful (2/12). 71.6 tok/s.
This is `quantization="cpu_int4"`.

The kernel dequantizes once per row of activations, so its cost is linear in
batch size: fine for decode, ruinous for prefill (453 tok/s if used for
everything). Large batches therefore take a second path — dequantize to bf16
once and hand it to the ordinary oneDNN/AMX GEMM.

### The weight layout, measured

That prefill path used to read a row-major *duplicate* of every weight, because
the tinygemm layout was treated as opaque. It is not. Packing an index pattern
and reading back where each nibble landed shows, on AVX-512:

```
[N/64][K][32 bytes]
```

K is the middle axis, so a K range is contiguous inside each n-block; and within
every 32-byte run, byte `d` carries output channel `d` in its low nibble and
`d + 32` in its high one — the same permutation for every `k` and every block.
Decoding it is a deinterleave, not a reverse-engineered ISA table.

So prefill can dequantize from the kernel's own layout and the duplicate is
never allocated: **615 MB**, bit-exact against the old path at 2048×2048,
3072×2048 and `gate_up`'s real 13312×2048.

It is not free. `to_split_halves` performed that deinterleave *once at load*, so
each prefill call read two contiguous planes; doing it per call costs **1.63×
prefill**. It is a dial, not a win:

- against `prefill_dequant: false` it is **strictly better** — same memory, 2.97×
  the prefill, and multi-row batches never reach the tinygemm kernel, avoiding
  the AMX tile-configuration SIGILL documented in `cpu_int4.py`. That setting is
  dominated.
- against keeping both layouts it trades 615 MB for ~1.6× prefill. Default on,
  because this path exists for devices chosen for their memory ceiling. Set
  `direct_dequant: false` in the checkpoint's `quantization_config` to revert.

`supports_direct_dequant` gates the reshape on N tiling the 64-channel block,
which every Spark linear does. An N like 80 packs as 64 + a 16 tail the reshape
would mis-read, so those layers keep the duplicate rather than being quietly
wrong.

### Why the path choice is a config field

The two prefill paths build *different graphs* — one passes `weight_halves` into
the custom op, the other passes `None`. vLLM keys its AOT compile cache on the
config, so an env-var-only switch shares one cached artifact between them, and a
warm cache loads the wrong graph: three benchmark passes died with
`KeyError: 'weight_halves'` inside an AOT-loaded forward. `direct_dequant` is a
`quantization_config` field because `ModelConfig.compute_hash` hashes those.

## `cpu_gemm_wna16` — the fallback

vLLM selects this when `use_w4a8` is off. Measured at 32.3 tok/s, **2.2× slower
than tinygemm**, so the selector does not route to it. It is documented because
`weight_path.py` previously asserted it did not exist.

## Building a checkpoint

```bash
# W4A16 (cpu_int4)
python benchmarks/edge_harness/quantize_int4.py \
    --model <bf16-model> --group-size 128 --out models/<name>

# W4A8 (GPTQ) — note the head must be untied and quantized
python benchmarks/edge_harness/quantize_gptq_int4.py \
    --model <bf16-model> --group-size 128 --out models/<name>
```

`quantization_config` keys the `cpu_int4` path understands: `group_size`,
`quantize_embedding`, `prefill_dequant`, `direct_dequant`, `dequant_threshold`,
`use_custom_op`.
