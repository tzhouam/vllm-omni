# The local text mode: one stage, one device, and a plan you can argue with

`vllm_omni.edge.local` runs a single autoregressive text stage on whichever
device *this process* can actually drive, and writes down why. It is the M0
milestone of the [single-device engine
proposal](../../../analysis/architecture_local_engine_20260915.md): the smallest
thing that turns "a configuration that happens to run" into "a configuration
that can be explained, reproduced and refused".

It adds no kernels, no scheduler and no graph compiler. vLLM keeps doing all of
that inside the Omni stage. What this adds is the layer above: which device,
which artifact, how many bytes, and what happens when the answer is no.

```bash
python -m vllm_omni.edge.local devices                       # what this process can drive
python -m vllm_omni.edge.local plan   --model <ckpt>         # exit 2 == explicit refusal
python -m vllm_omni.edge.local run    --model <ckpt> --prompt "..."
python -m vllm_omni.edge.local accept --model <ckpt> --json out.json
```

---

## 1. "It fits" is not "it runs"

`models/Spark-X2.5-1.7B-int8` is 1.84 GiB of compressed-tensors int8 W8A8. The
RTX 5090 Laptop in the reference machine has 23.89 GiB. Every capacity check
passes, the engine starts, the weights load — and the first forward dies:

```
dispatch_scaled_mm, csrc/.../c3x/scaled_mm_helper.hpp:34,
Int8 not supported on SM120. Use FP8 quantization instead, or run on older arch (SM < 100).
```

cutlass' `c3x` has no int8 `scaled_mm` for SM >= 100, and sm_120 is inside that
range. Roughly forty seconds and a stack trace from a subprocess to learn
something that is written in the checkpoint's `quantization_config` and the
device's compute capability.

The planner reads both and refuses in about a second:

```
REFUSED: no device in this process can run this checkpoint as asked.
  [weight_format_unsupported] cuda:0: the checkpoint is compressed-tensors:int8 and
      cuda:0 (NVIDIA GeForce RTX 5090 Laptop GPU, sm_120) executes only
      ['awq', 'compressed-tensors:fp8', 'dense', 'gptq', 'modelopt_fp4']
      remedy: this device has fp8 tensor cores but no int8 scaled_mm path
      (cutlass c3x has none for SM >= 100). Use an fp8 or a dense build here,
      or run the int8 build on the CPU backend (.venvs/omni-cpu).
```

The remedy is not decoration. The same checkpoint completes the full acceptance
run on the CPU backend, at 16.3 tok/s median — so this is a property of the
device, not a defect in the artifact, and the refusal says which.

## 2. What a device *is*, here

`hardware_probe` answers "what is in the box". That is the wrong question at
load time, because a vLLM wheel is built for one platform: the CUDA wheel
cannot run a CPU stage however many cores the probe finds, and the CPU wheel
cannot reach the card beside it. So a device is runnable only when the probe
finds it **and** `vllm.platforms.current_platform` is the platform that drives
it. Everything else is enumerated with the reason it is not available:

```
  [runnable] cuda:0           NVIDIA GeForce RTX 5090 Laptop GPU (23.9GiB vram) backend=vllm:cuda
  [not here] npu:amd          NPU Compute Accelerator Device (host_ram) backend=None
             An MCDM device; WSL2's GPU-PV forwards WDDM display adapters only...
  [not here] gpu_integrated:amd AMD Radeon(TM) 890M Graphics (host_ram) backend=None
             No /dev/dri under WSL2, but reachable through DirectML over /dev/dxg...
  [not here] cpu              AMD Ryzen AI 9 HX 370 (30.9GiB host_ram) backend=None
             vllm.platforms.current_platform is 'cuda', not 'cpu'...
```

The iGPU and the NPU are reported rather than dropped, because "the machine has
an NPU" and "a stage can run on it" are different claims and a plan has to be
able to show which one it relied on. They share `host_ram` with the CPU and
carry zero bytes, so nothing can accidentally add three copies of the same RAM
together.

`--mask gpu_discrete` hides the discrete GPU to exercise the no-NVIDIA
deployment class. It is a policy, not amnesia: the device stays in the record
with `masked: true` and its real memory.

## 3. The budget is a sum, and every term is visible

```
ADMITTED on cuda:0 (NVIDIA GeForce RTX 5090 Laptop GPU) via vllm:cuda
  artifact  43e867b180f1a3d2 dense 7.66GiB
  budget    12.51GiB of 23.89GiB vram (total)
    weights                 7843 MiB  vram  session_weights D  5 safetensors shard(s), on-disk size
    kv_cache                 720 MiB  vram  session         E  flat budget at 1x context; 248 MiB if
                                                               the backend accounts 27 sliding layers separately
    activations              120 MiB  vram  request         E  2048-token prefill chunk x 12 live tensors
    backend_workspace        512 MiB  vram  session         E  attention scratch, driver/allocator context
    load_transient          1569 MiB  vram  load            E  freed once the weights are resident
    external_reserve        1024 MiB  vram  session         E  memory this engine does not own
    safety_margin           1024 MiB  vram  session         E  fragmentation and allocator slack
```

`D` is read off the artifact; `E` is estimated. The peak is a **max**, not a
sum: the loader's transient is gone before vLLM sizes the KV pool, so adding
both would refuse plans that fit.

The KV line comes from `vllm_omni.edge.kv_budget`, which reads the real
`layer_types`. Spark-4B is 9 full-attention and 27 sliding@512 layers, so a
backend that accounts them separately needs 248 MiB where a flat charge needs
720. The budget takes the flat number (it is the safe one) and reports the
other.

**The estimates are checked, not believed.** After every run the engine reports
the budgeted peak against the sampled one:

| | CUDA / Spark-4B bf16 | CPU / Spark-1.7B int8 |
|---|---:|---:|
| counter | `nvml_device_used` (whole card) | `tree_rss` (process tree) |
| budgeted peak | 12.51 GiB | 6.55 GiB |
| measured attributable peak | 8.76 GiB | 5.48 GiB |
| budget / measured | 1.43x | 1.20x |

A budget above the measurement is the intended direction — admission needs a
provable upper bound, not a prediction — but 1.40x is loose enough to be worth
narrowing, and the number is in the record so somebody can.

### The flag that is not GPU-only

The plan sets `gpu_memory_utilization` on **both** pools, derived from its own
budget. On the CPU backend vLLM reads that same field as the fraction of *host
RAM* to reserve, so leaving it at its default there is not "no opinion" — it is
asking for 92% of the machine. A 6.55 GiB CPU plan failed to start with
`desired CPU memory utilization (0.92, 28.43 GiB)` until this was fixed.

The KV pool is set as an absolute `kv_cache_memory_bytes` rather than left as
whatever remains inside that fraction, so the cache is the size the request
limits imply. Both backends honour it exactly: CUDA logs 754974720 bytes, CPU
logs `Explicitly set (0.14/30.91) GiB for KV cache`.

## 4. Streaming, state and cancellation

Three contracts, which is what the proposal says the first version needs:

* **`ChunkEvent`** carries the *delta*, never the output so far, plus a `seq`
  that detects loss and an `epoch` that retires it.
* **`StateHandle`** is an opaque reference to state the backend owns. It is
  tied to the artifact that produced it (the same shapes from a different
  quantization are not the same state) and is never `migratable` — moving a KV
  cache between backends needs a measured layout conversion for the specific
  pair.
* **`BoundedEventStream`** bounds chunks *and* bytes and blocks the producer,
  which is the backpressure. A single event larger than the whole byte bound
  still passes on an empty queue, or it would deadlock.

Cancelling is not "stop calling `next()`". The backend has work in flight and
its tokens arrive after the cancel. `cancel()` retires the epoch first, then
aborts in the backend, then closes the stream — *before* waiting on the
producer, so a backend that hangs while unwinding delays cleanup rather than
the caller. In the recorded runs the fence discarded 3 in-flight events on CUDA
and 2 on CPU, and zero events from the retired epoch reached the consumer.

The device counter does **not** fall on cancel, because vLLM's KV pool is
pre-allocated. What falls is the in-flight count. The report says so, so that
"memory did not drop" is not read as "nothing was released".

## 5. Measuring memory without fooling yourself

Three counters, three different accountings, never summed:

* `nvml_device_used` is the whole card, every process. Under WSL2
  `nvmlDeviceGetComputeRunningProcesses` returns an empty list, so GPU bytes
  **cannot** be attributed to this engine's PIDs on this host. "Attributable
  peak" means peak minus a pre-load baseline (2.79 GiB here), not a private
  total.
* `tree_pss` includes a proportional share of file-backed pages, which on an
  mmap'd safetensors load is partly page cache.
* `host_available` moves for reasons that are not us.

Sampling is every 0.25 s, so every peak is a lower bound, and the record says
that rather than assuming it away.

## 6. What Spark needed before any of this worked

Two gaps that made "the model is in the registry" and "the model runs under
Omni" different statements:

1. **No `PipelineConfig`.** Spark's model definition had been registered for a
   while, but with no pipeline `Omni(model=<Spark checkpoint>)` had nothing to
   resolve, so the model was reachable only through plain `vllm.LLM`. Added as
   a single `LLM_AR` text stage in
   `vllm_omni/model_executor/models/spark2_5/pipeline.py`.
2. **A stage-call mismatch.** Omni's AR model runner hands every stage model
   `sampling_metadata`, `logits_index` and `sampler`. Spark's
   `ForCausalLM.forward` caught them in `**kwargs` and forwarded them into
   `Spark2_5Model.forward`, which takes four arguments — `TypeError` on the
   first forward. They now stop at the `ForCausalLM` boundary, where the
   decoder stack has no use for them anyway.

One stage is the design, not a placeholder. An autoregressive session stays on
one backend by default: prefill and decode share the KV cache, and a per-token
device boundary buys nothing and costs a synchronization. Multi-stage is M2,
and it needs the tensor/buffer contract that a single text stage — which hands
a buffer to nobody — cannot be used to design.

## 7. Does the Omni path change the answer?

This is the one claim the rest of M0 cannot make for itself. Everything above
is about the engine's own behaviour; this asks whether routing a decode through
the Omni stage changed the model's output. The validation plan makes it a
*hard* initial acceptance condition.

Same checkpoint, same device, same request limits, same environment, greedy
with `ignore_eos` -- the only difference is the path: plain `vllm.LLM` straight
into vLLM against `AsyncOmni` -> StageRuntime -> the stage's EngineCore.

**12 of 12 prompts, 128 greedy tokens each, token-for-token identical -- on
both paths**: CUDA with the bf16 build and CPU with the int8 one. The quantized
build gets its own run rather than inheriting the result, because quantization
is exactly what makes a near-tie likelier to flip.

The comparison is on **token ids, not text**: two different id sequences can
detokenize to the same string, and a text comparison would hide exactly the
drift worth catching. The record keeps the first divergent position rather than
a yes/no, because greedy decoding is not bitwise stable across batch shapes --
a different prefill chunking reorders reductions and can flip an argmax at a
near-tie, so "split at token 120" and "split at token 3" are different
findings. Neither occurred.

The two engines load sequentially in one process, reference first and released
before the second starts; two copies of a 7.66 GB checkpoint would not fit the
budget this milestone is about respecting.

```bash
python -m vllm_omni.edge.local parity --model <ckpt> --json out.json   # exit 3 on divergence
```

### The two paths do not need the same environment

The reference cannot start on this host with a default environment, and both
reasons are worth knowing:

* under WSL2, vLLM's own GPU model runner allocates a `UvaBuffer` for its input
  batch and raises `RuntimeError: UVA is not available` without pinned memory.
  Omni's AR model runner does not take that path and starts either way.
* FlashInfer's sampler JIT-compiles on first use and needs `nvcc`, which is not
  installed -- `nvidia-smi` reporting a CUDA version is not a toolkit. This one
  kills the engine at the *first token*, not at load.

Both now live in one `RUNTIME_ENV` applied by every entry point that starts an
engine, because a comparison whose reference is configured differently from its
subject is measuring the configuration.

## 8. Limits

* One model, one stage; no multi-stage, no tensor/buffer contract.
* `--mask` is a process-level policy, not a physical absence. A full no-card
  verification needs another machine.
* The iGPU and NPU are enumerated only; neither has a stage backend (M3).
* Reference consistency covers **one decode mode**: single sequence, greedy,
  128 tokens, on both paths. It does not cover concurrency above 1 (which
  changes batch shapes, and greedy is not bitwise stable across those), longer
  outputs, or non-greedy sampling -- and it is not a quality evaluation. It
  says the Omni path did not change the output, not that the output is good.
* Performance figures are one sample per prompt, not p50/p95, with no fixed
  power profile and no cache-cold guarantee.

## 9. Tests

```bash
# hermetic; no GPU, no weights, ~3 s
pytest tests/edge/local --ignore=tests/edge/local/test_e2e_local_text.py

# end to end against a real checkpoint; the venv decides which artifact
pytest tests/edge/local/test_e2e_local_text.py
```

The e2e tests run the acceptance criteria themselves: both startups reach a
decision, twelve fixed prompts each reach 128 tokens, the budget bounds the
measured peak, cancellation leaks nothing, and the Omni path decodes what plain
vLLM decodes. They skip rather than fail when no checkpoint is present.
**78 pass in each venv**; about 4 min on this machine either way.

The twelve prompts are the same twelve
`analysis/experiments/spark_edge/quality_int4.py` measures the quantized builds
against, so these numbers stay comparable with the existing W4A16 / W4A8 /
Q4_K_M fidelity work.

Full run records:
[`analysis/experiments/m0_local_text_20260915/`](../../../analysis/experiments/m0_local_text_20260915/README.md).
