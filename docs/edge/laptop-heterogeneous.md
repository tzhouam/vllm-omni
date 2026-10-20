# Qwen3.8-27B and Spark-X2.5-4B on a Ryzen AI laptop

Target machine: **AMD Ryzen AI 9 HX 370** (Zen 5, 12 cores / 24 threads,
AVX-512 with VNNI and BF16, **no AMX**), 30.9 GiB LPDDR5X, **RTX 5090 Laptop**
(24463 MiB, sm_120, PCIe 4.0 x8), **Radeon 890M** iGPU and an **XDNA2 NPU**,
running under **WSL2** (kernel 6.18.33.2).

Two models were added and both run:

| model | status | evidence |
|---|---|---|
| `XHToken/Spark-X2.5-4B` | runs; **24/24 greedy tokens identical to the HF reference** | `analysis/experiments/spark_edge/parity_hf_vs_vllm.py` |
| `Qwen/Qwen3.8-27B-FP8` | runs; correct output; 1.05 tok/s at the tightest fit | offload sweep below |

Neither needed a new model class. Spark-4B is the same `Spark2_5ForCausalLM`
the 1.7B uses and the implementation was already config-driven; Qwen3.8 is
`Qwen3_5ForConditionalGeneration`, which vLLM 0.28 already carries. What both
needed was the *sizing* and *placement* code around them, which is what this
document is about, and four environment faults, which are in
[§7](#7-what-was-actually-broken).

---

## 1. Three accelerators, and what each one costs to reach

The request was to use the CPU, the NPU and the GPU together. An earlier
version of this document said the iGPU and the NPU were simply unreachable
under WSL2. That was true of the default setup and false as a statement about
the machine: both have a route, the routes are not equivalent, and finding them
changed one of the two answers.

| device | reachable from WSL? | route | state |
|---|---|---|---|
| RTX 5090 Laptop | yes | `/dev/dxg` → `torch.cuda` | working |
| Radeon 890M iGPU | **yes** | DirectML over `/dev/dxg` (`torch-directml`) | **working, and slower than the CPU** |
| XDNA2 NPU | no | Windows-side ONNX Runtime + **VitisAI** EP | **working** — 2.6–5.3× the CPU on GEMMs |

### The iGPU: reachable, and measured not worth it

There is no `/dev/dri` under WSL2, and there does not need to be. Mesa ships a
D3D12 Gallium driver and Microsoft ships `libd3d12.so` / `libdxcore.so` in
`/usr/lib/wsl/lib`, so any WDDM adapter is addressable through `/dev/dxg`.
`glxinfo` already reports `D3D12 (AMD Radeon(TM) 890M Graphics)`, accelerated,
OpenGL 4.6. For compute, `torch-directml` offers it as a torch device:

```
DirectML device_count: 2
  [0] NVIDIA GeForce RTX 5090 Laptop GPU
  [1] AMD Radeon(TM) 890M Graphics
```

It computes correctly — a 512×512 matmul agrees with the CPU to **3.6e-07**.
Then it was benchmarked, which is why this section is short:

| | iGPU via DirectML | CPU | ratio |
|---|---:|---:|---|
| fp32 GEMM 2048³ | 1.12 TFLOP/s | 1.54 TFLOP/s | **0.73×** |
| memory bandwidth | 5.0–7.4 GB/s | 20.7 GB/s | **0.30×** |
| host↔device, 64 MB | 39.2 ms (3.3 GB/s) | — | — |

The iGPU is **slower than the CPU it shares memory with, on both axes**: the
D3D12 translation layer costs more than the hardware wins. It is genuinely
usable now, and the honest advice is to route work to it only when the point is
to free the CPU rather than to go faster. A native path — ROCm on bare-metal
Linux, or DirectML inside a Windows process — would not pay the translation and
is the version worth measuring.

Note the pin conflict: `torch-directml` requires `torch==2.4.1`, so it can
never share a virtualenv with the engine (torch 2.13). It gets its own.

### The NPU: working, at 2.6-5.3x the CPU

Not from WSL, and no driver install changes that: it is an **MCDM** device
(`IpuMcdmDriver`, AMD 32.0.203.329), and WSL2's GPU-PV forwards WDDM *display*
adapters only, so it has no device node in Linux. PCI passthrough is not a WSL2
feature. It has to be driven from a Windows-side process.

From Windows it works. Three things have to be right **at once**, and each one
failing on its own produces exactly the same symptom — a session that builds,
runs, returns correct numbers, and executes every node on the CPU:

1. **The right EP.** The appx package ships two.
   `onnxruntime_providers_ryzenai.dll` registers cleanly, enumerates the NPU as
   `OrtHardwareDeviceType.NPU`, opens a session — and claims **zero nodes** from
   everything: fp32, INT8 QDQ symmetric and asymmetric, per-channel,
   power-of-two scales, MatMul and Conv, and ahead-of-time `ModelCompiler`
   (which emitted zero `EPContext` nodes). It is the *Light* EP. The general one
   is `onnxruntime_vitisai_ep.dll`.
2. **The DLL search path.** That one fails to load from a bare
   `register_execution_provider_library` with `Error loading … But no
   dependency`. It needs its own directory added via `os.add_dll_directory`
   first, because it pulls `dyn_dispatch_core.dll` and `vaiml.dll` from beside
   it.
3. **A16W8, not A8W8.** The overlay names are embedded in the DLL and say so:

   ```
   4x2_psf_model_a8w8_qdq.xclbin
   8x4_psu_model_a16w8_qdq.xclbin      <- most of them
   2x4x4_bmm_model_a16w16.xclbin
   ```

   XDNA2 wants **16-bit activations with 8-bit weights**. A8W8 — the default
   shape of `quantize_static` — is rejected as completely as fp32, silently.

Also needed: **onnxruntime ≥ 1.25**, because the EP requests ORT API 25.
`onnxruntime-directml` stops at 1.24.4 and is refused, so the NPU and DirectML
cannot share a process.

With all of it right the partitioner starts talking and names its target,
`AMD_AIE2P_4x8_CMC_Overlay`. Measured against the same INT8 graph on the CPU:

| GEMM | CPU | NPU | speedup | NPU |
|---|---:|---:|---:|---:|
| 256 × 1024 × 1024 | 1.47 ms | 0.56 ms | 2.64× | 0.97 TOPS |
| 512 × 2048 × 2048 | 9.64 ms | 2.14 ms | 4.51× | 2.01 TOPS |
| 1024 × 2048 × 2048 | 19.89 ms | 3.76 ms | **5.29×** | **2.29 TOPS** |

Max relative error against the CPU: **3.2e-05**. It scales the right way — a
small GEMM cannot amortise the dispatch. Dequantize nodes stay on the CPU (the
partitioner says so explicitly), so a graph that is mostly requantization gains
nothing.

The recipe is code in `vllm_omni/edge/npu_ryzenai.py`, including the one check
worth trusting: **count nodes in an ORT profile.** A session reporting the EP in
`get_providers()` may still be running everything on the CPU — and when it does,
its outputs are bit-identical to the CPU, which is precisely what a *correct*
NPU run does not look like. A 0.000e+00 error is the bug signature, not the
success signature.

**What it is worth here.** Not decode: 2.29 TOPS against the dGPU's
48.2 TFLOP/s, over the same 20.7 GB/s system memory the CPU uses, cannot help a
bandwidth-bound step (§3). It earns its place on the compute-bound work off the
per-token path — a vision tower, a draft model for speculative decoding — where
5× the CPU on large GEMMs is real.

### Reaching a phone

The same question for mobile has a cleaner answer. The harness's `AdbDevice`
backend needs only an `adb` binary and a reachable device, and WSL2 is in NAT
mode, so **outbound** connections to the LAN work even though inbound do not.
That makes wireless debugging the path of least resistance:

```bash
sudo apt-get install -y adb          # done; adb 1.0.41 (34.0.5)
adb connect <phone-ip>:5555          # Android 11+ wireless debugging
```

No USB passthrough is involved, which matters because WSL2 has none by default
— the USB route needs `usbipd-win` on the Windows side, and is worth the extra
moving part only when the phone cannot be on the same network. `AdbDevice` also
takes an explicit binary path, so `adb="/mnt/c/.../adb.exe"` drives a
USB-attached phone through the Windows adb server with no passthrough at all,
at the cost of translating paths on every `push`/`pull`.

The mobile **export** path is now working as well: the QNN export tests needed
`onnxscript` and `tabulate`, and the ExecuTorch lowering needed `flatc`. All
three are installed, and the one remaining ExecuTorch failure turned out not to
be a failure — pytest run from the repository root lets the bare `executorch/`
source checkout shadow the installed package. Run from `vllm-omni/` the suite
is **124 passed, 1 skipped**.

## 2. The measurement that decides the design

Taken on an **idle** machine (see §7 for why that word is load-bearing):

| path | measured |
|---|---|
| GPU VRAM bandwidth | **472 GB/s** |
| GPU bf16 GEMM 8192³ | **48.2 TFLOP/s** (91 W, 1447 MHz) |
| PCIe H2D, pinned, bulk | **13.9 GB/s** |
| PCIe, as seen by offloaded weight reads | **10.6 GB/s** (§5) |
| CPU RAM bandwidth | **20.7 GB/s** |
| CPU bf16 GEMM 4096³ | **1.54 TFLOP/s** |

Two ratios follow, and everything else in this document is a consequence:

```
VRAM : system RAM   =  22.8 x
VRAM : PCIe link    =  34 x     (44x on the effective figure)
```

Autoregressive decode reads **every weight once per token** and does two FLOPs
per weight. At batch 1 it is bandwidth-bound with no room for argument, so
decode rate is set by *where the weights are*, and almost nothing else.

---

## 3. Why "use all three devices for decode" is the wrong goal

The intuition -- three processors, so split the model three ways -- inverts on
a bandwidth-bound workload. Splitting weights across pools does not add
bandwidth; the step ends when the *slowest* pool finishes, and the slow pool is
22.8x slower. Putting 10% of a model on the CPU makes the step
`0.9/472 + 0.1/20.7`, which is **3.2x slower** than putting all of it on the
GPU, not 10% slower.

The iGPU and the NPU do not escape this: both read the same LPDDR5X the CPU
does, so all three share one 20.7 GB/s pool — and the iGPU, measured, reaches
only 5-7.4 GB/s of it (§1). Adding them to a decode split adds arithmetic to a
problem that is not short of arithmetic, over a pool that is already the
bottleneck.

So the rule is: **for decode, one device, and it is the one with the memory
bandwidth.** The other devices earn their place on work that is *not* on the
per-token critical path:

| device | job | why it fits there |
|---|---|---|
| dGPU | every decode step, all resident layers | the only 472 GB/s in the machine |
| CPU | `embed_tokens` gather, detokenisation, scheduling, prompt templating | a gather reads one 10 KiB row per token, not the 2.54 GB table |
| NPU | the **vision tower** on image prompts; a **draft model** for speculative decoding | both are compute-bound and off the decode path — *pending the compiler in §1* |
| iGPU | the same class of work, when the goal is to free the CPU | reachable today, but measured at 0.73x the CPU's compute and 0.30x its bandwidth (§1) |

The vision tower is the clean case: 0.92 GB, run **once per image**, never per
token. Moving it off the GPU frees 0.92 GB of VRAM -- which by §5 is worth
86 ms/token -- and costs one prefill-time hop. That is a real heterogeneous win
and it does not touch the decode loop.

Speculative decoding is the other one, and it is the only way a second device
raises *decode* throughput: a draft model on the NPU proposes k tokens, the GPU
verifies them in a single batched pass, and the GPU's per-token weight read is
amortised over k tokens. Qwen3.8-27B **ships an MTP head for exactly this**
(`mtp_num_hidden_layers: 1`, 0.48 GB, in `mtp.safetensors`).

---

## 4. What the two models actually cost

`vllm_omni/edge/kv_budget.py` now prices both. It could price neither before:
it read `num_hidden_layers` off the top-level config, which `Qwen3_5Config`
does not have, and it charged every layer for a KV cache.

**Qwen3.8-27B-FP8** is a hybrid of a kind the old code had no concept of: of its
64 layers, **48 are gated DeltaNet**, which keep a fixed-size recurrent state
and *no KV cache at any context length*. Only the 16 full-attention layers
cache.

```
flat over 64 layers    256 KiB/token      <- what a layer-blind budget charges
16 full layers only     64 KiB/token      <- the truth, 4x smaller
GDN recurrent state    147 MiB/sequence   <- constant, and previously invisible
```

The state is constant and the cache is linear, so they cross at ~2350 tokens:
below that, the thing the old budget did not count at all is the larger half.
At a 4096-token context the whole cache plus state is **467 MiB** (147 MiB of
state, 320 MiB of cache), and **787 MiB** at 8192 -- against 30.87 GB of
weights. For this model, **memory is a weights problem and not a
cache problem**, and long context is nearly free. That is the architectural
gift of the 3:1 linear hybrid and it is the strongest reason to want this model
on an edge box.

**Spark-X2.5-4B** is the other hybrid: 27 sliding-window(512) layers to 9 full
ones, 144 KiB/token flat, and a **3.37x** discount if the backend accounts per
layer type. Both sizes now have their published geometry pinned in
`tests/model_executor/models/spark2_5/test_spark2_5.py`, so a change that
assumes the 1.7B's 8 query heads or 28 layers fails in CI rather than on a
checkpoint.

---

## 5. The cost of not fitting, measured

Qwen3.8-27B-FP8 is **30.87 GB** of weights against a 24 GiB card. It only runs
with part of it on the host. Sweeping how much (4096-token context, batch 1,
greedy, `enforce_eager`, idle machine):

| offloaded | tok/s | s/token | marginal |
|---|---:|---:|---|
| 10 GB (the tightest fit; 9 GB fails) | **1.05** | 0.952 | |
| 12 GB | 0.89 | 1.124 | +0.086 s/GB |
| 14 GB | 0.76 | 1.316 | +0.096 s/GB |
| 16 GB | 0.66 | 1.515 | +0.100 s/GB |

Straight line, slope **+94 ms per offloaded GB**, i.e. an effective
**10.6 GB/s** -- the 13.9 GB/s bulk figure less ~20% of overhead. The model

```
s/token  =  resident_GB / 472  +  offloaded_GB / 10.6
```

predicts the 10 GB point at 987 ms against 952 ms measured, and the whole curve
within 5%. At the tightest fit, **the 10 GB on the host is 94% of the step and
the 20.9 GB in VRAM is 4%.**

This is the number to design against: **a gigabyte that does not fit costs
44x what a gigabyte that fits costs** (94 ms against 2.1 ms). Fitting is not
one optimisation among several; it is the only one that matters until it is
solved.

### Which gigabytes to spend

vLLM's `cpu_offload_gb` offloads "non-selectively until the limit is reached",
which is the expensive way. Not every parameter is read once per token, and
`vllm_omni/edge/placement.py` sorts them:

```
lm_head         2.543 GB   HOT   one 248320 x 5120 GEMM per token
embed_tokens    2.543 GB   COLD  a gather: one row, ~10 KiB per token
visual.*        0.922 GB   COLD  unless the prompt carries an image
mtp.*           0.478 GB   COLD  unless speculative decoding is on
                ---------
cold set        3.94 GB    free to evict
```

`embed_tokens` and `lm_head` are the same shape and the same dtype and the
opposite decision. A size-ordered, name-blind policy cannot tell them apart;
`cpu_offload_params`, which matches name segments, can.

For FP8 the cold set is 3.94 GB against a ~10.6 GB deficit, so it helps but
cannot close it -- **selective placement does not rescue this checkpoint**.

And a larger caveat, found after this section was first written: *the cold set
cannot currently be offloaded at all*. `cpu_offload_params` is real and matches
name segments as documented, but the offloader is only ever offered decoder
layers (`make_layers`, `utils.py:812`), so `embed_tokens`, `lm_head`, the vision
tower and the MTP head are out of its reach. The classification below is right
about which bytes are cheap to move; moving them needs an upstream change. See
§6.

---

## 6. The design: make it fit — and the retraction that follows

The previous version of this section said: use a 4-bit checkpoint, it is worth
~20×, everything else is worth less than 2×. That was arithmetic —
`resident_GB / 472 GB/s` — and the arithmetic was **wrong by roughly 30×**. The
4-bit build was then measured, and it is *slower* than the 8-bit one.

| build | weights | offloaded | decode |
|---|---:|---:|---:|
| `Qwen/Qwen3.8-27B-FP8` | 30.87 GB | 10 GB | **1.05 tok/s** |
| `nvidia/Qwen3.8-27B-NVFP4` | 21.95 GB | 3 GB | **0.67 tok/s** |
| the same, torch.compile + CUDA graphs | 21.95 GB | 3 GB | 0.72 tok/s |
| the same, 4 GB offloaded | 21.95 GB | 4 GB | 0.56 tok/s |

Three things went wrong at once, and each is worth stating separately.

**It does not fit, even at 21.95 GB.** The weights load to 19.92 GiB and the
card has 22.58 GiB free (the Windows desktop holds the rest). vLLM's runtime
overhead — non-torch allocations plus the activation peak — is about **2.8 GiB**
on this model, which leaves the KV cache at **−0.31 GiB**. Shrinking
`max_model_len` from 4096 to 1024 moved that to −0.30: the overhead is not the
cache, so context length cannot buy the fit.

**The lever that would have made it fit does not exist.** §5 identified 4.31 GB
of cold parameters and prescribed `cpu_offload_params={embed_tokens, visual,
mtp}`. The flag is real and matches name segments as documented. But the
offloader is only ever *offered decoder layers*: `wrap_modules` is called from
inside `make_layers` (`vllm/model_executor/models/utils.py:812`) and nowhere
else. `embed_tokens`, `lm_head`, the vision tower and the MTP head are built
outside it. Asking for the cold set moved **~0.5 GiB** and the run failed
identically to one with no offload at all. The classification was right about
which bytes are cheap to move; the mechanism to move them is missing, and
closing it is an upstream change.

**And fewer bytes turned out not to mean faster.** This is the part that
invalidates the projection rather than merely complicating it. The traffic model
`resident/472 + offloaded/10.6` fits the FP8 offload curve to within 4.3% across
four points. Against NVFP4 it is **4.7× optimistic**: predicted 3.1 tok/s at
3 GB offloaded, measured 0.67. Its own offload slope is 0.293 s/GB — an
effective **3.4 GB/s**, against FP8's 10.6 — and extrapolating to zero offload
gives ~**1.6 tok/s** even if it fit. Enabling torch.compile and CUDA graphs
changed 0.67 to 0.72, so this is not an eager-mode artifact.

The conclusion is that **NVFP4 decode on this stack is compute-bound, not
bandwidth-bound**, and a model that counts bytes cannot see that. Why it is
compute-bound is not established here: the checkpoint is a `modelopt` mix of
FP8 and NVFP4 with group-16 scales, and this fork already carries a correctness
patch for the W4A4 NVFP4 linear path, which is a hint that the path is not a
well-trodden one on sm_120. Diagnosing it is the obvious next experiment, and
until someone does, the honest statement is the measurement.

### What the design actually is, then

1. **Run the FP8 checkpoint with the smallest offload that fits** — 10 GB, 1.05
   tok/s. This is the fastest configuration measured on this machine, and it is
   the 8-bit one.
2. **Do not change quantization format on an estimate.** The one estimate this
   document made without measuring is the one it got wrong, by 30×.
3. **Size the cache from the model** — 467 MiB at 4096 tokens (§4). Still right,
   and still worth little here, because weights dominate.
4. **Speculative decoding via the shipped MTP head** is now the largest
   *unexplored* lever, and the only one that attacks the per-token weight read
   rather than working within it. It is untested here.
5. **Getting the cold set offloadable upstream** would free 3.94 GB on FP8, worth
   ~370 ms/token by the traffic model — which, on the FP8 path where that model
   demonstrably holds, is the difference between 1.05 and roughly 1.6 tok/s.

The blunt summary: a 27B model on a 24 GB laptop GPU runs at **about one token
per second**, and the things that would change that by an order of magnitude —
a checkpoint that genuinely fits, or speculative decoding — are not yet in hand.

## 7. Where this beats llama.cpp, and where it does not

llama.cpp at the pinned revision **does** support this architecture -- `QWEN35`,
`QWEN35MOE` and `QWEN3NEXT` are in its arch registry and it has gated-DeltaNet
kernels. It is a real competitor here, and on a fitted 4-bit model on one
stream it is a strong one: this repository's own CPU measurements have
llama.cpp's Q4_K_M decode at 90 tok/s against our 74, and `docs/edge/` explains
that the gap is the execution model rather than the kernels.

So the case for vLLM-Omni on this machine is not raw single-stream decode. It is:

| axis | why |
|---|---|
| **Prefill** | measured ~2x llama.cpp on the CPU path (2190 vs 1121 tok/s, pp512). Prompt processing is compute-bound, which is the one regime where the hardware has headroom. |
| **Concurrency** | continuous batching and paged KV. At batch 1 they cost overhead; past ~4 streams they are the whole game, and llama.cpp has no equivalent. |
| **Speculative decoding** | the shipped MTP head is wired in vLLM and is not a GGUF concept. |
| **Multimodal pipelines** | the reason this fork exists: staged vision/audio/codec execution with an OpenAI-compatible server. |
| **Hybrid cache accounting** | 64 KiB/token instead of 256, once §4 is honoured. |

A design that claims to beat llama.cpp at batch-1 decode of a fitted model on
one GPU is claiming something this project has already measured to be false.
The defensible claim is the four rows above, and the way to establish it is
`benchmarks/edge_harness/` against `baselines/`, not assertion.

---

## 8. What was actually broken

Four faults stood between "the models are supported" and "the models run", none
of them in the model code:

1. **`RuntimeError: UVA is not available`.** vLLM disables pinned memory on WSL
   by default; its own gate is kernel ≥ 4.19.121 and this kernel is 6.18. Fixed
   by `VLLM_WSL2_ENABLE_PIN_MEMORY=1`. Without it the CUDA runner does not
   start **at all**, for any model.
2. **No C compiler**, so Triton could not JIT and the GDN kernels could not
   build. `gcc` plus `libc6-dev` (gcc alone did not pull the headers).
3. **FlashInfer's sampler wants `nvcc`**, which a runtime-only CUDA install does
   not have. `VLLM_USE_FLASHINFER_SAMPLER=0`. Note this surfaced as a failure in
   *warmup*, long after the model had loaded and run a forward pass.
4. **transformers 5 renamed `input_embeds` → `inputs_embeds`**, which broke the
   vendored Spark modeling file used as the parity *reference*. Shimmed in
   `parity_hf_vs_vllm.py` rather than by editing the cached module, so the
   reference stays the code the checkpoint ships.

The required environment, in full:

```bash
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0
sudo apt-get install -y gcc libc6-dev
```

### And one measurement error, which is the real lesson

The first roofline taken on this machine read **3.67 GB/s** H2D, **38 GB/s**
VRAM and **9 TFLOP/s**, with `nvidia-smi` reporting a 33 W power cap and a
220 MHz SM clock. The conclusion drawn from it -- that the dGPU had no
bandwidth advantage over the CPU and the design should lean on the CPU -- was
wrong in every part.

Those numbers were taken **while two model downloads were saturating the disk,
the network and the CPU**. Re-measured on an idle machine the same calls give
13.9 GB/s, 472 GB/s and 48.2 TFLOP/s, at 91 W and 1447 MHz, with the power
limit reading 100.98 W. The GPU was never capped; it was idling because the
benchmark could not feed it.

This repository already has [a methodology
document](methodology.md) that exists because of five retractions of exactly
this shape. This is the sixth, and the rule it re-teaches is the one already
written down: **a number taken on a contended machine is not a number.** The
harness has a contention guard for this; it was not in the path of an
interactive measurement, which is precisely where the mistake got made.
