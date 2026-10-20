# ncnn (Tencent/ncnn) — Engine Analysis

| Field | Value |
|---|---|
| Local path | `/data/zhoutaichang/embedding_infer/ncnn` |
| Remote | `https://github.com/Tencent/ncnn.git` (origin) |
| Branch | `master` |
| HEAD | `3b7bdba7fc8aea8fd46779533eee027df77c639d` — "Fix SIGFPE with missing or incomplete CPU topology (#6967)", 2026-09-07 |
| Dirty status | clean (`git status --porcelain` empty) |
| Submodules | **not initialized**: `glslang` (780b6a20…) and `python/pybind11` (d03662f0…) show `-` in `git submodule status` |
| Latest tag in checkout | `20260526` (README links the 20260526 release archives) |
| Analysis date | 2026-09-08 |
| Repo AGENTS.md / CLAUDE.md | none present (checked `AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`) |

Evidence labels used below: **[Verified]** read in code/docs, **[Inferred]** reasoned from structure, **[Proposal]** design suggestion, **[Unknown]** could not determine.

---

## 0. Summary

- **What it is.** ncnn is a dependency-free C++ inference *library* (not a serving stack): a static graph runtime with a `Net` (graph + weights) and `Extractor` (per-request lazy evaluator) API, hand-optimized CPU kernels (ARM NEON/fp16/bf16/dotprod/i8mm/SVE, x86 SSE2→AVX512-FP16/VNNI/BF16, RISC-V V/Zfh, MIPS MSA, LoongArch LSX/LASX) and a Vulkan compute GPU backend. [Verified] ([README](../../ncnn/README.md), [src/net.h](../../ncnn/src/net.h), [src/CMakeLists.txt](../../ncnn/src/CMakeLists.txt))
- **No serving layer.** There is no HTTP server, request queue, scheduler, or continuous batching; the "runtime" is `Net::load_param/load_model` + `Extractor::input/extract`. Request-level scheduling is entirely the application's job. [Verified] (§5)
- **Graph execution is lazy pull-based.** `Extractor::extract(blob)` recursively runs `NetPrivate::forward_layer` on the producer chain until the requested blob is materialized; unrequested branches never run. [Verified] ([src/net.cpp#L123](../../ncnn/src/net.cpp#L123), [#L2871](../../ncnn/src/net.cpp#L2871))
- **Backends in tree:** CPU (with runtime ISA dispatch) and Vulkan only. No CUDA, Metal, CoreML, OpenCL, NNAPI, QNN. Apple GPUs are reached through MoltenVK (`libMoltenVK.dylib` is dlopen'ed by the in-house loader). [Verified] ([src/simplevk.cpp#L558](../../ncnn/src/simplevk.cpp#L558), [FAQ-ncnn-vulkan.md](../../ncnn/docs/how-to-use-and-FAQ/FAQ-ncnn-vulkan.md))
- **Transformer/LLM support is real and recent (2025–2026).** Layers `MultiHeadAttention`, `SDPA` (with GQA and flash-attention style CPU/Vulkan kernels), `RotaryEmbed`, `RMSNorm`, `Embed`, `Gemm`, `GLU`, `GELU` exist with arch-specific kernels; an in-layer **KV cache** (`7=1` param, `cache_k/v_in/out` blobs, `Extractor::set_kvcache_allocator`, `set_kvcache_max_seqlen_hint`, `extract(..., type=1)`) is documented and tested. [Verified] ([docs/developer-guide/kvcache.md](../../ncnn/docs/developer-guide/kvcache.md), [src/layer/sdpa.cpp](../../ncnn/src/layer/sdpa.cpp), [src/net.h#L193](../../ncnn/src/net.h#L193))
- **LLM benchmark harness ships in-tree**: `benchncnn_llm` measures 256-token prefill and cached single-token decode for Qwen2.5-0.5B, Qwen3-0.6B, MiniCPM4-0.5B, Hunyuan-0.5B, TinyLlama-1.1B, Llama3.2-1B, Youtu-LLM-2B decoders (param files only; weights are random). [Verified] ([benchmark/benchncnn_llm.cpp](../../ncnn/benchmark/benchncnn_llm.cpp), [benchmark/CMakeLists.txt](../../ncnn/benchmark/CMakeLists.txt))
- **LLM weight quantization exists**: block-quantized Gemm/MHA (W4/W6 weight-only, W8A8 dynamic per-block on CPU) via `ncnnllm2table`/`ncnnllm2int` (minmax/mseclip/awq/gptq), gated by `NCNN_WEIGHT_QUANT`; plus classic PTQ int8 via `ncnn2table`/`ncnn2int8`. [Verified] ([docs/how-to-use-and-FAQ/quantized-int8-inference.md](../../ncnn/docs/how-to-use-and-FAQ/quantized-int8-inference.md), [tools/quantize](../../ncnn/tools/quantize))
- **TTS exists as an example, not a runtime**: `examples/piper.cpp` runs a VITS (Piper) pipeline split into 5 ncnn nets (enc_p, emb_g, dp, flow, dec) with custom layers registered from the app, a dictionary-based English phonemizer and WAV writer. `examples/whisper.cpp` runs Whisper ASR (fbank net → encoder → cached decoder with beam search). Audio ops `Spectrogram`/`InverseSpectrogram` (STFT/iSTFT) and 1D conv/deconv exist. [Verified] ([examples/piper.cpp](../../ncnn/examples/piper.cpp), [examples/whisper.cpp](../../ncnn/examples/whisper.cpp), [src/layer/spectrogram.h](../../ncnn/src/layer/spectrogram.h))
- **No tokenizer, no sampler, no vocoder library.** Whisper example carries a minimal reverse-vocab tokenizer; Piper carries a dict phonemizer. Nothing generic. [Verified]
- **VLA: nothing specific.** Vision encoders (CNN/ViT) are first-class; MLP/Gemm heads trivial; diffusion/flow policies would require the host to loop denoise steps calling one Extractor per step (no scheduler code). [Verified]/[Inferred] (§10)
- **Model format**: text `.param` + flat `.bin` (magic 7767517), produced mainly by **pnnx** (PyTorch/TorchScript/ONNX → ncnn; pip installable) with fp16 weights by default; legacy converters for Caffe/ONNX/Darknet/MXNet/Keras/TF-MLIR. [Verified] ([docs/developer-guide/param-and-model-file-structure.md](../../ncnn/docs/developer-guide/param-and-model-file-structure.md), [tools/pnnx/README.md](../../ncnn/tools/pnnx/README.md))
- **Dynamic shapes** are natural: every `Mat` carries its own dims; layers compute output shapes at forward time; param shape hints (`30=`) are optional; `Reshape` supports runtime **expressions** over input shapes. [Verified] ([docs/developer-guide/expression.md](../../ncnn/docs/developer-guide/expression.md))
- **Threading** is OpenMP-inside-kernels (`#pragma omp parallel for num_threads(opt.num_threads)`), with big.LITTLE affinity via `set_cpu_powersave`, a bundled minimal OpenMP runtime (`simpleomp`) and a bundled Vulkan loader (`simplevk`) so no external runtime deps are needed. [Verified] ([src/cpu.h](../../ncnn/src/cpu.h), [src/simpleomp.h](../../ncnn/src/simpleomp.h), [docs/developer-guide/vulkan-driver-loader.md](../../ncnn/docs/developer-guide/vulkan-driver-loader.md))
- **Vulkan shaders are GLSL embedded as source and compiled to SPIR-V at runtime** by the in-tree glslang, with a persistent `PipelineCache` (SPIR-V + driver blob) to cut startup. [Verified] ([cmake/ncnn_add_shader.cmake](../../ncnn/cmake/ncnn_add_shader.cmake), [docs/developer-guide/vulkan-pipeline-cache.md](../../ncnn/docs/developer-guide/vulkan-pipeline-cache.md))
- **Edge portability is the strongest asset**: 60 CMake toolchains (Android, iOS/tvOS/visionOS/watchOS via ios.toolchain, HarmonyOS, Jetson, QNX, ESP32, RISC-V bare-metal, HiSilicon, Loongson, MIPS, PowerPC, Windows XP, WebAssembly), 45 CI workflows, `NCNN_SIMPLESTL/SIMPLEOMP/SIMPLEOCV/SIMPLEMATH` shims, per-layer `WITH_LAYER_x` opt-out for size. [Verified] ([toolchains](../../ncnn/toolchains), [docs/how-to-use-and-FAQ/build-minimal-library.md](../../ncnn/docs/how-to-use-and-FAQ/build-minimal-library.md))
- **Batching**: `NCNN_BATCH` adds an `n` dimension to `Mat`; layers lacking `support_batch` are looped per item by the Net. KV cache supports **batch size 1 only**. [Verified] ([src/net.cpp#L662](../../ncnn/src/net.cpp#L662), [#L2882](../../ncnn/src/net.cpp#L2882))

---

## 1. Architecture overview

```mermaid
graph TD
  subgraph App["Application (C++/C/Python/JNI)"]
    A1[Mat::from_pixels / audio PCM / token ids] --> A2[Net.load_param + load_model]
    A2 --> A3[Extractor.input / extract]
  end

  subgraph Runtime["ncnn core (src/)"]
    N[Net / NetPrivate<br/>layers[], blobs[], custom registry] --> E[Extractor / ExtractorPrivate<br/>blob_mats[], blob_mats_gpu[], Option]
    E -->|lazy pull| FL[NetPrivate::forward_layer]
    FL --> CL[convert_layout<br/>fp16/bf16 cast + elempack]
    CL --> DL[do_forward_layer<br/>batch loop, lightmode inplace]
    DL --> L[Layer_final<br/>layer_cpu / layer_vulkan]
    O[Option<br/>num_threads, use_fp16_*, use_int8_*, use_bf16_storage, use_packing_layout, allocators, kvcache_*] --> FL
    P[ParamDict / ModelBin / DataReader] --> N
    AL[Allocator / PoolAllocator / UnlockedPoolAllocator] --> E
  end

  subgraph CPU["CPU backend"]
    L --> LC[layer/*.cpp naive] 
    L --> LA[layer/arm|x86|riscv|mips|loongarch<br/>runtime ISA dispatch]
    CPUI[cpu.cpp: cpu_support_*, big.LITTLE, affinity, omp] --> LA
  end

  subgraph GPU["Vulkan backend (NCNN_VULKAN)"]
    L --> LV[layer/vulkan/*_vulkan.cpp]
    LV --> PL[Pipeline / PipelineCache<br/>glslang GLSL->SPIR-V at runtime]
    LV --> CMD[VkCompute / VkTransfer<br/>record_upload/download/pipeline, submit_and_wait]
    CMD --> VD[VulkanDevice / GpuInfo<br/>queues, VkAllocator pools]
    VD --> SVK[simplevk loader<br/>dlopen libvulkan / MoltenVK / ICD]
  end

  subgraph Tools["Offline tools"]
    T1[pnnx: PyTorch/ONNX -> .ncnn.param/.bin] --> A2
    T2[ncnnoptimize: fuse/eliminate, fp16 weights] --> A2
    T3[ncnn2table/ncnn2int8, ncnnllm2table/ncnnllm2int] --> A2
  end
```

**Key modules** [Verified]

| Module | Path | Responsibility | Key symbols |
|---|---|---|---|
| Graph container | [src/net.h](../../ncnn/src/net.h), [src/net.cpp](../../ncnn/src/net.cpp) | Parse param, create layers (overwrite → vulkan → cpu → custom), load weights, create pipelines, upload to GPU, own local pool allocators | `Net`, `NetPrivate`, `Net::load_param` (L1306), `Net::load_model` (L2022), `Net::register_custom_layer` (L1215), `create_extractor` |
| Per-request executor | [src/net.cpp#L2676](../../ncnn/src/net.cpp#L2676) | Holds `blob_mats` (+GPU), per-extractor `Option`; lazy DAG evaluation; unpack/cast outputs | `Extractor`, `ExtractorPrivate`, `Extractor::extract` (L2871), `NetPrivate::forward_layer` (L123 CPU, L192 Vulkan), `convert_layout` (L358), `do_forward_layer` (L633) |
| Layer base + registry | [src/layer.h](../../ncnn/src/layer.h), [src/layer.cpp](../../ncnn/src/layer.cpp) | Virtual op interface, capability flags, factory tables | `Layer`, `support_*` flags (L45–90), `forward/forward_inplace`, `Layer_final` (L201), `layer_registry`, `create_layer_cpu/naive/vulkan`, `DEFINE_LAYER_CREATOR` (L199) |
| Tensor | [src/mat.h](../../ncnn/src/mat.h) | N-D tensor with `elemsize/elempack/cstep/n`; pixel conversion | `Mat` (L50), `VkMat` (L389), `VkImageMat` (L559), `from_pixels*` (L293–307), `substract_mean_normalize` (L335), `batch()` (L239) |
| Options | [src/option.h](../../ncnn/src/option.h), [src/option.cpp](../../ncnn/src/option.cpp) | All runtime knobs incl. precision, allocators, kv cache | `Option` |
| Allocators | [src/allocator.h](../../ncnn/src/allocator.h), [src/allocator.cpp](../../ncnn/src/allocator.cpp) | Aligned malloc, host pools, Vulkan buffer/image pools | `fastMalloc` (L56), `Allocator` (L142), `PoolAllocator` (L151), `UnlockedPoolAllocator` (L180), `VkAllocator` (L267), `VkBlobAllocator`, `VkWeightAllocator`, `VkStagingAllocator`, `VkAndroidHardwareBufferImageAllocator` (L405) |
| CPU topology | [src/cpu.h](../../ncnn/src/cpu.h), [src/cpu.cpp](../../ncnn/src/cpu.cpp) | ISA feature detection (ruapu), core counts, big.LITTLE, affinity, OpenMP wrappers, denormals | `cpu_support_arm_*`, `cpu_support_x86_*`, `get_big_cpu_count`, `set_cpu_powersave`, `set_cpu_thread_affinity`, `set_kmp_blocktime`, `set_flush_denormals` |
| Vulkan device | [src/gpu.h](../../ncnn/src/gpu.h), [src/gpu.cpp](../../ncnn/src/gpu.cpp) | Instance/device creation, capability query, queue + allocator pools, shader compile | `create_gpu_instance` (gpu.cpp L2919), `GpuInfo`, `VulkanDevice`, `get_gpu_device`, `compile_spirv_module` |
| Vulkan commands | [src/command.h](../../ncnn/src/command.h), [src/command.cpp](../../ncnn/src/command.cpp) | Command buffer recording/submission, staging copies | `VkCompute::record_upload/record_download/record_pipeline/submit_and_wait`, `VkTransfer` |
| Vulkan pipelines | [src/pipeline.h](../../ncnn/src/pipeline.h), [src/pipelinecache.h](../../ncnn/src/pipelinecache.h) | Shader module/pipeline objects, on-disk cache | `Pipeline`, `PipelineCache::save_cache/load_cache/get_pipeline` |
| Vulkan loader | [src/simplevk.h](../../ncnn/src/simplevk.h), [src/simplevk.cpp](../../ncnn/src/simplevk.cpp) | In-house loader (no libvulkan link) | `load_vulkan_driver` |
| Param/weights I/O | [src/paramdict.h](../../ncnn/src/paramdict.h), [src/modelbin.h](../../ncnn/src/modelbin.h), [src/datareader.h](../../ncnn/src/datareader.h) | Key=value params, weight blobs with type flags, stream abstraction | `ParamDict`, `ModelBin`, `DataReader` |
| Operator impls | [src/layer](../../ncnn/src/layer) (110 registered types), [src/layer/arm](../../ncnn/src/layer/arm), [src/layer/x86](../../ncnn/src/layer/x86), [src/layer/vulkan](../../ncnn/src/layer/vulkan), [src/layer/riscv](../../ncnn/src/layer/riscv), [src/layer/mips](../../ncnn/src/layer/mips), [src/layer/loongarch](../../ncnn/src/layer/loongarch) | Naive reference + arch kernels + GLSL shaders | e.g. `SDPA`, `MultiHeadAttention`, `Gemm`, `Convolution` |
| Shims | [src/simpleomp.h](../../ncnn/src/simpleomp.h), [src/simplestl.h](../../ncnn/src/simplestl.h), [src/simpleocv.h](../../ncnn/src/simpleocv.h), [src/simplemath.h](../../ncnn/src/simplemath.h) | Minimal OpenMP runtime, STL, OpenCV-like image I/O, libm | — |
| C API | [src/c_api.h](../../ncnn/src/c_api.h) | Flat C bindings incl. kv cache options | `ncnn_net_*`, `ncnn_extractor_*`, `ncnn_option_set_kvcache_allocator` |
| Python | [python/src/main.cpp](../../ncnn/python/src/main.cpp) | pybind11 bindings for Net/Extractor/Mat/Option/Layer/Vulkan | `py::class_<Extractor>` (L959) |
| Converters | [tools/pnnx](../../ncnn/tools/pnnx), [tools/onnx](../../ncnn/tools/onnx), [tools/ncnnoptimize.cpp](../../ncnn/tools/ncnnoptimize.cpp), [tools/quantize](../../ncnn/tools/quantize) | Model export, offline fusion, quantization | `pnnx`, `onnx2ncnn`, `ncnnoptimize`, `ncnn2table`, `ncnn2int8`, `ncnnllm2table`, `ncnnllm2int` |
| Build glue | [cmake/ncnn_add_layer.cmake](../../ncnn/cmake/ncnn_add_layer.cmake), [cmake/ncnn_add_shader.cmake](../../ncnn/cmake/ncnn_add_shader.cmake), [src/CMakeLists.txt](../../ncnn/src/CMakeLists.txt) | Layer registry generation, ISA-variant source generation, shader embedding | `ncnn_add_layer`, `ncnn_add_arch_opt_layer`, `ncnn_add_shader` |

---

## 2. Device layer

### Backends actually in tree [Verified]
- **CPU**: naive C++ in `src/layer/*.cpp` plus arch dirs: arm (56 `*_arm.cpp` + ISA variants `_asimdhp/_asimddp/_asimdfhm/_bf16/_i8mm/_vfpv4`), x86 (59 base + `_avx2/_fma/_fma4/_xop/_f16c/_avx512*/_avxvnni*/_avxneconvert`), riscv (52), mips (56), loongarch (56). Counts from `ls src/layer/<arch>/*_<arch>.cpp`.
- **Vulkan GPU**: 64 `*_vulkan.cpp` layer implementations and 300 `.comp` shaders under [src/layer/vulkan/shader](../../ncnn/src/layer/vulkan/shader). Enabled by `NCNN_VULKAN` (default **OFF**) ([CMakeLists.txt#L82](../../ncnn/CMakeLists.txt#L82)).
- **Not present**: CUDA, Metal, CoreML, OpenCL, NNAPI, QNN/Hexagon, TensorRT. A grep for those names over `src/` only hits Vulkan loader/portability code (`VK_KHR_portability_subset`, MoltenVK dlopen). The FAQ explicitly states Vulkan was chosen over CUDA/OpenCL/Metal for portability ([FAQ-ncnn-vulkan.md](../../ncnn/docs/how-to-use-and-FAQ/FAQ-ncnn-vulkan.md), "why using vulkan over cuda/opencl/metal").

### Device discovery / selection [Verified]
- CPU: `get_cpu_count / get_big_cpu_count / get_little_cpu_count / get_physical_*` ([src/cpu.h#L124](../../ncnn/src/cpu.h#L124)); big.LITTLE is derived from sysfs `cpufreq` max frequency (`get_max_freq_khz`, [src/cpu.cpp#L1486](../../ncnn/src/cpu.cpp#L1486)) and topology files (`thread_siblings`, [src/cpu.cpp#L961](../../ncnn/src/cpu.cpp#L961)). ISA features: `cpu_support_arm_asimdhp/asimddp/bf16/i8mm/sve*`, `cpu_support_x86_avx2/avx512_*`, `cpu_support_riscv_v/zfh/zvfh`, `cpu_riscv_vlenb` via bundled [src/ruapu.h](../../ncnn/src/ruapu.h). `Option::num_threads` defaults to `get_physical_big_cpu_count()` ([src/option.cpp](../../ncnn/src/option.cpp)).
- Vulkan: lazy singleton `create_gpu_instance()` under a mutex ([src/gpu.cpp#L2919](../../ncnn/src/gpu.cpp#L2919)); loader order: env `VK_ICD_FILENAMES`/`NCNN_VULKAN_DRIVER` → system `libvulkan.so/.dylib/vulkan-1.dll` → direct driver file search (`libvulkan_radeon.so`, `libMoltenVK.dylib`, Android HAL `HMI`) ([src/simplevk.cpp#L287](../../ncnn/src/simplevk.cpp#L287), [#L558](../../ncnn/src/simplevk.cpp#L558), [vulkan-driver-loader.md](../../ncnn/docs/developer-guide/vulkan-driver-loader.md)). Each physical device gets a `GpuInfo` with a `rough_score` (device type, subgroup size, fp16/int8/coopmat support, memory) used to choose `get_default_gpu_index()` ([src/gpu.cpp#L1297](../../ncnn/src/gpu.cpp#L1297), [#L3624](../../ncnn/src/gpu.cpp#L3624)). `Net::set_vulkan_device(int|VulkanDevice*)` or `opt.vulkan_device_index` selects a device before `load_param` ([src/net.cpp#L1338](../../ncnn/src/net.cpp#L1338)). If no valid device, `opt.use_vulkan_compute` is silently set to false (CPU fallback) ([src/net.cpp#L1348](../../ncnn/src/net.cpp#L1348)).
- Capability sanitization: at `load_param`, unsupported `use_fp16_packed/storage/arithmetic`, `use_int8_*`, `use_bf16_*`, `use_cooperative_matrix`, `use_subgroup_ops` are cleared from the Net option based on `GpuInfo` ([src/net.cpp#L1350–1372](../../ncnn/src/net.cpp#L1350)). `GpuInfo` exposes ~50 extension flags including `VK_KHR_cooperative_matrix`, `VK_NV_cooperative_matrix2`, `VK_KHR_shader_bfloat16`, `VK_EXT_shader_float8`, `VK_ANDROID_external_memory_android_hardware_buffer` ([src/gpu.h#L327–378](../../ncnn/src/gpu.h#L327)).

### Memory allocation / pools [Verified]
- Host: `fastMalloc` = aligned alloc with `NCNN_MALLOC_ALIGN` and 64-byte over-read pad ([src/allocator.h#L56](../../ncnn/src/allocator.h#L56)). `PoolAllocator` (mutex) / `UnlockedPoolAllocator` keep `budgets`/`payouts` lists; reuse if a free block is ≥ size and within `size_compare_ratio`; drop when > `size_drop_threshold` (default 10) ([src/allocator.cpp#L98](../../ncnn/src/allocator.cpp#L98)). Option roles: `blob_allocator` (named blobs, lifetime = Extractor), `workspace_allocator` (per-layer scratch), `kvcache_allocator` (must differ from blob allocator, [src/net.cpp#L2876](../../ncnn/src/net.cpp#L2876)). With `use_local_pool_allocator` (default true) the Net lazily creates its own pools at `load_model` ([src/net.cpp#L2136](../../ncnn/src/net.cpp#L2136)) and clones outputs out of them on `extract` so the Net can be destroyed early ([src/net.cpp#L3060](../../ncnn/src/net.cpp#L3060)).
- Device: `VkAllocator` family: `VkBlobAllocator` (activations), `VkWeightAllocator` (device-local weights; `use_weights_in_host_memory` flips to host-visible), `VkStagingAllocator`/`VkWeightStagingAllocator` (host-visible staging), with `flush/invalidate` for non-coherent memory ([src/allocator.h#L267–395](../../ncnn/src/allocator.h#L267)). `VulkanDevice::acquire_blob_allocator/acquire_staging_allocator` hand out pooled allocators; an Extractor acquires them on first `extract` if not set ([src/net.cpp#L2915](../../ncnn/src/net.cpp#L2915)). Memory type choice via `find_memory_index(required, preferred, preferred_not)` and `is_mappable/is_coherent/is_device_local` ([src/gpu.h#L465](../../ncnn/src/gpu.h#L465)). Android `AHardwareBuffer` zero-copy import allocator exists (API ≥ 26) ([src/allocator.h#L405](../../ncnn/src/allocator.h#L405), [use-ncnn-with-android-hardware-buffer.md](../../ncnn/docs/how-to-use-and-FAQ/use-ncnn-with-android-hardware-buffer.md)).
- Model weights: `Net::load_model(const unsigned char*)` references external memory without copying; `opt.use_mapped_model_loading` mmaps the `.bin` on Linux/Android/OHOS/Windows/Apple ([src/net.cpp#L2261](../../ncnn/src/net.cpp#L2261)).

### Host↔device transfers and synchronization [Verified]
- All GPU work is recorded into a `VkCompute` command buffer: `record_upload` (staging copy + optional fp32→fp16 cast + packing), `record_download`, `record_clone`, `record_pipeline(pipeline, bindings, push constants, dispatcher)`, then `submit_and_wait()` (fence wait) ([src/command.h#L22–87](../../ncnn/src/command.h#L22)). Weights are uploaded once at `load_model` via `VkTransfer`, committed every 256 MB ([src/net.cpp#L2115](../../ncnn/src/net.cpp#L2115)).
- Inside a Vulkan extract, the Net records the whole reachable subgraph into one command buffer; it force-submits when a CPU-only layer needs a downloaded input, or when `pending_dispatch_total()` exceeds a threshold scaled by `rough_score` (32 KB … 8 MB "for avoiding driver timeout") ([src/net.cpp#L250–283](../../ncnn/src/net.cpp#L250)). Final output is downloaded and waited on in `Extractor::extract` ([src/net.cpp#L2940](../../ncnn/src/net.cpp#L2940)). Barriers are inserted automatically by `barrier_readwrite` bookkeeping ([src/command.h#L82](../../ncnn/src/command.h#L82)).
- Queues: `VulkanDevice::acquire_queue/reclaim_queue` pool per family; `GpuInfo::compute_queue_count()` is documented as the practical max number of concurrent extractors ([vulkan-notes.md](../../ncnn/docs/how-to-use-and-FAQ/vulkan-notes.md), "batch inference"). Zero-copy `VkMat::mapped()` on unified-memory devices and cross-net `VkMat` chaining through a shared `VkCompute` are documented there too.
- Timestamps/profiling: `NCNN_BENCHMARK` build records per-layer GPU timestamps via query pool ([src/net.cpp#L2952](../../ncnn/src/net.cpp#L2952)).

### Portability [Verified]
- OS: Linux, Android (NDK ≥ r18b; Vulkan needs API 24 with system loader, or lower with simplevk), iOS/macOS/Catalyst/tvOS/visionOS/watchOS, Windows (incl. XP), HarmonyOS (`__OHOS__`), QNX, WebAssembly (Emscripten), bare-metal RISC-V/ESP32 — evidenced by [toolchains](../../ncnn/toolchains) and [.github/workflows](../../ncnn/.github/workflows) (android.yml, ios.yml, harmonyos.yml, web-assembly.yml, esp32.yml, elf-riscv64.yml, windows-arm.yml, linux-ppc64.yml, linux-loongarch64.yml …).
- Vulkan platform matrix (repo doc): Intel/AMD/NVIDIA on Windows/Linux, Qualcomm/ARM Mali on Android, Apple on mac/iOS via MoltenVK ([vulkan-notes.md](../../ncnn/docs/how-to-use-and-FAQ/vulkan-notes.md)). Jetson: aarch64 CPU + Tegra Vulkan driver ([docs/how-to-build/how-to-build.md#L66](../../ncnn/docs/how-to-build/how-to-build.md#L66), [toolchains/jetson.toolchain.cmake](../../ncnn/toolchains/jetson.toolchain.cmake)).

---

## 3. Kernel layer

### Operator set & registration [Verified]
- 110 layer types registered via `ncnn_add_layer(Class)` in [src/CMakeLists.txt](../../ncnn/src/CMakeLists.txt) (two marked OFF by default: `ArgMax`, `SPP`). Full list with params in [docs/developer-guide/operators.md](../../ncnn/docs/developer-guide/operators.md) and [operation-param-weight-table.md](../../ncnn/docs/developer-guide/operation-param-weight-table.md). Transformer/audio-relevant: `MultiHeadAttention`, `SDPA`, `RotaryEmbed`, `RMSNorm`, `LayerNorm`, `GroupNorm`, `Embed`, `Gemm`, `MatMul`, `Einsum`, `GLU`, `GELU`, `Softmax`, `Convolution1D/DepthWise1D/Deconvolution1D`, `Pooling1D`, `Spectrogram`, `InverseSpectrogram`, `LSTM/GRU/RNN`, `GridSample`, `DeformableConv2D`, `Interp`, `PixelShuffle`, `CumulativeSum`.
- Registration is **build-time code generation**: `ncnn_add_layer` (macro at [cmake/ncnn_add_layer.cmake#L82](../../ncnn/cmake/ncnn_add_layer.cmake#L82)) appends `#include` + `DEFINE_LAYER_CREATOR(Class)` lines into generated `layer_declaration.h` and a `layer_registry` table per ISA level (`layer_registry_arm82`, `_avx512`, …) into `layer_registry.h` (templates: [src/layer_declaration.h.in](../../ncnn/src/layer_declaration.h.in), [src/layer_registry.h.in](../../ncnn/src/layer_registry.h.in), [src/layer_type_enum.h.in](../../ncnn/src/layer_type_enum.h.in)). ISA variants are produced by rewriting the base arch source with `cmake/ncnn_generate_<opt>_source.cmake` (renaming the class, e.g. `Convolution_arm_asimdhp`) and compiling with the ISA flags ([cmake/ncnn_add_layer.cmake#L2](../../ncnn/cmake/ncnn_add_layer.cmake#L2)). `WITH_LAYER_<name>=OFF` removes a layer and its kernels (registry gets a null creator).
- `layer_to_index(name)` is a linear strcmp scan ([src/layer.cpp#L150](../../ncnn/src/layer.cpp#L150)); `create_layer_cpu(index)` picks the best registry table at runtime based on `cpu_support_*` when `NCNN_RUNTIME_CPU=ON` (see generated `layer_registry.h` consumers in [src/layer.cpp](../../ncnn/src/layer.cpp) after L400) [Verified for mechanism; exact selection code lives in the generated portion].

### Dispatch mechanism (op → backend impl) [Verified]
1. `Net::load_param` creates each layer as: overwrite-builtin registry → `create_layer_vulkan` (if `use_vulkan_compute` and device) → `create_layer_cpu` → custom registry; missing type is a hard error ([src/net.cpp#L1400–1420](../../ncnn/src/net.cpp#L1400)).
2. `create_layer_vulkan/cpu` return a `Layer_final` wrapper holding `layer_cpu` and optionally `layer_vulkan`; `load_param/create_pipeline` try the Vulkan layer first and **delete it and fall back to CPU** if `support_vulkan` becomes false after `load_param` (layer-specific param not supported on GPU) ([src/layer.cpp#L280–344](../../ncnn/src/layer.cpp#L280)). The Net additionally recreates a CPU layer when the Vulkan variant rejects the params or the layer's `featmask` disables Vulkan ([src/net.cpp#L1536](../../ncnn/src/net.cpp#L1536)).
3. At forward time, `Layer_final::forward(Mat…)` → `layer_cpu`, `forward(VkMat…)` → `layer_vulkan` ([src/layer.cpp#L359–400](../../ncnn/src/layer.cpp#L359)). The Vulkan `forward_layer` inspects `layer->support_vulkan` per layer: GPU layers get `record_upload` of any host-only inputs; CPU-only layers get `record_download` + forced `submit_and_wait`, then run on host ([src/net.cpp#L215–248](../../ncnn/src/net.cpp#L215)). Mixed CPU/GPU graphs therefore work automatically at the cost of syncs.
4. Inside an arch kernel, further dispatch is by data type: e.g. `Clip_arm::forward_inplace` checks `elembits()==16 && opt.use_fp16_storage` → fp16 path, bf16 path, else fp32 ([layer-support-behavior.md](../../ncnn/docs/developer-guide/layer-support-behavior.md)). `SDPA_arm` declares `support_fp16_storage = cpu_support_arm_asimdhp()` and `support_bf16_storage = true` ([src/layer/arm/sdpa_arm.cpp#L14](../../ncnn/src/layer/arm/sdpa_arm.cpp#L14)).
5. Layout negotiation is done by the Net, not the kernel: `convert_layout` casts fp32→fp16/bf16 when `layer->support_fp16_storage/support_bf16_storage` and the option is on, and re-packs to the arch-optimal `elempack` (4 NEON, 8 AVX, 16 AVX-512, `vlenb/4` RVV) unless `support_any_packing` ([src/net.cpp#L358–470](../../ncnn/src/net.cpp#L358); [element-packing.md](../../ncnn/docs/developer-guide/element-packing.md)). KV-cache blobs skip layout conversion ([src/net.cpp#L364](../../ncnn/src/net.cpp#L364)).
6. Per-layer overrides: `31=<featmask>` bits disable fp16 arithmetic/storage, bf16, int8, vulkan, sgemm, winograd, threading for that layer ([docs/developer-guide/layer-feat-mask.md](../../ncnn/docs/developer-guide/layer-feat-mask.md), [src/net.cpp#L100](../../ncnn/src/net.cpp#L100)).

### Quantization formats & where dequant happens [Verified]
- **Weight storage flags** in `.bin`: `0` fp32, `0x01306B47` fp16, `0x01348B83` bf16, other = quantized int8 with table ([param-and-model-file-structure.md](../../ncnn/docs/developer-guide/param-and-model-file-structure.md)). pnnx writes fp16 weights by default (`fp16=1`, [tools/pnnx/src/save_ncnn.cpp#L240](../../ncnn/tools/pnnx/src/save_ncnn.cpp#L240)); `ModelBin` converts to the layer's requested type at load.
- **Activation precision**: fp16 storage/arithmetic on ARMv8.2 (`asimdhp`), VFPv4 fp16 storage on ARMv7, bf16 storage on any ARM/x86 (`NCNN_BF16`), fp16/bf16/int8 packed+storage+arithmetic on Vulkan, chosen by `Option` flags and clamped by device caps. Casting happens in `convert_layout` at layer boundaries and in `Extractor::extract` back to fp32 unless `type=1` ([src/net.cpp#L2985–3055](../../ncnn/src/net.cpp#L2985)).
- **PTQ int8** (`NCNN_INT8`): `ncnn2table` (KL/ACIQ, from images or `.npy`, also static weight scales for RNN/GRU/LSTM/MHA/Embed) → `ncnn2int8` rewrites weights and adds `8=`/`18=` int8 scale params; `Quantize/Dequantize/Requantize` layers exist and conv/innerproduct/gemm/mha/sdpa/embed have `forward_int8` paths; dequant to fp32 happens inside those layers or at `extract` (int8 output cast, [src/net.cpp#L3043](../../ncnn/src/net.cpp#L3043)). x86 int8 uses AVX-VNNI/AVX512-VNNI/AVX-VNNI-INT8 kernels; ARM uses dotprod/i8mm ([src/layer/arm/convolution_arm_i8mm.cpp](../../ncnn/src/layer/arm/convolution_arm_i8mm.cpp), [src/layer/x86/convolution_x86_avx512vnni.cpp](../../ncnn/src/layer/x86/convolution_x86_avx512vnni.cpp)).
- **LLM block quant** (`NCNN_WEIGHT_QUANT`, default ON): `Gemm`/`MultiHeadAttention` `quantize_term = bits*100 + input_scale*10 + block_code` (bits 4/6/8, block 32/64/128); W4/W6 are weight-only with fp32 activations, W8 is dynamic W8A8 per-block int32 accumulate, CPU only; when enabled the layer turns off packing/fp16/bf16/Vulkan (`support_vulkan=false`) ([src/layer/multiheadattention.cpp#L84–100](../../ncnn/src/layer/multiheadattention.cpp#L84), [src/layer/gemm.h](../../ncnn/src/layer/gemm.h), [quantized-int8-inference.md](../../ncnn/docs/how-to-use-and-FAQ/quantized-int8-inference.md)). Tools: [tools/quantize/ncnnllm2table.cpp](../../ncnn/tools/quantize/ncnnllm2table.cpp), [tools/quantize/ncnnllm2int.cpp](../../ncnn/tools/quantize/ncnnllm2int.cpp); test [tests/test_gemm_block_quant.cpp](../../ncnn/tests/test_gemm_block_quant.cpp).

### Compilation / JIT / graph capture [Verified]
- CPU: no JIT; all kernels are AOT C++/intrinsics/inline asm, selected at runtime by ISA table. Winograd/sgemm/packing variants are selected in each layer's `create_pipeline` based on shape hints and `Option` (e.g. `use_winograd23/43/63_convolution`).
- Vulkan: GLSL `.comp` sources are preprocessed into hex headers at build time ([cmake/ncnn_add_shader.cmake](../../ncnn/cmake/ncnn_add_shader.cmake)) and **compiled to SPIR-V at runtime** by in-tree glslang with option-dependent macros (`compile_spirv_module`, [src/gpu.h#L572](../../ncnn/src/gpu.h#L572); glslang linked privately, [src/CMakeLists.txt#L341](../../ncnn/src/CMakeLists.txt#L341)). Specialization constants + push constants parameterize shapes; `PipelineCache` deduplicates pipelines in-process and persists SPIR-V plus driver `VkPipelineCache` blobs to a file ([docs/developer-guide/vulkan-pipeline-cache.md](../../ncnn/docs/developer-guide/vulkan-pipeline-cache.md)). No command-buffer replay/graph capture across extracts: each `extract` records a fresh `VkCompute`.
- No runtime graph optimization: fusions (conv+bn, conv+activation, etc.) are done offline by `ncnnoptimize` ([tools/ncnnoptimize.cpp#L42–77](../../ncnn/tools/ncnnoptimize.cpp#L42)) or pnnx (`optlevel`). `grep fuse src/net.cpp` finds nothing.

### Backend coverage gaps (computed from file presence) [Verified]
Computed by checking, for each of the 110 `ncnn_add_layer(...)` entries in [src/CMakeLists.txt](../../ncnn/src/CMakeLists.txt), whether `src/layer/vulkan/<name>_vulkan.cpp` / `src/layer/arm/<name>_arm.cpp` exist.

| Backend | Layers **without** a dedicated kernel (run naive C++ on CPU, or CPU fallback for Vulkan) |
|---|---|
| Vulkan (64/110 covered) | ArgMax, Bias, BNLL, **Embed**, Exp, Input, Log, MVN, Power, Proposal, ROIPooling, SPP, Threshold, Tile, **RNN, LSTM, GRU**, Squeeze, ExpandDims, DetectionOutput, YoloDetectionOutput, Yolov3DetectionOutput, PSROIPooling, ROIAlign, StatisticsPooling, **Pooling1D, ConvolutionDepthWise1D, Deconvolution1D, DeconvolutionDepthWise1D**, Convolution3D, ConvolutionDepthWise3D, Pooling3D, Deconvolution3D, DeconvolutionDepthWise3D, **MatMul, Einsum, GLU**, DeformableConv2D, Fold, **GridSample**, CumulativeSum, CopyTo, Diag, **Spectrogram, InverseSpectrogram**, Flip |
| ARM NEON (56/110 covered) | ArgMax, BNLL, **Embed**, Exp, Input, Log, MemoryData, MVN, Power, Proposal, Reduction, ROIPooling, Split, SPP, Threshold, Tile, Squeeze, ExpandDims, Normalize, Permute, PriorBox, DetectionOutput, Reorg, Yolo*DetectionOutput, PSROIPooling, ROIAlign, Noop, DeepCopy, StatisticsPooling, Softplus, **Pooling1D, ConvolutionDepthWise1D, Deconvolution1D, DeconvolutionDepthWise1D**, Convolution3D/DepthWise3D, Pooling3D, Deconvolution3D/DepthWise3D, **Einsum, GLU**, DeformableConv2D, Fold, Unfold, **GridSample**, CumulativeSum, CopyTo, Diag, CELU, Shrink, **Spectrogram, InverseSpectrogram**, Flip |

Implications for TTS/VLA: `Embed`, `GLU`, depthwise/transposed 1-D convs (common in vocoders), `Spectrogram`, and RNNs always run on CPU — on a Vulkan graph each such layer forces a download + `submit_and_wait` ([src/net.cpp#L237](../../ncnn/src/net.cpp#L237)). `SDPA`, `MultiHeadAttention`, `Gemm`, `RMSNorm`, `LayerNorm`, `RotaryEmbed`, `Convolution1D`, `Deconvolution` are covered on both ARM and Vulkan. [Verified]

### Precision / performance option matrix [Verified]
Defaults from [src/option.cpp](../../ncnn/src/option.cpp); effects from [src/net.cpp](../../ncnn/src/net.cpp) and layer `support_*` checks.

| Option | Default | CPU effect | Vulkan effect |
|---|---|---|---|
| `use_fp16_storage` / `use_fp16_arithmetic` / `use_fp16_packed` | true | ARMv8.2 (`asimdhp`), ARMv7 VFPv4 storage, RISC-V Zfh/Zvfh: activations cast to fp16 between layers that `support_fp16_storage` | fp16 SSBO storage, fp16 math, packed fp16x4; cleared if device lacks support |
| `use_bf16_storage` | false | Any ARM/x86/LoongArch with `NCNN_BF16`: bf16 activations (takes precedence over fp16 when set, [src/net.cpp#L400](../../ncnn/src/net.cpp#L400)) | `use_bf16_packed/storage` need `VK_KHR_shader_bfloat16` |
| `use_int8_inference` (+`use_int8_*`) | true | Runs `forward_int8` on layers carrying int8 scales | int8 storage/arithmetic shaders where present |
| `use_packing_layout` | true | elempack 4/8/16 SIMD layout negotiated per layer | elempack 4 (`support_vulkan_packing`) |
| `use_winograd_convolution` / `use_sgemm_convolution` (+`winograd23/43/63`) | true | Conv algorithm choice at `create_pipeline` (more memory) | Winograd/GEMM shaders incl. cooperative-matrix `_cm` variants |
| `use_cooperative_matrix`, `use_subgroup_ops`, `use_shader_local_memory` | true | — | Tensor-core style matmul (`VK_KHR_cooperative_matrix`/NV), subgroup reductions, shared memory (discrete GPUs only) |
| `lightmode` | true | Free intermediates early, inplace when refcount==1 | same |
| `use_local_pool_allocator` | true | Net-owned pools when app passes none | — |
| `num_threads`, `openmp_blocktime` (20 ms), `flush_denormals` (3=DAZ+FTZ) | big-core count | OpenMP width, spin time, denormal handling around each extract | CPU-side upload/cast still threaded |
| `kvcache_allocator`, `kvcache_max_seqlen_hint`, `kvcache_vkallocator` | null / 0 | KV cache placement and initial reservation | device-resident cache |
| `use_mapped_model_loading`, `use_weights_in_host_memory` | false | mmap `.bin` | keep GPU weights in host-visible memory (iGPU/UMA) |
| `vulkan_device_index` | -1 (auto by `rough_score`) | — | device choice |

### Fallback when op unsupported [Verified]
- Unknown layer type in param → `load_param` fails ("layer %s not exists or registered", [src/net.cpp#L1417](../../ncnn/src/net.cpp#L1417)); the app must register it via `register_custom_layer` first (Piper example does this).
- Vulkan-unsupported layer/param → transparent CPU execution with upload/download around it ([src/net.cpp#L215](../../ncnn/src/net.cpp#L215); [FAQ-ncnn-vulkan.md](../../ncnn/docs/how-to-use-and-FAQ/FAQ-ncnn-vulkan.md) "layers without vulkan support").
- Arch kernel missing → naive `src/layer/<name>.cpp` is compiled and registered in its place ([cmake/ncnn_add_layer.cmake#L39](../../ncnn/cmake/ncnn_add_layer.cmake#L39)).
- Precision unsupported by a layer → Net converts inputs to fp32/elempack 1 for it (`support_*` false).

### Extension points [Verified]
- Custom op: subclass `ncnn::Layer`, implement `load_param/load_model/forward(_inplace)`, set `one_blob_only/support_inplace/support_packing…`, then `net.register_custom_layer("Type", creator, destroyer, userdata)` (string or index) — tutorial in [how-to-implement-custom-layer-step-by-step.md](../../ncnn/docs/developer-guide/how-to-implement-custom-layer-step-by-step.md), [add-custom-layer.zh.md](../../ncnn/docs/developer-guide/add-custom-layer.zh.md); pnnx exports unknown modules as `moduleop` so names round-trip ([tools/pnnx/README.md](../../ncnn/tools/pnnx/README.md)). Python can register custom layers too ([python/src/main.cpp#L1046](../../ncnn/python/src/main.cpp#L1046)).
- Overwriting a builtin op with your own implementation uses the same API (`create_overwrite_builtin_layer`, [src/net.cpp#L1215](../../ncnn/src/net.cpp#L1215)).
- Custom Vulkan op: add shader, `upload_model`, `create_pipeline`, `forward(VkMat…)` ([vulkan-notes.md](../../ncnn/docs/how-to-use-and-FAQ/vulkan-notes.md) "add vulkan compute support to layer"; GLSL helper macros in [docs/developer-guide/glsl-extension.md](../../ncnn/docs/developer-guide/glsl-extension.md)).
- Custom allocator: subclass `Allocator`/`VkAllocator` ([custom-allocator.md](../../ncnn/docs/developer-guide/custom-allocator.md)).
- Low-level op API: create/forward a single layer without a Net ([low-level-operation-api.md](../../ncnn/docs/developer-guide/low-level-operation-api.md)).
- New backend (e.g. NPU): **no plugin interface**; you would add a third `layer_xxx` slot to `Layer_final` and a third registry, mirroring how Vulkan is wired in [src/layer.cpp#L201](../../ncnn/src/layer.cpp#L201) and [src/net.cpp](../../ncnn/src/net.cpp). [Inferred]

---

## 4. Model runner

### Formats & conversion pipeline [Verified]
- Runtime format: `model.param` (text: magic `7767517`, `layer_count blob_count`, then `Type name in out blobs... k=v`) + `model.bin` (concatenated 32-bit-aligned weight blobs with optional type flag) ([param-and-model-file-structure.md](../../ncnn/docs/developer-guide/param-and-model-file-structure.md)). Binary param (`.param.bin`) + `ncnn2mem` header for string-less builds ([tools/ncnn2mem.cpp](../../ncnn/tools/ncnn2mem.cpp)); loading table in [ncnn-load-model.md](../../ncnn/docs/how-to-use-and-FAQ/ncnn-load-model.md).
- Recommended path: PyTorch → `pip install pnnx` → `pnnx.export(model, "m.pt", (x,))` → `m.ncnn.param/bin` + `m_ncnn.py` ([README](../../ncnn/README.md)). pnnx also loads ONNX and TNN ([tools/pnnx/src/load_onnx.cpp](../../ncnn/tools/pnnx/src/load_onnx.cpp), [load_tnn.cpp](../../ncnn/tools/pnnx/src/load_tnn.cpp)); 253 ncnn lowering passes in [tools/pnnx/src/pass_ncnn](../../ncnn/tools/pnnx/src/pass_ncnn) including `F_scaled_dot_product_attention.cpp`, `nn_MultiheadAttention.cpp`, `nn_RMSNorm.cpp`, `fuse_convert_rotaryembed.cpp`, `torch_stft.cpp`, `torchaudio_F_spectrogram.cpp`. CLI options `inputshape`, `inputshape2` (dynamic dim discovery), `fp16`, `optlevel`, `moduleop`, `customop` ([tools/pnnx/README.md#L85](../../ncnn/tools/pnnx/README.md#L85)).
- Legacy converters: [tools/onnx/onnx2ncnn.cpp](../../ncnn/tools/onnx/onnx2ncnn.cpp), [tools/caffe](../../ncnn/tools/caffe), [tools/darknet](../../ncnn/tools/darknet), [tools/mxnet](../../ncnn/tools/mxnet), [tools/keras](../../ncnn/tools/keras), [tools/tensorflow](../../ncnn/tools/tensorflow), [tools/mlir](../../ncnn/tools/mlir). Post-processing: [tools/ncnnoptimize.cpp](../../ncnn/tools/ncnnoptimize.cpp) (fusion, dead-op removal, fp16/fp32 storage choice), [tools/ncnnmerge.cpp](../../ncnn/tools/ncnnmerge.cpp).
- LLM decoders: export with sequence length 1, then **patch the param** to add `cache_k_in/v_in/k_out/v_out` blobs + `7=1` on every MHA/SDPA and set `Gemm 7=0` (dynamic M) — reference Python in [kvcache.md](../../ncnn/docs/developer-guide/kvcache.md); `benchncnn_llm` further rewrites `Gemm 18=<quantize_term>` in memory for block-quant variants ([benchmark/benchncnn_llm.cpp#L72](../../ncnn/benchmark/benchncnn_llm.cpp#L72)).

### Execution lifecycle [Verified]
1. **Init**: `Net net; net.opt.* = …; net.load_param(path|mem|AAsset|DataReader); net.load_model(...)`. `load_param` creates layers and resolves blob shape hints; `load_model` reads weights, calls `Layer::create_pipeline(opt)` (weight repacking, winograd transform, GPU pipeline creation) and `upload_model` for Vulkan ([src/net.cpp#L2022–2170](../../ncnn/src/net.cpp#L2022)). Option changes must precede loading (comments in [src/option.h](../../ncnn/src/option.h)).
2. **Warmup**: no explicit API; first `extract` triggers pool allocator growth and (Vulkan) SPIR-V compile/pipeline creation unless a `PipelineCache` file is preloaded. `benchncnn` uses 8 warmup loops by convention ([benchmark/benchncnn_llm.cpp#L42](../../ncnn/benchmark/benchncnn_llm.cpp#L42)).
3. **Step**: `Extractor ex = net.create_extractor(); ex.input(name|idx, Mat); ex.extract(name|idx, out[, type])`. `Extractor` copies `net.opt` at creation; `set_light_mode`, `set_blob_allocator`, `set_workspace_allocator`, `set_kvcache_allocator`, `set_kvcache_max_seqlen_hint`, `set_*_vkallocator` per extractor ([src/net.h#L167–256](../../ncnn/src/net.h#L167)). Light mode (default) releases intermediate blobs as soon as consumed and does inplace ops when the refcount is 1 ([src/net.cpp#L644](../../ncnn/src/net.cpp#L644)). Extract also sets OpenMP blocktime and denormal flags around the run ([src/net.cpp#L2896](../../ncnn/src/net.cpp#L2896)).
4. **Teardown**: `Extractor::clear`, `Net::clear` (destroys pipelines, frees allocators), `ncnn::destroy_gpu_instance()`.

### Dynamic shapes [Verified]
- Tensors carry shape at runtime; layers derive output shapes in `forward`. Param shape hints (`30=` on Input/any layer) only feed `create_pipeline` heuristics ([src/net.cpp#L1488](../../ncnn/src/net.cpp#L1488)); they are not enforced.
- `Reshape 6="expr"` and `Slice` accept expressions over input dims (`0w`, `1c`, `+`, `*`, `max`, …) resolved on the host at forward time, avoiding shape-op subgraphs ([docs/developer-guide/expression.md](../../ncnn/docs/developer-guide/expression.md), [src/expression.h](../../ncnn/src/expression.h)).
- Gemm `7=` constant-M flag must be 0 for variable sequence length; MHA/SDPA accept arbitrary `seqlen` (`Mat::h`). `Input` layers accept any size; Piper feeds a variable-length phoneme id vector and gets variable-length audio ([examples/piper.cpp#L529](../../ncnn/examples/piper.cpp#L529)).
- Batch: `Mat::n` (NCNN_BATCH ON by default); the Net loops layers that lack `support_batch` per batch element and validates consistent `n` across inputs ([src/net.cpp#L662–712](../../ncnn/src/net.cpp#L662), [#L789](../../ncnn/src/net.cpp#L789)). Test: [tests/test_mat_batch.cpp](../../ncnn/tests/test_mat_batch.cpp).

### State & KV cache [Verified]
- **Allocation & layout**: cache is a `Mat(head_dim, seqlen, num_kv_head)` whose `cstep` reserves extra sequence capacity; `Mat::h` is the logical length. `SDPA::kvcache_capacity` reserves `max_seqlen_hint` on first allocation, else `max(16, …)` up to +256, then grows geometrically by ×1.5; `create_or_grow_kvcache` reuses in place if `new_seqlen <= capacity` and the cache came from the kv allocator, otherwise allocates and copies valid history per head ([src/layer/sdpa.cpp#L28–92](../../ncnn/src/layer/sdpa.cpp#L28); same in [src/layer/multiheadattention.cpp#L230](../../ncnn/src/layer/multiheadattention.cpp#L230)). Layout is backend-private (head-contiguous on CPU, device-resident `VkMat` on Vulkan); Vulkan uses `sdpa_kvcache_append.comp` / `sdpa_kvcache_copy.comp` shaders ([src/layer/vulkan/shader/sdpa_kvcache_append.comp](../../ncnn/src/layer/vulkan/shader/sdpa_kvcache_append.comp)).
- **Paging / eviction**: none. The cache is a contiguous per-layer buffer; no paged attention, no sliding window, no eviction policy. Growth = realloc + memcpy. [Verified]
- **Reuse across calls**: consume-and-replace: `ex.input(cache_in_idx, cache); cache.release(); ex.extract(cache_out_idx, cache, 1)`; `type=1` preserves fp16/bf16/packing/allocator/capacity. A session-owned `UnlockedPoolAllocator` with `size_compare_ratio=0` (or `VkAllocator`) must be set on every Extractor of the session and differ from the blob allocator; shallow `Mat` copies are not snapshots, so beam search must clone ([kvcache.md](../../ncnn/docs/developer-guide/kvcache.md), [src/net.cpp#L2876](../../ncnn/src/net.cpp#L2876)). Whisper's beam search keeps a `std::vector<ncnn::Mat> kvcache` per beam ([examples/whisper.cpp#L291](../../ncnn/examples/whisper.cpp#L291)).
- **Cross-attention**: MHA with separate q and k/v inputs reuses a static cache without appending (encoder–decoder), SDPA always appends ([kvcache.md](../../ncnn/docs/developer-guide/kvcache.md) §2).
- **Attention mask / RoPE** are model inputs: `benchncnn_llm` builds a causal `attention_mask (dst_seqlen × cur_seqlen)` with −10000 and `cos/sin` caches on the host per step ([benchmark/benchncnn_llm.cpp#L285–309](../../ncnn/benchmark/benchncnn_llm.cpp#L285)); `RotaryEmbed` takes `(x, cos_cache, sin_cache)` ([src/layer/rotaryembed.cpp#L19](../../ncnn/src/layer/rotaryembed.cpp#L19)).
- **Other recurrent state**: LSTM/GRU/RNN take/return hidden-state blobs; RVM example threads 4 recurrent tensors between frames ([examples/rvm.cpp#L197](../../ncnn/examples/rvm.cpp#L197)).
- Tests: [tests/test_sdpa_kvcache.cpp](../../ncnn/tests/test_sdpa_kvcache.cpp), [tests/test_sdpa_kvcache_session.cpp](../../ncnn/tests/test_sdpa_kvcache_session.cpp), [tests/test_multiheadattention_kvcache.cpp](../../ncnn/tests/test_multiheadattention_kvcache.cpp), [tests/test_multiheadattention_kvcache_allocator.cpp](../../ncnn/tests/test_multiheadattention_kvcache_allocator.cpp); perf: [tests/perf/perf_sdpa_kvcache.cpp](../../ncnn/tests/perf/perf_sdpa_kvcache.cpp), [perf_sdpa_decode.cpp](../../ncnn/tests/perf/perf_sdpa_decode.cpp), [perf_sdpa_prefill.cpp](../../ncnn/tests/perf/perf_sdpa_prefill.cpp).

### Multimodal stages [Verified]
- ncnn has no notion of stages; each stage is a separate `Net`, glued by host code. Whisper: `fbank` net (waveform → log-mel, uses Spectrogram-based graph [Inferred]) → `encoder` → `embed_token` + `embed_position` → `decoder` (self-attn with dynamic cache, cross-attn with static encoder k/v) → `proj_out` ([examples/whisper.cpp#L310–318](../../ncnn/examples/whisper.cpp#L310)). Piper: `enc_p` → `emb_g` (speaker) → `dp` (duration predictor with rational-quadratic spline custom layer) → host `path_attention` (length regulation + noise) → `flow` → `dec` (HiFi-GAN-style vocoder inside VITS) → PCM ([examples/piper.cpp#L518–675](../../ncnn/examples/piper.cpp#L518)).
- Vision: `Mat::from_pixels_resize` + `substract_mean_normalize` → detector net → host NMS/decode (e.g. [examples/yolo11.cpp#L352](../../ncnn/examples/yolo11.cpp#L352)); Vulkan zero-copy camera path via AHB import ([docs/how-to-use-and-FAQ/use-ncnn-with-android-hardware-buffer.md](../../ncnn/docs/how-to-use-and-FAQ/use-ncnn-with-android-hardware-buffer.md)).
- Diffusion: no sampler/scheduler code in tree. [Verified by absence: grep for "diffusion|ddim|scheduler" in src/examples returns nothing relevant]

---

## 5. Scheduler

### (a) Request / token / pipeline-level scheduling — **N/A** [Verified]
- There is no request queue, no batch former, no continuous batching, no prefill/decode scheduler, no multi-stage pipeline orchestrator. The unit of work is one synchronous `Extractor::extract` call on the calling thread. Prefill vs decode is purely the host choosing `seqlen>1` vs `seqlen=1` inputs against the same Net ([benchmark/benchncnn_llm.cpp#L406–469](../../ncnn/benchmark/benchncnn_llm.cpp#L406), [kvcache.md](../../ncnn/docs/developer-guide/kvcache.md) §5).
- Concurrency model: a `Net` is immutable after load and can be shared by many threads, each creating its own `Extractor` (documented pattern with per-thread `UnlockedPoolAllocator` + shared locked workspace allocator, [custom-allocator.md](../../ncnn/docs/developer-guide/custom-allocator.md); Vulkan "batch inference" with `omp parallel for` over extractors bounded by `compute_queue_count()`, [vulkan-notes.md](../../ncnn/docs/how-to-use-and-FAQ/vulkan-notes.md)). An `Extractor` itself is single-threaded state (`blob_mats` vector) and not safe to share. [Inferred]
- Cancellation/abort: none; a running `extract` cannot be interrupted. Backpressure: none (host's job). Timeouts: only the Vulkan chunked-submit heuristic to avoid GPU driver watchdogs ([src/net.cpp#L250](../../ncnn/src/net.cpp#L250)).
- Batching across requests: `NCNN_BATCH` allows `n>1` in a single extract but the KV cache rejects `n>1` ([src/net.cpp#L2882](../../ncnn/src/net.cpp#L2882)), so LLM decode is strictly one sequence per extract.

### (b) Graph / operator / thread-level scheduling [Verified]
- **Op ordering**: depth-first recursion from the requested blob to its producers (`forward_layer` recursion over `layer->bottoms`), i.e. topological order induced by param order; memoized in `blob_mats` so shared subgraphs run once per Extractor ([src/net.cpp#L123–150](../../ncnn/src/net.cpp#L123)). Multiple `extract` calls on one Extractor reuse computed blobs (e.g. Piper extracts `out0/out1/out2` from one run).
- **Intra-op threading**: OpenMP `parallel for` inside kernels with `num_threads(opt.num_threads)`; per-layer `featmask` bit 7 forces single thread; `set_omp_num_threads`, `set_kmp_blocktime` (default 20 ms spin), `set_cpu_powersave(0|1|2)` binds to all/little/big cores (Android/Linux), `set_cpu_thread_affinity(CpuSet)` ([src/cpu.h#L145–170](../../ncnn/src/cpu.h#L145); [openmp-best-practice.md](../../ncnn/docs/how-to-use-and-FAQ/openmp-best-practice.md)). `NCNN_SIMPLEOMP` provides a minimal LLVM-ABI OpenMP runtime supporting only `parallel for num_threads` ([src/simpleomp.h](../../ncnn/src/simpleomp.h)); `NCNN_THREADS=OFF` builds single-threaded (WASM CI does this).
- **Inter-op parallelism**: none on CPU (sequential layer execution). On Vulkan, layers are recorded back-to-back into one command buffer with automatic barriers; no multi-queue/multi-stream overlap inside one extract. Overlap is achievable only by running several Extractors on several threads/queues.
- **Stream assignment**: one `VkCompute` (compute queue) per extract; `VkTransfer` (transfer queue if `unified_compute_transfer_queue()` false) for weight upload ([src/gpu.h#L257–264](../../ncnn/src/gpu.h#L257)).
- **Memory scheduling**: light mode frees blobs after last use; pool allocators recycle; no static memory planning.
- **A53/A55 tuning**: `use_a53_a55_optimized_kernel` auto-set from the current thread's core ([src/option.cpp](../../ncnn/src/option.cpp), [docs/developer-guide/arm-a53-a55-dual-issue.md](../../ncnn/docs/developer-guide/arm-a53-a55-dual-issue.md)).

---

## 6. I/O layer

- **Image preprocessing** [Verified]: `Mat::from_pixels / from_pixels_resize / from_pixels_roi / from_pixels_roi_resize` with `PIXEL_RGB/BGR/GRAY/RGBA/BGRA` and `X2Y` conversions, stride support, `to_pixels*`, `substract_mean_normalize`, `from_android_bitmap*` ([src/mat.h#L255–335](../../ncnn/src/mat.h#L255)); resize/rotate/affine/drawing helpers ([src/mat_pixel_resize.cpp](../../ncnn/src/mat_pixel_resize.cpp), [src/mat_pixel_rotate.cpp](../../ncnn/src/mat_pixel_rotate.cpp), [src/mat_pixel_affine.cpp](../../ncnn/src/mat_pixel_affine.cpp), [src/mat_pixel_drawing.cpp](../../ncnn/src/mat_pixel_drawing.cpp); doc [efficient-roi-resize-rotate.md](../../ncnn/docs/how-to-use-and-FAQ/efficient-roi-resize-rotate.md)). `NCNN_SIMPLEOCV` supplies `cv::Mat/imread/imwrite` stand-ins ([src/simpleocv.h](../../ncnn/src/simpleocv.h)) backed by [src/stb_image.h](../../ncnn/src/stb_image.h). Vulkan YCbCr camera import shader [src/convert_ycbcr.comp](../../ncnn/src/convert_ycbcr.comp).
- **Audio preprocessing** [Verified]: no audio I/O library. `Spectrogram` (n_fft, hop, win, hann/hamming, center/pad, power, onesided, normalized) and `InverseSpectrogram` layers implement STFT/iSTFT in-graph with a naive DFT (cos/sin loops) ([src/layer/spectrogram.cpp](../../ncnn/src/layer/spectrogram.cpp), [src/layer/inversespectrogram.cpp](../../ncnn/src/layer/inversespectrogram.cpp)); Whisper reads 16-bit WAV by hand (`load_wav_samples`, [examples/whisper.cpp#L829](../../ncnn/examples/whisper.cpp#L829)) and Piper writes a WAV header by hand ([examples/piper.cpp#L677](../../ncnn/examples/piper.cpp#L677)).
- **Tokenization** [Verified]: none in the library. Whisper example: plain reverse-vocab list (`whisper_vocab.txt`) with hard-coded special-token ids ([examples/whisper.cpp#L31–200](../../ncnn/examples/whisper.cpp#L31)); Piper: binary word→phoneme-id dictionary + rule-based sentence splitting (`simple_phonemize`, [examples/piper.cpp#L323](../../ncnn/examples/piper.cpp#L323)); `benchncnn_llm` feeds random embeddings (no tokenizer). `Embed` layer maps ids→vectors in-graph ([src/layer/embed.h](../../ncnn/src/layer/embed.h)).
- **Streaming** [Verified]: no streaming API. Token streaming falls out naturally from the per-step decode loop (the app owns the loop). Audio streaming/chunking is not provided; Piper synthesizes the whole utterance then normalizes/clips to int16 ([examples/piper.cpp#L646](../../ncnn/examples/piper.cpp#L646)). Chunked/streaming vocoding would need the model to be exported in a chunkable form. [Inferred]
- **Buffering** [Verified]: `Mat` refcounted buffers; `Extractor::input` stores a shallow reference (no copy) ([src/net.cpp#L2861](../../ncnn/src/net.cpp#L2861)); external memory `Mat(w,h,c,data)` constructors allow zero-copy input.
- **Postprocessing** [Verified]: host code in examples (argmax/greedy in whisper `run_decoder_*`, beam search with log-softmax, NMS in detectors). No sampling library (temperature/top-p) in tree.
- **Application boundaries** [Verified]:
  - C++ API: [src/net.h](../../ncnn/src/net.h), [src/mat.h](../../ncnn/src/mat.h), [src/option.h](../../ncnn/src/option.h).
  - C API (`NCNN_C_API`, default ON): [src/c_api.h](../../ncnn/src/c_api.h) incl. `ncnn_mat_create_*_batch`, `ncnn_option_set_kvcache_*`, `ncnn_pipelinecache_*`.
  - Python (`pip install ncnn`, pybind11): `ncnn.Net/Extractor/Mat/Option/Layer/PoolAllocator/VulkanDevice…`, `extract(name, type)` returning `(ret, Mat)`, kv-cache setters ([python/src/main.cpp#L959](../../ncnn/python/src/main.cpp#L959), [python/README.md](../../ncnn/python/README.md)); a model zoo of detection/pose/classification wrappers ([python/ncnn/model_zoo](../../ncnn/python/ncnn/model_zoo)).
  - Android: `AAssetManager` loaders ([src/net.h#L110](../../ncnn/src/net.h#L110)), `Mat::from_android_bitmap`, AHB Vulkan import; the JNI demo project moved out of tree ([examples/squeezencnn/README.md](../../ncnn/examples/squeezencnn/README.md)).
  - iOS: `Info.plist` + `ios.toolchain.cmake`; MoltenVK for GPU.
  - HTTP server / CLI serving: none. Only benchmark/example binaries ([benchmark/benchncnn.cpp](../../ncnn/benchmark/benchncnn.cpp)).

---

## 7. Execution flow

### (a) Text LLM decode with KV cache (as in `benchncnn_llm` / kvcache.md) [Verified]

```mermaid
sequenceDiagram
  participant App
  participant Net as Net(decoder)
  participant Ex as Extractor
  participant NP as NetPrivate::forward_layer
  participant SDPA as SDPA layer (per block)
  participant Gemm as Gemm/RMSNorm/RoPE layers
  participant Proj as Net(proj_out)

  App->>Net: load_param(decoder.param patched 7=1, Gemm 7=0) ; load_model(bin)
  Note over Net: create_pipeline per layer, (Vulkan: upload weights, compile SPIR-V)
  App->>App: prefill: tokens -> embeds Mat(hidden, seqlen), causal mask, cos/sin cache
  App->>Ex: create_extractor(); set_kvcache_allocator(pool); set_kvcache_max_seqlen_hint(N)
  App->>Ex: input(in0..in3), input(cache_k_in_i, empty) for all i
  App->>Ex: extract(cache_k_out_i, cache[i], type=1) for all i
  Ex->>NP: forward_layer(producer of cache_k_out_i)
  NP->>Gemm: RMSNorm -> Gemm(q,k,v) -> RotaryEmbed
  NP->>SDPA: forward(q,k,v,mask,past_k,past_v)
  SDPA->>SDPA: create_or_grow_kvcache (reserve hint) ; append ; flash-attn QK^T softmax V
  SDPA-->>NP: out, cache_k_out, cache_v_out (private layout)
  NP-->>Ex: blob_mats filled (memoized)
  App->>Ex: extract("out0", hidden)  (already computed for last block, no re-run)
  App->>Proj: Extractor.input(last row).extract("out0", logits)
  App->>App: argmax/sample -> next token
  loop decode step
    App->>Ex: new Extractor; input(embed of 1 token, mask (past+1 x 1), cos/sin row)
    App->>Ex: input(cache_k_in_i, cache[i]); cache[i].release()
    App->>Ex: extract(cache_k_out_i, cache[i], 1) ; extract("out0")
    Note over SDPA: cache reused in place when seqlen <= capacity, else grow x1.5 + memcpy
    App->>Proj: logits -> next token
  end
```

### (b) TTS / ASR path — Whisper ASR (encoder–decoder with cross-attention cache) and Piper TTS [Verified]

```mermaid
sequenceDiagram
  participant App as whisper.cpp
  participant FB as Net(fbank)
  participant Enc as Net(encoder)
  participant Emb as Net(embed_token/position)
  participant Dec as Net(decoder, MHA 7=1)
  participant PO as Net(proj_out)

  App->>App: load_wav_samples(16-bit PCM) -> Mat waveform
  App->>FB: Extractor.input("in0", waveform).extract("out0", input_features)
  Note over FB: log-mel via Spectrogram-based graph [Inferred]
  App->>Enc: input(input_features).extract("out0", encoder_states)
  App->>App: prompt ids (SOT, lang, task, no_timestamps)
  App->>Emb: ids -> token_embeds ; positions -> position_embeds ; sum
  App->>Dec: input(in0=embeds, in1=encoder_states, in2=attention_mask)
  App->>Dec: extract(cache_k/v_out_i, kv[i], type=1) for all MHA ; extract("out0")
  Note over Dec: self-attn MHA appends to dynamic cache; cross-attn MHA uses static k/v from encoder_states
  App->>PO: last state -> logits ; log_softmax ; beam search candidates keep own kv vector
  loop until EOT
    App->>Dec: input(1 token embeds, encoder_states, mask, kv[i]) ; extract(kv_out, "out0")
    App->>PO: logits -> beam update
  end
  App->>App: detokenize via reverse vocab -> text

  Note over App: Piper TTS (examples/piper.cpp): text -> simple_phonemize (dict) -> Net enc_p (custom relative_embeddings layers) -> Net emb_g(speaker) -> Net dp (custom spline layer, noise) -> host path_attention (durations, noise) -> Net flow -> Net dec (vocoder) -> normalize -> int16 PCM -> WAV 22050 Hz
```

---

## 8. Tests and examples

- **Unit tests** [Verified]: 181 `tests/test_*.cpp` (per-layer randomized tests comparing naive vs optimized vs Vulkan via [tests/testutil.h](../../ncnn/tests/testutil.h); `_oom` variants exercise allocation failure). Build with `-DNCNN_BUILD_TESTS=ON`; each becomes a CTest via [cmake/run_test.cmake](../../ncnn/cmake/run_test.cmake) ([tests/CMakeLists.txt](../../ncnn/tests/CMakeLists.txt)). Non-layer tests: `test_c_api`, `test_cpu`, `test_expression`, `test_modelbin`, `test_paramdict`, `test_mat_batch`, `test_mat_pixel*`, `test_squeezenet`, and with Vulkan `test_command`, `test_pipeline_cache`. Transformer-specific: `test_sdpa*`, `test_multiheadattention*` (incl. `_kvcache`, `_block_quant`), `test_rotaryembed`, `test_rmsnorm`, `test_gemm_*` (incl. `_block_quant`, `_nt_int8`), `test_embed`, `test_spectrogram`, `test_inversespectrogram`, `test_glu`, `test_lstm/gru`.
- **How to run (per repo docs/CI)** [Verified]:
  ```shell
  # library + tests (CPU)
  mkdir build && cd build
  cmake -DNCNN_BUILD_TESTS=ON ..            # add -DNCNN_VULKAN=ON for GPU tests (needs glslang submodule)
  cmake --build . -j $(nproc)
  ctest --output-on-failure -j $(nproc)     # exact invocation used by CI: .github/workflows/linux-x64-cpu-gcc.yml
  # model benchmark (CPU, 4 big-core threads, 4 loops, cooldown) / GPU device 0
  ./benchmark/benchncnn 4 4 2 -1 1
  ./benchmark/benchncnn 8 1 2 0 0
  # LLM prefill/decode tokens-per-second table
  ./benchmark/benchncnn_llm [loop count] [num threads] [powersave] [gpu device] [cooling down]
  # examples (need OpenCV or -DNCNN_SIMPLEOCV=ON) — piper needs en_*.ncnn.param/bin + en-word_id.bin in cwd
  ./examples/piper "Hello World" 0 out.wav
  ./examples/whisper <wav> ...              # needs whisper_vocab.txt and exported nets
  ```
  Sources: [docs/how-to-build/how-to-build.md#L40](../../ncnn/docs/how-to-build/how-to-build.md#L40), [.github/workflows/linux-x64-cpu-gcc.yml#L53](../../ncnn/.github/workflows/linux-x64-cpu-gcc.yml#L53), [benchmark/README.md](../../ncnn/benchmark/README.md), [examples/piper.cpp#L718](../../ncnn/examples/piper.cpp#L718). Note this checkout cannot build Vulkan or Python until `git submodule update --init` (glslang, pybind11).
- **Perf micro-benchmarks** [Verified]: [tests/perf](../../ncnn/tests/perf) (`perf_sdpa_prefill/decode/kvcache`, conv, gemm-ish ops).
- **Model benchmarks** [Verified]: `benchncnn` (36 built-in params incl. `vision_transformer.param`, int8 variants; usage `./benchncnn [loops] [threads] [powersave] [gpu] [cooldown] param=… shape=…`) and `benchncnn_llm` (7 LLM decoders, prints prefill/decode tokens/s; `[loop] [threads] [powersave] [gpu] [cooldown]`) ([benchmark/README.md](../../ncnn/benchmark/README.md), [benchmark/benchncnn_llm.cpp#L643](../../ncnn/benchmark/benchncnn_llm.cpp#L643)). README carries repo-reported timing tables for many boards (Jetson AGX Orin/Orin Nano/Nano/TX2 NX, RK3588, Snapdragon 8xx/X Elite, Raspberry Pi 3/4/5, Apple A7, Loongson…) — repo-reported, not reproduced.
- **Examples** [Verified] ([examples/CMakeLists.txt](../../ncnn/examples/CMakeLists.txt)): vision — squeezenet (+C API), yolov2…yolo11 (det/seg/pose/obb/cls), yoloworld, yolox, nanodet, scrfd, retinaface, arcface, simplepose, yolact, rvm (video matting w/ recurrent state), ppocrv5 (OCR), p2pnet, mobilenet/squeezenet SSD; audio — [piper.cpp](../../ncnn/examples/piper.cpp) (TTS), [whisper.cpp](../../ncnn/examples/whisper.cpp) (ASR). Python examples mirror detection models ([python/examples](../../ncnn/python/examples)). Mobile app demos are external (nihui/ncnn-android-*), only a README remains in [examples/squeezencnn](../../ncnn/examples/squeezencnn).
- **CI** [Verified]: 45 workflows in [.github/workflows](../../ncnn/.github/workflows): per-OS/arch CPU & GPU (Linux x64 GPU via SwiftShader/lavapipe [Inferred]), Android (armeabi-v7a/arm64-v8a/x86/x86_64/riscv64, Vulkan ON), iOS/macOS/Catalyst/tvOS/visionOS/watchOS, HarmonyOS, WebAssembly (3 variants incl. simpleomp threads), ESP32, RISC-V ELF, Windows (MSVC/clang/mingw/XP/ARM), Intel SDE for AVX-512 variants, code-format, CodeQL, coverage, `compare-binary-size` (reports `libncnn.so` delta for x86_64/armhf/aarch64 on PRs), pnnx and python wheel builds.

---

## 9. TTS relevance

**What exists** [Verified]
- End-to-end TTS example: Piper/VITS split into 5 nets with 3 app-registered custom layers, speaker embedding, duration predictor, flow, and neural vocoder (inside `dec`), 22.05 kHz int16 output; Vulkan enabled per net (`opt.use_vulkan_compute = true`) ([examples/piper.cpp](../../ncnn/examples/piper.cpp)). Export recipe references nihui/ncnn-android-piper `export_ncnn.py` (external).
- ASR counterpart with the full encoder–decoder + KV-cache machinery (Whisper tiny…large-v3-turbo) ([examples/whisper.cpp](../../ncnn/examples/whisper.cpp)).
- Audio ops: `Spectrogram`/`InverseSpectrogram` (STFT/iSTFT; pnnx maps `torch.stft/istft`, `torchaudio.functional.spectrogram/inverse_spectrogram` — [tools/pnnx/src/pass_ncnn/torch_stft.cpp](../../ncnn/tools/pnnx/src/pass_ncnn/torch_stft.cpp), [torchaudio_F_inverse_spectrogram.cpp](../../ncnn/tools/pnnx/src/pass_ncnn/torchaudio_F_inverse_spectrogram.cpp)), `Convolution1D/ConvolutionDepthWise1D/Deconvolution1D/DeconvolutionDepthWise1D` (with ARM fp16 and Vulkan kernels: [src/layer/arm/convolution1d_arm_asimdhp.cpp](../../ncnn/src/layer/arm/convolution1d_arm_asimdhp.cpp), [src/layer/vulkan/convolution1d_vulkan.cpp](../../ncnn/src/layer/vulkan/convolution1d_vulkan.cpp)), `Pooling1D`, `LSTM/GRU`, `LayerNorm/GroupNorm/InstanceNorm`, `Gemm/MatMul`, `MultiHeadAttention/SDPA` with cache — sufficient for VITS, HiFi-GAN/Vocos-style vocoders, Tacotron-like RNNs and transformer TTS decoders.
- Transformer-decoder TTS (e.g. token-based LLM-TTS such as CosyVoice/Fish-Speech-style AR heads) can reuse the LLM decode loop with `SDPA 7=1`; speech-token → mel/flow-matching and vocoder stages would be additional nets. [Inferred]

**What is missing** [Verified unless noted]
- No tokenizer/G2P/phonemizer library (only an English dictionary hack in the example; comment says "works for english only", [examples/piper.cpp#L325](../../ncnn/examples/piper.cpp#L325)).
- No audio I/O, resampling, mel filterbank utility, or vocoder library; no streaming/chunked synthesis API; no FFT acceleration (Spectrogram is a naive DFT [Verified from cos/sin loops], likely slow for long inputs [Inferred]).
- No sampler (temperature/top-k/top-p/repetition penalty) or beam search utility outside the Whisper example.
- No neural audio codec (EnCodec/SNAC/DAC) implementations or examples; codebook lookup can be done with `Embed` [Inferred].
- No batched multi-utterance decode (KV cache is batch-1).
- Piper export requires patches to the upstream repo and an external script (not in tree).

---

## 10. VLA relevance

**What exists** [Verified]
- Vision encoders: CNN backbones are the historical core (winograd/sgemm/int8/fp16 conv on ARM/x86/Vulkan); ViT is exercised by `benchmark/models/vision_transformer.param` and `MultiHeadAttention/LayerNorm/GELU/Gemm` kernels ([benchmark/models/vision_transformer.param](../../ncnn/benchmark/models/vision_transformer.param)); `Interp`, `GridSample`, `DeformableConv2D`, `ROIAlign` for spatial ops. Camera zero-copy on Android via AHB.
- Structured / proprioceptive inputs: any number of `Input` blobs (1-D `Mat`), `Concat`, `Gemm/InnerProduct`, `Embed` for discrete state; expression-based reshape for variable shapes.
- Action heads: MLP (`InnerProduct`/`Gemm` + activations), Gaussian/mixture heads (host post-process), autoregressive token heads via the LLM path (`Embed` → decoder with `SDPA 7=1` → `Gemm` proj).
- Diffusion/flow policy building blocks: `Convolution1D`/`GroupNorm`/`Mish`/`SiLU`(Swish) (Diffusion Policy U-Net 1D), transformer blocks, `Embed` for timestep tables, `RotaryEmbed`. Each denoise step = one `Extractor` call with `(x_t, t_embed, cond)` inputs; host implements the DDIM/flow-matching update. [Inferred]
- Low-latency loop support: light mode + pooled allocators, `set_cpu_powersave(2)` big-core pinning, `openmp_blocktime` tuning, Vulkan `PipelineCache` to avoid first-run shader compilation, mmap model loading, `VkMat` chaining across nets without downloads ([vulkan-notes.md](../../ncnn/docs/how-to-use-and-FAQ/vulkan-notes.md)).

**What is missing** [Verified unless noted]
- No VLA/robotics example, no diffusion-policy or flow-matching sampler, no action chunking utilities.
- No multi-input scheduling: vision encoder and language decoder are separate Nets executed serially by the host; no async overlap primitive beyond threads + separate Extractors [Inferred].
- No paged/shared KV cache across parallel action samples (batch-1 cache); sampling K action candidates needs K cache clones.
- No NPU/DSP delegates (Hexagon/QNN, Ethos, RKNPU, Jetson DLA/CUDA); GPU only via Vulkan, so Jetson runs CPU or Tegra-Vulkan, not TensorRT.
- Vision-language projector/token merging (e.g. SigLIP+MLP) must be exported as ordinary Gemm graphs; multimodal embedding concatenation is host code [Inferred].

---

## 11. Edge deployment profile

- **Build system** [Verified]: CMake ≥ 2.8.12 (3.10 policies), C++03/11 compatible core (`host.gcc-c++03.toolchain.cmake` exists), no runtime third-party deps. Key options ([CMakeLists.txt#L61–110](../../ncnn/CMakeLists.txt#L61)): `NCNN_VULKAN` (OFF), `NCNN_SIMPLEVK` (ON), `NCNN_SYSTEM_GLSLANG` (OFF), `NCNN_OPENMP` (ON), `NCNN_SIMPLEOMP/SIMPLESTL/SIMPLEOCV/SIMPLEMATH` (OFF), `NCNN_THREADS` (ON), `NCNN_RUNTIME_CPU` (ON), `NCNN_INT8` (ON), `NCNN_WEIGHT_QUANT` (ON), `NCNN_BF16` (ON), `NCNN_BATCH` (ON), `NCNN_C_API` (ON), `NCNN_PLATFORM_API` (ON), `NCNN_PIXEL*` (ON), `NCNN_STDIO/STRING` (ON), `NCNN_SHARED_LIB` (OFF), `NCNN_DISABLE_RTTI/EXCEPTION` (ON when tools/examples off), `NCNN_BUILD_TOOLS/EXAMPLES/BENCHMARK/TESTS`, `NCNN_PYTHON`, per-ISA `NCNN_ARM82/ARM82DOT/ARM84BF16/ARM84I8MM/ARM86SVE*`, `NCNN_AVX2/AVX512*/AVXVNNI*`, `NCNN_RVV/ZFH/ZVFH/XTHEADVECTOR`, `NCNN_MSA/LSX/LASX`, `WITH_LAYER_<name>`.
- **Binary size** [Verified/Unknown]: the repo does not publish absolute sizes; CI `compare-binary-size` computes `libncnn.so` deltas per PR for x86_64/armhf/aarch64 ([.github/workflows/compare-binary-size.yml](../../ncnn/.github/workflows/compare-binary-size.yml)). The size cheatsheet ([build-minimal-library.md](../../ncnn/docs/how-to-use-and-FAQ/build-minimal-library.md)) lists reductions: disable RTTI/exceptions, Vulkan, STDIO/STRING, BF16/INT8/WEIGHT_QUANT, pixel functions, OpenMP, unused ISA kernels, runtime dispatch, unused layers, and STL (`simplestl`, `-nodefaultlibs -fno-builtin -nostdinc++`). Absolute numbers: [Unknown].
- **Dependencies** [Verified]: none at runtime. Optional: OpenMP runtime (or simpleomp), Vulkan driver (loaded dynamically; no libvulkan link with simplevk), glslang (submodule, statically linked when `NCNN_VULKAN=ON`; not initialized in this checkout), OpenCV or simpleocv for examples/tools, protobuf for legacy converters, libtorch for building pnnx from source (pip wheel avoids it).
- **Platforms** [Verified]:
  - Embedded NVIDIA/Jetson: aarch64 toolchain + Tegra Vulkan ([how-to-build.md#L66](../../ncnn/docs/how-to-build/how-to-build.md#L66)); benchmark tables for AGX Orin/Orin Nano/Nano/TX2 NX (repo-reported). No CUDA/TensorRT.
  - ARM CPU: ARMv7 (NEON, VFPv4 fp16), ARMv8/8.2/8.4/8.6 (fp16, dotprod, fhm, bf16, i8mm, SVE/SVE2 flags), A53/A55 dual-issue tuning, big.LITTLE affinity; toolchains for Cortex-A cross builds, HiSilicon Hi35xx, Rockchip, Allwinner, Raspberry Pi ([toolchains](../../ncnn/toolchains)).
  - Apple: macOS/iOS/tvOS/visionOS/watchOS builds via [toolchains/ios.toolchain.cmake](../../ncnn/toolchains/ios.toolchain.cmake) and CI; GPU only through MoltenVK (Vulkan→Metal), no native Metal/CoreML/ANE.
  - Android: NDK builds for all ABIs incl. riscv64, Vulkan on Adreno/Mali (API 24+ with system loader; simplevk allows <24), `AAssetManager` loading, AHB zero-copy (API 26+), JNI bitmap helpers. Prebuilt release archives are built with `-DANDROID_PLATFORM=android-19/21 -DNCNN_VULKAN=ON` ([build-android.cmd](../../ncnn/build-android.cmd)). No NNAPI/QNN.
  - HarmonyOS (`ohos.toolchain.cmake`, `__OHOS__` mmap path), QNX, Windows XP/ARM, WebAssembly (single-thread or simpleomp+pthreads, SIMD128 via SSE2 path), ESP32/RISC-V bare metal (`riscv64-unknown-elf`, XuanTie C906/C907/C908/C910, SpacemiT K1), LoongArch, MIPS, PowerPC.
- **Model conversion path & constraints** [Verified]: PyTorch → pnnx (needs `inputshape` for shape propagation; `inputshape2` to mark dynamic dims; unsupported modules → `moduleop` + custom layer; fp16 weights default) → optional `ncnnoptimize` (skip if pnnx) → optional `ncnn2table/ncnn2int8` or `ncnnllm2table/ncnnllm2int`. ONNX also via pnnx (preferred) or `onnx2ncnn`. Constraints: batch dim removed by pnnx (`Mat` is CHW-ish, WHC ordering in tools), ops must exist among the 110 types, control flow must be unrolled or hosted, LLM decoders need param patching for cache and dynamic-M Gemm, Vulkan-unsupported layers (e.g. `Embed`, `RNN/LSTM/GRU`, `Spectrogram` — no `*_vulkan.cpp` in [src/layer/vulkan](../../ncnn/src/layer/vulkan)) fall back to CPU with sync cost.

---

## 12. Limitations and unknowns

1. [Verified] No request scheduler, batching server, or streaming API; all orchestration is application code.
2. [Verified] KV cache is per-layer contiguous with geometric growth and batch size 1; no paging, sliding window, prefix sharing or eviction.
3. [Verified] Attention mask and RoPE cos/sin tables are host-provided inputs each step; positions are implicit in the mask shape.
4. [Verified] Vulkan shaders compile at runtime (glslang); first-run latency unless a `PipelineCache` file is shipped; cache is not portable across devices/drivers/builds ([vulkan-pipeline-cache.md#L288](../../ncnn/docs/developer-guide/vulkan-pipeline-cache.md#L288)).
5. [Verified] No CUDA/Metal/CoreML/OpenCL/NNAPI/QNN backends; NPUs on phones and Jetson tensor cores are unreachable except through Vulkan cooperative-matrix shaders (15 `*_cm.comp` shaders incl. `sdpa_fa_cm.comp`, `gemm_cm.comp`).
6. [Verified] W4/W6/W8 block-quantized Gemm/MHA run on CPU only and disable fp16/bf16/packing/Vulkan for those layers.
7. [Verified] Piper/Whisper export scripts and Android demo apps live in external nihui/* repos; Piper needs a patched upstream.
8. [Verified] `Spectrogram` uses a direct DFT (no FFT library); [Inferred] cost grows as O(n_fft²) per frame.
9. [Verified] glslang and pybind11 submodules are not initialized in this checkout, so Vulkan and Python builds are not possible from it as-is.
10. [Unknown] Absolute library binary sizes per configuration (only CI deltas exist).
11. [Unknown] Real throughput of `benchncnn_llm` on target edge boards; README LLM section describes the harness but no LLM numbers were found in the tables reviewed.
12. [Unknown] Thread-safety guarantees of `VulkanDevice` allocator/queue pools under heavy multi-thread use beyond the documented `compute_queue_count()` bound.
13. [Inferred] Multi-Net pipelines (encoder → decoder) pay a host round-trip unless both are Vulkan and chained with `VkMat` + shared `VkCompute`; CPU-fallback layers inside a GPU graph force `submit_and_wait`.
14. [Inferred] Very large vocab `Gemm` proj_out on CPU dominates decode time for small LLMs; no fused sampling.
15. [Proposal] For a vLLM-Omni edge backend, wrap `Net`+`Extractor` per stage behind an executor that owns the kv-cache allocator per sequence, keeps a `PipelineCache` per device, and pins decode threads to big cores; use `extract(type=1)` handles as opaque cache objects.

---

## 13. Reference index

| Path | Role |
|---|---|
| [../../ncnn/README.md](../../ncnn/README.md) | Project overview, quick start (pnnx → ncnn), release links |
| [../../ncnn/CMakeLists.txt](../../ncnn/CMakeLists.txt) | Top-level build options and ISA feature flags |
| [../../ncnn/build-android.cmd](../../ncnn/build-android.cmd) | Flags used for prebuilt Android archives |
| [../../ncnn/src/net.h](../../ncnn/src/net.h) | `Net`/`Extractor` public API incl. kv-cache setters |
| [../../ncnn/src/net.cpp](../../ncnn/src/net.cpp) | Graph load, layer creation/fallback, lazy forward, layout conversion, batch loop, Vulkan submit heuristics |
| [../../ncnn/src/layer.h](../../ncnn/src/layer.h) | `Layer` interface, `support_*` flags, creator macros |
| [../../ncnn/src/layer.cpp](../../ncnn/src/layer.cpp) | Registry lookup, `Layer_final` CPU/Vulkan wrapper |
| [../../ncnn/src/layer_declaration.h.in](../../ncnn/src/layer_declaration.h.in) / [layer_registry.h.in](../../ncnn/src/layer_registry.h.in) / [layer_type_enum.h.in](../../ncnn/src/layer_type_enum.h.in) | Generated registry templates |
| [../../ncnn/src/mat.h](../../ncnn/src/mat.h) | `Mat`/`VkMat`/`VkImageMat`, pixel conversion API, batch dim |
| [../../ncnn/src/option.h](../../ncnn/src/option.h) / [option.cpp](../../ncnn/src/option.cpp) | Runtime options and defaults |
| [../../ncnn/src/allocator.h](../../ncnn/src/allocator.h) / [allocator.cpp](../../ncnn/src/allocator.cpp) | Host/Vulkan allocators and pools |
| [../../ncnn/src/cpu.h](../../ncnn/src/cpu.h) / [cpu.cpp](../../ncnn/src/cpu.cpp) | ISA detection, topology, affinity, OpenMP wrappers |
| [../../ncnn/src/ruapu.h](../../ncnn/src/ruapu.h) | Bundled ISA detection header |
| [../../ncnn/src/gpu.h](../../ncnn/src/gpu.h) / [gpu.cpp](../../ncnn/src/gpu.cpp) | Vulkan instance/device/GpuInfo, shader compile |
| [../../ncnn/src/simplevk.h](../../ncnn/src/simplevk.h) / [simplevk.cpp](../../ncnn/src/simplevk.cpp) | In-house Vulkan loader (dlopen, MoltenVK, Android HAL) |
| [../../ncnn/src/command.h](../../ncnn/src/command.h) / [command.cpp](../../ncnn/src/command.cpp) | `VkCompute`/`VkTransfer` recording and submission |
| [../../ncnn/src/pipeline.h](../../ncnn/src/pipeline.h) / [pipelinecache.h](../../ncnn/src/pipelinecache.h) | Vulkan pipeline objects and persistent cache |
| [../../ncnn/src/paramdict.h](../../ncnn/src/paramdict.h) / [modelbin.h](../../ncnn/src/modelbin.h) / [datareader.h](../../ncnn/src/datareader.h) | Param/weight parsing abstractions |
| [../../ncnn/src/expression.h](../../ncnn/src/expression.h) | Runtime shape expression evaluator |
| [../../ncnn/src/c_api.h](../../ncnn/src/c_api.h) | C API |
| [../../ncnn/src/simpleomp.h](../../ncnn/src/simpleomp.h) / [simplestl.h](../../ncnn/src/simplestl.h) / [simpleocv.h](../../ncnn/src/simpleocv.h) / [simplemath.h](../../ncnn/src/simplemath.h) | Minimal runtime shims |
| [../../ncnn/src/stb_image.h](../../ncnn/src/stb_image.h) | Image decoding for simpleocv |
| [../../ncnn/src/convert_ycbcr.comp](../../ncnn/src/convert_ycbcr.comp) | Camera YCbCr → RGB Vulkan shader |
| [../../ncnn/src/mat_pixel_resize.cpp](../../ncnn/src/mat_pixel_resize.cpp) / [mat_pixel_rotate.cpp](../../ncnn/src/mat_pixel_rotate.cpp) / [mat_pixel_affine.cpp](../../ncnn/src/mat_pixel_affine.cpp) / [mat_pixel_drawing.cpp](../../ncnn/src/mat_pixel_drawing.cpp) | Pixel preprocessing utilities |
| [../../ncnn/src/CMakeLists.txt](../../ncnn/src/CMakeLists.txt) | Layer registration list, shader embedding, glslang link |
| [../../ncnn/src/layer](../../ncnn/src/layer) | Naive layer implementations (110 types) |
| [../../ncnn/src/layer/sdpa.h](../../ncnn/src/layer/sdpa.h) / [sdpa.cpp](../../ncnn/src/layer/sdpa.cpp) | SDPA with GQA and kv cache growth logic |
| [../../ncnn/src/layer/multiheadattention.h](../../ncnn/src/layer/multiheadattention.h) / [multiheadattention.cpp](../../ncnn/src/layer/multiheadattention.cpp) | MHA with cache, int8 and block-quant paths |
| [../../ncnn/src/layer/rotaryembed.h](../../ncnn/src/layer/rotaryembed.h) / [rotaryembed.cpp](../../ncnn/src/layer/rotaryembed.cpp) | RoPE from host cos/sin caches |
| [../../ncnn/src/layer/embed.h](../../ncnn/src/layer/embed.h) | Token embedding lookup |
| [../../ncnn/src/layer/gemm.h](../../ncnn/src/layer/gemm.h) | Gemm params incl. block quantization |
| [../../ncnn/src/layer/spectrogram.h](../../ncnn/src/layer/spectrogram.h) / [spectrogram.cpp](../../ncnn/src/layer/spectrogram.cpp) / [inversespectrogram.h](../../ncnn/src/layer/inversespectrogram.h) / [inversespectrogram.cpp](../../ncnn/src/layer/inversespectrogram.cpp) | STFT / iSTFT layers |
| [../../ncnn/src/layer/arm](../../ncnn/src/layer/arm) / [x86](../../ncnn/src/layer/x86) / [riscv](../../ncnn/src/layer/riscv) / [mips](../../ncnn/src/layer/mips) / [loongarch](../../ncnn/src/layer/loongarch) | Arch-optimized kernels |
| [../../ncnn/src/layer/arm/sdpa_arm.cpp](../../ncnn/src/layer/arm/sdpa_arm.cpp) | ARM SDPA (fp16/bf16 storage flags) |
| [../../ncnn/src/layer/arm/convolution_arm_i8mm.cpp](../../ncnn/src/layer/arm/convolution_arm_i8mm.cpp) / [convolution1d_arm_asimdhp.cpp](../../ncnn/src/layer/arm/convolution1d_arm_asimdhp.cpp) | ISA-variant kernel examples |
| [../../ncnn/src/layer/x86/convolution_x86_avx512vnni.cpp](../../ncnn/src/layer/x86/convolution_x86_avx512vnni.cpp) | x86 int8 VNNI kernel |
| [../../ncnn/src/layer/vulkan](../../ncnn/src/layer/vulkan) / [shader](../../ncnn/src/layer/vulkan/shader) | Vulkan layer impls and 300 GLSL shaders |
| [../../ncnn/src/layer/vulkan/shader/sdpa_kvcache_append.comp](../../ncnn/src/layer/vulkan/shader/sdpa_kvcache_append.comp) | GPU kv-cache append shader |
| [../../ncnn/src/layer/vulkan/convolution1d_vulkan.cpp](../../ncnn/src/layer/vulkan/convolution1d_vulkan.cpp) | 1-D conv on GPU |
| [../../ncnn/cmake/ncnn_add_layer.cmake](../../ncnn/cmake/ncnn_add_layer.cmake) / [ncnn_add_shader.cmake](../../ncnn/cmake/ncnn_add_shader.cmake) / [run_test.cmake](../../ncnn/cmake/run_test.cmake) | Build macros |
| [../../ncnn/toolchains](../../ncnn/toolchains) / [jetson.toolchain.cmake](../../ncnn/toolchains/jetson.toolchain.cmake) / [ios.toolchain.cmake](../../ncnn/toolchains/ios.toolchain.cmake) | Cross-compile toolchains |
| [../../ncnn/tools/pnnx](../../ncnn/tools/pnnx) / [README.md](../../ncnn/tools/pnnx/README.md) / [src/save_ncnn.cpp](../../ncnn/tools/pnnx/src/save_ncnn.cpp) / [src/load_onnx.cpp](../../ncnn/tools/pnnx/src/load_onnx.cpp) / [src/load_tnn.cpp](../../ncnn/tools/pnnx/src/load_tnn.cpp) / [src/pass_ncnn](../../ncnn/tools/pnnx/src/pass_ncnn) | PyTorch/ONNX → ncnn exporter |
| [../../ncnn/tools/pnnx/src/pass_ncnn/torch_stft.cpp](../../ncnn/tools/pnnx/src/pass_ncnn/torch_stft.cpp) / [torchaudio_F_inverse_spectrogram.cpp](../../ncnn/tools/pnnx/src/pass_ncnn/torchaudio_F_inverse_spectrogram.cpp) | Audio op lowering |
| [../../ncnn/tools/onnx/onnx2ncnn.cpp](../../ncnn/tools/onnx/onnx2ncnn.cpp) | Legacy ONNX converter |
| [../../ncnn/tools/caffe](../../ncnn/tools/caffe) / [darknet](../../ncnn/tools/darknet) / [mxnet](../../ncnn/tools/mxnet) / [keras](../../ncnn/tools/keras) / [tensorflow](../../ncnn/tools/tensorflow) / [mlir](../../ncnn/tools/mlir) | Other converters |
| [../../ncnn/tools/ncnnoptimize.cpp](../../ncnn/tools/ncnnoptimize.cpp) / [ncnnmerge.cpp](../../ncnn/tools/ncnnmerge.cpp) / [ncnn2mem.cpp](../../ncnn/tools/ncnn2mem.cpp) | Offline graph optimization / packaging |
| [../../ncnn/tools/quantize](../../ncnn/tools/quantize) / [ncnnllm2table.cpp](../../ncnn/tools/quantize/ncnnllm2table.cpp) / [ncnnllm2int.cpp](../../ncnn/tools/quantize/ncnnllm2int.cpp) | PTQ int8 and LLM block quantization tools |
| [../../ncnn/benchmark/benchncnn.cpp](../../ncnn/benchmark/benchncnn.cpp) / [benchncnn_llm.cpp](../../ncnn/benchmark/benchncnn_llm.cpp) / [CMakeLists.txt](../../ncnn/benchmark/CMakeLists.txt) / [README.md](../../ncnn/benchmark/README.md) / [models/vision_transformer.param](../../ncnn/benchmark/models/vision_transformer.param) | Benchmarks and repo-reported results |
| [../../ncnn/examples/CMakeLists.txt](../../ncnn/examples/CMakeLists.txt) | Example targets |
| [../../ncnn/examples/piper.cpp](../../ncnn/examples/piper.cpp) | Piper/VITS TTS example |
| [../../ncnn/examples/whisper.cpp](../../ncnn/examples/whisper.cpp) | Whisper ASR with kv cache and beam search |
| [../../ncnn/examples/yolo11.cpp](../../ncnn/examples/yolo11.cpp) / [rvm.cpp](../../ncnn/examples/rvm.cpp) / [squeezencnn/README.md](../../ncnn/examples/squeezencnn/README.md) | Vision / recurrent-state / Android pointers |
| [../../ncnn/tests/CMakeLists.txt](../../ncnn/tests/CMakeLists.txt) / [testutil.h](../../ncnn/tests/testutil.h) | Test harness |
| [../../ncnn/tests/test_sdpa_kvcache.cpp](../../ncnn/tests/test_sdpa_kvcache.cpp) / [test_sdpa_kvcache_session.cpp](../../ncnn/tests/test_sdpa_kvcache_session.cpp) / [test_multiheadattention_kvcache.cpp](../../ncnn/tests/test_multiheadattention_kvcache.cpp) / [test_multiheadattention_kvcache_allocator.cpp](../../ncnn/tests/test_multiheadattention_kvcache_allocator.cpp) / [test_gemm_block_quant.cpp](../../ncnn/tests/test_gemm_block_quant.cpp) / [test_mat_batch.cpp](../../ncnn/tests/test_mat_batch.cpp) | Transformer/kv-cache/batch tests |
| [../../ncnn/tests/perf](../../ncnn/tests/perf) / [perf_sdpa_kvcache.cpp](../../ncnn/tests/perf/perf_sdpa_kvcache.cpp) / [perf_sdpa_decode.cpp](../../ncnn/tests/perf/perf_sdpa_decode.cpp) / [perf_sdpa_prefill.cpp](../../ncnn/tests/perf/perf_sdpa_prefill.cpp) | Micro-benchmarks |
| [../../ncnn/python/src/main.cpp](../../ncnn/python/src/main.cpp) / [python/README.md](../../ncnn/python/README.md) / [python/ncnn/model_zoo](../../ncnn/python/ncnn/model_zoo) / [python/examples](../../ncnn/python/examples) | Python bindings |
| [../../ncnn/.github/workflows](../../ncnn/.github/workflows) / [compare-binary-size.yml](../../ncnn/.github/workflows/compare-binary-size.yml) | CI matrix, binary-size tracking |
| [../../ncnn/docs/developer-guide/kvcache.md](../../ncnn/docs/developer-guide/kvcache.md) | KV cache design and usage |
| [../../ncnn/docs/developer-guide/layer-support-behavior.md](../../ncnn/docs/developer-guide/layer-support-behavior.md) | Meaning of `support_*` flags |
| [../../ncnn/docs/developer-guide/layer-feat-mask.md](../../ncnn/docs/developer-guide/layer-feat-mask.md) | Per-layer feature mask `31=` |
| [../../ncnn/docs/developer-guide/expression.md](../../ncnn/docs/developer-guide/expression.md) | Dynamic shape expressions |
| [../../ncnn/docs/developer-guide/element-packing.md](../../ncnn/docs/developer-guide/element-packing.md) | elempack layout |
| [../../ncnn/docs/developer-guide/param-and-model-file-structure.md](../../ncnn/docs/developer-guide/param-and-model-file-structure.md) | File formats |
| [../../ncnn/docs/developer-guide/operators.md](../../ncnn/docs/developer-guide/operators.md) / [operation-param-weight-table.md](../../ncnn/docs/developer-guide/operation-param-weight-table.md) | Operator/param reference |
| [../../ncnn/docs/developer-guide/custom-allocator.md](../../ncnn/docs/developer-guide/custom-allocator.md) | Allocator roles and concurrency patterns |
| [../../ncnn/docs/developer-guide/low-level-operation-api.md](../../ncnn/docs/developer-guide/low-level-operation-api.md) | Single-layer usage |
| [../../ncnn/docs/developer-guide/how-to-implement-custom-layer-step-by-step.md](../../ncnn/docs/developer-guide/how-to-implement-custom-layer-step-by-step.md) / [add-custom-layer.zh.md](../../ncnn/docs/developer-guide/add-custom-layer.zh.md) | Custom layer tutorials |
| [../../ncnn/docs/developer-guide/vulkan-driver-loader.md](../../ncnn/docs/developer-guide/vulkan-driver-loader.md) / [vulkan-pipeline-cache.md](../../ncnn/docs/developer-guide/vulkan-pipeline-cache.md) / [glsl-extension.md](../../ncnn/docs/developer-guide/glsl-extension.md) | Vulkan internals |
| [../../ncnn/docs/developer-guide/arm-a53-a55-dual-issue.md](../../ncnn/docs/developer-guide/arm-a53-a55-dual-issue.md) | Little-core kernel tuning |
| [../../ncnn/docs/how-to-use-and-FAQ/quantized-int8-inference.md](../../ncnn/docs/how-to-use-and-FAQ/quantized-int8-inference.md) | PTQ and LLM block quant workflow |
| [../../ncnn/docs/how-to-use-and-FAQ/build-minimal-library.md](../../ncnn/docs/how-to-use-and-FAQ/build-minimal-library.md) | Binary size reduction |
| [../../ncnn/docs/how-to-use-and-FAQ/openmp-best-practice.md](../../ncnn/docs/how-to-use-and-FAQ/openmp-best-practice.md) | Threading/affinity guidance |
| [../../ncnn/docs/how-to-use-and-FAQ/vulkan-notes.md](../../ncnn/docs/how-to-use-and-FAQ/vulkan-notes.md) / [FAQ-ncnn-vulkan.md](../../ncnn/docs/how-to-use-and-FAQ/FAQ-ncnn-vulkan.md) | Vulkan usage, platform matrix, fallback |
| [../../ncnn/docs/how-to-use-and-FAQ/use-ncnn-with-android-hardware-buffer.md](../../ncnn/docs/how-to-use-and-FAQ/use-ncnn-with-android-hardware-buffer.md) | Zero-copy camera input |
| [../../ncnn/docs/how-to-use-and-FAQ/ncnn-load-model.md](../../ncnn/docs/how-to-use-and-FAQ/ncnn-load-model.md) / [efficient-roi-resize-rotate.md](../../ncnn/docs/how-to-use-and-FAQ/efficient-roi-resize-rotate.md) | Loading and preprocessing docs |
| [../../ncnn/docs/how-to-build/how-to-build.md](../../ncnn/docs/how-to-build/how-to-build.md) | Per-platform build instructions (Jetson, Android, iOS, WASM, HarmonyOS, ESP32…) |
