# Mobile engine candidates and evidence boundaries

`candidates.json` pins source files and publisher LFS SHA256 values audited on
2026-10-08. It does not download weights, pin a locally built Android runtime,
or make a candidate eligible for automatic Agent routing.

| Candidate | Required artifact bytes | First execution candidates |
|---|---:|---|
| Gemma 4 E2B LiteRT-LM | 2,588,147,712 | CPU / GPU, generic `.litertlm` bundle |
| Gemma 4 E4B LiteRT-LM | 3,659,530,240 | CPU / GPU, generic `.litertlm` bundle |
| Qwen3.8-27B UD-IQ1_S | 6,192,222,208 + 927,607,488 F16 projector | llama.cpp CPU / GPU |

The publisher exposes additional web/GPU and SoC-specific files. They are not
silently interchangeable with the generic mobile bundle. NPU execution is not
claimed by this catalog. Exact runtime, SoC, ABI and numerical/full-request
validation must precede its admission.

Sources: [E2B pinned metadata](https://huggingface.co/api/models/litert-community/gemma-4-E2B-it-litert-lm/revision/b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1?blobs=true),
[E4B pinned metadata](https://huggingface.co/api/models/litert-community/gemma-4-E4B-it-litert-lm/revision/2eee7ac325f20eb8c9ac1d0e972f7c84663062da?blobs=true),
[27B pinned metadata](https://huggingface.co/api/models/unsloth/Qwen3.8-27B-GGUF/revision/4ca720788d1e01f1bff70c033e0d0028fd02e502?blobs=true).
License strings are those source model cards' assertions; upstream lineage and
redistribution terms require their own review.

`vllm_omni.edge.mobile_routes` exposes these candidate identities, distinguishes
host conformance / AI Hub component / AI Hub chain / device-local evidence, and
uses one `host_ram` pool for CPU, GPU and NPU. It refuses hosted evidence that
attempts to claim device-resident whole-request latency, peak shared RAM,
cancel/release or thermal duration. Publisher file size is not runtime peak.

The C++17 boundary in `packages/omni-stage-controller` now passes host
conformance against the existing v2 fixture. It contains no model kernels or
sampling loop. Concrete JNI adapters for llama.cpp and LiteRT-LM, an Android
NDK build, and device shell execution remain required. Only AI Hub access is
currently available; no candidate here has a newly measured hosted full chain
or resident mobile qualification.
