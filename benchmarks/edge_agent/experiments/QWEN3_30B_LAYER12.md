# Qwen3-30B partial-layer experimental route

The [native configuration](../configs/windows_laptop_qwen3_30b_a3b_q4_k_m_layer12.experimental.json)
is a **proposed, not yet loaded** text-only alternative to the Qwen3-30B CPU-expert
and `Vulkan_Host` routes. Its separate
[lineage record](../configs/windows_laptop_qwen3_30b_a3b_q4_k_m_layer12.lineage.experimental.json)
keeps the exact base checkpoint revision unverified. The official Q4_K_M GGUF
is pinned to artifact revision `e4d4bafdfb96a411a163846265362aceb0b9c63a`,
18,556,685,824 bytes, SHA-256
`0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5`.
The native llama-server executable is pinned to SHA-256
`9ffc5919acb4cb43c7be5f3053b59014cce70a87d3a801fcad49c4f459984f52`.

The proposal requests 12 GPU layers and the remaining model layers on CPU,
with `cpu_moe_layers=0` and `host_mapped_expert_layers=0`. It therefore does
not request `--n-cpu-moe` or claim that `Vulkan_Host` expert storage is CPU
execution. The separate route and log names prevent mixing its evidence with
the earlier host-mapped experiment. The number 12 is an initial feasibility
choice, **not a measured optimal split**; loading must verify the actual layer
count, buffer locations, memory use, and complete-request behavior.

The static claims reuse the prior Qwen3-30B experimental budget: 16.0 GB CPU
model buffers plus 8.0 GB host loading/KV/workspace headroom within a 24.0 GB
host reservation, and 8.0 GB GPU model buffers plus 4.0 GB VRAM headroom
within a 12.0 GB VRAM reservation. The two model-buffer ceilings sum to
24.0 GB, exceeding the pinned 18.56 GB GGUF lower bound. These are admission
limits, **not measured peaks**; either buffer ceiling may prove too small and
must cause an explicit refusal. At the previous Gemma browser-profile start,
the laptop reported 25,048,182,784 B available host RAM and 19,467,522,048 B
available VRAM, so these demands fit that *historical* snapshot. They do not
establish capacity during a resident Gemma run or at the later Qwen load.
The target remains native Windows 11 build 26200, Ryzen AI 9 HX 370,
RTX 5090 Laptop GPU, NVIDIA driver 610.71, AC power, batch size 1 and one
active request.

The first [native admission attempt](../public_evidence/agent_qwen3_30b_layer12_capacity_refusal_20261006.json)
on 2026-10-06 refused **before model load or request**: the live Windows
controller ceiling was 23,213,023,232 B host RAM against this configuration's
24,000,000,000 B declaration. The private raw JSONL is bound by SHA-256 in
that receipt. VRAM was sufficient; no claim about layer placement or speed
follows. Recheck live physical-pool availability. If admitted, load this exact
config and preserve the startup log. Require `observed_model_placement` to
equal `cpu+Vulkan0`, an unambiguous 12-layer GPU offload with CPU-assigned
remaining layers, separate model-buffer sizes below their ceilings, a bound
worker generation, and one full Agent text request with ordered events and
reference answer. A refusal or mismatch is a result for this configuration,
not proof that Qwen3-30B is unsupported. The startup log can establish
reported model-layer and buffer placement; it cannot prove every operation's
compute location. Only after the smoke passes should paired batch-one task
profiling and independent quality/lineage/placement gates be considered.
