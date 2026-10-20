# Hardware-class overlays for `deploy_profile="auto"`

When a deploy profile is `auto`, vLLM-Omni probes the host
(`vllm_omni/edge/hardware_probe.py`), maps it to one of the classes below, and
materializes a deploy YAML as: `edge/<model_type>.yaml` (or the model default)
→ overlay `edge/hardware/<class>.yaml` (this directory; same `stages:` /
`connectors:` shape as a deploy file, merged per stage id) → derived overrides
from `vllm_omni/edge/adapt.py` (dtype, KV budgets, thread binding, eager
switches, adaptive chunking, timeouts) → cached calibration
(`vllm_omni/edge/calibrate.py`) if one exists for this hardware fingerprint.

Class files are optional; add one to pin values that the derivation should not
touch for that class. Emulate a class on any host with
`VLLM_OMNI_HW_PROFILE=<json>` (see `benchmarks/tts/hw_emulation.py`).

| class | selected when |
|---|---|
| `jetson` | CUDA on unified memory (Tegra release file present) |
| `cuda_discrete` | CUDA device with its own memory |
| `arm64_cpu` | aarch64, no accelerator |
| `x86_cpu` | x86_64, no accelerator |
| `npu_phone` | `accelerator: npu:<vendor>` (only via profile override; vLLM-Omni does not run on the NPU itself) |
