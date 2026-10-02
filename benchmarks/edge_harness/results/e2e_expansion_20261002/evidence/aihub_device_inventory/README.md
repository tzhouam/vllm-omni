# AI Hub hosted target inventory (2026-10-02)

[Raw SDK response fields](device_inventory.json) identify exact hosted registrations
for the five Qualcomm targets in the device/model matrix. The query used
`qai-hub` SDK 0.55.0 and the already configured client, without submitting a
compile, inference or profile job. Reproduce with:

```bash
/home/zhout/project/edge_infer/.venvs/aihub/bin/python \
  benchmarks/edge_harness/experiments/audit_aihub_device_inventory.py \
  --output /tmp/aihub_device_inventory.json
```

This is **D (hosted registration)** evidence. Each target has one exact name
match and reports OS, chipset and supported framework attributes. The SDK's
`Device` fields and attributes report **no RAM capacity or usable-RAM budget**.
Consequently, neither a Qwen3.8-27B capacity refusal nor a claim that the
checkpoint fits can be made for these exact leased SKUs from this inventory.
It contains no batch-size or request measurements and no device-resident Omni
state, shared-memory or thermal evidence.
