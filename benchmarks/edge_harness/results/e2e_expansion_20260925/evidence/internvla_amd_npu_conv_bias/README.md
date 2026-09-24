# Cosmos Conv13 FP32-bias control

The [real-weight bias rewrite](../../../../experiments/rewrite_internvla_conv_bias.py)
moved Conv13's unchanged FP32 bias to a following Add. ONNX Runtime CPU
reproduced the original ramp and pattern outputs bitwise
([rewrite report](rewrite_report.json)). Its A16W8 Conv candidate passed the
same bounded CPU numerical gate at 0.619%/0.529% relative L2
([quantization report](quantization_report.json)). VitisAI executed one NPU
node plus four CPU nodes on the six-frame pattern but differed by **2.311
relative L2** from the same QDQ graph on CPU ([NPU probe](npu_probe.json)).
The paired tensor is pinned outside Git by that report. Its frame 0 matched
CPU within 1.03e-5; frames 1–5 were identically wrong at 2.531 relative L2
each. This control narrows, but does not repair, the batch-index failure.
The [batch-one follow-up](../internvla_amd_npu_batch1/README.md) passed the
same-fixture component gate; neither is a full encoder or policy run.
