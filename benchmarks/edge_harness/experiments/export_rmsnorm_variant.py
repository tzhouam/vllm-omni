"""Export the sliding layer with F.rms_norm at a higher opset.

The calibrated layer spends ~11% of its cycles in an unfused RMSNorm
(Pow/ReduceMean/Div/Mul), where the earlier build had a single fused `rms`
op at 1%. ONNX opset 23 has a real RMSNormalization node; if the exporter
emits it and QNN maps it to its fused kernel, that 11% largely goes away.
"""
import json, sys
from pathlib import Path
import torch

sys.path.insert(0, "/data/zhoutaichang/embedding_infer/vllm-omni")
import vllm_omni.edge.spark_export as se

SNAP = ("/data/zhoutaichang/.cache/huggingface/hub/models--XHToken--Spark-X2.5-1.7B/"
        "snapshots/448e61eb392c00f2c403185c5b56d5e0665bfaab")
HERE = Path("/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge")

_orig = se._rms


def _rms_fused(x, weight, eps):
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)


hf = json.loads(Path(f"{SNAP}/config.json").read_text())
i = hf["layer_types"].index(se.SLIDING)

for tag, fn, opset in (("rmsfused_op23", _rms_fused, 23),
                       ("rmsfused_op18", _rms_fused, 18)):
    se._rms = fn
    cfg = se.config_from_spark(hf, slice(i, i + 1), include_lm_head=False,
                               cache_layout="ring")
    step = se.SparkDecodeStep(cfg)
    se.load_spark_weights(step, SNAP)
    out = HERE / f"onnx/{tag}_sliding.onnx"
    try:
        info = se.export_onnx(step, 1024, out, opset=opset)
        import onnx
        g = onnx.load(str(out), load_external_data=False)
        ops = {n.op_type for n in g.graph.node}
        print(tag, "OK ->", out.name,
              "| RMSNormalization" if "RMSNormalization" in ops else "| decomposed",
              "| ops:", len(g.graph.node), flush=True)
    except Exception as e:
        print(tag, "export FAILED:", str(e)[:200], flush=True)
se._rms = _orig
