"""Capture real layer inputs for calibration and parity.

Quantisation ranges inferred from random tensors describe a distribution the
model never sees. This runs the reference on real text and records what
actually enters a sliding layer at several positions, together with the fp32
output of that layer, so the same tensors can calibrate the quantiser and
then judge it.
"""
import importlib.util, json, sys, types
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/zhoutaichang/embedding_infer/vllm-omni")
HERE = Path("/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


for pkg in ("vllm_omni", "vllm_omni.edge"):
    mod = types.ModuleType(pkg); mod.__path__ = []; sys.modules[pkg] = mod
_load("vllm_omni.edge.qnn_export", ROOT / "vllm_omni/edge/qnn_export.py")
se = _load("vllm_omni.edge.spark_export", ROOT / "vllm_omni/edge/spark_export.py")

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "XHToken/Spark-X2.5-1.7B"
SNAP = ("/data/zhoutaichang/.cache/huggingface/hub/models--XHToken--Spark-X2.5-1.7B/"
        "snapshots/448e61eb392c00f2c403185c5b56d5e0665bfaab")
TEXTS = [
    "The quick brown fox jumps over the lazy dog. " * 300,
    "In a distant galaxy, engineers argued about memory bandwidth and cache lines. " * 200,
    "def forward(self, x):\n    return self.down_proj(self.act_fn(self.gate_proj(x)))\n" * 200,
    "人工智能模型在移动设备上的推理速度取决于内存带宽和算子调度。" * 300,
]
C = 1024

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True,
                                         dtype=torch.float32).eval()
hf = json.loads(Path(f"{SNAP}/config.json").read_text())
import os
WHICH = os.environ.get("CAPTURE", "sliding")
li = hf["layer_types"].index(se.SLIDING if WHICH == "sliding" else se.FULL)
cfg = se.config_from_spark(hf, slice(li, li + 1), include_lm_head=False,
                           cache_layout="ring" if WHICH == "sliding" else "roll")
step = se.SparkDecodeStep(cfg)
se.load_spark_weights(step, SNAP)
step.eval()

LT = se.SLIDING if WHICH == "sliding" else se.FULL
sw = cfg.cache_len(LT, C)
cos, sin = se._angles(C, cfg.rotary_dim(LT), cfg.rope_theta[LT], torch.float32)
samples = {k: [] for k in ("x", "cos_sw", "sin_sw", "mask_sw", "k_cache_0",
                           "v_cache_0", "hidden_out")}

for text in TEXTS:
    ids = tok(text, add_special_tokens=False)["input_ids"][: C + 1]
    assert len(ids) == C + 1
    with torch.no_grad():
        pre = m(torch.tensor([ids[:-1]]), use_cache=True)
        cache = pre.past_key_values
        try:
            k, v = cache.layers[li].keys, cache.layers[li].values
        except AttributeError:
            k, v = cache.key_cache[li], cache.value_cache[li]
        k, v = k[:, :, -sw:].float(), v[:, :, -sw:].float()
        if k.shape[2] < sw:                       # pad the ring, mask it out
            pad = sw - k.shape[2]
            z = torch.zeros(k.shape[0], k.shape[1], pad, k.shape[3])
            k, v = torch.cat([z, k], 2), torch.cat([z, v], 2)
        mask = torch.zeros(1, 1, 1, sw + 1)
        # Real hidden state entering this layer, not a random vector.
        # Real hidden state entering this layer: run the prefix through the
        # layers below it rather than assuming the embedding.
        x = m.model.embedding(torch.tensor([[ids[-1]]])).float()
        if li:
            pos = torch.tensor([[C]])
            hs = x
            for j in range(li):
                lt = hf["layer_types"][j]
                prf = m.config.get_partial_rotary_factor(lt)
                cj_, sj_ = __import__("importlib").import_module(
                    m.__class__.__module__).compute_rope_cos_sin(
                        pos[0], hf["head_dim"], m.config.get_rope_theta(lt),
                        partial_rotary_factor=prf)
                hs = m.model.layers[j](hs, position_embeddings=(cj_, sj_),
                                       past_key_values=cache, cache_position=pos[0])
                hs = hs[0] if isinstance(hs, tuple) else hs
            x = hs.float()
        out = step(x, cos, sin, cos, sin, mask, mask, k, v)[0]
    for name, arr in (("x", x), ("cos_sw", cos), ("sin_sw", sin),
                      ("mask_sw", mask), ("k_cache_0", k), ("v_cache_0", v),
                      ("hidden_out", out)):
        samples[name].append(arr.numpy().astype(np.float32))

np.savez(HERE / f"real_acts_{WHICH}.npz",
         **{k: np.stack(v) for k, v in samples.items()})
print("captured", len(TEXTS), "real samples for", WHICH, "layer", li)
for k, v in samples.items():
    a = np.stack(v)
    print(f"  {k:<12} {str(a.shape):<26} absmax {np.abs(a).max():.3f}")
