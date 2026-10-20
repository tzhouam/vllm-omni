"""Check the exported Spark decode step against the HF reference.

Runs the reference over a C-token prompt, hands its KV cache to
SparkDecodeStep (sliding layers get only the last 512 entries, which is the
whole point of the hybrid layout) and compares next-token logits.
"""
import argparse, importlib.util, json, sys, types
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/zhoutaichang/embedding_infer/vllm-omni")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


for pkg_name in ("vllm_omni", "vllm_omni.edge"):
    mod = types.ModuleType(pkg_name)
    mod.__path__ = []
    sys.modules[pkg_name] = mod
_load("vllm_omni.edge.qnn_export", ROOT / "vllm_omni/edge/qnn_export.py")
se = _load("vllm_omni.edge.spark_export", ROOT / "vllm_omni/edge/spark_export.py")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="XHToken/Spark-X2.5-1.7B")
ap.add_argument("--context", type=int, default=1024)
ap.add_argument("--cache-layout", default="auto", choices=["auto", "ring", "roll"])
ap.add_argument("--gelu", default="exact", choices=["exact", "tanh", "sigmoid"])
ap.add_argument("--out", default="export_parity.json")
a = ap.parse_args()

from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
text = ("The quick brown fox jumps over the lazy dog. " * 400)
ids = tok(text, add_special_tokens=False)["input_ids"][: a.context + 1]
assert len(ids) == a.context + 1, f"only got {len(ids)} tokens"

m = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True,
                                         dtype=torch.float32).eval()
import copy
with torch.no_grad():
    pre = m(torch.tensor([ids[:-1]]), use_cache=True)
    # The decode call appends to the cache in place, so snapshot it first.
    cache = copy.deepcopy(pre.past_key_values)
    ref = m(torch.tensor([[ids[-1]]]), past_key_values=pre.past_key_values,
            use_cache=True)
ref_logits = ref.logits[0, -1].float().numpy()

def kv(i):
    try:
        return cache.layers[i].keys, cache.layers[i].values
    except AttributeError:
        return cache.key_cache[i], cache.value_cache[i]

hf_cfg = json.loads((Path(m.config._name_or_path) / "config.json").read_text()) \
    if Path(str(m.config._name_or_path)).exists() else m.config.to_dict()
cfg = se.config_from_spark(hf_cfg, None, include_lm_head=True,
                           cache_layout=a.cache_layout, gelu_mode=a.gelu)
step = se.SparkDecodeStep(cfg)

from huggingface_hub import snapshot_download
wdir = snapshot_download(a.model, allow_patterns=["*.safetensors"])
print("loaded", se.load_spark_weights(step, wdir), "tensors")
step.eval()

C = a.context
pos = C
emb = m.model.embedding(torch.tensor([[ids[-1]]])).float()
cos_sw, sin_sw = se._angles(pos, cfg.rotary_dim(se.SLIDING), cfg.rope_theta[se.SLIDING], torch.float32)
cos_f, sin_f = se._angles(pos, cfg.rotary_dim(se.FULL), cfg.rope_theta[se.FULL], torch.float32)
sw_len = cfg.cache_len(se.SLIDING, C)
# Ring masks only cache slots; roll masks the rebuilt window (cache + new).
mask_sw = torch.zeros(1, 1, 1, sw_len + (1 if cfg.layout_for(se.SLIDING) == "ring" else 0))
mask_full = torch.zeros(1, 1, 1, C + 1)

shapes = {}
caches = []
for i, t in enumerate(cfg.layer_types):
    k, v = kv(i)
    shapes.setdefault(t, k.shape[2])
    if t == se.SLIDING:
        # transformers already truncates sliding layers to the window; take
        # the newest sw_len entries and left-pad (masked out) if it holds fewer.
        k, v = k[:, :, -sw_len:], v[:, :, -sw_len:]
        have = k.shape[2]
        if have < sw_len:
            pad = sw_len - have
            z = torch.zeros(k.shape[0], k.shape[1], pad, k.shape[3])
            # No mask needed: the window roll drops exactly these pad slots.
            k, v = torch.cat([z, k.float()], 2), torch.cat([z, v.float()], 2)
    caches += [k.float(), v.float()]
print("hf cache lens by layer type:", shapes, "| sw_len used:", sw_len)

with torch.no_grad():
    out = step(emb, cos_sw, sin_sw, cos_f, sin_f, mask_sw, mask_full, *caches)
got = out[0][0].float().numpy()

def snr(ref, got):
    return 20 * np.log10(np.linalg.norm(ref) / max(np.linalg.norm(ref - got), 1e-12))

rep = {
    "context": C,
    "cache_layout": a.cache_layout,
    "gelu_mode": a.gelu,
    "logits_snr_db": round(float(snr(ref_logits, got)), 2),
    "max_abs_diff": float(np.max(np.abs(ref_logits - got))),
    "ref_top1": int(np.argmax(ref_logits)),
    "export_top1": int(np.argmax(got)),
    "top1_agree": int(np.argmax(ref_logits)) == int(np.argmax(got)),
    "top5_agree": np.argsort(-ref_logits)[:5].tolist() == np.argsort(-got)[:5].tolist(),
    "kv_bytes": se.kv_cache_bytes(cfg, C),
}
json.dump(rep, open(a.out, "w"), indent=2)
print(json.dumps(rep, indent=2))
