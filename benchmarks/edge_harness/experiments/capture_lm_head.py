"""Real inputs for the output head: the last layer's output, pre-final-norm."""
import importlib.util, json, sys, types
from pathlib import Path
import numpy as np, torch

ROOT = Path("/data/zhoutaichang/embedding_infer/vllm-omni")
HERE = Path("/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge")
def _load(n, p):
    sp = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(sp)
    sys.modules[n] = m; sp.loader.exec_module(m); return m
for pkg in ("vllm_omni", "vllm_omni.edge"):
    mod = types.ModuleType(pkg); mod.__path__ = []; sys.modules[pkg] = mod
_load("vllm_omni.edge.qnn_export", ROOT / "vllm_omni/edge/qnn_export.py")
se = _load("vllm_omni.edge.spark_export", ROOT / "vllm_omni/edge/spark_export.py")

from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL = "XHToken/Spark-X2.5-1.7B"
SNAP = ("/data/zhoutaichang/.cache/huggingface/hub/models--XHToken--Spark-X2.5-1.7B/"
        "snapshots/448e61eb392c00f2c403185c5b56d5e0665bfaab")
TEXTS = ["The quick brown fox jumps over the lazy dog. " * 60,
         "Engineers argued about memory bandwidth on the train home. " * 60,
         "def forward(self, x):\n    return self.down(self.act(self.gate(x)))\n" * 60,
         "移动端推理的瓶颈通常是内存带宽而不是算力。" * 80]

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True,
                                         dtype=torch.float32).eval()
hf = json.loads(Path(f"{SNAP}/config.json").read_text())
cfg = se.config_from_spark(hf, slice(0, 0), include_lm_head=True)
step = se.SparkDecodeStep(cfg); se.load_spark_weights(step, SNAP); step.eval()

grabbed = {}
h = m.model.layers[-1].register_forward_hook(
    lambda mod, inp, out: grabbed.__setitem__("h", out[0] if isinstance(out, tuple) else out))
xs, outs = [], []
for t in TEXTS:
    ids = tok(t, add_special_tokens=False)["input_ids"][:256]
    with torch.no_grad():
        m(torch.tensor([ids]))
        x = grabbed["h"][:, -1:, :].float()
        logits = step(x, *[torch.zeros(1)] * 0) if False else step(
            x, torch.zeros(1,1,1,4), torch.zeros(1,1,1,4),
            torch.zeros(1,1,1,4), torch.zeros(1,1,1,4),
            torch.zeros(1,1,1,1), torch.zeros(1,1,1,1))[0]
    xs.append(x.numpy().astype(np.float32)); outs.append(logits.numpy().astype(np.float32))
h.remove()
np.savez(HERE / "real_acts_lmhead.npz", x=np.stack(xs), hidden_out=np.stack(outs))
print("captured", len(TEXTS), "| x absmax", np.abs(np.stack(xs)).max(),
      "| logits absmax", np.abs(np.stack(outs)).max())
