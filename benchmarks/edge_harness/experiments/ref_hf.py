"""Ground-truth Spark-X2.5 run using the official remote code on
transformers 4.57.1 (the version config.json was written for)."""
import argparse, json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "XHToken/Spark-X2.5-1.7B"

def build_msgs(scenario):
    if scenario == "short":
        return [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
    if scenario == "long":
        # >512 tokens so the 512-window sliding layers actually slide and
        # diverge from the full-attention layers.
        filler = " ".join(
            f"Item {i}: the code for warehouse district {i} is {1000 + i * 7}."
            for i in range(1, 121)
        )
        return [{"role": "user", "content":
                 "Here is an inventory listing.\n" + filler +
                 "\nWhat is the code for warehouse district 73? Answer with just the number."}]
    if scenario == "tools":
        return [
            {"role": "system", "content": "You are a helpful assistant with tools."},
            {"role": "user", "content": "What is the weather in Shenzhen right now, in celsius?"},
        ]
    raise ValueError(scenario)


TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}]


ap = argparse.ArgumentParser()
ap.add_argument("--n-new", type=int, default=24)
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--scenario", default="short", choices=["short","long","tools"])
ap.add_argument("--out", default="ref.json")
ap.add_argument("--logits-out", default="ref_logits.npy")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
kw = {"tools": TOOLS} if a.scenario == "tools" else {}
prompt = tok.apply_chat_template(build_msgs(a.scenario), tokenize=False,
                                 add_generation_prompt=True, **kw)
ids = tok(prompt, add_special_tokens=False)["input_ids"]
m = AutoModelForCausalLM.from_pretrained(
    MODEL, trust_remote_code=True, dtype=getattr(torch, a.dtype)).eval()
t = torch.tensor([ids])
with torch.no_grad():
    logits = m(t).logits[0, -1].float().numpy()
    gen = m.generate(t, max_new_tokens=a.n_new, do_sample=False, pad_token_id=2)
new = gen[0, len(ids):].tolist()
np.save(a.logits_out, logits)
json.dump({"prompt": prompt, "prompt_ids": ids, "new_ids": new,
           "text": tok.decode(new), "dtype": a.dtype,
           "top1": int(np.argmax(logits)), "scenario": a.scenario}, open(a.out, "w"), indent=2)
print("prompt_len", len(ids), "| top1", int(np.argmax(logits)))
print("text:", repr(tok.decode(new)))
