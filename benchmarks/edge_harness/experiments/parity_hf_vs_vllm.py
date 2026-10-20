"""Numerical parity: HF reference Spark-X2.5 vs the vLLM-Omni implementation.

Runs both on CPU, greedy, and compares generated ids plus the first-step
logit vector.
"""
import argparse, json, os, sys
import numpy as np
import torch

MODEL = os.environ.get("SPARK_MODEL", "XHToken/Spark-X2.5-1.7B")
PROMPT_MSGS = [
    {"role": "user", "content": "What is the capital of France? Answer in one word."},
]


def hf_reference(model_path, prompt_ids, n_new, dtype):
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    # modeling_spark.py targets transformers 4.57; v5 expects the tied-weight
    # map to be a dict rather than a list of keys.
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    cls = get_class_from_dynamic_module(
        cfg.auto_map["AutoModelForCausalLM"], model_path
    )
    if isinstance(getattr(cls, "_tied_weights_keys", None), list):
        cls._tied_weights_keys = {"lm_head.weight": "model.embedding.weight"}
    m = cls.from_pretrained(model_path, dtype=dtype).eval()
    ids = torch.tensor([prompt_ids])
    with torch.no_grad():
        out = m(ids)
        first_logits = out.logits[0, -1].float().numpy()
        gen = m.generate(ids, max_new_tokens=n_new, do_sample=False,
                         pad_token_id=2)
    return first_logits, gen[0, len(prompt_ids):].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-new", type=int, default=24)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--out", default="parity.json")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompt = tok.apply_chat_template(PROMPT_MSGS, tokenize=False,
                                     add_generation_prompt=True)
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    print(f"prompt tokens: {len(prompt_ids)}")

    hf_logits, hf_ids = hf_reference(MODEL, prompt_ids, args.n_new, dtype)
    print("HF  :", repr(tok.decode(hf_ids)))

    import vllm_omni  # noqa: F401  (registers Spark2_5ForCausalLM)
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, dtype=args.dtype, max_model_len=args.max_model_len,
              enforce_eager=True, gpu_memory_utilization=0.30,
              max_num_seqs=1, trust_remote_code=False)
    sp = SamplingParams(temperature=0.0, max_tokens=args.n_new, logprobs=0,
                        prompt_logprobs=0)
    res = llm.generate([{"prompt_token_ids": prompt_ids}], sp)[0]
    v_ids = list(res.outputs[0].token_ids)
    print("vLLM:", repr(res.outputs[0].text))

    n = min(len(hf_ids), len(v_ids))
    match = sum(int(a == b) for a, b in zip(hf_ids[:n], v_ids[:n]))
    # first generated token's logit agreement
    first_hf_top = int(np.argmax(hf_logits))
    report = {
        "prompt_len": len(prompt_ids),
        "dtype": args.dtype,
        "hf_text": tok.decode(hf_ids),
        "vllm_text": res.outputs[0].text,
        "hf_ids": hf_ids, "vllm_ids": v_ids,
        "greedy_token_match": f"{match}/{n}",
        "exact_match": hf_ids[:n] == v_ids[:n],
        "hf_first_top1": first_hf_top,
        "vllm_first_token": v_ids[0] if v_ids else None,
        "first_token_agree": bool(v_ids and v_ids[0] == first_hf_top),
    }
    json.dump(report, open(args.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("hf_ids", "vllm_ids")}, indent=2))
    return 0 if report["exact_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
