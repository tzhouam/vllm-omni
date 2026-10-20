"""Greedy-decode parity for llama.cpp against the same fp32 reference vLLM is
scored on.

The point of the comparison is that "4-bit" is not one thing: llama.cpp's
Q4_K_M and our `cpu_int4` spend a similar number of bits per weight but pick
different grids, so the speed comparison is only meaningful next to a quality
comparison run on identical prompts.  llama-server takes a prompt as raw token
ids, so both engines see exactly the token sequence the reference did.
"""
import argparse, json, subprocess, time, urllib.request

import numpy as np


def post(url, payload, timeout=600):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ref", default="ref.json")
    ap.add_argument("--ref-logits", default="ref_logits.npy")
    ap.add_argument("--bin", default="/data/zhoutaichang/embedding_infer/llama.cpp/build/bin/llama-server")
    ap.add_argument("--cores", default="56-87")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--port", type=int, default=18099)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--out", default="parity_llamacpp.json")
    a = ap.parse_args()

    ref = json.load(open(a.ref))
    ref_logits = np.load(a.ref_logits)
    n_new = len(ref["new_ids"])

    proc = subprocess.Popen(
        ["taskset", "-c", a.cores, a.bin, "-m", a.gguf, "--port", str(a.port),
         "-t", str(a.threads), "-c", str(a.ctx), "--no-warmup", "--host", "127.0.0.1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{a.port}"
    try:
        for _ in range(180):
            try:
                urllib.request.urlopen(url + "/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("llama-server did not become healthy")

        res = post(url + "/completion", {
            "prompt": ref["prompt_ids"], "n_predict": n_new,
            "temperature": 0.0, "top_k": 1, "n_probs": 20,
            "cache_prompt": False, "post_sampling_probs": False,
        })
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

    probs = res.get("completion_probabilities") or []
    ids = [p["id"] for p in probs] if probs and "id" in probs[0] else []
    n = min(n_new, len(ids)) if ids else 0
    match = sum(int(x == y) for x, y in zip(ref["new_ids"][:n], ids[:n]))

    rep = {
        "engine": "llama.cpp", "gguf": a.gguf.split("/")[-1],
        "ref_text": ref["text"], "llamacpp_text": res.get("content", ""),
        "greedy_token_match": f"{match}/{n}" if n else "n/a",
        "exact_match": bool(n) and ref["new_ids"][:n] == ids[:n],
        "ref_top1": ref["top1"],
        "llamacpp_first_token": ids[0] if ids else None,
        "first_token_agree": bool(ids) and ids[0] == ref["top1"],
    }
    if probs:
        import math
        top = (probs[0].get("top_logprobs") or probs[0].get("top_probs")
               or probs[0].get("probs") or [])
        first = {}
        for e in top:
            if "id" not in e:
                continue
            first[int(e["id"])] = (float(e["prob"]) if "prob" in e
                                   else math.exp(float(e["logprob"])))
        ref_top = np.argsort(-ref_logits)[:20]
        ref_p = np.exp(ref_logits - ref_logits.max())
        ref_p /= ref_p.sum()
        pairs = [(int(t), float(ref_p[t]), first[int(t)]) for t in ref_top if int(t) in first]
        rep["top20_prob_overlap"] = f"{len(pairs)}/20"
        if pairs:
            d = [abs(r - v) for _, r, v in pairs]
            rep["top20_prob_max_abs_diff"] = max(d)
            rep["top20_prob_mean_abs_diff"] = sum(d) / len(d)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
