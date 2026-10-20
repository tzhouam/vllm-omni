"""How much does 4-bit cost, measured the same way on both engines.

Absolute quality numbers are not comparable across engines -- different
tokenizers' edge cases, different chat handling.  What *is* comparable is how
far each engine's own 4-bit build drifts from its own bf16 build on the same
prompts under greedy decoding.  That isolates the quantization, which is the
question: is our W4A16 as faithful as llama.cpp's Q4_K_M?
"""
import argparse, json, subprocess, time
from pathlib import Path

HERE = Path(__file__).parent

PLAIN = [
    "The capital of Australia is",
    "In 1969, the Apollo 11 mission",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    "The three laws of thermodynamics state that",
    "To make a proper espresso you need",
    "The difference between TCP and UDP is",
    "Photosynthesis converts",
    "A binary search tree is a data structure that",
    "The Treaty of Westphalia was signed in",
]


def prompts():
    out = []
    for f, name in (("ref.json", "chat_short"), ("ref_long.json", "chat_long"),
                    ("ref_tools.json", "chat_tools")):
        p = HERE / f
        if p.exists():
            out.append((name, json.loads(p.read_text())["prompt"]))
    out += [(f"plain_{i}", t) for i, t in enumerate(PLAIN)]
    return out


def run_vllm(model, n_predict):
    import vllm_omni  # noqa: F401
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, dtype="bfloat16", max_model_len=4096,
              max_num_seqs=1, enable_prefix_caching=False,
              trust_remote_code=False, disable_log_stats=True)
    sp = SamplingParams(temperature=0.0, max_tokens=n_predict, ignore_eos=False)
    names, texts = zip(*prompts())
    outs = llm.generate(list(texts), sp, use_tqdm=False)
    return {n: {"text": o.outputs[0].text,
                "token_ids": list(o.outputs[0].token_ids)}
            for n, o in zip(names, outs)}


def run_llamacpp(binary, gguf, n_predict, threads, port=8399):
    """Drive llama.cpp through its server's /completion endpoint.

    llama-cli in this build defaults to interactive conversation mode and
    refuses Spark's chat template without --jinja, so it never emits a plain
    completion. The server takes the prompt as raw text, which is what a
    like-for-like comparison needs, and loads the model once instead of once
    per prompt.
    """
    import json as _json
    import urllib.request

    log = open("/tmp/llama_server_quality.log", "w")
    proc = subprocess.Popen(
        [binary, "-m", gguf, "-t", str(threads), "-c", "4096",
         "--port", str(port), "--host", "127.0.0.1", "-np", "1"],
        stdout=log, stderr=subprocess.STDOUT,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(600):
            try:
                urllib.request.urlopen(base + "/health", timeout=2).read()
                break
            except Exception:
                if proc.poll() is not None:
                    raise RuntimeError("llama-server exited; see /tmp/llama_server_quality.log")
                time.sleep(1)
        else:
            raise RuntimeError("llama-server did not become healthy")

        res = {}
        for name, text in prompts():
            body = _json.dumps({
                "prompt": text, "n_predict": n_predict,
                "temperature": 0.0, "top_k": 1, "seed": 0,
                "cache_prompt": False,
            }).encode()
            req = urllib.request.Request(
                base + "/completion", data=body,
                headers={"Content-Type": "application/json"})
            out = _json.loads(urllib.request.urlopen(req, timeout=1800).read())
            res[name] = {"text": out["content"], "token_ids": None}
        return res
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def compare(ref, got, label):
    rows, exact, tot_pref, tot_len = [], 0, 0, 0
    for name in ref:
        r, g = ref[name]["text"], got[name]["text"]
        cp = common_prefix(r, g)
        exact += int(r == g)
        tot_pref += cp
        tot_len += max(len(r), 1)
        rows.append({"prompt": name, "exact": r == g,
                     "common_prefix_chars": cp, "ref_chars": len(r)})
    return {"label": label, "prompts": len(ref),
            "exact_match": f"{exact}/{len(ref)}",
            "mean_prefix_kept": round(tot_pref / max(tot_len, 1), 4),
            "detail": rows}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--engine", choices=["vllm", "llamacpp"], required=True)
    g.add_argument("--model", default=None)
    g.add_argument("--gguf", default=None)
    g.add_argument("--binary",
                   default="/data/zhoutaichang/embedding_infer/llama.cpp/build/bin/llama-server")
    g.add_argument("--threads", type=int, default=32)
    g.add_argument("--n-predict", type=int, default=64)
    g.add_argument("--out", required=True)
    c = sub.add_parser("cmp")
    c.add_argument("--ref", required=True)
    c.add_argument("--got", required=True)
    c.add_argument("--label", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.cmd == "gen":
        res = (run_vllm(a.model, a.n_predict) if a.engine == "vllm"
               else run_llamacpp(a.binary, a.gguf, a.n_predict, a.threads))
        Path(a.out).write_text(json.dumps(res, indent=2))
        print(f"wrote {a.out} ({len(res)} prompts)")
    else:
        rep = compare(json.loads(Path(a.ref).read_text()),
                      json.loads(Path(a.got).read_text()), a.label)
        Path(a.out).write_text(json.dumps(rep, indent=2))
        print(json.dumps({k: v for k, v in rep.items() if k != "detail"}, indent=2))


if __name__ == "__main__":
    main()
