"""Drive Spark-X2.5 through Android control steps against an OpenAI endpoint.

Measures whether the model picks the right tool and the right element, and how
long a decision takes -- the two things that decide whether a 1.7B model can
actually run a phone agent locally.
"""
import argparse, json, time
from urllib import request as urlrequest

from action_space import SYSTEM_PROMPT, TOOLS
from scenarios import SCENARIOS


def chat(base_url, model, messages, tools, timeout=900, max_tokens=1024,
         think=True):
    payload = {
        "model": model, "messages": messages, "tools": tools,
        "tool_choice": "auto", "temperature": 0.0, "max_tokens": max_tokens,
    }
    if not think:
        # Spark's template opens <think> by default; closing it up front makes
        # the model answer directly, which is what a phone agent wants.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    req = urlrequest.Request(f"{base_url}/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urlrequest.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8077/v1")
    ap.add_argument("--model", default="XHToken/Spark-X2.5-1.7B")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", default="android_agent_results.json")
    a = ap.parse_args()

    rows, correct_tool, correct_args = [], 0, 0
    for sc in SCENARIOS:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                f"Task: {sc['task']}\n\nCurrent screen:\n{sc['screen']}\n\n"
                "Choose the next action."},
        ]
        try:
            out, dt = chat(a.base_url, a.model, msgs, TOOLS,
                           max_tokens=a.max_tokens, think=not a.no_think)
        except Exception as e:
            rows.append({"id": sc["id"], "error": str(e)[:200]})
            print(f"{sc['id']:<20} ERROR {str(e)[:90]}", flush=True)
            continue
        msg = out["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        got_name = calls[0]["function"]["name"] if calls else None
        try:
            got_args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
        except Exception:
            got_args = {"_raw": calls[0]["function"]["arguments"]}

        name_ok = got_name in sc["accept_names"]
        args_ok = name_ok and all(
            str(got_args.get(k)) == str(v) for k, v in sc["expect"]["args"].items()
        ) if got_name == sc["expect"]["name"] else name_ok and not sc["expect"]["args"]
        correct_tool += int(name_ok)
        correct_args += int(bool(args_ok))
        rows.append({
            "id": sc["id"], "latency_s": round(dt, 2),
            "expected": sc["expect"], "got_name": got_name, "got_args": got_args,
            "tool_ok": name_ok, "args_ok": bool(args_ok),
            "reasoning_chars": len(msg.get("reasoning_content") or ""),
            "completion_tokens": out.get("usage", {}).get("completion_tokens"),
        })
        flag = "OK " if args_ok else ("~  " if name_ok else "X  ")
        print(f"{flag}{sc['id']:<20} {got_name}({got_args}) {dt:.1f}s", flush=True)

    n = len(SCENARIOS)
    summary = {
        "scenarios": n,
        "thinking": not a.no_think,
        "tool_accuracy": f"{correct_tool}/{n}",
        "action_accuracy": f"{correct_args}/{n}",
        "median_latency_s": round(
            sorted(r["latency_s"] for r in rows if "latency_s" in r)[
                max(0, len([r for r in rows if "latency_s" in r]) // 2)] , 2)
        if any("latency_s" in r for r in rows) else None,
        "rows": rows,
    }
    json.dump(summary, open(a.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
