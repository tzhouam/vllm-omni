"""Drive Spark-X2.5 through a whole phone task, not one decision.

    # against the in-process simulated phone (no device needed)
    python run_episode.py --task "Turn on Wi-Fi." --device sim

    # against a real phone
    adb devices                      # confirm it is listed
    python run_episode.py --task "Turn on Wi-Fi." --device adb

The server side is vLLM's stock OpenAI endpoint; Spark emits GLM-4.5's tool
format, so no custom parser is needed:

    vllm serve <model> --port 8077 --enable-auto-tool-choice \\
        --tool-call-parser glm45 --reasoning-parser glm45
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from agent import AgentConfig, run_episode
from device import AdbDevice, AdbError, demo_phone

TASKS = {
    "wifi": "Turn on Wi-Fi.",
    "alarm": "Set an alarm for 7:30 AM.",
}


def build_device(kind: str, serial: str | None):
    if kind == "sim":
        return demo_phone(), "simulated phone (settings + clock)"
    try:
        device = AdbDevice(serial=serial)
        w, h = device.size()
        return device, f"adb device {serial or '(default)'} {w}x{h}"
    except AdbError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=TASKS["wifi"],
                    help=f"free text, or one of: {', '.join(TASKS)}")
    ap.add_argument("--device", choices=("sim", "adb"), default="sim")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--base-url", default="http://127.0.0.1:8077/v1")
    ap.add_argument("--model", default="XHToken/Spark-X2.5-1.7B")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--no-think-fallback", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    task = TASKS.get(a.task, a.task)
    device, described = build_device(a.device, a.serial)
    cfg = AgentConfig(
        base_url=a.base_url, model=a.model, max_steps=a.max_steps,
        max_tokens=a.max_tokens, think=not a.no_think,
        think_fallback=not a.no_think_fallback,
    )

    print(f"device: {described}")
    print(f"task:   {task}")
    t0 = time.perf_counter()
    episode = run_episode(device, task, cfg)
    wall = time.perf_counter() - t0

    print()
    print(episode.transcript())
    latencies = [s.latency_s for s in episode.steps]
    print(f"\n{episode.n_steps} steps in {wall:.1f} s"
          f"  (median decision {sorted(latencies)[len(latencies) // 2]:.2f} s)"
          if latencies else f"\n{episode.n_steps} steps in {wall:.1f} s")
    if hasattr(device, "state"):
        print(f"device state: {device.state}")

    if a.out:
        payload = {
            "task": task, "device": described, "done": episode.done,
            "summary": episode.summary, "stop_reason": episode.stop_reason,
            "wall_s": round(wall, 2),
            "think": cfg.think, "max_steps": cfg.max_steps,
            "steps": [
                {"n": s.n, "tool": s.tool, "args": s.args,
                 "ok": s.result.ok if s.result else False,
                 "message": s.result.message if s.result else s.note,
                 "latency_s": round(s.latency_s, 3), "thinking": s.thinking}
                for s in episode.steps
            ],
            "final_state": getattr(device, "state", None),
        }
        with open(a.out, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"wrote {a.out}")

    raise SystemExit(0 if episode.done else 1)


if __name__ == "__main__":
    main()
