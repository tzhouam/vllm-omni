"""Run the task suite and score it, so an agent change can be compared.

Success is decided by the phone's own settings and screen, not by the
transcript: an agent that says it turned Wi-Fi on and left it off fails here,
which is exactly the failure this model exhibits.

Two things changed after the plan review:

* ``--repeat N`` now walks the task's **variants** rather than re-running one
  fixture. At temperature 0 against fixed fixtures, repeats measure
  repeatability; they only become trials if the phone differs between them.
  The summary reports a pass rate per task with the variants named, so a build
  comparison rests on more than one cell.
* ``--device adb`` scores a real phone. Each run resets the device to the
  task's declared starting state first, and a task that cannot be scored from
  a phone (because its predicate needs flip history) is skipped and reported
  as skipped, rather than silently scored from final state.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict

from agent import AgentConfig, run_episode, wait_for_server
from phone_state import read_state
from tasks import SUITE, SUITE_BY_ID


def _reset_real_device(device, task) -> None:
    if not task.reset_adb:
        raise SystemExit(
            f"task {task.task_id!r} declares no reset_adb, so a phone run would "
            f"be scored against an unknown starting state. Add one or exclude "
            f"the task with --tasks."
        )
    for cmd in task.reset_adb:
        device._run(*cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8077/v1")
    ap.add_argument("--model", default="spark")
    ap.add_argument("--tasks", default="all",
                    help="comma-separated task ids, or 'all'")
    ap.add_argument("--repeat", type=int, default=1,
                    help="runs per task; each walks the next variant")
    ap.add_argument("--device", choices=("sim", "adb"), default="sim")
    ap.add_argument("--serial", default=None, help="adb device serial")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tasks = SUITE if a.tasks == "all" else [SUITE_BY_ID[t] for t in a.tasks.split(",")]
    cfg = AgentConfig(base_url=a.base_url, model=a.model,
                      think=not a.no_think, max_tokens=a.max_tokens)

    phone = None
    if a.device == "adb":
        from device import AdbDevice
        phone = AdbDevice(serial=a.serial)
        skipped = [t.task_id for t in tasks if t.needs]
        if skipped:
            print(f"skipping on a real phone (needs flip history): "
                  f"{', '.join(skipped)}")
        tasks = [t for t in tasks if not t.needs]

    if not wait_for_server(cfg, timeout_s=600):
        raise SystemExit(f"no model server at {a.base_url}")

    rows, t0 = [], time.perf_counter()
    for task in tasks:
        for run in range(a.repeat):
            if phone is not None:
                _reset_real_device(phone, task)
                device = phone
            else:
                device = task.build(run)
            cfg.max_steps = task.max_steps
            episode = run_episode(device, task.prompt, cfg)
            score = task.score(read_state(device), episode.done, run)
            score.update(
                run=run + 1, steps=episode.n_steps,
                stop_reason=episode.stop_reason,
                recovered_steps=sum(1 for s in episode.steps
                                    if s.parsed_via not in ("tool_calls", "none")),
                median_latency_s=round(
                    sorted(s.latency_s for s in episode.steps)[episode.n_steps // 2], 2)
                if episode.n_steps else None,
            )
            rows.append(score)
            mark = "PASS" if score["success"] else "fail"
            detail = []
            if not score.get("scorable", True):
                mark, detail = "skip", [score.get("reason", "")]
            else:
                if score["reached_goal"] and not score["declared_done"]:
                    detail.append("goal reached but never stopped")
                if score["declared_done"] and not score["reached_goal"]:
                    detail.append("claimed done, goal NOT reached")
                if score["collateral_damage"]:
                    detail.append("collateral damage")
            print(f"  {mark}  {task.task_id:18s} {score['variant']:10s} "
                  f"steps={score['steps']:2d} "
                  f"{score['stop_reason'][:34]:36s} {'; '.join(detail)}", flush=True)

    wall = time.perf_counter() - t0
    scored = [r for r in rows if r.get("scorable", True)]
    n_pass = sum(1 for r in scored if r["success"])
    reached = sum(1 for r in scored if r["reached_goal"])
    claimed = sum(1 for r in scored if r["declared_done"] and not r["reached_goal"])

    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in scored:
        by_task[r["task_id"]].append(r)
    print("\npass rate per task")
    for task_id, rs in by_task.items():
        passes = sum(1 for r in rs if r["success"])
        failed = [r["variant"] for r in rs if not r["success"]]
        print(f"  {task_id:18s} {passes}/{len(rs)}"
              + (f"   failed on: {', '.join(failed)}" if failed else ""))

    rates = [sum(1 for r in rs if r["success"]) / len(rs) for rs in by_task.values()]
    spread = (f"{min(rates):.0%}-{max(rates):.0%}" if rates else "-")
    print(f"\n{n_pass}/{len(scored)} passed in {wall:.0f}s"
          f"   (goal reached {reached}/{len(scored)}; "
          f"falsely claimed done {claimed}; per-task rate {spread}"
          + (f"; median {statistics.median(rates):.0%}" if rates else "") + ")")
    if len(scored) and len(by_task) and max(len(rs) for rs in by_task.values()) < 3:
        print("  NOTE: fewer than 3 runs per task -- enough to show a failure "
              "reproduces, not to rank builds.")

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"label": a.label, "base_url": a.base_url,
                       "model": a.model, "think": cfg.think,
                       "device": a.device, "repeat": a.repeat,
                       "passed": n_pass, "total": len(scored),
                       "skipped": len(rows) - len(scored),
                       "wall_s": round(wall, 1), "rows": rows}, f, indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
