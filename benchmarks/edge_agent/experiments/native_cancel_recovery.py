"""One batch-1 native Windows cancel/restart smoke on an experimental route.

This is a functional check only. It cannot qualify model quality, latency,
memory peaks, or 30-minute sustained operation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import threading
from concurrent.futures import CancelledError
from pathlib import Path

from vllm_omni.edge.agent.native_app import build_controller


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--record", required=True, type=Path)
    args = parser.parse_args()
    source_config = json.loads(args.config.read_text(encoding="utf-8"))
    events: list[dict] = []
    cancel_requested = threading.Event()
    cancelled = threading.Event()
    outcome: dict[str, object] = {"batch_size": 1, "concurrency": 1,
                                  "scope": "cancel_and_recover_functional_smoke"}
    with tempfile.TemporaryDirectory(prefix="omni-agent-cancel-") as directory:
        root = Path(directory)
        config = dict(source_config)
        config["memory_file"] = str(root / "memory.sqlite")
        config["routes"] = [dict(route) for route in source_config["routes"]]
        for route in config["routes"]:
            route["log_file"] = str(root / f"{route['route_id']}.log")
        config_file = root / "config.json"
        config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")
        controller, hardware = build_controller(config_file)
        outcome["hardware"] = hardware
        outcome["config"] = config
        route_id = config["routes"][0]["route_id"]

        def listener(event: dict) -> None:
            events.append(event)
            if (event["kind"] == "text_delta" and not cancel_requested.is_set()
                    and event["epoch"] == 1):
                first_plan = controller.backends[route_id].execution_plan
                outcome["first_worker"] = {
                    "generation": first_plan["worker_generation"],
                    "pid": first_plan["worker_pid"],
                    "placement": first_plan["requested_device"],
                }
                cancel_requested.set()
                controller.cancel()
            if event["kind"] == "cancelled":
                cancelled.set()

        controller.add_listener(listener)
        try:
            first = controller.submit(
                "List each integer from 1 to 96, one number per line. Continue until 96."
            )
            try:
                first.result(timeout=120)
            except CancelledError:
                pass
            if not cancel_requested.wait(5) or not cancelled.wait(15):
                raise AssertionError("first Agent request was not cancelled after a streamed delta")
            if not controller._turn_done.wait(15):
                raise AssertionError("cancelled turn did not release its graph gate")
            releases = [event for event in events if event["kind"] == "state_released"
                        and event["epoch"] == 1]
            if len(releases) != 1:
                raise AssertionError("cancelled request lacks one verified state release")
            release = releases[0]["payload"]
            outcome["release_evidence"] = release
            if (release.get("worker_pid_before") != outcome["first_worker"]["pid"] or
                release.get("worker_exit_confirmed") is not True or
                release.get("stage_ledger_empty") is not True or
                release.get("host_claim_released") is not True or
                release.get("host_ledger_empty") is not True):
                raise AssertionError("worker exit or memory release proof is incomplete")
            second = controller.submit("Reply with the single word ready.")
            answer = second.result(timeout=120)
            second_plan = controller.backends[route_id].execution_plan
            outcome["second_worker"] = {
                "generation": second_plan["worker_generation"],
                "pid": second_plan["worker_pid"],
                "placement": second_plan["requested_device"],
            }
            if outcome["first_worker"]["generation"] == outcome["second_worker"]["generation"]:
                raise AssertionError("cancelled worker was reused instead of restarted")
            outcome["recovery_answer"] = answer
            outcome["exact_answer_pass"] = answer == "ready"
            if answer.strip().rstrip(".") != "ready":
                raise AssertionError(f"restarted route gave unexpected answer: {answer!r}")
            outcome["result"] = "cancelled_stream_then_restarted_request_passed"
        except Exception as exc:
            outcome["result"] = "failed"
            outcome["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                controller.close()
            finally:
                args.record.parent.mkdir(parents=True, exist_ok=True)
                for route in config["routes"]:
                    log = Path(route["log_file"])
                    if log.exists():
                        shutil.copyfile(log, args.record.with_name(
                            args.record.stem + f".{route['route_id']}.log"))
                with args.record.open("x", encoding="utf-8") as output:
                    output.write(json.dumps({"record_type": "manifest", **outcome},
                                            ensure_ascii=False) + "\n")
                    for event in events:
                        output.write(json.dumps(event, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
