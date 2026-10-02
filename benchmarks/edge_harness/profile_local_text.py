# SPDX-License-Identifier: Apache-2.0
"""Measure the existing Omni local text path; do not infer model quality from speed."""

import argparse
import asyncio
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

from gpu_telemetry import GpuTelemetry
from single_request_protocol import LENGTH_BANDS, metadata as protocol_metadata, validate_settings


def configure_host_environment():
    """Reuse the native Windows settings established by the acceptance runs."""
    if os.name == "nt":
        cudart = Path(sys.prefix) / "Lib/site-packages/torch/lib/cudart64_13.dll"
        if cudart.is_file():
            os.environ.setdefault("VLLM_CUDART_SO_PATH", str(cudart))
        os.environ.setdefault("VLLM_CACHE_ROOT", str(Path.home() / "c29"))
    return {
        key: os.environ[key]
        for key in ("VLLM_CUDART_SO_PATH", "VLLM_CACHE_ROOT", "CUDA_HOME", "PYTHONUTF8", "PYTHONIOENCODING")
        if key in os.environ
    }


HOST_ENVIRONMENT = configure_host_environment()


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    try:
        temp.replace(path)
    except PermissionError as error:
        # Windows readers may deny replacement while holding the previous
        # snapshot open. Keep request JSONL authoritative and defer a running
        # snapshot rather than blocking concurrent inference on file I/O.
        if value.get("status") == "running" and path.is_file():
            value["report_write_permission_errors"] = value.get("report_write_permission_errors", 0) + 1
            value["last_report_write_error"] = repr(error)
            print(f"Deferred report snapshot after PermissionError: {path}", file=sys.stderr, flush=True)
            return
        # Initial/final snapshots must persist. This wait is outside measured
        # request execution; a persistent permission failure remains an error.
        for _ in range(20):
            time.sleep(0.05)
            try:
                temp.replace(path)
                return
            except PermissionError:
                pass
        raise


async def run(args):
    validate_settings(args.batch_size, args.concurrency)
    from vllm_omni.edge.local.engine import LocalTextEngine, apply_runtime_env
    from vllm_omni.edge.local.plan import plan_text_session

    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "scope": "complete local text generation; quality gate separate",
        "argv": sys.argv,
        "platform": platform.platform(),
        "python": sys.executable,
        "start_unix": time.time(),
        "runtime_env": apply_runtime_env(),
        "host_environment": HOST_ENVIRONMENT,
        "settings": vars(args),
        "profile_protocol": protocol_metadata(),
        "cache_condition": "existing disk/JIT cache; not cold disk",
        "trace": "untraced; API event timings",
        "requests": 0,
    }
    save(args.out / "report.json", report)
    engine = None
    gpu_telemetry = None
    if args.gpu_telemetry_interval_s > 0:
        gpu_telemetry = GpuTelemetry(args.out / "gpu_telemetry.jsonl", args.gpu_telemetry_interval_s)
        gpu_telemetry.start()
    try:
        plan = plan_text_session(
            args.model,
            max_model_len=4096,
            max_num_seqs=1,
            max_num_batched_tokens=512,
            enforce_eager=True,
            digest_weights=True,
        )
        report["plan"] = plan.to_dict()
        if not plan.admitted:
            report["status"] = "rejected"
            return
        engine = LocalTextEngine(plan)
        start = time.perf_counter()
        await engine.start()
        report["startup_s"] = time.perf_counter() - start
        report["placement"] = engine.report_placement()
        save(args.out / "report.json", report)
        # Repeated natural text gives reproducible length bands; record actual token counts.
        prompts = {
            name: ("The garden has trees, flowers, and a small pond. " * n) + "\nDescribe the garden in detail."
            for name, n in [("short", 4), ("medium", 40), ("long", 160)]
        }
        samples = (args.out / "requests.jsonl").open("a", encoding="utf-8")
        counter = 0

        async def request(name, concurrency, phase, slow=False):
            nonlocal counter
            counter += 1
            session = engine.open_session()
            start = time.perf_counter()
            rid, stream = await engine.submit(
                session, prompts[name], max_tokens=128, temperature=0.0, ignore_eos=True, max_chunks=4 if slow else 64
            )
            arrivals = []
            try:
                async for event in stream:
                    arrivals.append(
                        {
                            "elapsed_s": time.perf_counter() - start,
                            "kind": event.kind,
                            "sequence": event.seq,
                            "epoch": event.epoch,
                        }
                    )
                    if slow:
                        await asyncio.sleep(0.03)
                rec = engine.records[rid].to_dict()
                rec.update(
                    phase=phase,
                    length_band=name,
                    concurrency=concurrency,
                    batch_size=1,
                    wall_s=time.perf_counter() - start,
                    arrivals=arrivals,
                    stream=stream.stats(),
                )
                samples.write(json.dumps(rec) + "\n")
                samples.flush()
                if rec["error"] or rec["output_tokens"] != 128 or not rec["finished"]:
                    raise RuntimeError(f"Incomplete request: {rid}: {rec['error']}")
                report["requests"] = counter
                save(args.out / "report.json", report)
            finally:
                engine.close_session(session.session_id)

        for name in LENGTH_BANDS:
            await request(name, 1, "warmup")
            for _ in range(args.repeats):
                await request(name, 1, "measured")
            print(f"profiled {name} batch_size=1 concurrency=1", flush=True)
        thermal_start = time.perf_counter()
        report["sustained_start_unix"] = time.time()
        while time.perf_counter() - thermal_start < args.sustained_seconds:
            await request("medium", 1, "sustained")
        report["sustained_wall_s"] = time.perf_counter() - thermal_start
        await request("short", 1, "slow_consumer", slow=True)
        session = engine.open_session()
        rid, stream = await engine.submit(session, prompts["short"], max_tokens=512, ignore_eos=True)
        seen = 0
        async for event in stream:
            if event.kind == "token":
                seen += 1
            if seen >= 8:
                break
        start = time.perf_counter()
        report["cancel"] = await engine.cancel(rid)
        report["cancel"]["latency_s"] = time.perf_counter() - start
        report["cancel"]["remaining_events"] = [e.to_dict() async for e in stream]
        engine.close_session(session.session_id)
        await request("short", 1, "after_cancel")
        samples.close()
        report["usage"] = engine.report_usage()
        report["status"] = "completed"
        report["quality_gate"] = "not rerun against high precision reference; see separate acceptance evidence"
    except Exception as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if engine is not None:
            await engine.close()
            report["usage_after_close"] = engine.report_usage()
        if gpu_telemetry is not None:
            report["gpu_telemetry"] = gpu_telemetry.stop()
        report["end_unix"] = time.time()
        save(args.out / "report.json", report)
    return report["status"] == "completed"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--sustained-seconds", type=float, default=1800)
    p.add_argument("--batch-size", type=int, choices=(1,), default=1)
    p.add_argument("--concurrency", type=int, choices=(1,), default=1)
    p.add_argument("--gpu-telemetry-interval-s", type=float, default=0.0,
                   help="Sample device-wide NVML and host telemetry; 0 disables sampling.")
    args = p.parse_args()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm_omni.windows.aio import install_selector_policy

    install_selector_policy()
    raise SystemExit(0 if asyncio.run(run(args)) else 1)
