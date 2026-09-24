# SPDX-License-Identifier: Apache-2.0
"""Check Spark BF16 CPU and its NPU output-head graph resident together.

The NPU calls replay captured activations. They are not connected to the live
vLLM requests, so this is a co-residency/interference gate, not split E2E.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil

from vllm_omni.edge.hardware_probe import load_profile
from vllm_omni.edge.local.capabilities import FORMAT_ONNX_A16W8, enumerate_devices
from vllm_omni.edge.local.engine import LocalTextEngine
from vllm_omni.edge.local.external.stage import ExternalStage, plan_external_stage
from vllm_omni.edge.local.manifest import build_graph_artifact, runtime_versions
from vllm_omni.edge.local.plan import plan_text_session
from vllm_omni.edge.local.prompts import acceptance_prompts


GRAPH_SHA = "e2ae9c3fd7071e8ff628f209e6140c8a023bfaad59814b80a936c6b263f33eaf"
FIXTURE_SHA = "cfa2ed8439f8fe01730319667bfd01abf2e39cf5851490346a0331304cf63194"
BF16_INDEX_SHA = "cc2b212985f5d0469bf926903e4bd7ee81856687baa0c6c6b55ff200d4cdc63f"
BF16_REVISION = "14d6e83c13c7add2b62a7c39b2131f4ed1cddcf8"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def windows_available_bytes() -> int:
    # Windows physical RAM is the parent pool of WSL's quota and the NPU worker.
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command",
         "[long](Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1024"],
        text=True, capture_output=True, timeout=30, check=True,
    )
    return int(result.stdout.strip())


def rank(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


async def profile(args: argparse.Namespace) -> dict:
    checkout = Path(__file__).resolve().parents[3]
    if Path(inspect.getfile(plan_external_stage)).resolve() != checkout / "vllm_omni/edge/local/external/stage.py":
        raise RuntimeError("loaded Omni from another checkout; set PYTHONPATH=. from this repository")
    if digest(args.graph) != GRAPH_SHA or digest(args.fixture) != FIXTURE_SHA:
        raise ValueError("NPU graph or activation fixture differs from the pinned artifact")
    if digest(args.model / "model.safetensors.index.json") != BF16_INDEX_SHA:
        raise ValueError("BF16 checkpoint index differs from the pinned revision")
    identity = json.loads(args.weight_identity.read_text())
    if (identity.get("status") != "output_head_weights_equal_only"
        or identity.get("source_graph_sha256") != "f5f0c31eb438e9940ab242da8bf6f0f5dad6ebc9f6ef02d5475c21b744e595bc"
        or identity.get("bf16_checkpoint_index_sha256") != BF16_INDEX_SHA
        or not all(item.get("bitwise_equal_after_conversion") for item in identity.get("comparisons", []))
        or len(identity.get("comparisons", [])) != 2):
        raise ValueError("head weight identity is not established for this BF16 checkpoint")
    reference = json.loads(args.reference_profile.read_text())
    expected_tokens = reference["runs"][0]["token_ids_sha256"]
    if reference["runs"][0]["output_tokens"] != args.max_tokens:
        raise ValueError("reference profile has a different output length")
    with np.load(args.fixture, allow_pickle=False) as archive:
        activations, reference_logits = archive["x"].copy(), archive["hidden_out"].copy()

    cpu = plan_text_session(
        str(args.model), max_model_len=4096, max_num_seqs=1,
        max_num_batched_tokens=2048, enforce_eager=True,
    )
    graph = build_graph_artifact(
        args.graph, fmt=FORMAT_ONNX_A16W8, opset=21,
        source_model="XHToken/Spark-X2.5-1.7B",
        source_revision="448e61eb392c00f2c403185c5b56d5e0665bfaab",
        component="spark_output_head", exporter="probe_spark_amd_npu_lm_head.py --composite-with-norm",
        calibration={"fixture_sha256": FIXTURE_SHA, "samples": 4},
        parity={"previous_top1_matches": 4},
    )
    npu = plan_external_stage(
        graph, enumerate_devices(load_profile(use_torch=False)),
        require="npu:amd", min_fraction_on_target=0.125,
        worker_peak_rss_hint_bytes=args.worker_peak_rss_hint_bytes,
    )
    if not cpu.admitted or cpu.selected is None or cpu.selected.device_id != "cpu":
        raise RuntimeError("Spark BF16 CPU plan was not admitted")
    if not npu.admitted:
        raise RuntimeError(npu.summary())
    capacity = {
        "wsl_available_before_bytes": psutil.virtual_memory().available,
        "windows_available_before_bytes": windows_available_bytes(),
        "cpu_plan_peak_bytes": cpu.peak_bytes,
        "npu_plan_budget_bytes": npu.budget_bytes,
    }
    capacity["combined_peak_budget_bytes"] = cpu.peak_bytes + npu.budget_bytes
    if capacity["combined_peak_budget_bytes"] > min(
        capacity["wsl_available_before_bytes"],
        capacity["windows_available_before_bytes"],
    ):
        raise RuntimeError(f"joint shared-RAM admission refused: {capacity}")

    result: dict = {
        "status": "running",
        "scope": "BF16 Omni CPU complete requests with an idle co-resident AMD NPU head; separate captured-activation NPU calls",
        "bf16_revision": BF16_REVISION,
        "bf16_index_sha256": BF16_INDEX_SHA,
        "npu_graph_sha256": GRAPH_SHA,
        "fixture_sha256": FIXTURE_SHA,
        "runtime": runtime_versions().to_dict(),
        "started_unix": time.time(),
        "capacity": capacity,
        "cpu_plan": cpu.to_dict(),
        "npu_plan": npu.to_dict(),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "max_tokens": args.max_tokens,
        "cpu_requests": [],
        "npu_replay": [],
        "power_thermal": "not measured",
    }
    engine = LocalTextEngine(cpu)
    args.profile_dir.mkdir(parents=True, exist_ok=True)
    try:
        await engine.start()
        with ExternalStage(npu) as stage:
            placement = stage.open({"x": activations[0]}, profile_dir=args.profile_dir)
            result["npu_placement"] = placement.to_dict()
            result["windows_available_after_both_loads_bytes"] = windows_available_bytes()
            result["wsl_available_after_both_loads_bytes"] = psutil.virtual_memory().available
            name, prompt = acceptance_prompts()[0]
            for index in range(args.warmups + args.repeats):
                session = engine.open_session()
                started = time.perf_counter()
                request_id, stream = await engine.submit(
                    session, prompt, max_tokens=args.max_tokens,
                    temperature=0.0, ignore_eos=True,
                )
                async for _ in stream:
                    pass
                wall = time.perf_counter() - started
                record = engine.records[request_id]
                token_hash = hashlib.sha256(json.dumps(record.output_token_ids).encode()).hexdigest()
                if record.error or record.cancelled or not record.finished or token_hash != expected_tokens:
                    raise RuntimeError(f"CPU request {index} failed or changed tokens: {record.error}")
                result["cpu_requests"].append({
                    "kind": "warmup" if index < args.warmups else "measured",
                    "prompt": name, "prompt_tokens": record.prompt_tokens,
                    "output_tokens": record.output_tokens, "token_ids_sha256": token_hash,
                    "wall_s": wall, "ttft_s": record.ttft_s,
                })
                engine.close_session(session.session_id)
            for index in range(args.repeats):
                case = index % len(activations)
                outputs, timing = stage.run({"x": activations[case]})
                logits = outputs["logits_concatenated"]
                result["npu_replay"].append({
                    "case": case, **timing.to_dict(),
                    "top1_match": bool(np.argmax(logits) == np.argmax(reference_logits[case])),
                    "finite": bool(np.isfinite(logits).all()),
                })
            result["npu_stats"] = stage.stats()
            result["cpu_placement"] = engine.report_placement()
            result["cpu_usage"] = engine.report_usage()
            result["cpu_measured_peak"] = engine.measured_peak()
        result["status"] = "co_resident_independent_stages_profiled"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        await engine.close()
        result["ended_unix"] = time.time()
        result["windows_available_after_close_bytes"] = windows_available_bytes()
        result["wsl_available_after_close_bytes"] = psutil.virtual_memory().available
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    measured = [r["wall_s"] for r in result["cpu_requests"] if r["kind"] == "measured"]
    result["cpu_wall_p50_p95_s"] = [rank(measured, 0.5), rank(measured, 0.95)]
    result["npu_round_trip_p50_p95_s"] = [
        rank([r["round_trip_s"] for r in result["npu_replay"]], 0.5),
        rank([r["round_trip_s"] for r in result["npu_replay"]], 0.95),
    ]
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--reference-profile", type=Path, required=True)
    parser.add_argument("--weight-identity", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--worker-peak-rss-hint-bytes", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.warmups < 1 or args.repeats < 20 or args.max_tokens < 1:
        parser.error("expected at least one warmup, 20 measured runs and positive token count")
    result = asyncio.run(profile(args))
    print(json.dumps({key: result[key] for key in (
        "status", "cpu_wall_p50_p95_s", "npu_round_trip_p50_p95_s"
    )}, indent=2))


if __name__ == "__main__":
    main()
