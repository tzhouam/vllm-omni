#!/usr/bin/env python3
"""Profile an admitted InternVLA action stage under Omni StageRuntime."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rank(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "cosmos-dir", "processor-dir", "python-bin", "log-file", "output-report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--graph-file", type=Path)
    parser.add_argument("--prefix-file", type=Path)
    parser.add_argument("--suffix-file", type=Path)
    parser.add_argument("--ep-dir", type=Path)
    parser.add_argument("--placement", choices=("cpu", "cuda", "radeon-cosmos", "amd-npu-conv13"), required=True)
    parser.add_argument("--output-actions", type=Path)
    parser.add_argument("--reference-actions", type=Path)
    parser.add_argument("--capacity-gib", type=int, default=30)
    parser.add_argument("--reserve-gib", type=int, default=16)
    parser.add_argument("--vram-capacity-gib", type=int)
    parser.add_argument("--vram-reserve-gib", type=int)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--admission-refusal", action="store_true")
    parser.add_argument("--abort-check", action="store_true")
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1 or args.reserve_gib < 1 or args.capacity_gib < args.reserve_gib:
        parser.error("invalid profile counts or explicit memory budget")
    if args.placement == "radeon-cosmos" and args.graph_file is None:
        parser.error("Radeon placement requires --graph-file")
    if args.placement == "amd-npu-conv13" and not all((
        args.graph_file, args.prefix_file, args.suffix_file, args.ep_dir,
    )):
        parser.error("AMD NPU placement requires graph, prefix, suffix and EP directory")
    if args.placement == "cuda" and (
        args.vram_capacity_gib is None or args.vram_reserve_gib is None
        or args.vram_reserve_gib < 1 or args.vram_capacity_gib < args.vram_reserve_gib
    ):
        parser.error("CUDA placement requires an explicit positive VRAM capacity and reservation")
    if args.placement != "cuda" and (args.vram_capacity_gib is not None or args.vram_reserve_gib is not None):
        parser.error("VRAM budget is only valid for CUDA placement")

    import numpy as np
    import psutil

    host_available_bytes = psutil.virtual_memory().available
    if args.capacity_gib << 30 > host_available_bytes:
        raise RuntimeError("declared host-RAM capacity exceeds OS available RAM before load")

    from vllm_omni.config.stage_config import DeployConfig, StageDeployConfig, merge_pipeline_deploy
    from vllm_omni.diffusion.models.internvla_a1_whole_pipeline import INTERNVLA_A1_WHOLE_POLICY_PIPELINE
    from vllm_omni.engine.stage_runtime import StageRuntime

    model = args.model_dir.resolve(strict=True)
    cosmos = args.cosmos_dir.resolve(strict=True)
    processor = args.processor_dir.resolve(strict=True)
    python = args.python_bin.absolute()
    graph = args.graph_file.resolve(strict=True) if args.graph_file else None
    prefix = args.prefix_file.resolve(strict=True) if args.prefix_file else None
    suffix = args.suffix_file.resolve(strict=True) if args.suffix_file else None
    ep_dir = args.ep_dir.resolve(strict=True) if args.ep_dir else None
    files = {
        "python": python, "model": model / "model.safetensors",
        "model_config": model / "config.json", "train_config": model / "train_config.json",
        "stats": model / "stats.json", "cosmos_encoder": cosmos / "encoder.safetensors",
        "cosmos_decoder": cosmos / "decoder.safetensors",
        "processor_tokenizer": processor / "tokenizer.json",
        "processor_config": processor / "preprocessor_config.json",
    }
    if graph is not None:
        files["graph"] = graph
    if args.placement == "amd-npu-conv13":
        files.update(prefix=prefix, suffix=suffix, ep_dll=ep_dir / "onnxruntime_vitisai_ep.dll")
    hashes = {name: sha256(path) for name, path in files.items()}
    import torch

    cuda_before = None
    if args.placement == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA placement requested but no CUDA device is available")
        free_vram, total_vram = torch.cuda.mem_get_info(0)
        cuda_before = {"free_bytes": free_vram, "total_bytes": total_vram,
                       "device_name": torch.cuda.get_device_name(0)}
        if args.vram_capacity_gib << 30 > free_vram:
            raise RuntimeError("declared CUDA capacity exceeds observed free VRAM before load")

    backend = {
        "name": "external.internvla.policy.v1", "placement": args.placement,
        "expected_cuda_device_name": cuda_before["device_name"] if cuda_before else None,
        "python_bin": str(python), "expected_torch": str(torch.__version__),
        "model_dir": str(model), "cosmos_dir": str(cosmos), "processor_dir": str(processor),
        "graph_file": str(graph) if graph else None, "artifact_sha256": hashes,
        "prefix_file": str(prefix) if prefix else None,
        "suffix_file": str(suffix) if suffix else None,
        "ep_dir": str(ep_dir) if ep_dir else None,
        "log_file": str(args.log_file), "memory_overhead_bytes": 8 << 30,
        "max_input_bytes": 8 << 20, "max_action_bytes": 1 << 20,
        "start_timeout_s": 180, "request_timeout_s": 60,
    }
    budget = {"capacities": {"host_ram": args.capacity_gib << 30},
              "demands": {"host_ram": args.reserve_gib << 30}}
    if args.placement == "cuda":
        budget["capacities"]["cuda:0"] = args.vram_capacity_gib << 30
        budget["demands"]["cuda:0"] = args.vram_reserve_gib << 30
    deploy = DeployConfig(async_chunk=False, stages=[StageDeployConfig(
        stage_id=0, backend=backend, resource_budget=budget,
    )])
    configs = [item.to_omegaconf() for item in merge_pipeline_deploy(
        INTERNVLA_A1_WHOLE_POLICY_PIPELINE, deploy,
    )]
    runtime = StageRuntime(configs, "local-internvla-policy", "", stage_init_timeout=180, async_chunk=False)
    report = {
        "scope": "real InternVLA Place_Markpen checkpoint via bounded Omni whole-policy graph stage; synthetic patterned observations/noise; no robot-task quality claim",
        "placement": args.placement, "artifact_sha256": hashes, "budget": budget,
        "host_available_bytes_before": host_available_bytes,
        "cuda_before": cuda_before,
        "warmups": args.warmups, "repeats": args.repeats,
    }
    try:
        started = time.perf_counter()
        if args.admission_refusal:
            try:
                runtime.initialize()
            except Exception as exc:
                if "exceed reservation" not in str(exc):
                    raise
                report["admission_refusal"] = f"{type(exc).__name__}: {exc}"
                report["status"] = "passed"
                return
            raise AssertionError("InternVLA admitted the insufficient memory reservation")
        runtime.initialize()
        report["startup_s"] = time.perf_counter() - started
        pool = runtime.stage_pools[0]
        report["execution_plan_start"] = dict(pool.stage_client.execution_plan)
        state = SimpleNamespace(sampling_params_list=[None])
        images = [np.zeros((1, 2, 3, 224, 224), dtype=np.float32) for _ in range(3)]
        images[0][:, :, 0, 56:168, 56:168] = 1.0
        images[1][:, :, 1, 56:168, 56:168] = .5
        prompt = {
            **{f"image{i}": images[i] for i in range(3)},
            **{f"mask{i}": np.ones((1,), dtype=np.bool_) for i in range(3)},
            "state": np.zeros((1, 32), dtype=np.float32),
            "noise": np.zeros((1, 50, 32), dtype=np.float32),
            "task": "Place the marker pen in its holder.",
        }

        async def request_one(request_id: str):
            prompt["observation_timestamp_ns"] = time.time_ns()
            started = time.perf_counter()
            await pool.submit_initial(request_id, state, prompt)
            deadline = time.monotonic() + 75
            while True:
                output = pool.poll_graph_output(0)
                if output is not None:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("InternVLA Omni action request timed out")
                await asyncio.sleep(.005)
            try:
                if output.error:
                    raise RuntimeError(output.error)
                values = np.asarray(output.custom_output["actions"])
                event = output.custom_output["stage_event"]
                metadata = output.custom_output["action_metadata"]
                if values.dtype != np.float32 or values.shape != (1, 50, 32) or not np.isfinite(values).all():
                    raise RuntimeError("InternVLA returned invalid actions")
                if not event["terminal"] or event["kind"] != "action" or metadata["control_ready"]:
                    raise RuntimeError("InternVLA action event or metadata differs from contract")
                return {"wall_s": time.perf_counter() - started,
                        "action_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                        "stage_event": event, "action_metadata": metadata,
                        "metrics": output.metrics, "actions": values.copy()}
            finally:
                output.release_stage_buffers()
                await asyncio.sleep(0)

        report["warmup_results"] = []
        for index in range(args.warmups):
            row = await request_one(f"internvla-warmup-{index}")
            row.pop("actions")
            report["warmup_results"].append(row)
        measured = []
        report["measured"] = measured
        first_actions = None
        for index in range(args.repeats):
            row = await request_one(f"internvla-measured-{index}")
            values = row.pop("actions")
            if first_actions is None:
                first_actions = values
            measured.append(row)
        assert first_actions is not None
        report["nearest_rank_p50_wall_s"] = rank([r["wall_s"] for r in measured], .5)
        report["nearest_rank_p95_wall_s"] = rank([r["wall_s"] for r in measured], .95)
        report["unique_action_hashes"] = sorted({r["action_sha256"] for r in measured})
        if args.output_actions:
            args.output_actions.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.output_actions, first_actions)
            report["output_actions"] = str(args.output_actions)
        if args.reference_actions:
            reference = np.load(args.reference_actions, allow_pickle=False)
            if reference.shape != first_actions.shape:
                raise RuntimeError("reference action shape differs")
            difference = first_actions - reference
            report["reference_comparison"] = {
                "max_abs": float(np.max(np.abs(difference))),
                "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(reference)),
                "cosine": float(np.dot(first_actions.ravel(), reference.ravel()) / (
                    np.linalg.norm(first_actions) * np.linalg.norm(reference)
                )),
            }
        report["execution_plan_end"] = dict(pool.stage_client.execution_plan)
        if args.abort_check:
            prompt["observation_timestamp_ns"] = time.time_ns()
            await pool.submit_initial("internvla-abort", state, prompt)
            await asyncio.sleep(.05)
            await pool.abort_requests(["internvla-abort"])
            await asyncio.sleep(.1)
            report["abort_check"] = {
                "stale_output": pool.poll_graph_output(0) is not None,
                "ledger": runtime.resource_ledger.snapshot(),
            }
            if report["abort_check"]["stale_output"]:
                raise RuntimeError("InternVLA cancellation leaked a stale action")
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        runtime.shutdown()
        if runtime.resource_ledger is not None:
            report["ledger_after_shutdown"] = runtime.resource_ledger.snapshot()
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "measured"}, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
