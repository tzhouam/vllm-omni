#!/usr/bin/env python3
"""Check a bounded InternVLA action request through public AsyncOmni."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import tempfile
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "cosmos-dir", "processor-dir", "python-bin", "log-file", "output-report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--graph-file", type=Path)
    parser.add_argument("--prefix-file", type=Path)
    parser.add_argument("--suffix-file", type=Path)
    parser.add_argument("--ep-dir", type=Path)
    parser.add_argument("--dml-python-bin", type=Path)
    parser.add_argument("--placement", choices=("cpu", "cuda", "radeon-cosmos", "amd-npu-conv13", "amd-npu-radeon-cosmos"), required=True)
    parser.add_argument("--capacity-gib", type=int, default=30)
    parser.add_argument("--reserve-gib", type=int, default=16)
    parser.add_argument("--vram-capacity-gib", type=int)
    parser.add_argument("--vram-reserve-gib", type=int)
    args = parser.parse_args()
    if args.placement == "radeon-cosmos" and args.graph_file is None:
        parser.error("Radeon placement requires graph-file")
    if args.placement in {"amd-npu-conv13", "amd-npu-radeon-cosmos"} and not all((
        args.graph_file, args.prefix_file, args.suffix_file, args.ep_dir,
    )):
        parser.error("AMD NPU placement requires graph, prefix, suffix and EP directory")
    if args.placement == "amd-npu-radeon-cosmos" and args.dml_python_bin is None:
        parser.error("joint AMD placement requires a DirectML interpreter")
    if args.placement != "amd-npu-radeon-cosmos" and args.dml_python_bin is not None:
        parser.error("DirectML interpreter is only valid for joint AMD placement")
    if args.reserve_gib < 1 or args.capacity_gib < args.reserve_gib:
        parser.error("invalid explicit host-RAM budget")
    if args.placement == "cuda" and (
        args.vram_capacity_gib is None or args.vram_reserve_gib is None
        or args.vram_reserve_gib < 1 or args.vram_capacity_gib < args.vram_reserve_gib
    ):
        parser.error("CUDA placement requires an explicit positive VRAM capacity and reservation")
    if args.placement != "cuda" and (args.vram_capacity_gib is not None or args.vram_reserve_gib is not None):
        parser.error("VRAM budget is only valid for CUDA placement")

    import numpy as np
    import psutil
    import torch
    import yaml

    from vllm_omni.diffusion.models.internvla_a1_whole_pipeline import INTERNVLA_A1_WHOLE_POLICY_PIPELINE
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    host_available_bytes = psutil.virtual_memory().available
    if args.capacity_gib << 30 > host_available_bytes:
        raise RuntimeError("declared host-RAM capacity exceeds OS available RAM before load")
    cuda_before = None
    if args.placement == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA placement requested but no CUDA device is available")
        free_vram, total_vram = torch.cuda.mem_get_info(0)
        cuda_before = {"free_bytes": free_vram, "total_bytes": total_vram,
                       "device_name": torch.cuda.get_device_name(0)}
        if args.vram_capacity_gib << 30 > free_vram:
            raise RuntimeError("declared CUDA capacity exceeds observed free VRAM before load")

    model = args.model_dir.resolve(strict=True)
    cosmos = args.cosmos_dir.resolve(strict=True)
    processor = args.processor_dir.resolve(strict=True)
    python = args.python_bin.absolute()
    graph = args.graph_file.resolve(strict=True) if args.graph_file else None
    prefix = args.prefix_file.resolve(strict=True) if args.prefix_file else None
    suffix = args.suffix_file.resolve(strict=True) if args.suffix_file else None
    ep_dir = args.ep_dir.resolve(strict=True) if args.ep_dir else None
    dml_python = args.dml_python_bin.resolve(strict=True) if args.dml_python_bin else None
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
    if args.placement in {"amd-npu-conv13", "amd-npu-radeon-cosmos"}:
        files.update(prefix=prefix, suffix=suffix, ep_dll=ep_dir / "onnxruntime_vitisai_ep.dll")
    if dml_python is not None:
        files["dml_python"] = dml_python
    backend = {
        "name": "external.internvla.policy.v1", "placement": args.placement,
        "expected_cuda_device_name": cuda_before["device_name"] if cuda_before else None,
        "python_bin": str(python), "expected_torch": str(torch.__version__),
        "model_dir": str(model), "cosmos_dir": str(cosmos), "processor_dir": str(processor),
        "graph_file": str(graph) if graph else None,
        "prefix_file": str(prefix) if prefix else None,
        "suffix_file": str(suffix) if suffix else None,
        "ep_dir": str(ep_dir) if ep_dir else None,
        "dml_python_bin": str(dml_python) if dml_python else None,
        "artifact_sha256": {name: sha256(path) for name, path in files.items()},
        "log_file": str(args.log_file), "memory_overhead_bytes": 8 << 30,
        "max_input_bytes": 8 << 20, "max_action_bytes": 1 << 20,
        "start_timeout_s": 180, "request_timeout_s": 60,
    }
    budget = {"capacities": {"host_ram": args.capacity_gib << 30},
              "demands": {"host_ram": args.reserve_gib << 30}}
    if args.placement == "cuda":
        budget["capacities"]["cuda:0"] = args.vram_capacity_gib << 30
        budget["demands"]["cuda:0"] = args.vram_reserve_gib << 30
    report = {"entrypoint": "AsyncOmni.generate", "placement": args.placement,
              "pipeline": INTERNVLA_A1_WHOLE_POLICY_PIPELINE.model_type,
              "host_available_bytes_before": host_available_bytes,
              "cuda_before": cuda_before, "budget": budget}
    engine = None
    try:
        with tempfile.TemporaryDirectory(prefix="omni-internvla-") as directory:
            deployment = Path(directory) / "deploy.yaml"
            deployment.write_text(yaml.safe_dump({
                "pipeline": INTERNVLA_A1_WHOLE_POLICY_PIPELINE.model_type,
                "async_chunk": False,
                "stages": [{"stage_id": 0, "backend": backend,
                            "resource_budget": budget}],
            }), encoding="utf-8")
            started = time.perf_counter()
            engine = AsyncOmni(
                model=str(model), deploy_config=str(deployment),
                stage_init_timeout=180, init_timeout=240,
            )
            report["startup_s"] = time.perf_counter() - started
            images = [np.zeros((1, 2, 3, 224, 224), dtype=np.float32) for _ in range(3)]
            images[0][:, :, 0, 56:168, 56:168] = 1
            images[1][:, :, 1, 56:168, 56:168] = .5
            prompt = {
                **{f"image{i}": images[i] for i in range(3)},
                **{f"mask{i}": np.ones((1,), dtype=np.bool_) for i in range(3)},
                "state": np.zeros((1, 32), dtype=np.float32),
                "noise": np.zeros((1, 50, 32), dtype=np.float32),
                "task": "Place the marker pen in its holder.",
                "observation_timestamp_ns": time.time_ns(),
            }
            outputs = []
            started = time.perf_counter()
            async for output in engine.generate(prompt, request_id="public-internvla-1"):
                if output.error:
                    raise RuntimeError(output.error)
                actions = np.asarray(output.custom_output["actions"])
                outputs.append({
                    "request_id": output.request_id,
                    "action_shape": list(actions.shape),
                    "action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                    "finite": bool(np.isfinite(actions).all()),
                    "metadata": output.custom_output.get("action_metadata"),
                    "stage_event": output.custom_output.get("stage_event"),
                })
            report["request_wall_s"] = time.perf_counter() - started
            report["outputs"] = outputs
            assert len(outputs) == 1 and outputs[0]["request_id"] == "public-internvla-1"
            assert outputs[0]["action_shape"] == [1, 50, 32] and outputs[0]["finite"]
            assert outputs[0]["metadata"]["control_ready"] is False
            report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if engine is not None:
            engine.shutdown()
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
