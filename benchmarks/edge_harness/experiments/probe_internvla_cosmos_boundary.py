#!/usr/bin/env python3
"""Isolate a real Cosmos activation-to-convolution boundary on HX370 VitisAI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections import Counter
from pathlib import Path


SOURCE_SHA = "fee79864e393df35475ba243e14b4995c5651aa33e7e182cd1fda726425668b7"
WEIGHTS_SHA = "b9bc3411c10f05daec2ff977698ee100979ef756ed7f0d685275403517558b58"
FIXTURES_SHA = "dcbaf1391ca914d4bbccc1cb384daf05666e771d9cdae90c4f069ec8fcf05368"
PREFIX148_SHA = "07d1a1fcf33c6731a5c2cc3c629b31f8ba32ae58ed939fae5254fb935c4e5b62"
PREFIX151_SHA = "f28908b7fdd8b04574f66880837f23059bad7dfd63403d8482d5d6a0c0c6efe2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def relative_l2(expected, actual) -> float:
    import numpy as np

    difference = actual.astype(np.float64) - expected.astype(np.float64)
    return float(np.linalg.norm(difference) / max(np.linalg.norm(expected.astype(np.float64)), 1e-12))


def prepare(args: argparse.Namespace) -> None:
    import numpy as np
    import onnx
    import onnxruntime as ort

    if sha256(args.source) != SOURCE_SHA or sha256(args.source.with_name(args.source.name + ".data")) != WEIGHTS_SHA:
        raise ValueError("Cosmos source graph or external weights changed")
    if sha256(args.prefix148) != PREFIX148_SHA or sha256(args.prefix151) != PREFIX151_SHA or sha256(args.fixtures) != FIXTURES_SHA:
        raise ValueError("Cosmos boundary prefix or fixture changed")
    graph = onnx.load(str(args.source), load_external_data=True)
    suffix = onnx.utils.Extractor(graph).extract_model(["group_norm"], ["conv2d_13"])
    onnx.checker.check_model(suffix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(suffix, str(args.output))
    if [node.op_type for node in suffix.graph.node] != ["Sigmoid", "Mul", "Conv"]:
        raise ValueError("extracted boundary ops changed")
    cpu_prefix = ort.InferenceSession(str(args.prefix148), providers=["CPUExecutionProvider"])
    cpu_full_prefix = ort.InferenceSession(str(args.prefix151), providers=["CPUExecutionProvider"])
    cpu_suffix = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    inputs = {}
    parity = {}
    with np.load(args.fixtures, allow_pickle=False) as fixtures:
        for name in ("ramp", "pattern"):
            pixels = np.ascontiguousarray(fixtures[f"{name}_pixels"])
            activation = cpu_prefix.run(None, {"pixels": pixels})[0]
            if activation.shape != (6, 128, 64, 64) or activation.dtype != np.float32:
                raise ValueError("boundary activation contract changed")
            output = cpu_suffix.run(None, {"group_norm": activation})[0]
            full_output = cpu_full_prefix.run(None, {"pixels": pixels})[0]
            inputs[f"{name}_group_norm"] = activation
            parity[name] = {"output_finite": bool(np.isfinite(output).all()), "output_shape": list(output.shape),
                            "relative_l2_vs_source_prefix151": relative_l2(full_output, output)}
    args.boundary_fixtures.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.boundary_fixtures, **inputs)
    write(args.report, {
        "scope": "real-weight Cosmos Sigmoid+Mul+Conv subgraph with CPU-produced activation; no full encoder or policy claim",
        "source_sha256": SOURCE_SHA,
        "weights_sha256": WEIGHTS_SHA,
        "source_fixtures_sha256": FIXTURES_SHA,
        "prefix148_sha256": PREFIX148_SHA,
        "prefix151_sha256": PREFIX151_SHA,
        "candidate_sha256": sha256(args.output),
        "candidate_bytes": args.output.stat().st_size,
        "boundary_fixtures_sha256": sha256(args.boundary_fixtures),
        "boundary_fixtures_bytes": args.boundary_fixtures.stat().st_size,
        "boundary_shapes": parity,
        "extracted_ops": [node.op_type for node in suffix.graph.node],
    })


def probe(args: argparse.Namespace) -> None:
    import numpy as np
    import onnxruntime as ort

    prepared = json.loads(args.preparation_report.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate_report.read_text(encoding="utf-8")) if args.candidate_report else None
    if candidate is not None and (candidate["source_sha256"] != prepared["candidate_sha256"]
                                  or candidate["boundary_fixtures_sha256"] != prepared["boundary_fixtures_sha256"]):
        raise ValueError("quantized boundary candidate does not derive from pinned source")
    expected_model_hash = candidate["candidate_sha256"] if candidate is not None else prepared["candidate_sha256"]
    if sha256(args.model) != expected_model_hash or sha256(args.boundary_fixtures) != prepared["boundary_fixtures_sha256"]:
        raise ValueError("boundary model or fixture changed")
    with np.load(args.boundary_fixtures, allow_pickle=False) as data:
        activation = np.ascontiguousarray(data["pattern_group_norm"])
    if activation.shape != (6, 128, 64, 64) or activation.dtype != np.float32:
        raise ValueError("boundary activation contract changed")
    report = dict(prepared)
    if candidate is not None:
        report["fp32_boundary_sha256"] = prepared["candidate_sha256"]
        report["candidate_sha256"] = candidate["candidate_sha256"]
        report["candidate_bytes"] = candidate["candidate_bytes"]
        report["quantization"] = candidate["quantization"]
        report["candidate_cpu_parity_vs_fp32"] = candidate["relative_l2_vs_fp32_cpu"]
    report.update({"hardware": "Ryzen AI 9 HX 370 AMD NPU", "os": platform.platform(),
                   "onnxruntime": ort.__version__, "status": "cpu_session_start"})
    write(args.report, report)
    cpu = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    expected = cpu.run(None, {"group_norm": activation})[0]
    report["status"] = "cpu_component_pass"
    report["cpu_output_shape"] = list(expected.shape)
    report["cpu_output_finite"] = bool(np.isfinite(expected).all())
    write(args.report, report)
    if not report["cpu_output_finite"]:
        raise ValueError("CPU boundary output is nonfinite")
    ep_dir = args.ep_dir.resolve(strict=True)
    os.environ["PATH"] = str(ep_dir) + os.pathsep + os.environ.get("PATH", "")
    os.add_dll_directory(str(ep_dir))
    ort.register_execution_provider_library("vitisai", str(ep_dir / "onnxruntime_vitisai_ep.dll"))
    devices = [device for device in ort.get_ep_devices()
               if device.ep_name == "vitisai" and str(device.device.type).endswith("NPU")]
    report["npu_devices"] = len(devices)
    report["status"] = "vitisai_session_creation_started"
    write(args.report, report)
    if not devices:
        raise RuntimeError("VitisAI did not expose an NPU")
    options = ort.SessionOptions()
    provider_options: dict[str, str] = {}
    if args.config_file is not None:
        if args.config_sha256 is None or sha256(args.config_file) != args.config_sha256:
            raise ValueError("VitisAI config hash differs from pinned value")
        provider_options["config_file"] = str(args.config_file.resolve(strict=True))
        report["vitisai_config_sha256"] = args.config_sha256
        write(args.report, report)
    options.add_provider_for_devices(devices, provider_options)
    options.enable_profiling = True
    options.profile_file_prefix = str(args.profile_prefix)
    start = time.perf_counter()
    session = ort.InferenceSession(str(args.model), sess_options=options)
    report["session_create_s"] = time.perf_counter() - start
    report["status"] = "vitisai_session_created"
    write(args.report, report)
    start = time.perf_counter()
    actual = session.run(None, {"group_norm": activation})[0]
    report["inference_wall_s"] = time.perf_counter() - start
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("VitisAI boundary output contract failed")
    report["relative_l2_vs_cpu"] = relative_l2(expected, actual)
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    report["max_abs_vs_cpu"] = float(np.max(np.abs(difference)))
    report["cpu_range"] = [float(np.min(expected)), float(np.max(expected))]
    report["vitisai_range"] = [float(np.min(actual)), float(np.max(actual))]
    report["cosine_vs_cpu"] = float(
        np.sum(expected.astype(np.float64) * actual.astype(np.float64))
        / max(np.linalg.norm(expected.astype(np.float64)) * np.linalg.norm(actual.astype(np.float64)), 1e-12))
    if args.save_paired is not None:
        if args.save_paired.suffix.lower() != ".npz":
            raise ValueError("paired tensor file must be .npz")
        args.save_paired.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_paired, cpu=expected, vitisai=actual)
        report["paired_sha256"] = sha256(args.save_paired)
        report["paired_bytes"] = args.save_paired.stat().st_size
    profile = Path(session.end_profiling())
    report["profile_file"] = str(profile)
    events = json.loads(profile.read_text(encoding="utf-8"))
    providers = [(event.get("args") or {}).get("provider") for event in events if event.get("cat") == "Node"]
    report["node_providers"] = dict(Counter(providers))
    report["status"] = "npu_component_numeric_pass" if providers.count("vitisai") and report["relative_l2_vs_cpu"] <= 0.01 else "component_not_qualified"
    write(args.report, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--prefix148", type=Path, required=True)
    prep.add_argument("--prefix151", type=Path, required=True)
    prep.add_argument("--fixtures", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--boundary-fixtures", type=Path, required=True)
    prep.add_argument("--report", type=Path, required=True)
    run = sub.add_parser("probe")
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--preparation-report", type=Path, required=True)
    run.add_argument("--candidate-report", type=Path)
    run.add_argument("--boundary-fixtures", type=Path, required=True)
    run.add_argument("--ep-dir", type=Path, required=True)
    run.add_argument("--profile-prefix", type=Path, required=True)
    run.add_argument("--report", type=Path, required=True)
    run.add_argument("--config-file", type=Path)
    run.add_argument("--config-sha256")
    run.add_argument("--save-paired", type=Path)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else probe)(args)


if __name__ == "__main__":
    main()
