#!/usr/bin/env python3
"""Extract a pinned Code2Wav ONNX prefix and probe AMD VitisAI compilation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def extract(args: argparse.Namespace) -> None:
    import onnx

    source_hash = sha256(args.source)
    if source_hash != args.expected_source_sha256:
        raise ValueError("source ONNX hash changed")
    source = onnx.load(args.source)
    if not 1 <= args.cut <= len(source.graph.node):
        raise ValueError("cut must be a one-based node ordinal")
    terminal = source.graph.node[args.cut - 1]
    output_name = terminal.output[0]
    if not output_name:
        raise ValueError("selected node has an empty first output")
    inputs = args.input_tensor or [item.name for item in source.graph.input]
    candidate = onnx.utils.Extractor(source).extract_model(inputs, [output_name])
    onnx.checker.check_model(candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(candidate, args.output)
    report = {
        "scope": "compiler bisection component, not a vocoder waveform or TTS stream",
        "source_sha256": source_hash,
        "source_node_count": len(source.graph.node),
        "cut_one_based": args.cut,
        "terminal_node": terminal.name,
        "terminal_op": terminal.op_type,
        "terminal_output": output_name,
        "boundary_inputs": inputs,
        "extracted_node_count": len(candidate.graph.node),
        "extracted_ops": dict(Counter(node.op_type for node in candidate.graph.node)),
        "output_type": str(candidate.graph.output[0].type),
        "candidate_file": str(args.output.resolve()),
        "candidate_bytes": args.output.stat().st_size,
        "candidate_sha256": sha256(args.output),
    }
    write_report(args.report, report)
    print(json.dumps({key: report[key] for key in
                      ("cut_one_based", "extracted_node_count", "candidate_sha256")}))


def probe(args: argparse.Namespace) -> None:
    import numpy as np
    import onnxruntime as ort

    report = json.loads(args.extraction_report.read_text(encoding="utf-8"))
    expected_model_hash = report.get("candidate_sha256", report.get("rewritten_sha256"))
    if sha256(args.model) != expected_model_hash:
        raise ValueError("candidate ONNX hash changed")
    if args.synthetic_input:
        fixture_hash = None
        quantized = None
    else:
        if args.fixture is None or args.expected_fixture_sha256 is None:
            raise ValueError("fixture and expected hash are required for real input")
        if sha256(args.fixture) != args.expected_fixture_sha256:
            raise ValueError("fixture hash changed")
        with np.load(args.fixture, allow_pickle=False) as source:
            quantized = np.ascontiguousarray(source["quantized"])
        if quantized.shape != (1, 512, 74) or quantized.dtype != np.float32:
            raise ValueError("fixture shape/dtype changed")
        fixture_hash = args.expected_fixture_sha256
    report.update({
        "os": platform.platform(),
        "onnxruntime_version": ort.__version__,
        "fixture_sha256": fixture_hash,
        "input_provenance": "synthetic fixed-seed normal" if args.synthetic_input else "pinned real-weight vocoder fixture",
        "status": "started",
    })
    write_report(args.report, report)
    cpu = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    cpu_input = cpu.get_inputs()[0]
    if args.synthetic_input:
        shape = cpu_input.shape
        if any(not isinstance(size, int) or size <= 0 for size in shape):
            raise ValueError("synthetic input requires a fully static tensor shape")
        quantized = np.random.default_rng(42).standard_normal(shape).astype(np.float32)
    started = time.perf_counter()
    cpu_output = cpu.run(None, {cpu_input.name: quantized})[0]
    report.update({
        "cpu_output_shape": list(cpu_output.shape),
        "cpu_output_finite": bool(np.isfinite(cpu_output).all()),
        "cpu_inference_s": time.perf_counter() - started,
        "status": "cpu_component_pass",
    })
    write_report(args.report, report)
    if not report["cpu_output_finite"]:
        raise ValueError("CPU prefix output is nonfinite")
    if args.cpu_only:
        return

    ep_dir = args.ep_dir.resolve(strict=True)
    os.environ["PATH"] = str(ep_dir) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(ep_dir))
    ort.register_execution_provider_library(
        "vitisai", str(ep_dir / "onnxruntime_vitisai_ep.dll"))
    devices = [device for device in ort.get_ep_devices()
               if device.ep_name == "vitisai" and str(device.device.type).endswith("NPU")]
    report["vitisai_npu_device_count"] = len(devices)
    provider_options: dict[str, str] = {}
    if args.config_file is not None:
        if args.expected_config_sha256 is None:
            raise ValueError("expected config hash is required")
        config_hash = sha256(args.config_file)
        if config_hash != args.expected_config_sha256:
            raise ValueError("VitisAI config hash changed")
        provider_options["config_file"] = str(args.config_file.resolve(strict=True))
        report["vitisai_config_sha256"] = config_hash
        report["vitisai_config_file"] = provider_options["config_file"]
    report["status"] = "vitisai_session_creation_started"
    write_report(args.report, report)
    if not devices:
        raise RuntimeError("VitisAI EP did not expose an NPU device")
    options = ort.SessionOptions()
    options.add_provider_for_devices(devices, provider_options)
    options.enable_profiling = True
    options.profile_file_prefix = str(args.profile_prefix)
    started = time.perf_counter()
    session = ort.InferenceSession(str(args.model), sess_options=options)
    report["vitisai_session_create_s"] = time.perf_counter() - started
    report["status"] = "vitisai_session_created"
    write_report(args.report, report)
    started = time.perf_counter()
    actual = session.run(None, {cpu_input.name: quantized})[0]
    report["vitisai_inference_s"] = time.perf_counter() - started
    if actual.shape != cpu_output.shape:
        raise ValueError("VitisAI prefix output shape changed")
    report["vitisai_output_finite"] = bool(np.isfinite(actual).all())
    report["relative_l2_vs_cpu"] = float(
        np.linalg.norm((actual - cpu_output).astype(np.float64))
        / max(np.linalg.norm(cpu_output.astype(np.float64)), 1e-12))
    if args.save_output is not None:
        if args.save_output.suffix.lower() != ".npz":
            raise ValueError("paired output path must end in .npz")
        args.save_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_output, cpu=cpu_output, npu=actual)
        report["paired_output_file"] = str(args.save_output.resolve())
        report["paired_output_sha256"] = sha256(args.save_output)
    profile_path = Path(session.end_profiling())
    report["profile_file"] = str(profile_path)
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    providers = [(event.get("args") or {}).get("provider") for event in events
                 if event.get("cat") == "Node"]
    report["node_providers"] = dict(Counter(providers))
    report["status"] = (
        "npu_component_numeric_pass"
        if providers.count("vitisai") > 0
        and report["vitisai_output_finite"]
        and report["relative_l2_vs_cpu"] <= 0.01
        else "component_not_qualified"
    )
    write_report(args.report, report)
    print(json.dumps({"status": report["status"],
                      "relative_l2_vs_cpu": report["relative_l2_vs_cpu"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extraction = subparsers.add_parser("extract")
    extraction.add_argument("--source", type=Path, required=True)
    extraction.add_argument("--expected-source-sha256", required=True)
    extraction.add_argument("--cut", type=int, required=True)
    extraction.add_argument("--input-tensor", action="append")
    extraction.add_argument("--output", type=Path, required=True)
    extraction.add_argument("--report", type=Path, required=True)
    execution = subparsers.add_parser("probe")
    execution.add_argument("--model", type=Path, required=True)
    execution.add_argument("--extraction-report", type=Path, required=True)
    execution.add_argument("--fixture", type=Path)
    execution.add_argument("--expected-fixture-sha256")
    execution.add_argument("--synthetic-input", action="store_true")
    execution.add_argument("--ep-dir", type=Path, required=True)
    execution.add_argument("--report", type=Path, required=True)
    execution.add_argument("--profile-prefix", type=Path, required=True)
    execution.add_argument("--config-file", type=Path)
    execution.add_argument("--expected-config-sha256")
    execution.add_argument("--save-output", type=Path)
    execution.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    (extract if args.command == "extract" else probe)(args)


if __name__ == "__main__":
    main()
