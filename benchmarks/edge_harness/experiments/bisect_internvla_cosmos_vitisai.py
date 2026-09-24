#!/usr/bin/env python3
"""Extract and test real-weight Cosmos encoder prefixes on HX370 VitisAI.

Each probe writes status before native session creation so a compiler abort
does not masquerade as a missing experiment. A prefix is component evidence.
"""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def extract(args: argparse.Namespace) -> None:
    import onnx

    if sha256(args.source) != SOURCE_SHA or sha256(args.source.with_name(args.source.name + ".data")) != WEIGHTS_SHA:
        raise ValueError("Cosmos source graph or external weights changed")
    source = onnx.load(str(args.source), load_external_data=True)
    if not 1 <= args.cut <= len(source.graph.node):
        raise ValueError("cut is outside the one-based graph node range")
    terminal = source.graph.node[args.cut - 1]
    if not terminal.output or not terminal.output[0]:
        raise ValueError("cut has no first output")
    candidate = onnx.utils.Extractor(source).extract_model(["pixels"], [terminal.output[0]])
    onnx.checker.check_model(candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(candidate, str(args.output))
    write_report(args.report, {
        "source_sha256": SOURCE_SHA,
        "external_weights_sha256": WEIGHTS_SHA,
        "cut_one_based": args.cut,
        "terminal_op": terminal.op_type,
        "terminal_name": terminal.name,
        "terminal_output": terminal.output[0],
        "extracted_nodes": len(candidate.graph.node),
        "extracted_ops": dict(Counter(node.op_type for node in candidate.graph.node)),
        "candidate_sha256": sha256(args.output),
        "candidate_bytes": args.output.stat().st_size,
        "output_type": str(candidate.graph.output[0].type),
        "scope": "real-weight image encoder prefix; no complete policy or E2E claim",
    })


def probe(args: argparse.Namespace) -> None:
    import numpy as np
    import onnxruntime as ort

    extraction = json.loads(args.extraction_report.read_text(encoding="utf-8"))
    if sha256(args.model) != extraction["candidate_sha256"] or sha256(args.fixtures) != FIXTURES_SHA:
        raise ValueError("candidate or fixtures changed")
    with np.load(args.fixtures, allow_pickle=False) as data:
        pixels = np.ascontiguousarray(data["pattern_pixels"])
    if pixels.shape != (6, 3, 256, 256) or pixels.dtype != np.float32:
        raise ValueError("pattern fixture contract changed")
    report = dict(extraction)
    report.update({
        "fixture_sha256": FIXTURES_SHA,
        "fixture": "pattern_pixels",
        "hardware": "Ryzen AI 9 HX 370 AMD NPU",
        "os": platform.platform(),
        "onnxruntime": ort.__version__,
        "status": "cpu_session_start",
    })
    write_report(args.report, report)
    cpu = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    start = time.perf_counter()
    expected = cpu.run(None, {"pixels": pixels})[0]
    report.update({
        "cpu_wall_s": time.perf_counter() - start,
        "output_shape": list(expected.shape),
        "output_finite": bool(np.isfinite(expected).all()),
        "status": "cpu_component_pass",
    })
    write_report(args.report, report)
    if not report["output_finite"]:
        raise ValueError("CPU prefix produced a nonfinite output")
    if args.cpu_only:
        return

    ep_dir = args.ep_dir.resolve(strict=True)
    os.environ["PATH"] = str(ep_dir) + os.pathsep + os.environ.get("PATH", "")
    os.add_dll_directory(str(ep_dir))
    ort.register_execution_provider_library("vitisai", str(ep_dir / "onnxruntime_vitisai_ep.dll"))
    devices = [device for device in ort.get_ep_devices()
               if device.ep_name == "vitisai" and str(device.device.type).endswith("NPU")]
    report["npu_devices"] = len(devices)
    report["status"] = "vitisai_session_creation_started"
    write_report(args.report, report)
    if not devices:
        raise RuntimeError("VitisAI did not expose an NPU")
    options = ort.SessionOptions()
    provider_options: dict[str, str] = {}
    if args.config_file is not None:
        if args.config_sha256 is None or sha256(args.config_file) != args.config_sha256:
            raise ValueError("VitisAI config hash differs from pinned value")
        provider_options["config_file"] = str(args.config_file.resolve(strict=True))
        report["vitisai_config_sha256"] = args.config_sha256
        write_report(args.report, report)
    options.add_provider_for_devices(devices, provider_options)
    options.enable_profiling = True
    options.profile_file_prefix = str(args.profile_prefix)
    start = time.perf_counter()
    session = ort.InferenceSession(str(args.model), sess_options=options)
    report["session_create_s"] = time.perf_counter() - start
    report["status"] = "vitisai_session_created"
    write_report(args.report, report)
    start = time.perf_counter()
    actual = session.run(None, {"pixels": pixels})[0]
    report["npu_wall_s"] = time.perf_counter() - start
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("NPU output shape or finiteness failed")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    report["relative_l2_vs_cpu"] = float(
        np.linalg.norm(difference) / max(np.linalg.norm(expected.astype(np.float64)), 1e-12))
    profile = Path(session.end_profiling())
    report["profile_file"] = str(profile)
    events = json.loads(profile.read_text(encoding="utf-8"))
    providers = [(event.get("args") or {}).get("provider") for event in events if event.get("cat") == "Node"]
    report["node_providers"] = dict(Counter(providers))
    report["status"] = "npu_component_numeric_pass" if providers.count("vitisai") and report["relative_l2_vs_cpu"] <= 0.01 else "component_not_qualified"
    write_report(args.report, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extraction = sub.add_parser("extract")
    extraction.add_argument("--source", type=Path, required=True)
    extraction.add_argument("--cut", type=int, required=True)
    extraction.add_argument("--output", type=Path, required=True)
    extraction.add_argument("--report", type=Path, required=True)
    execution = sub.add_parser("probe")
    execution.add_argument("--model", type=Path, required=True)
    execution.add_argument("--extraction-report", type=Path, required=True)
    execution.add_argument("--fixtures", type=Path, required=True)
    execution.add_argument("--ep-dir", type=Path, required=True)
    execution.add_argument("--profile-prefix", type=Path, required=True)
    execution.add_argument("--report", type=Path, required=True)
    execution.add_argument("--cpu-only", action="store_true")
    execution.add_argument("--config-file", type=Path)
    execution.add_argument("--config-sha256")
    args = parser.parse_args()
    (extract if args.command == "extract" else probe)(args)


if __name__ == "__main__":
    main()
