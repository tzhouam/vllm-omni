#!/usr/bin/env python3
"""Replay a measured VitisAI Code2Wav prefix through the unchanged CPU suffix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def relative_l2(reference, actual) -> float:
    import numpy as np

    reference = reference.astype(np.float64)
    actual = actual.astype(np.float64)
    return float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "fixture", "reference", "prefix-model", "prefix-report",
                 "npu-report", "paired-output", "suffix-output", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--waveforms-output", type=Path)
    parser.add_argument("--profile-requests", type=int, default=0)
    parser.add_argument("--reuse-suffix-sha256")
    args = parser.parse_args()
    if args.profile_requests < 0:
        parser.error("profile requests cannot be negative")

    import numpy as np
    import onnx
    import onnxruntime as ort

    extraction = json.loads(args.prefix_report.read_text(encoding="utf-8-sig"))
    observed = json.loads(args.npu_report.read_text(encoding="utf-8-sig"))
    boundary_name = extraction["terminal_output"]
    if (extraction["cut_one_based"] not in (100, 105)
            or boundary_name != "val_340"
            or extraction["source_sha256"] != sha256(args.source)
            or extraction["candidate_sha256"] != sha256(args.prefix_model)
            or observed["candidate_sha256"] != extraction["candidate_sha256"]
            or observed["status"] != "npu_component_numeric_pass"
            or observed["node_providers"].get("vitisai", 0) < 1
            or observed["paired_output_sha256"] != sha256(args.paired_output)
            or observed["fixture_sha256"] != sha256(args.fixture)):
        raise ValueError("Code2Wav source, NPU placement, fixture or captured boundary changed")
    with np.load(args.fixture, allow_pickle=False) as fixture:
        values = np.ascontiguousarray(fixture["quantized"])
    with np.load(args.reference, allow_pickle=False) as reference:
        reference_wave = np.ascontiguousarray(reference["wav"])
    with np.load(args.paired_output, allow_pickle=False) as pair:
        cpu_boundary = np.ascontiguousarray(pair["cpu"])
        npu_boundary = np.ascontiguousarray(pair["npu"])
    if (values.shape not in ((1, 512, 74), (1, 512, 97))
            or values.dtype != np.float32
            or reference_wave.shape != (1, 3840)
            or cpu_boundary.shape != npu_boundary.shape
            or cpu_boundary.ndim != 4 or cpu_boundary.shape[:2] != (1, 16)
            or cpu_boundary.shape[2:] != (values.shape[2], values.shape[2])
            or not np.isfinite(npu_boundary).all()):
        raise ValueError("fixed two-frame Code2Wav input/output contract changed")

    if args.reuse_suffix_sha256 is not None:
        if sha256(args.suffix_output) != args.reuse_suffix_sha256.lower():
            raise ValueError("reused CPU suffix hash differs from prior extraction")
        input_name, output_name = "quantized", "wav"
    else:
        source = onnx.load(str(args.source))
        input_name = source.graph.input[0].name
        output_name = source.graph.output[0].name
        suffix = onnx.utils.Extractor(source).extract_model(
            [input_name, boundary_name], [output_name]
        )
        onnx.checker.check_model(suffix)
        args.suffix_output.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(suffix, str(args.suffix_output))
    full_cpu = ort.InferenceSession(str(args.source), providers=["CPUExecutionProvider"])
    prefix_cpu = ort.InferenceSession(str(args.prefix_model), providers=["CPUExecutionProvider"])
    suffix_cpu = ort.InferenceSession(str(args.suffix_output), providers=["CPUExecutionProvider"])
    if ({item.name for item in suffix_cpu.get_inputs()} != {input_name, boundary_name}
            or [item.name for item in suffix_cpu.get_outputs()] != [output_name]):
        raise ValueError("extracted suffix contract changed")
    full = full_cpu.run(None, {input_name: values})[0]
    prefix_control = prefix_cpu.run(None, {input_name: values})[0]
    if relative_l2(prefix_control, cpu_boundary) > 1e-5:
        raise ValueError("captured CPU prefix differs from exact candidate")
    started = time.perf_counter()
    cpu_wave = suffix_cpu.run(None, {input_name: values, boundary_name: cpu_boundary})[0]
    cpu_suffix_s = time.perf_counter() - started
    started = time.perf_counter()
    npu_wave = suffix_cpu.run(None, {input_name: values, boundary_name: npu_boundary})[0]
    npu_suffix_s = time.perf_counter() - started
    zero_wave = suffix_cpu.run(None, {
        input_name: values, boundary_name: np.zeros_like(cpu_boundary)
    })[0]
    if (full.shape != cpu_wave.shape or full.shape != npu_wave.shape
            or full.shape != zero_wave.shape
            or not np.isfinite(npu_wave).all() or not np.isfinite(zero_wave).all()):
        raise ValueError("CPU suffix waveform shape or finiteness changed")
    if relative_l2(full, cpu_wave) > 1e-5:
        raise ValueError("CPU prefix plus CPU suffix does not reproduce full source")
    report = {
        "scope": "one real-weight two-frame Code2Wav NPU prefix plus unchanged CPU suffix on one pinned fixture; not complete TTS",
        "source_sha256": extraction["source_sha256"],
        "fixture_sha256": observed["fixture_sha256"],
        "reference_sha256": sha256(args.reference),
        "prefix_sha256": extraction["candidate_sha256"],
        "paired_output_sha256": observed["paired_output_sha256"],
        "suffix_sha256": sha256(args.suffix_output),
        "suffix_bytes": args.suffix_output.stat().st_size,
        "onnxruntime": ort.__version__,
        "npu_node_events": observed["node_providers"]["vitisai"],
        "prefix_boundary_relative_l2": relative_l2(cpu_boundary, npu_boundary),
        "prefix_boundary_cpu_max_abs": float(np.max(np.abs(cpu_boundary))),
        "prefix_boundary_npu_max_abs": float(np.max(np.abs(npu_boundary))),
        "full_cpu_vs_retained_reference_relative_l2": relative_l2(reference_wave, full),
        "cpu_suffix_vs_full_cpu_relative_l2": relative_l2(full, cpu_wave),
        "npu_prefix_cpu_suffix_waveform_relative_l2": relative_l2(cpu_wave, npu_wave),
        "npu_prefix_cpu_suffix_waveform_max_abs": float(np.max(np.abs(npu_wave - cpu_wave))),
        "zero_boundary_cpu_suffix_waveform_relative_l2": relative_l2(cpu_wave, zero_wave),
        "npu_prefix_cpu_suffix_waveform_snr_db": float(
            20 * np.log10(max(np.linalg.norm(cpu_wave), 1e-12)
                          / max(np.linalg.norm(npu_wave - cpu_wave), 1e-12))
        ),
        "cpu_suffix_one_call_s": cpu_suffix_s,
        "npu_injected_cpu_suffix_one_call_s": npu_suffix_s,
        "status": "measured_one_fixture",
    }
    if args.waveforms_output is not None:
        if args.waveforms_output.suffix.lower() != ".npz":
            raise ValueError("waveform output must end in .npz")
        args.waveforms_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.waveforms_output, cpu_full=full,
                            cpu_suffix=cpu_wave, npu_prefix_cpu_suffix=npu_wave,
                            zero_boundary_cpu_suffix=zero_wave)
        report["waveforms_sha256"] = sha256(args.waveforms_output)
    if args.profile_requests:
        timings = {"cpu_full_s": [], "cpu_suffix_s": [], "npu_captured_suffix_s": []}
        calls = (
            ("cpu_full_s", full_cpu, {input_name: values}, full),
            ("cpu_suffix_s", suffix_cpu,
             {input_name: values, boundary_name: cpu_boundary}, cpu_wave),
            ("npu_captured_suffix_s", suffix_cpu,
             {input_name: values, boundary_name: npu_boundary}, npu_wave),
        )
        for index in range(args.profile_requests):
            for name, session, inputs, expected in (
                calls if index % 2 == 0 else reversed(calls)
            ):
                start = time.perf_counter()
                actual = session.run(None, inputs)[0]
                timings[name].append(time.perf_counter() - start)
                if not np.array_equal(actual, expected):
                    raise RuntimeError(f"repeated {name} waveform changed")
        report["repeated_cpu_only_profile"] = {
            "scope": "same-process fixed-fixture CPU full graph versus CPU suffix with captured NPU tensor; NPU not rerun per request",
            "warmup": "one correctness call for each route before measurement",
            "requests": args.profile_requests,
            "seconds": timings,
            "nearest_rank_p50_p95_s": {
                name: [sorted(series)[math.ceil(len(series) * fraction) - 1]
                       for fraction in (.5, .95)]
                for name, series in timings.items()
            },
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "npu_prefix_cpu_suffix_waveform_relative_l2", "status"
    )}))


if __name__ == "__main__":
    main()
