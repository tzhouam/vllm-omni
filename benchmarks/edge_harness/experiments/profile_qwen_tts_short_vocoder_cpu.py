#!/usr/bin/env python3
"""Profile the pinned two-frame Code2Wav ONNX artifact on local ORT CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rank(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("export-report", "model", "fixture", "reference-output", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.threads < 1 or args.warmups < 1 or args.repeats < 20:
        raise ValueError("expected positive threads and at least one warmup / 20 samples")
    export = json.loads(args.export_report.read_text())
    if (sha256(args.model) != export["onnx_sha256"]
            or sha256(args.fixture) != export["short_fixture_sha256"]
            or sha256(args.reference_output) != export["ort_output_sha256"]):
        raise ValueError("model, fixture or reference differs from the audited export")
    with np.load(args.fixture, allow_pickle=False) as data:
        if data.files != ["quantized"]:
            raise ValueError("unexpected fixture schema")
        x = np.asarray(data["quantized"])
    with np.load(args.reference_output, allow_pickle=False) as data:
        if data.files != ["wav"]:
            raise ValueError("unexpected reference schema")
        reference = np.asarray(data["wav"])
    if x.shape != (1, 512, 74) or x.dtype != np.float32 or reference.shape != (1, 3840):
        raise ValueError("short-window tensor contract differs")
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    started = time.perf_counter()
    session = ort.InferenceSession(str(args.model), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    load_s = time.perf_counter() - started
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("the profile did not use ORT CPU")
    process = psutil.Process()
    loaded_rss = process.memory_info().rss
    samples: list[float] = []
    for index in range(args.warmups + args.repeats):
        started = time.perf_counter()
        output = np.asarray(session.run(["wav"], {"quantized": x})[0])
        elapsed = time.perf_counter() - started
        if output.shape != reference.shape or not np.isfinite(output).all():
            raise ValueError("profile output violates waveform contract")
        relative_l2 = float(np.linalg.norm((output - reference).astype(np.float64))
                            / np.linalg.norm(reference.astype(np.float64)))
        if relative_l2 > 1e-6:
            raise ValueError(f"profile output drifted from its own CPU reference: {relative_l2}")
        if index >= args.warmups:
            samples.append(elapsed)
    report = {
        "scope": "one fixed two-frame Code2Wav component on HX370 WSL ORT CPU; no complete TTS stream",
        "status": "component_cpu_profiled",
        "platform": platform.platform(),
        "cpu": next((line.split(":", 1)[1].strip()
                     for line in Path("/proc/cpuinfo").read_text().splitlines()
                     if line.startswith("model name")), platform.processor()),
        "python_version": sys.version.split()[0],
        "checkpoint_revision": export["model_snapshot_revision"],
        "power_condition": "not recorded",
        "ort_version": ort.__version__,
        "provider": session.get_providers(),
        "threads": args.threads,
        "model_sha256": export["onnx_sha256"],
        "fixture_sha256": export["short_fixture_sha256"],
        "load_s": load_s,
        "rss_after_load_bytes": loaded_rss,
        "warmups": args.warmups,
        "sample_count": len(samples),
        "samples_s": samples,
        "nearest_rank_p50_s": rank(samples, 0.5),
        "nearest_rank_p95_s": rank(samples, 0.95),
        "audio_duration_s": 3840 / 24000,
        "median_component_rtf": rank(samples, 0.5) / (3840 / 24000),
        "limits": [
            "The input is one retained synthetic window, and no talker or predictor ran.",
            "This host CPU timing cannot be transplanted to the S25 or another SoC.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("sample_count", "nearest_rank_p50_s", "nearest_rank_p95_s",
                       "median_component_rtf")}, indent=2))


if __name__ == "__main__":
    main()
