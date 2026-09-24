#!/usr/bin/env python3
"""Probe the pinned InternVLA Cosmos post-Conv13 suffix on DirectML."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
import traceback
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def rank(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def relative_l2(reference, observed) -> float:
    import numpy as np

    a = reference.astype(np.float64)
    b = observed.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prefix", "conv", "suffix", "report", "outputs", "profile-prefix"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.device_id < 0 or args.warmups < 0 or args.repeats < 1:
        parser.error("invalid DirectML device or sample counts")

    import numpy as np
    import onnxruntime as ort

    report = {
        "scope": "fixed synthetic six-frame real-weight Cosmos CPU-prefix/Conv13 and DirectML-suffix component; not a full policy",
        "status": "started",
        "os": platform.platform(),
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "device_id_requested": args.device_id,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "artifact_sha256": {name: sha256(getattr(args, name))
                            for name in ("prefix", "conv", "suffix")},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        if "DmlExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("DirectML EP unavailable in this interpreter")
        pixels = np.zeros((6, 3, 256, 256), dtype=np.float32)
        pixels[:, 0, 48:208, 48:208] = 1
        pixels[:, 1, 48:208, 48:208] = -1
        pixels[:, 2] = np.linspace(-1, 1, 256, dtype=np.float32)[None, None, :]
        prefix = ort.InferenceSession(str(args.prefix), providers=["CPUExecutionProvider"])
        conv = ort.InferenceSession(str(args.conv), providers=["CPUExecutionProvider"])
        suffix_cpu = ort.InferenceSession(str(args.suffix), providers=["CPUExecutionProvider"])
        if ([item.name for item in prefix.get_inputs()] != ["pixels"]
                or [item.name for item in prefix.get_outputs()] != ["group_norm"]
                or [item.name for item in conv.get_inputs()] != ["group_norm"]
                or [item.name for item in conv.get_outputs()] != ["conv2d_13"]
                or {item.name for item in suffix_cpu.get_inputs()} != {"pixels", "conv2d_13"}
                or [item.name for item in suffix_cpu.get_outputs()] != ["latent"]):
            raise ValueError("pinned Cosmos boundary names changed")
        boundary = prefix.run(None, {"pixels": pixels})[0]
        parts = [conv.run(None, {"group_norm": np.ascontiguousarray(boundary[i:i+1])})[0]
                 for i in range(6)]
        conv_output = np.ascontiguousarray(np.concatenate(parts, axis=0))
        if boundary.shape != (6, 128, 64, 64) or conv_output.shape != (6, 256, 64, 64):
            raise ValueError("pinned Cosmos boundary shapes changed")
        values = {"pixels": pixels, "conv2d_13": conv_output}
        cpu_latent = suffix_cpu.run(None, values)[0]
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_profiling = True
        options.profile_file_prefix = str(args.profile_prefix)
        started = time.perf_counter()
        suffix_dml = ort.InferenceSession(
            str(args.suffix), sess_options=options,
            providers=[("DmlExecutionProvider", {"device_id": str(args.device_id)}),
                       "CPUExecutionProvider"],
        )
        report["dml_session_create_s"] = time.perf_counter() - started
        report["session_providers"] = suffix_dml.get_providers()
        dml_latent = suffix_dml.run(None, values)[0]
        profile_path = Path(suffix_dml.end_profiling())
        events = json.loads(profile_path.read_text(encoding="utf-8-sig"))
        providers = Counter((event.get("args") or {}).get("provider")
                            for event in events if event.get("cat") == "Node")
        report["profile_file"] = str(profile_path)
        report["node_provider_events"] = dict(providers)
        if (providers["DmlExecutionProvider"] < 1
                or dml_latent.shape != cpu_latent.shape != (6, 16, 32, 32)
                or not np.isfinite(dml_latent).all()):
            raise RuntimeError("DirectML suffix placement or output contract failed")
        report["relative_l2_vs_cpu_suffix"] = relative_l2(cpu_latent, dml_latent)
        report["max_abs_vs_cpu_suffix"] = float(np.max(np.abs(cpu_latent - dml_latent)))
        args.outputs.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.outputs, pixels=pixels, cpu_conv13=conv_output,
                            cpu_latent=cpu_latent, dml_latent=dml_latent)
        report["outputs_sha256"] = sha256(args.outputs)
        timings = {"cpu_suffix_s": [], "dml_suffix_s": []}
        for index in range(args.warmups + args.repeats):
            for name, session, expected in (
                ("cpu_suffix_s", suffix_cpu, cpu_latent),
                ("dml_suffix_s", suffix_dml, dml_latent),
            ) if index % 2 == 0 else (
                ("dml_suffix_s", suffix_dml, dml_latent),
                ("cpu_suffix_s", suffix_cpu, cpu_latent),
            ):
                started = time.perf_counter()
                actual = session.run(None, values)[0]
                elapsed = time.perf_counter() - started
                if not np.array_equal(actual, expected):
                    raise RuntimeError(f"{name} changed output during profiling")
                if index >= args.warmups:
                    timings[name].append(elapsed)
        report["timing_s"] = timings
        report["nearest_rank_p50_p95_s"] = {
            name: [rank(series, .5), rank(series, .95)] for name, series in timings.items()
        }
        report["status"] = "dml_suffix_component_measured"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: report.get(key) for key in (
            "status", "dml_session_create_s", "node_provider_events",
            "relative_l2_vs_cpu_suffix", "nearest_rank_p50_p95_s", "error",
        )}, indent=2))


if __name__ == "__main__":
    main()
