#!/usr/bin/env python3
"""Submit/resume the first MiniCPM-o vision stage on a real photo in AI Hub.

This executes only patch/position embedding. It must never be counted as a
complete image+speech to text+audio request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import qai_hub as hub


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--device", default="Samsung Galaxy S25")
    parser.add_argument("--profile-placement", action="store_true")
    args = parser.parse_args()
    fixture_dir = args.fixture_dir.resolve()
    evidence_dir = args.evidence_dir.resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    fixture = load(fixture_dir / "report.json")
    if fixture.get("revision") != "503e754207c94da6bb26850b4469f367c9ea3582":
        raise RuntimeError("MiniCPM-o checkpoint revision is not pinned")
    for name, expected in fixture["artifacts"].items():
        if sha256(fixture_dir / name) != expected:
            raise RuntimeError(f"fixture artifact changed: {name}")
    with np.load(fixture_dir / "astronaut_pixels.npz", allow_pickle=False) as archive:
        pixels = np.ascontiguousarray(archive["pixels"])
    with np.load(fixture_dir / "reference_output.npz", allow_pickle=False) as archive:
        reference = np.asarray(archive["fp32"])
        bf16_reference = np.asarray(archive["bf16"])
    if (pixels.shape != (1, 3, 448, 448) or pixels.dtype != np.float32
            or reference.shape != (1, 1024, 1152)):
        raise RuntimeError("vision embedding I/O contract changed")
    record_path = evidence_dir / "job_record.json"
    record = load(record_path)
    required = {
        "scope": "real-image vision embedding component only; no target audio/thinker/codec",
        "checkpoint": fixture["checkpoint"],
        "revision": fixture["revision"],
        "device": args.device,
        "source_sha256": fixture["artifacts"]["vision_embedding_448.onnx"],
        "input_sha256": fixture["artifacts"]["astronaut_pixels.npz"],
        "precision": "FP32 ONNX from pinned BF16 weights",
        "runtime": "tflite",
        "compute_unit_requested": "cpu",
        "batch_size": 1,
        "concurrency": 1,
        "inference_profile": False,
        "placement_profile_separate": True,
        "qai_hub_version": str(hub.__version__),
    }
    for key, value in required.items():
        if key in record and record[key] != value:
            raise RuntimeError(f"cannot resume changed request: {key}")
        record[key] = value
    save(record_path, record)
    device = hub.Device(args.device)
    if "source_model_id" not in record:
        source = hub.upload_model(str(fixture_dir / "vision_embedding_448.onnx"),
                                  name="MiniCPM-o 4.5 pinned real-image embedding")
        record["source_model_id"] = source.model_id
        save(record_path, record)
        print("uploaded", source.model_id, flush=True)
    else:
        source = hub.get_model(record["source_model_id"])
    if "compile_job_id" not in record:
        job = hub.submit_compile_job(
            model=source, device=device,
            input_specs={"pixels": ((1, 3, 448, 448), "float32")},
            options="--target_runtime tflite",
            name="MiniCPM-o pinned real-image vision embedding CPU candidate",
        )
        record["compile_job_id"] = job.job_id
        save(record_path, record)
        print("compile submitted", job.job_id, flush=True)
    else:
        job = hub.get_job(record["compile_job_id"])
    status = job.wait()
    record["compile_status"] = status.code
    record["compile_message"] = status.message or ""
    if job.device.name != args.device:
        raise RuntimeError("compile target registration changed")
    record["target_registration"] = {
        "name": job.device.name, "os": job.device.os,
        "attributes": list(job.device.attributes),
    }
    save(record_path, record)
    if not status.success:
        print("compile failed", status.message, flush=True)
        return
    target = job.get_target_model()
    record["target_model_id"] = target.model_id
    target_path = fixture_dir / f"vision_embedding_{target.model_id}.tflite"
    if not target_path.exists():
        downloaded = job.download_target_model(str(target_path))
        if downloaded is not None:
            target_path = Path(downloaded)
    if not target_path.exists():
        raise RuntimeError("compiled target was not downloaded")
    target_sha = sha256(target_path)
    if record.get("target_sha256", target_sha) != target_sha:
        raise RuntimeError("compiled target changed during resume")
    record["target_sha256"] = target_sha
    record["target_artifact_path"] = str(target_path)
    save(record_path, record)
    if "inference_job_id" not in record:
        inference = hub.submit_inference_job(
            model=target, device=device, inputs={"pixels": [pixels]},
            options="--compute_unit cpu", profile=False,
            name="MiniCPM-o pinned real-photo first vision stage",
        )
        record["inference_job_id"] = inference.job_id
        record["input_tensor_sha256"] = hashlib.sha256(pixels.tobytes()).hexdigest()
        save(record_path, record)
        print("inference submitted", inference.job_id, flush=True)
    else:
        inference = hub.get_job(record["inference_job_id"])
    result_status = inference.wait()
    record["inference_status"] = result_status.code
    record["inference_message"] = result_status.message or ""
    record["updated_utc"] = datetime.now(timezone.utc).isoformat()
    if inference.device.name != args.device or inference.model.model_id != target.model_id:
        raise RuntimeError("inference target or artifact changed")
    if result_status.success:
        outputs = inference.download_output_data()
        if len(outputs) != 1:
            raise RuntimeError(f"expected one hidden tensor, found {list(outputs)}")
        observed = np.asarray(next(iter(outputs.values()))[0])
        if observed.shape != reference.shape or not np.isfinite(observed).all():
            raise RuntimeError("device embedding output violated contract")
        np.savez_compressed(evidence_dir / "device_output.npz", hidden=observed)
        record["output_sha256"] = sha256(evidence_dir / "device_output.npz")
        record["output_tensor_sha256"] = hashlib.sha256(observed.tobytes()).hexdigest()
        lhs = observed.astype(np.float64)
        fp32 = reference.astype(np.float64)
        bf16 = bf16_reference.astype(np.float64)
        record["device_vs_fp32_relative_l2"] = float(
            np.linalg.norm(lhs - fp32) / max(np.linalg.norm(fp32), 1e-12))
        record["device_vs_bf16_relative_l2"] = float(
            np.linalg.norm(lhs - bf16) / max(np.linalg.norm(bf16), 1e-12))
        record["numerical_gate_fp32_pass"] = record["device_vs_fp32_relative_l2"] <= 1e-4
        record["input_dataset_id"] = inference.inputs.dataset_id
    save(record_path, record)
    print(json.dumps({key: record.get(key) for key in
                      ("compile_status", "inference_status", "inference_job_id",
                       "device_vs_fp32_relative_l2", "device_vs_bf16_relative_l2")}), flush=True)
    if result_status.success and not record["numerical_gate_fp32_pass"]:
        raise RuntimeError("device embedding failed local FP32 numerical gate")
    if args.profile_placement and result_status.success:
        if "profile_job_id" not in record:
            profile_job = hub.submit_inference_job(
                model=target, device=device, options="--compute_unit cpu",
                profile=True, name="MiniCPM-o first vision stage placement",
            )
            record["profile_job_id"] = profile_job.job_id
            save(record_path, record)
            print("profile submitted", profile_job.job_id, flush=True)
        else:
            profile_job = hub.get_job(record["profile_job_id"])
        placement_status = profile_job.wait()
        record["profile_status"] = placement_status.code
        record["profile_message"] = placement_status.message or ""
        if placement_status.success:
            profile = profile_job.download_profile()
            save(evidence_dir / "profile.json", profile)
            counts: dict[str, int] = {}
            for item in profile.get("execution_detail", []):
                unit = item.get("compute_unit", "unknown")
                counts[unit] = counts.get(unit, 0) + 1
            record["profile_compute_unit_rows"] = counts
            record["profile_sha256"] = sha256(evidence_dir / "profile.json")
        save(record_path, record)
        print(json.dumps({key: record.get(key) for key in
                          ("profile_job_id", "profile_status", "profile_compute_unit_rows")}), flush=True)


if __name__ == "__main__":
    main()
