#!/usr/bin/env python3
"""Run three pinned vision layers on the prior AI Hub embedding output.

This is a dependent two-stage *vision component* replay, not a complete
MiniCPM-o image+speech to text+audio request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import qai_hub as hub


REVISION = "503e754207c94da6bb26850b4469f367c9ea3582"
PREFIX_SHA = "b68fe07f9ae74d2c4d3a953b43ce5b33e28f86a41f413f857d2e00edf67b9efb"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    lhs = a.astype(np.float64)
    rhs = b.astype(np.float64)
    return float(np.linalg.norm(lhs - rhs) / max(np.linalg.norm(lhs), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-evidence", type=Path, required=True)
    parser.add_argument("--embedding-fixture", type=Path, required=True)
    parser.add_argument("--prefix-source", type=Path, required=True)
    parser.add_argument("--prefix-report", type=Path, required=True)
    parser.add_argument("--prefix-fixture", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--device", default="Samsung Galaxy S24")
    parser.add_argument("--reference-python", type=Path, required=True,
                        help="Python with ONNX Runtime, used only for same-input reference comparison")
    parser.add_argument("--profile-placement", action="store_true",
                        help="Submit separate representative placement profile after functional inference")
    args = parser.parse_args()
    output = args.evidence_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    first = load(args.embedding_evidence / "job_record.json")
    first_fixture = load(args.embedding_fixture / "report.json")
    prefix = load(args.prefix_report)
    if (first.get("inference_status") != "SUCCESS"
            or not first.get("numerical_gate_fp32_pass")
            or first.get("device") != args.device
            or first.get("inference_profile") is not False
            or first.get("target_registration", {}).get("name") != args.device
            or first.get("output_sha256") != sha256(args.embedding_evidence / "device_output.npz")
            or first.get("revision") != REVISION
            or prefix.get("model_revision") != REVISION
            or prefix.get("image_sha256") != first_fixture.get("image_sha256")
            or prefix.get("output_sha256") != PREFIX_SHA
            or sha256(args.prefix_source) != PREFIX_SHA
            or prefix.get("fixture_sha256") != sha256(args.prefix_fixture)):
        raise RuntimeError("upstream AI Hub image tensor or pinned prefix identity changed")
    with np.load(args.embedding_evidence / "device_output.npz", allow_pickle=False) as archive:
        hidden = np.ascontiguousarray(archive["hidden"])
    with np.load(args.prefix_fixture, allow_pickle=False) as archive:
        fp32_reference = np.asarray(archive["torch_fp32"])
        bf16_reference = np.asarray(archive["torch_bf16"])
    if hidden.shape != (1, 1024, 1152) or hidden.dtype != np.float32:
        raise RuntimeError("upstream hidden-state layout changed")
    # Keep the AI Hub SDK environment independent from the ONNX Runtime environment.
    # This host execution supplies only a numerical reference, never a chain output.
    paired_path = output / "paired_reference.npz"
    reference_code = (
        "import numpy as np, onnxruntime as ort, sys; "
        "s=ort.InferenceSession(sys.argv[1],providers=['CPUExecutionProvider']); "
        "x=np.load(sys.argv[2],allow_pickle=False)['hidden']; "
        "y=s.run(['hidden_after_prefix'],{'hidden':x})[0]; "
        "np.savez_compressed(sys.argv[3],hidden_after_prefix=y)"
    )
    subprocess.run([str(args.reference_python), "-c", reference_code,
                    str(args.prefix_source), str(args.embedding_evidence / "device_output.npz"),
                    str(paired_path)], check=True)
    with np.load(paired_path, allow_pickle=False) as archive:
        paired_reference = np.asarray(archive["hidden_after_prefix"])
    record_path = output / "job_record.json"
    record = load(record_path)
    required = {
        "scope": "real-photo dependent 3-layer vision prefix after prior AI Hub embedding; component only",
        "checkpoint": "openbmb/MiniCPM-o-4_5",
        "revision": REVISION,
        "device": args.device,
        "source_sha256": PREFIX_SHA,
        "image_sha256": prefix["image_sha256"],
        "upstream_job_id": first["inference_job_id"],
        "upstream_output_sha256": first["output_sha256"],
        "paired_reference_sha256": sha256(paired_path),
        "precision": "FP32 ONNX from pinned BF16 weights",
        "runtime": "tflite",
        "compute_unit_requested": "cpu",
        "batch_size": 1,
        "concurrency": 1,
        "inference_profile": False,
        "placement_profile_separate": args.profile_placement,
        "qai_hub_version": str(hub.__version__),
    }
    for key, value in required.items():
        if key in record and record[key] != value:
            raise RuntimeError(f"cannot resume changed prefix request: {key}")
        record[key] = value
    save(record_path, record)
    device = hub.Device(args.device)
    if "source_model_id" not in record:
        source = hub.upload_model(str(args.prefix_source), name="MiniCPM-o pinned real-photo vision 3-layer prefix")
        record["source_model_id"] = source.model_id
        save(record_path, record)
        print("uploaded", source.model_id, flush=True)
    else:
        source = hub.get_model(record["source_model_id"])
    if "compile_job_id" not in record:
        job = hub.submit_compile_job(
            model=source, device=device,
            input_specs={"hidden": ((1, 1024, 1152), "float32")},
            options="--target_runtime tflite",
            name="MiniCPM-o pinned 3-layer vision prefix CPU candidate",
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
        raise RuntimeError("prefix compile target registration changed")
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
    target_path = args.prefix_source.parent / f"minicpmo_prefix3_{target.model_id}.tflite"
    if not target_path.exists():
        downloaded = job.download_target_model(str(target_path))
        if downloaded is not None:
            target_path = Path(downloaded)
    if not target_path.exists():
        raise RuntimeError("prefix target download missing")
    target_sha = sha256(target_path)
    if record.get("target_sha256", target_sha) != target_sha:
        raise RuntimeError("prefix target artifact changed")
    record["target_sha256"] = target_sha
    record["target_artifact_path"] = str(target_path)
    save(record_path, record)
    if "inference_job_id" not in record:
        inference = hub.submit_inference_job(
            model=target, device=device, inputs={"hidden": [hidden]},
            options="--compute_unit cpu", profile=False,
            name="MiniCPM-o dependent real-photo vision prefix",
        )
        record["inference_job_id"] = inference.job_id
        record["input_tensor_sha256"] = hashlib.sha256(hidden.tobytes()).hexdigest()
        save(record_path, record)
        print("inference submitted", inference.job_id, flush=True)
    else:
        inference = hub.get_job(record["inference_job_id"])
    result = inference.wait()
    record["inference_status"] = result.code
    record["inference_message"] = result.message or ""
    record["updated_utc"] = datetime.now(timezone.utc).isoformat()
    if inference.device.name != args.device or inference.model.model_id != target.model_id:
        raise RuntimeError("prefix inference target/artifact changed")
    if result.success:
        outputs = inference.download_output_data()
        if len(outputs) != 1:
            raise RuntimeError(f"expected one prefix tensor, found {list(outputs)}")
        observed = np.asarray(next(iter(outputs.values()))[0])
        if observed.shape != fp32_reference.shape or not np.isfinite(observed).all():
            raise RuntimeError("prefix device output violated shape/finiteness")
        np.savez_compressed(output / "device_output.npz", hidden_after_prefix=observed)
        record["output_sha256"] = sha256(output / "device_output.npz")
        record["device_vs_paired_onnx_relative_l2"] = relative_l2(paired_reference, observed)
        record["device_vs_photo_fp32_relative_l2"] = relative_l2(fp32_reference, observed)
        record["device_vs_photo_bf16_relative_l2"] = relative_l2(bf16_reference, observed)
        record["numerical_gate_paired_pass"] = record["device_vs_paired_onnx_relative_l2"] <= 1e-4
        record["input_dataset_id"] = inference.inputs.dataset_id
    save(record_path, record)
    print(json.dumps({key: record.get(key) for key in
                      ("compile_status", "inference_status", "inference_job_id",
                       "device_vs_paired_onnx_relative_l2", "device_vs_photo_bf16_relative_l2")}), flush=True)
    if result.success and not record["numerical_gate_paired_pass"]:
        raise RuntimeError("hosted vision prefix differs from same-input ONNX reference")
    if args.profile_placement and result.success:
        if "profile_job_id" not in record:
            profile_job = hub.submit_inference_job(
                model=target, device=device, inputs={"hidden": [hidden]},
                options="--compute_unit cpu", profile=True,
                name="MiniCPM-o dependent real-photo vision prefix placement",
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
            save(output / "profile.json", profile)
            counts: dict[str, int] = {}
            for item in profile.get("execution_detail", []):
                unit = item.get("compute_unit", "unknown")
                counts[unit] = counts.get(unit, 0) + 1
            record["profile_compute_unit_rows"] = counts
            record["profile_sha256"] = sha256(output / "profile.json")
        save(record_path, record)
        print(json.dumps({key: record.get(key) for key in
                          ("profile_job_id", "profile_status", "profile_compute_unit_rows")}), flush=True)


if __name__ == "__main__":
    main()
