#!/usr/bin/env python3
"""Submit/resume a real-input, revision-pinned Qwen3-TTS predictor component.

This records one hosted neural stage.  It never labels the local-generation
fixture plus hosted predictor as a hosted complete TTS request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import qai_hub as hub


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--device", default="Samsung Galaxy S25")
    parser.add_argument("--source-model-id", help="Reuse the identical uploaded ONNX source")
    parser.add_argument("--runtime", choices=("tflite", "onnx"), default="tflite")
    parser.add_argument("--profile-placement", action="store_true",
                        help="Run a separate profile to verify execution-unit placement")
    args = parser.parse_args()
    fixture_dir = args.fixture_dir.resolve()
    evidence = args.evidence_dir.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    provenance = load(fixture_dir / "report.json")
    if provenance.get("revision") != "85e237c12c027371202489a0ec509ded67b5e4b5":
        raise RuntimeError("predictor fixture lacks the pinned CustomVoice revision")
    for name, expected in provenance["artifacts"].items():
        if sha256(fixture_dir / name) != expected:
            raise RuntimeError(f"artifact changed: {name}")
    with np.load(fixture_dir / "real_predictor_input.npz", allow_pickle=False) as data:
        x = np.ascontiguousarray(data["x"])
    with np.load(fixture_dir / "reference_output.npz", allow_pickle=False) as data:
        reference = np.asarray(data["hidden"])
    if x.shape != (1, 17, 1024) or x.dtype != np.float32:
        raise RuntimeError("predictor input contract changed")
    device = hub.Device(args.device)
    record_path = evidence / "job_record.json"
    record = load(record_path)
    record.setdefault("scope", "AI Hub predictor-body component on real generated input; not TTS E2E")
    record.setdefault("checkpoint", provenance["checkpoint"])
    record.setdefault("revision", provenance["revision"])
    record.setdefault("device", args.device)
    record.setdefault("precision", f"float32 source; {args.runtime} CPU-requested target")
    record.setdefault("runtime", args.runtime)
    if record["runtime"] != args.runtime:
        raise RuntimeError("cannot resume a job with another target runtime")
    record.setdefault("source_sha256", provenance["artifacts"]["predictor_body.onnx"])
    record.setdefault("fixture_sha256", provenance["artifacts"]["real_predictor_input.npz"])
    record.setdefault("batch_size", 1)
    record.setdefault("concurrency", 1)
    record.setdefault("inference_profile", False)
    record.setdefault("placement_profile_separate", True)
    record.setdefault("qai_hub_version", str(hub.__version__))
    save(record_path, record)
    if args.source_model_id and "source_model_id" not in record:
        record["source_model_id"] = args.source_model_id
        save(record_path, record)
    if args.source_model_id and record.get("source_model_id") != args.source_model_id:
        raise RuntimeError("source model ID changed during resume")
    if "source_model_id" not in record:
        model = hub.upload_model(str(fixture_dir / "predictor_body.onnx"),
                                 name="Qwen3-TTS pinned real predictor body")
        record["source_model_id"] = model.model_id
        save(record_path, record)
        print("uploaded", model.model_id, flush=True)
    else:
        model = hub.get_model(record["source_model_id"])
    if "compile_job_id" not in record:
        compile_job = hub.submit_compile_job(
            model=model, device=device,
            input_specs={"x": ((1, 17, 1024), "float32")},
            options=f"--target_runtime {args.runtime}",
            name="Qwen3-TTS pinned predictor real-input CPU candidate",
        )
        record["compile_job_id"] = compile_job.job_id
        save(record_path, record)
        print("compile submitted", compile_job.job_id, flush=True)
    else:
        compile_job = hub.get_job(record["compile_job_id"])
    compile_status = compile_job.wait()
    record["compile_status"] = compile_status.code
    record["compile_message"] = compile_status.message or ""
    save(record_path, record)
    if not compile_status.success:
        print("compile failed", compile_status.message, flush=True)
        return
    if compile_job.device.name != args.device:
        raise RuntimeError("compile ran on another target registration")
    record["target_registration"] = {
        "name": compile_job.device.name,
        "os": compile_job.device.os,
        "attributes": list(compile_job.device.attributes),
    }
    target = compile_job.get_target_model()
    record["target_model_id"] = target.model_id
    save(record_path, record)
    target_name = re.sub(r"[^a-z0-9]+", "_", args.device.lower()).strip("_")
    target_path = Path(record["target_artifact_path"]) if "target_artifact_path" in record else (
        fixture_dir / f"predictor_{target_name}_{target.model_id}.{args.runtime}")
    if "target_sha256" not in record:
        if not target_path.exists():
            existing = list(fixture_dir.glob(target_path.name + "*"))
            if len(existing) == 1:
                target_path = existing[0]
            else:
                downloaded = compile_job.download_target_model(str(target_path))
                if downloaded is not None:
                    target_path = Path(downloaded)
        if not target_path.exists():
            raise RuntimeError("compiled target download did not create an artifact")
        record["target_sha256"] = sha256(target_path)
        record["target_artifact_path"] = str(target_path)
        save(record_path, record)
    else:
        if not target_path.exists():
            downloaded = compile_job.download_target_model(str(target_path))
            if downloaded is not None:
                target_path = Path(downloaded)
        if not target_path.exists() or sha256(target_path) != record["target_sha256"]:
            raise RuntimeError("compiled target artifact changed during resume")
    if "inference_job_id" not in record:
        job = hub.submit_inference_job(
            model=target, device=device, inputs={"x": [x]},
            options="--compute_unit cpu", profile=False,
            name="Qwen3-TTS pinned predictor body real-input CPU",
        )
        record["inference_job_id"] = job.job_id
        record["inference_input_sha256"] = hashlib.sha256(x.tobytes()).hexdigest()
        save(record_path, record)
        print("inference submitted", job.job_id, flush=True)
    else:
        job = hub.get_job(record["inference_job_id"])
    status = job.wait()
    if job.device.name != args.device or job.model.model_id != target.model_id:
        raise RuntimeError("inference ran on another target or artifact")
    record["inference_status"] = status.code
    record["inference_message"] = status.message or ""
    record["updated_utc"] = datetime.now(timezone.utc).isoformat()
    if status.success:
        output = job.download_output_data()
        names = list(output)
        if len(names) != 1:
            raise RuntimeError(f"expected one predictor hidden output, found {names}")
        observed = np.asarray(output[names[0]][0])
        if observed.shape != reference.shape or not np.isfinite(observed).all():
            raise RuntimeError("hosted predictor output violates shape/finiteness")
        np.savez_compressed(evidence / "device_output.npz", hidden=observed)
        record["output_sha256"] = sha256(evidence / "device_output.npz")
        record["output_tensor_sha256"] = hashlib.sha256(observed.tobytes()).hexdigest()
        ref = reference.astype(np.float64)
        actual = observed.astype(np.float64)
        record["reference_relative_l2"] = float(np.linalg.norm(actual - ref) /
                                                max(np.linalg.norm(ref), 1e-12))
        record["numerical_pass"] = record["reference_relative_l2"] <= 1e-4
        record["output_shape"] = list(observed.shape)
        record["input_dataset_id"] = job.inputs.dataset_id
    save(record_path, record)
    print(json.dumps({k: record.get(k) for k in
                      ("device", "compile_job_id", "compile_status", "inference_job_id",
                       "inference_status", "reference_relative_l2")}), flush=True)
    if status.success and not record["numerical_pass"]:
        raise RuntimeError("hosted predictor failed the component numerical gate")
    if args.profile_placement and status.success:
        if "profile_job_id" not in record:
            profile_job = hub.submit_inference_job(
                model=target, device=device, options="--compute_unit cpu",
                profile=True,
                name="Qwen3-TTS pinned predictor CPU placement",
            )
            record["profile_job_id"] = profile_job.job_id
            save(record_path, record)
            print("profile submitted", profile_job.job_id, flush=True)
        else:
            profile_job = hub.get_job(record["profile_job_id"])
        profile_status = profile_job.wait()
        record["profile_status"] = profile_status.code
        record["profile_message"] = profile_status.message or ""
        if profile_status.success:
            profile = profile_job.download_profile()
            save(evidence / "profile.json", profile)
            counts: dict[str, int] = {}
            for detail in profile.get("execution_detail", []):
                unit = detail.get("compute_unit", "unknown")
                counts[unit] = counts.get(unit, 0) + 1
            record["profile_compute_unit_rows"] = counts
            record["profile_sha256"] = sha256(evidence / "profile.json")
        save(record_path, record)
        print(json.dumps({k: record.get(k) for k in
                          ("profile_job_id", "profile_status", "profile_compute_unit_rows")}),
              flush=True)


if __name__ == "__main__":
    main()
