#!/usr/bin/env python3
"""Check real-input hosted predictor hidden states against all 15 code heads.

The heads are run locally only as an offline quality reference. This does not
make a predictor-body component into hosted complete speech generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    fixture = json.loads((args.fixture_dir / "report.json").read_text())
    record = json.loads((args.evidence_dir / "job_record.json").read_text())
    if (record.get("inference_status") != "SUCCESS"
            or record.get("source_sha256") != fixture["artifacts"]["predictor_body.onnx"]
            or fixture["revision"] != args.model.resolve().name
            or fixture["source_sha256"] != sha256(args.model / "model.safetensors")
            or record.get("output_sha256") != sha256(args.evidence_dir / "device_output.npz")):
        raise RuntimeError("predictor output or pinned checkpoint identity changed")
    with np.load(args.fixture_dir / "reference_output.npz", allow_pickle=False) as archive:
        reference = np.asarray(archive["hidden"], dtype=np.float32)
    with np.load(args.evidence_dir / "device_output.npz", allow_pickle=False) as archive:
        device = np.asarray(archive["hidden"], dtype=np.float32)
    if reference.shape != (1, 17, 1024) or device.shape != reference.shape:
        raise RuntimeError("predictor output shape changed")
    expected = np.asarray(fixture["code_groups"], dtype=np.int64).reshape(15)
    reference_top1 = []
    device_top1 = []
    with safe_open(str(args.model / "model.safetensors"), framework="pt", device="cpu") as weights:
        for index in range(15):
            weight = weights.get_tensor(
                f"talker.code_predictor.lm_head.{index}.weight").float().numpy()
            reference_top1.append(int(np.argmax(reference[0, index + 1] @ weight.T)))
            device_top1.append(int(np.argmax(device[0, index + 1] @ weight.T)))
    report = {
        "scope": "offline code-head audit of one real-input AI Hub predictor component",
        "device": record["device"],
        "checkpoint": fixture["checkpoint"],
        "revision": fixture["revision"],
        "inference_job_id": record["inference_job_id"],
        "source_sha256": record["source_sha256"],
        "target_sha256": record["target_sha256"],
        "output_sha256": record["output_sha256"],
        "generated_code_groups": expected.tolist(),
        "subtalker_dosample": fixture["subtalker_dosample"],
        "reference_top1": reference_top1,
        "device_top1": device_top1,
        "device_vs_reference_top1_matches": int(sum(a == b for a, b in
                                                     zip(reference_top1, device_top1))),
        "reference_vs_generated_top1_matches": int(sum(a == b for a, b in
                                                       zip(reference_top1, expected))),
        "device_vs_generated_top1_matches": int(sum(a == b for a, b in
                                                    zip(device_top1, expected))),
        "code_groups": 15,
        "batch_size": 1,
        "concurrency": 1,
    }
    output = args.evidence_dir / "code_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"device": report["device"],
                      "device_vs_reference": report["device_vs_reference_top1_matches"],
                      "reference_vs_generated": report["reference_vs_generated_top1_matches"]}))
    if report["device_vs_reference_top1_matches"] != 15:
        raise RuntimeError("hosted predictor changed a reference code-head top-1")


if __name__ == "__main__":
    main()
