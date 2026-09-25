#!/usr/bin/env python3
"""Audit bounded RTX-only versus RTX+AMD-NPU MiniCPM-o requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import ntpath
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def one_event(events: list[dict], phase: str) -> dict:
    found = [event for event in events if event["phase"] == phase]
    if len(found) != 1:
        raise ValueError(f"expected one {phase} event, found {len(found)}")
    return found[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence_dir
    joint_path = root / "joint_384_report.json"
    cuda_path = root / "cuda_384_report.json"
    joint = json.loads(joint_path.read_text(encoding="utf-8"))
    cuda = json.loads(cuda_path.read_text(encoding="utf-8"))
    joint_config_path = root / "cuda_npu_image_384.yaml"
    cuda_config_path = root / "cuda_image_384.yaml"
    joint_config = yaml.safe_load(joint_config_path.read_text(encoding="utf-8"))
    cuda_config = yaml.safe_load(cuda_config_path.read_text(encoding="utf-8"))
    hook = joint_config["stages"][0].pop("runtime")
    if (joint_config != cuda_config or not hook["env"]["VLLM_OMNI_MINICPMO_KV_GRAPH"]
            or joint["deploy_config_sha256"] != sha256(joint_config_path)
            or cuda["deploy_config_sha256"] != sha256(cuda_config_path)
            or joint["model"] != cuda["model"]
            or joint["model"] != str(args.model_dir.resolve())
            or len(joint["requests"]) != 2 or len(cuda["requests"]) != 2):
        raise ValueError("checkpoint, plan or request count differs across routes")
    graph = Path(hook["env"]["VLLM_OMNI_MINICPMO_KV_GRAPH"])
    if sha256(graph) != "330bbbd0d18caaf6903aa41f836dbeaef57643a8afbaba333df68e3c1e722aeb":
        raise ValueError("NPU graph differs from the as-run candidate")
    shard = args.model_dir / "model-00004-of-00004.safetensors"
    if sha256(shard) != "f61addf4747c94fedcaee059e5d9918ed15543beec494404139a99f2f86c9b31":
        raise ValueError("checkpoint shard differs from the NPU adapter contract")

    events_path = root / "joint_384_events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    placement = one_event(events, "placement")["placement"]
    installed = one_event(events, "adapter_installed")
    closed = one_event(events, "close")
    completed = [event for event in events if event["phase"] == "request_complete"]
    runs = [event for event in events if event["phase"] == "run"]
    if (len({event["pid"] for event in events}) != 1
            or installed["model_revision"] != "503e754207c94da6bb26850b4469f367c9ea3582"
            or placement["ep"] != "vitisai" or placement["node_counts"].get("vitisai") != 1
            or len(completed) != 3 or [event["request"] for event in completed] != [1, 2, 3]
            or [event["tiles"] for event in completed] != [11, 1, 1]
            or len(runs) != 13 or closed["calls"] != len(runs)
            or closed["worker_stats"]["peak_rss_bytes"] > installed["planner_budget_bytes"]):
        raise ValueError("NPU placement, lifecycle or admission evidence failed")
    for complete in completed:
        request_runs = [event for event in runs if event["request"] == complete["request"]]
        if (len(request_runs) != complete["tiles"]
                or [event["tile"] for event in request_runs] != list(range(len(request_runs)))
                or sum(event["valid_tokens"] for event in request_runs)
                != complete["input_shape"][0] * complete["input_shape"][1]
                or complete["source_device"] != "cuda:0"
                or not 0 <= complete["projection_relative_l2"] < .01):
            raise ValueError("incomplete CUDA-to-NPU tiles or failed numerical gate")
    profile_name = ntpath.basename(placement["profile_path"])
    profile_path = root / "kv_profile" / profile_name
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    providers = [event.get("args", {}).get("provider") for event in profile
                 if event.get("cat") == "Node"]
    if providers.count("vitisai") != 1:
        raise ValueError("retained raw profile lacks the NPU node")

    comparisons = []
    for index, (candidate, reference) in enumerate(zip(joint["requests"], cuda["requests"])):
        if (candidate["index"] != index or reference["index"] != index
                or candidate["image_sha256"] != reference["image_sha256"]
                or not candidate["text"] or not reference["text"]
                or not candidate["token_ids"] or not reference["token_ids"]):
            raise ValueError(f"request {index}: input or text contract failed")
        for route, row in (("joint", candidate), ("cuda", reference)):
            archive = root / f"{route}_384_audio" / f"request_{index:02d}.npy"
            wav = archive.with_suffix(".wav")
            waveform = np.load(archive, allow_pickle=False)
            wav_values, sample_rate = sf.read(wav, dtype="float32")
            if (sha256(archive) != row["audio_archive_sha256"]
                    or waveform.size != row["audio_samples"] or sample_rate != 24000
                    or not np.array_equal(waveform, wav_values)
                    or not np.isfinite(waveform).all() or np.max(np.abs(waveform)) == 0):
                raise ValueError(f"request {index}: {route} audio contract failed")
        complete = completed[index + 1]  # First event is vLLM's dummy profile.
        request_runs = [event for event in runs if event["request"] == index + 2]
        comparisons.append({
            "index": index,
            "image_sha256": candidate["image_sha256"],
            "joint_text": candidate["text"], "cuda_text": reference["text"],
            "token_ids_equal": candidate["token_ids"] == reference["token_ids"],
            "joint_audio_samples": candidate["audio_samples"],
            "cuda_audio_samples": reference["audio_samples"],
            "joint_wall_s": candidate["wall_s"], "cuda_wall_s": reference["wall_s"],
            "projection_relative_l2": complete["projection_relative_l2"],
            "npu_round_trip_s": sum(event["timing"]["round_trip_s"] for event in request_runs),
            "cuda_to_host_s": complete["input_transfer_s"],
            "host_to_cuda_s": complete["output_transfer_s"],
        })

    asr = {}
    for route, path in (("joint", joint_path), ("cuda", cuda_path)):
        asr_path = root / f"{route}_384_asr.json"
        report = json.loads(asr_path.read_text(encoding="utf-8"))
        if (report["request_report_sha256"] != sha256(path)
                or len(report["requests"]) != 2
                or any(row["word_error_rate"] != 0 for row in report["requests"])):
            raise ValueError(f"{route}: speech proxy audit failed")
        asr[route] = {"report_sha256": sha256(asr_path),
                      "word_error_rates": [row["word_error_rate"] for row in report["requests"]]}
    driver_hashes = {}
    for route in ("joint", "cuda"):
        driver_path = root / f"{route}_384_driver.log"
        log = driver_path.read_text(encoding="utf-8", errors="replace")
        if any(f"Stage {stage} replica 0 shut down" not in log for stage in range(3)):
            raise ValueError(f"{route}: three-stage shutdown is not recorded")
        driver_hashes[route] = sha256(driver_path)
    output = {
        "status": "scoped_joint_execution_passed_default_selection_unqualified",
        "checkpoint_shard_sha256": sha256(shard), "graph_sha256": sha256(graph),
        "joint_report_sha256": sha256(joint_path), "cuda_report_sha256": sha256(cuda_path),
        "events_sha256": sha256(events_path), "raw_npu_profile_sha256": sha256(profile_path),
        "driver_log_sha256": driver_hashes,
        "raw_npu_profile_node_providers": {name: providers.count(name) for name in sorted(set(providers))},
        "joint_startup_s": joint["startup_s"], "cuda_startup_s": cuda["startup_s"],
        "npu_worker_peak_rss_bytes": closed["worker_stats"]["peak_rss_bytes"],
        "asr": asr, "requests": comparisons,
        "limits": "Two separate serial runs; text and speech differ, output lengths differ, no paired speedup, gold quality, loading peak, cancellation, concurrency or sustained-power gate.",
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "requests": len(comparisons)}))


if __name__ == "__main__":
    main()
