"""Snapshot a completed native llama.cpp route's private placement evidence.

This is an independent observation, not a signed qualification receipt. It
copies the sanitized startup log without overwriting either evidence file,
rehashes the local artifacts, and binds the recovered request to the worker
generation recorded by the Omni stage event. The startup log itself has no
worker PID or generation, so its worker association is harness-derived.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_OFFLOAD = re.compile(r"^load_tensors: offloaded (\d+)/(\d+) layers to GPU$", re.M)
_LAYER = re.compile(r"^load_tensors: layer (\d+) assigned to device (CPU|Vulkan\d+)$", re.M)
_BUFFER = re.compile(r"^load_tensors: (CPU|Vulkan\d+) model buffer size = ([0-9]+(?:\.[0-9]+)?) MiB$", re.M)
_USING = re.compile(r"^llama_model_load: using device (Vulkan\d+) \(([^\r\n]*)\)$", re.M)
_LOADED = re.compile(r"^llama\.cpp status: model loaded$", re.M)
_REPO = Path(__file__).resolve().parents[3]
_PRIVATE_RESULTS = _REPO / "benchmarks" / "edge_agent" / "results"


class PlacementSnapshotError(ValueError):
    """The raw trace, artifacts, or loader report cannot support a snapshot."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def record_digest(record: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical({key: value for key, value in record.items()
                                      if key != "record_sha256"})).hexdigest()


def verify_record(record: dict[str, Any]) -> bool:
    digest = record.get("record_sha256")
    return isinstance(digest, str) and bool(_HEX64.fullmatch(digest)) and digest == record_digest(record)


def _stable_bytes(path: Path, *, max_bytes: int) -> bytes:
    before = path.stat()
    if not path.is_file() or before.st_size > max_bytes:
        raise PlacementSnapshotError("input is not a bounded regular file")
    content = path.read_bytes()
    after = path.stat()
    if ((before.st_size, before.st_mtime_ns, before.st_ino) !=
        (after.st_size, after.st_mtime_ns, after.st_ino) or len(content) != before.st_size):
        raise PlacementSnapshotError("input changed during snapshot")
    return content


def _artifact(path_value: Any, expected_sha: Any) -> dict[str, Any]:
    if not isinstance(path_value, str) or not path_value:
        raise PlacementSnapshotError("artifact path is absent")
    if not isinstance(expected_sha, str) or not _HEX64.fullmatch(expected_sha):
        raise PlacementSnapshotError("artifact digest is absent or invalid")
    path = Path(path_value)
    before = path.stat()
    if not path.is_file():
        raise PlacementSnapshotError("artifact is not a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    after = path.stat()
    if ((before.st_size, before.st_mtime_ns, before.st_ino) !=
        (after.st_size, after.st_mtime_ns, after.st_ino)):
        raise PlacementSnapshotError("artifact changed during hashing")
    if digest.hexdigest() != expected_sha:
        raise PlacementSnapshotError("local artifact differs from the recorded hash")
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "size_bytes": before.st_size}


def _parse_log(raw: bytes, route: dict[str, Any]) -> dict[str, Any]:
    try:
        log = raw.decode("utf-8").replace("\r\n", "\n")
    except UnicodeDecodeError as exc:
        raise PlacementSnapshotError("startup log is not UTF-8") from exc
    offloads = _OFFLOAD.findall(log)
    if len(offloads) != 1 or len(_LOADED.findall(log)) != 1:
        raise PlacementSnapshotError("startup log lacks one offload report and one exact load marker")
    offloaded, total = (int(value) for value in offloads[0])
    if total < 1 or offloaded > total:
        raise PlacementSnapshotError("invalid offload count")
    assignments = [(int(index), device) for index, device in _LAYER.findall(log)]
    if len(assignments) != total or [index for index, _ in assignments] != list(range(total)):
        raise PlacementSnapshotError("layer assignments are missing, duplicated, or out of order")
    gpu = [(index, device) for index, device in assignments if device != "CPU"]
    if len(gpu) != offloaded:
        raise PlacementSnapshotError("layer assignment count disagrees with offload report")
    placement = route.get("placement")
    using = _USING.findall(log)
    expected_device = route.get("expected_device_name")
    if placement == "cpu":
        if offloaded != 0 or using:
            raise PlacementSnapshotError("CPU placement disagrees with startup log")
        device_name = None
    else:
        match = re.fullmatch(r"(?:(?:cpu|Vulkan_Host)\+)?(Vulkan\d+)", str(placement))
        if not match or offloaded == 0 or len(using) != 1:
            raise PlacementSnapshotError("GPU placement lacks an exact device and offload")
        device, device_name = using[0]
        if (device != match.group(1) or device_name != expected_device or
            {name for _, name in gpu} != {device}):
            raise PlacementSnapshotError("assigned GPU differs from requested device")
        if placement.startswith("cpu+") and offloaded == total:
            raise PlacementSnapshotError("hybrid route has no observed CPU layers")
    if route.get("gpu_layers") != offloaded:
        raise PlacementSnapshotError("observed offload differs from route configuration")
    buffers = _BUFFER.findall(log)
    if not buffers or len({device for device, _ in buffers}) != len(buffers):
        raise PlacementSnapshotError("model-buffer report is missing or ambiguous")
    buffer_mib = {device: float(amount) for device, amount in buffers}
    assigned_devices = {device for _, device in assignments}
    if assigned_devices != set(buffer_mib) or any(amount <= 0 for amount in buffer_mib.values()):
        raise PlacementSnapshotError("model-buffer report differs from layer devices")
    return {
        "reported_placement": placement, "selected_gpu_name": device_name,
        "offloaded_layers": offloaded, "total_layers": total,
        "cpu_layer_count": total - offloaded,
        "gpu_layer_indices": [index for index, _ in gpu],
        "model_buffer_mib_reported": buffer_mib,
        "evidence_level": "startup_log_offload_and_model_buffers",
        "compute_placement_verified": False,
    }


def _load_trace(path: Path) -> tuple[bytes, dict[str, Any], list[dict[str, Any]]]:
    raw = _stable_bytes(path, max_bytes=16 << 20)
    try:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlacementSnapshotError("native trace is not valid JSONL") from exc
    if (not rows or not isinstance(rows[0], dict) or
        rows[0].get("record_type") != "manifest" or
        any(not isinstance(row, dict) or row.get("record_type") == "manifest"
            for row in rows[1:])):
        raise PlacementSnapshotError("native trace has no single leading manifest")
    return raw, rows[0], rows[1:]


def _request_proof(manifest: dict[str, Any], events: list[dict[str, Any]],
                   route: dict[str, Any]) -> dict[str, Any]:
    first, second = manifest.get("first_worker"), manifest.get("second_worker")
    release = manifest.get("release_evidence")
    if not all(isinstance(item, dict) for item in (first, second, release)):
        raise PlacementSnapshotError("trace lacks worker and cancellation evidence")
    for worker in (first, second):
        if (not isinstance(worker.get("generation"), str) or
            not isinstance(worker.get("pid"), int) or worker["pid"] <= 0 or
            worker.get("placement") != route.get("placement")):
            raise PlacementSnapshotError("worker identity or placement is invalid")
    if first["generation"] == second["generation"] or first["pid"] == second["pid"]:
        raise PlacementSnapshotError("cancelled worker was not replaced")
    if (release.get("worker_pid_before") != first["pid"] or
        release.get("release_mode") != "worker_shutdown" or
        type(release.get("worker_exit_code")) is not int or
        any(release.get(key) is not True for key in (
            "backend_request_state_verified", "graph_gate_released",
            "tool_request_finished", "worker_exit_confirmed", "stage_ledger_empty",
            "host_claim_released", "host_ledger_empty",
        ))):
        raise PlacementSnapshotError("first worker has no verified release")
    if manifest.get("result") != "cancelled_stream_then_restarted_request_passed":
        raise PlacementSnapshotError("native recovery smoke did not pass")
    seen: set[tuple[str, int]] = set()
    last_sequence: dict[str, int] = {}
    for event in events:
        request_id, seq = event.get("request_id"), event.get("seq")
        if (not isinstance(request_id, str) or type(seq) is not int or seq < 1 or
            (request_id, seq) in seen or seq <= last_sequence.get(request_id, 0)):
            raise PlacementSnapshotError("event sequence is invalid, duplicated, or out of order")
        seen.add((request_id, seq))
        last_sequence[request_id] = seq
    cancelled = [event for event in events if event.get("kind") == "cancelled"]
    if len(cancelled) != 1 or cancelled[0].get("epoch") != 1:
        raise PlacementSnapshotError("trace lacks one first-epoch cancellation")
    cancelled_id = cancelled[0]["request_id"]
    if release.get("request_id") != cancelled_id:
        raise PlacementSnapshotError("release belongs to a different request")
    cancelled_turn = sorted((event for event in events if event.get("request_id") == cancelled_id),
                            key=lambda event: event["seq"])
    if ([event["seq"] for event in cancelled_turn] != list(range(1, len(cancelled_turn) + 1)) or
        any(event.get("epoch") != 1 for event in cancelled_turn)):
        raise PlacementSnapshotError("cancelled request event order is invalid")
    releases = [event for event in cancelled_turn if event.get("kind") == "state_released"]
    if (len(releases) != 1 or releases[0].get("payload") != release or
        releases[0]["seq"] <= cancelled[0]["seq"] or
        any(event.get("kind") in {"text_delta", "final"} and
            event["seq"] > cancelled[0]["seq"] for event in cancelled_turn)):
        raise PlacementSnapshotError("release event is missing or post-cancellation output exists")
    recovered = [event for event in events if event.get("kind") == "final" and
                 event.get("epoch") == 2]
    if len(recovered) != 1:
        raise PlacementSnapshotError("trace lacks one recovered final request")
    final = recovered[0]
    request_id = final["request_id"]
    turn = sorted((event for event in events if event.get("request_id") == request_id),
                  key=lambda event: event["seq"])
    if ([event["seq"] for event in turn] != list(range(1, len(turn) + 1)) or
        not turn or any(event.get("epoch") != 2 for event in turn) or
        turn[-1] is not final):
        raise PlacementSnapshotError("recovered request event order is invalid")
    routes = [event for event in turn if event.get("kind") == "route"]
    metrics = [event for event in turn if event.get("kind") == "model_metrics"]
    if len(routes) != 1 or len(metrics) != 1:
        raise PlacementSnapshotError("recovered request lacks route or stage proof")
    actual = routes[0].get("payload", {})
    stage = metrics[0].get("payload", {}).get("metrics", {}).get("stage_event", {})
    if (actual.get("route_id") != route.get("route_id") or
        actual.get("actual_placement") != route.get("placement") or
        actual.get("artifact_id") != route.get("artifact_id") or
        stage.get("worker_generation") != second["generation"] or
        stage.get("request_id") != f"{request_id}-step-0" or
        stage.get("terminal") is not True or stage.get("error") is not None):
        raise PlacementSnapshotError("recovered request does not bind to second worker and route")
    deltas = [event.get("payload", {}).get("text") for event in turn
              if event.get("kind") == "text_delta"]
    answer = final.get("payload", {}).get("answer")
    if (not deltas or any(not isinstance(delta, str) for delta in deltas) or
        not isinstance(answer, str) or "".join(deltas) != answer or
        answer != manifest.get("recovery_answer")):
        raise PlacementSnapshotError("recovered output differs from streamed deltas")
    return {
        "worker_generation": second["generation"], "worker_pid": second["pid"],
        "recovered_request_id": request_id, "recovered_epoch": 2,
        "route_seq": routes[0]["seq"], "stage_seq": metrics[0]["seq"],
        "final_seq": final["seq"], "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
        "stage_event_request_id": stage["request_id"],
        "first_worker_release_pid": first["pid"],
        "worker_binding_source": "cancel_harness_manifest_and_recovered_stage_event",
        "startup_log_embeds_worker_identity": False,
    }


def snapshot(record_path: Path, startup_log_path: Path, output_path: Path) -> dict[str, Any]:
    """Write a private, content-addressed log copy and no-clobber JSON record."""
    absolute_output = output_path.resolve()
    if ("public_evidence" in output_path.parts or "public_evidence" in startup_log_path.parts or
        absolute_output.is_relative_to(_REPO) and not absolute_output.is_relative_to(_PRIVATE_RESULTS)):
        raise PlacementSnapshotError("raw placement evidence must not use public_evidence")
    raw_trace, manifest, events = _load_trace(record_path)
    config = manifest.get("config")
    if (not isinstance(config, dict) or manifest.get("batch_size") != 1 or
        manifest.get("concurrency") != 1 or not isinstance(config.get("routes"), list) or
        len(config["routes"]) != 1):
        raise PlacementSnapshotError("trace lacks one batch-1 configured route")
    route = config["routes"][0]
    if not isinstance(route, dict):
        raise PlacementSnapshotError("trace route is invalid")
    proof = _request_proof(manifest, events, route)
    log_bytes = _stable_bytes(startup_log_path, max_bytes=2 << 20)
    placement = _parse_log(log_bytes, route)
    artifacts = {
        "model": _artifact(route.get("model_file"), route.get("model_sha256")),
        "server": _artifact(route.get("server_bin"), route.get("server_sha256")),
    }
    if route.get("mmproj_file") is not None:
        artifacts["projector"] = _artifact(route.get("mmproj_file"), route.get("mmproj_sha256"))
    log_sha = hashlib.sha256(log_bytes).hexdigest()
    log_name = f"{output_path.stem}.startup_{log_sha}.log"
    log_copy = output_path.with_name(log_name)
    if output_path.exists() or log_copy.exists():
        raise PlacementSnapshotError("snapshot output already exists; refusing overwrite")
    result: dict[str, Any] = {
        "schema": "omni-agent-placement-snapshot-v1",
        "record_type": "runtime_placement_observation_v1",
        "status": "observed_not_qualified",
        "source_record": {"path": str(record_path.resolve()),
                          "sha256": hashlib.sha256(raw_trace).hexdigest(),
                          "size_bytes": len(raw_trace)},
        "startup_log": {"path": log_name, "sha256": log_sha,
                        "size_bytes": len(log_bytes), "copied_after_recovery": True},
        "config": {"canonical_sha256": hashlib.sha256(_canonical(config)).hexdigest(),
                   "route_canonical_sha256": hashlib.sha256(_canonical(route)).hexdigest(),
                   "route_id": route["route_id"], "artifact_id": route["artifact_id"]},
        "artifacts": artifacts, "placement": placement, "request_proof": proof,
        "limitations": [
            "Startup log is copied after the recovery run; it has no embedded PID or generation.",
            "Layer offload and model buffers do not establish per-op compute placement.",
            "This observation is not an independent signed qualification gate or latency profile.",
        ],
    }
    result["record_sha256"] = record_digest(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_copy.open("xb") as stream:
            stream.write(log_bytes)
        if hashlib.sha256(log_copy.read_bytes()).hexdigest() != log_sha:
            raise PlacementSnapshotError("copied startup log differs from source")
        with output_path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, sort_keys=True, indent=2, ensure_ascii=False)
            stream.write("\n")
    except BaseException:
        if not output_path.exists():
            log_copy.unlink(missing_ok=True)
        raise
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--startup-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = snapshot(args.record, args.startup_log, args.output)
    print(json.dumps({"status": result["status"], "record_sha256": result["record_sha256"],
                      "observed_placement": result["placement"]["reported_placement"],
                      "offloaded_layers": result["placement"]["offloaded_layers"],
                      "total_layers": result["placement"]["total_layers"]}))


if __name__ == "__main__":
    main()
