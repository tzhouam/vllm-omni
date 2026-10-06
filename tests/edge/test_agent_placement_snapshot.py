"""The private placement snapshot rejects ambiguous provenance and loader data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.edge_agent.experiments.snapshot_runtime_placement import (
    PlacementSnapshotError, snapshot, verify_record,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict, str]:
    model = tmp_path / "model.gguf"
    server = tmp_path / "llama-server.exe"
    projector = tmp_path / "mmproj.gguf"
    for path, data in ((model, b"model"), (server, b"server"), (projector, b"projector")):
        path.write_bytes(data)
    route = {
        "route_id": "gemma-test", "artifact_id": "a1", "model_file": str(model),
        "model_sha256": _sha(b"model"), "server_bin": str(server),
        "server_sha256": _sha(b"server"), "mmproj_file": str(projector),
        "mmproj_sha256": _sha(b"projector"), "placement": "cpu+Vulkan0",
        "expected_device_name": "Test GPU", "gpu_layers": 1,
    }
    first_id, second_id = "cancelled", "recovered"
    manifest = {
        "record_type": "manifest", "batch_size": 1, "concurrency": 1,
        "config": {"routes": [route]},
        "first_worker": {"generation": "first-generation", "pid": 101,
                         "placement": "cpu+Vulkan0"},
        "release_evidence": {"request_id": first_id, "release_mode": "worker_shutdown",
                             "worker_pid_before": 101, "worker_exit_code": 1,
                             "backend_request_state_verified": True,
                             "graph_gate_released": True, "tool_request_finished": True,
                             "worker_exit_confirmed": True, "stage_ledger_empty": True,
                             "host_claim_released": True, "host_ledger_empty": True},
        "second_worker": {"generation": "second-generation", "pid": 202,
                          "placement": "cpu+Vulkan0"},
        "recovery_answer": "ready", "result": "cancelled_stream_then_restarted_request_passed",
    }
    events = [
        {"request_id": first_id, "epoch": 1, "seq": 1, "kind": "text_delta", "payload": {"text": "1"}},
        {"request_id": first_id, "epoch": 1, "seq": 2, "kind": "cancelled", "payload": {}},
        {"request_id": first_id, "epoch": 1, "seq": 3, "kind": "state_released",
         "payload": manifest["release_evidence"]},
        {"request_id": second_id, "epoch": 2, "seq": 1, "kind": "user_observation", "payload": {}},
        {"request_id": second_id, "epoch": 2, "seq": 2, "kind": "route", "payload": {
            "route_id": "gemma-test", "artifact_id": "a1", "actual_placement": "cpu+Vulkan0",
        }},
        {"request_id": second_id, "epoch": 2, "seq": 3, "kind": "text_delta",
         "payload": {"text": "ready"}},
        {"request_id": second_id, "epoch": 2, "seq": 4, "kind": "model_metrics",
         "payload": {"metrics": {"stage_event": {
             "request_id": "recovered-step-0", "worker_generation": "second-generation",
             "terminal": True, "error": None,
         }}}},
        {"request_id": second_id, "epoch": 2, "seq": 5, "kind": "final",
         "payload": {"answer": "ready"}},
    ]
    record = tmp_path / "cancel.jsonl"
    record.write_text("".join(json.dumps(row) + "\n" for row in [manifest, *events]),
                      encoding="utf-8")
    log_text = "\n".join((
        "llama_model_load: using device Vulkan0 (Test GPU)",
        "load_tensors: layer 0 assigned to device CPU",
        "load_tensors: layer 1 assigned to device CPU",
        "load_tensors: layer 2 assigned to device Vulkan0",
        "load_tensors: offloaded 1/3 layers to GPU",
        "load_tensors: CPU model buffer size = 2.00 MiB",
        "load_tensors: Vulkan0 model buffer size = 1.00 MiB",
        "llama.cpp status: model loaded",
    )) + "\n"
    log = tmp_path / "startup.log"
    log.write_text(log_text, encoding="utf-8")
    return record, log, tmp_path / "results" / "snapshot.json", manifest, log_text


def _rewrite(record: Path, rows: list[dict]) -> None:
    record.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_snapshot_binds_exact_bytes_worker_and_recovered_request(tmp_path: Path) -> None:
    record, log, output, manifest, log_text = _fixture(tmp_path)
    observed = snapshot(record, log, output)
    assert verify_record(observed)
    assert observed["source_record"]["sha256"] == _sha(record.read_bytes())
    assert observed["config"]["canonical_sha256"]
    assert observed["artifacts"]["model"]["sha256"] == manifest["config"]["routes"][0]["model_sha256"]
    assert observed["artifacts"]["server"]["sha256"] == manifest["config"]["routes"][0]["server_sha256"]
    assert observed["startup_log"]["sha256"] == _sha(log.read_bytes())
    assert (output.parent / observed["startup_log"]["path"]).read_bytes() == log.read_bytes()
    assert observed["placement"]["offloaded_layers"] == 1
    assert observed["placement"]["cpu_layer_count"] == 2
    assert observed["placement"]["compute_placement_verified"] is False
    assert observed["request_proof"]["worker_pid"] == 202
    assert observed["request_proof"]["worker_generation"] == "second-generation"
    assert observed["request_proof"]["startup_log_embeds_worker_identity"] is False
    assert observed["request_proof"]["answer_sha256"] == _sha(b"ready")
    assert json.loads(output.read_text(encoding="utf-8")) == observed
    with pytest.raises(PlacementSnapshotError, match="already exists"):
        snapshot(record, log, output)
    assert json.loads(output.read_text(encoding="utf-8")) == observed


@pytest.mark.parametrize("mutation", [
    "duplicate_report", "missing_marker", "wrong_layer", "wrong_device", "wrong_buffer",
])
def test_rejects_ambiguous_or_false_loader_report(tmp_path: Path, mutation: str) -> None:
    record, log, output, _, text = _fixture(tmp_path)
    edits = {
        "duplicate_report": text + "load_tensors: offloaded 1/3 layers to GPU\n",
        "missing_marker": text.replace("llama.cpp status: model loaded\n", ""),
        "wrong_layer": text.replace("layer 2 assigned to device Vulkan0", "layer 1 assigned to device Vulkan0"),
        "wrong_device": text.replace("using device Vulkan0 (Test GPU)", "using device Vulkan1 (Test GPU)"),
        "wrong_buffer": text.replace("Vulkan0 model buffer size", "Vulkan1 model buffer size"),
    }
    log.write_text(edits[mutation], encoding="utf-8")
    with pytest.raises(PlacementSnapshotError):
        snapshot(record, log, output)
    assert not output.exists()


@pytest.mark.parametrize("mutation", [
    "wrong_generation", "wrong_pid", "wrong_answer", "duplicate_seq", "wrong_route",
    "wrong_release", "post_cancel_delta",
])
def test_rejects_wrong_worker_or_request_proof(tmp_path: Path, mutation: str) -> None:
    record, log, output, _, _ = _fixture(tmp_path)
    rows = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]
    recovered_route = next(row for row in rows if row.get("kind") == "route")
    recovered_metrics = next(row for row in rows if row.get("kind") == "model_metrics")
    recovered_final = next(row for row in rows if row.get("kind") == "final")
    if mutation == "wrong_generation":
        recovered_metrics["payload"]["metrics"]["stage_event"]["worker_generation"] = "first-generation"
    elif mutation == "wrong_pid":
        rows[0]["second_worker"]["pid"] = rows[0]["first_worker"]["pid"]
    elif mutation == "wrong_answer":
        recovered_final["payload"]["answer"] = "not ready"
    elif mutation == "duplicate_seq":
        recovered_final["seq"] = recovered_metrics["seq"]
    elif mutation == "wrong_route":
        recovered_route["payload"]["actual_placement"] = "CPU"
    elif mutation == "wrong_release":
        rows[0]["release_evidence"]["request_id"] = "another-request"
    elif mutation == "post_cancel_delta":
        release_event = next(row for row in rows if row.get("kind") == "state_released")
        release_event["kind"] = "text_delta"
    _rewrite(record, rows)
    with pytest.raises(PlacementSnapshotError):
        snapshot(record, log, output)


@pytest.mark.parametrize("mutation", ["duplicate_manifest", "cancel_order", "recovery_order"])
def test_rejects_duplicate_manifest_or_reordered_raw_events(tmp_path: Path, mutation: str) -> None:
    record, log, output, _, _ = _fixture(tmp_path)
    rows = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]
    if mutation == "duplicate_manifest":
        rows.insert(1, dict(rows[0]))
    elif mutation == "cancel_order":
        rows[1], rows[2] = rows[2], rows[1]
    else:
        rows[4], rows[5] = rows[5], rows[4]
    _rewrite(record, rows)
    with pytest.raises(PlacementSnapshotError, match="manifest|out of order"):
        snapshot(record, log, output)
    assert not output.exists()


def test_rehashes_artifacts_and_keeps_raw_evidence_private(tmp_path: Path) -> None:
    record, log, output, manifest, _ = _fixture(tmp_path)
    model = Path(manifest["config"]["routes"][0]["model_file"])
    model.write_bytes(b"tampered")
    with pytest.raises(PlacementSnapshotError, match="recorded hash"):
        snapshot(record, log, output)
    assert not output.exists()
    with pytest.raises(PlacementSnapshotError, match="public_evidence"):
        snapshot(record, log, tmp_path / "public_evidence" / "snapshot.json")


def test_snapshot_record_digest_detects_mutation(tmp_path: Path) -> None:
    record, log, output, _, _ = _fixture(tmp_path)
    observed = snapshot(record, log, output)
    observed["placement"]["offloaded_layers"] = 2
    assert verify_record(observed) is False
