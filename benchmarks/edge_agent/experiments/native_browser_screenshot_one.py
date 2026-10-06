# SPDX-License-Identifier: Apache-2.0
"""One native Windows Qwen3.6 browser-screenshot Agent request, never a profile.

This benchmark isolates the browser-image case from the fixed visual suite.
It makes exactly one Agent submission, with no warmup or desktop capture.
Output is private, uniquely named, hash-bound functional evidence and cannot
qualify a route or provide a p50/p95 latency estimate.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, TextIO
from urllib.request import urlopen

from benchmarks.edge_agent.native_profile import (
    NativeProfileBridge,
    WindowsTelemetry,
    _conditions,
    load_profile_routes,
)
from benchmarks.edge_agent.paired_suite import (
    FixtureSite,
    build_paired_cases,
    evaluate_case,
)
from benchmarks.edge_agent.profile import _one_request


ROUTE_ID = "qwen3.6-35b-a3b-iq4-xs-windows-host-mapped-experts40"
CASE_ID = "screen_vision-en-short"
PINNED_DOWNLOAD_MANIFEST = (
    Path(__file__).resolve().parents[1] / "configs" /
    "qwen3_6_35b_a3b_ud_iq4_xs_download.json"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_record(handle: TextIO, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _write_new_json(path: Path, record: Mapping[str, Any]) -> str:
    data = (json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256(data)


def _select_case(origin: str, mouse_speed: int):
    cases = build_paired_cases(origin, mouse_speed)["browser_vision"]["short"]
    selected = [case for case in cases if case.case_id == CASE_ID]
    if len(selected) != 1:
        raise ValueError("fixed browser-screenshot case is missing or duplicated")
    case = selected[0]
    if (case.metadata.get("kind") != "screen_vision" or
            tuple(case.metadata.get("required_operations", ())) !=
            ("browser_open", "browser_screenshot") or
            case.metadata.get("source") != origin + "/visual/en" or
            case.reference != "ORBIT-7391"):
        raise ValueError("fixed browser-screenshot reference changed")
    return case


def _request_passed(row: Mapping[str, Any] | None) -> bool:
    return bool(
        row is not None and row.get("error") is None and
        row.get("e2e_complete") is True and
        row.get("placement_matches") is True and
        isinstance(row.get("evaluation"), Mapping) and
        row["evaluation"].get("success") is True and
        row["evaluation"].get("tool_safe") is True
    )


def _validate_pinned_bundle(entry: Mapping[str, Any],
                            lineage_item: Mapping[str, Any],
                            download: Mapping[str, Any]) -> None:
    if download.get("repo") != "unsloth/Qwen3.6-35B-A3B-GGUF":
        raise ValueError("unexpected Qwen3.6 artifact repository")
    if lineage_item.get("artifact_revision") != download.get("revision"):
        raise ValueError("Qwen3.6 route artifact revision differs from pinned download")
    files = download.get("files")
    if not isinstance(files, list) or len(files) != 2:
        raise ValueError("Qwen3.6 model and projector must both be pinned")
    pinned = {item["filename"]: item for item in files}
    if len(pinned) != 2:
        raise ValueError("duplicate Qwen3.6 artifact filename")
    for path_key, hash_key in (("model_file", "model_sha256"),
                               ("mmproj_file", "mmproj_sha256")):
        filename = PureWindowsPath(str(entry[path_key])).name
        item = pinned.get(filename)
        if item is None or item.get("sha256") != entry.get(hash_key):
            raise ValueError(f"{path_key} does not match pinned Qwen3.6 artifact")


async def run_one(*, config_path: Path, lineage_path: Path,
                  output_dir: Path) -> tuple[Path, bool]:
    if sys.platform != "win32":
        raise RuntimeError("native browser-screenshot evidence requires Windows Python")

    from vllm_omni.edge.agent.native_app import _hardware_snapshot
    from vllm_omni.edge.agent.tools import WindowsSettings

    config_bytes = config_path.read_bytes()
    lineage_bytes = lineage_path.read_bytes()
    download_bytes = PINNED_DOWNLOAD_MANIFEST.read_bytes()
    native_config = json.loads(config_bytes)
    lineage = json.loads(lineage_bytes)
    download = json.loads(download_bytes)
    routes, provenance = load_profile_routes(native_config, lineage)
    if len(routes) != 1 or routes[0].route_id != ROUTE_ID:
        raise ValueError("this one-case runner requires exactly the pinned Qwen3.6 route")
    route = routes[0]
    entry = native_config["routes"][0]
    if (not entry.get("mmproj_file") or not entry.get("mmproj_sha256") or
            "image" not in entry.get("modalities", ())):
        raise ValueError("the Qwen3.6 route must pin its image projector")
    _validate_pinned_bundle(entry, lineage["routes"][ROUTE_ID], download)

    run_id = "native_visual_one_" + uuid.uuid4().hex
    work_root = output_dir.resolve() / run_id
    work_root.mkdir(parents=True, exist_ok=False)
    local_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    private_root = local_data / "OmniEdgeAgent" / "profile-memory" / run_id
    private_root.mkdir(parents=True, exist_ok=False)
    raw_path = work_root / "samples.jsonl"
    row: dict[str, Any] | None = None
    failure: str | None = None
    sampler: WindowsTelemetry | None = None
    bridge: NativeProfileBridge | None = None
    with raw_path.open("x", encoding="utf-8") as raw:
        try:
            hardware = _hardware_snapshot()
            conditions = _conditions(hardware, native_config)
            sampler = WindowsTelemetry(conditions.power_condition)
            mouse_speed = int(WindowsSettings().read("mouse_speed")["value"])
            with FixtureSite() as fixture:
                case = _select_case(fixture.origin, mouse_speed)
                # Hash the exact input page and rendered-image source. These
                # fixture GETs are setup reads, not Agent submissions.
                with urlopen(str(case.metadata["source"]), timeout=5) as page:
                    page_bytes = page.read(1_000_001)
                with urlopen(fixture.origin + "/assets/en.svg", timeout=5) as image:
                    image_bytes = image.read(1_000_001)
                if len(page_bytes) > 1_000_000 or len(image_bytes) > 1_000_000:
                    raise ValueError("fixed visual fixture exceeded the 1 MB evidence cap")
                _write_record(raw, {
                    "record_type": "manifest", "schema": "edge-agent-single-visual-v1",
                    "run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(),
                    "scope": "complete_agent_request_fixed_browser_screenshot",
                    "protocol": "single_request_functional_smoke_unqualified",
                    "batch_size": 1, "concurrency": 1,
                    "protocol_compliant": False, "release_qualified": False,
                    "route": asdict(route), "case_id": case.case_id,
                    "case_prompt_sha256": _sha256(case.prompt.encode("utf-8")),
                    "fixture_page_sha256": _sha256(page_bytes),
                    "fixture_image_svg_sha256": _sha256(image_bytes),
                    "source_config": str(config_path.resolve()),
                    "source_config_sha256": _sha256(config_bytes),
                    "lineage_manifest": str(lineage_path.resolve()),
                    "lineage_sha256": _sha256(lineage_bytes),
                    "artifact_revision": lineage["routes"][ROUTE_ID]["artifact_revision"],
                    "license_claim": lineage["routes"][ROUTE_ID].get("license"),
                    "download_manifest_sha256": _sha256(download_bytes),
                    "artifact_provenance": provenance[route.route_id],
                    "hardware": hardware, "conditions": asdict(conditions),
                    "private_encrypted_memory_root": str(private_root),
                })
                bridge = NativeProfileBridge(
                    native_config=native_config,
                    config_root=work_root / "derived_configs",
                    private_root=private_root,
                    fixture_origin=fixture.origin,
                    telemetry=sampler,
                )
                load_started_ns = time.perf_counter_ns()
                prepared = await bridge.prepare(route)
                load_seconds = (time.perf_counter_ns() - load_started_ns) / 1e9
                cold_confirmed = bool(
                    prepared.cold_start_confirmed and
                    prepared.artifact_id == route.artifact_id and
                    prepared.actual_placement == route.expected_placement
                )
                _write_record(raw, {
                    "record_type": "route_prepare", "run_id": run_id,
                    "route_id": route.route_id,
                    "cold_start_s": load_seconds if cold_confirmed else None,
                    "preparation": asdict(prepared),
                })
                if not cold_confirmed:
                    raise RuntimeError("cold route preparation did not match the pinned route")
                setup = bridge.before_request(route, case, "measured", 0)
                row = await _one_request(
                    run_id=run_id, route=route, case=case,
                    phase="measured", repetition=0,
                    runner=bridge.run, evaluator=evaluate_case,
                    telemetry=sampler.sample, telemetry_interval_s=0.1,
                )
                row["fixture_setup"] = dict(setup)
                _write_record(raw, row)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            _write_record(raw, {
                "record_type": "failure", "run_id": run_id, "error": failure,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
        finally:
            if bridge is not None:
                try:
                    await asyncio.to_thread(bridge.close)
                except Exception as exc:
                    failure = f"cleanup {type(exc).__name__}: {exc}"
                    _write_record(raw, {
                        "record_type": "cleanup_failure", "run_id": run_id,
                        "error": failure,
                    })
            if sampler is not None:
                try:
                    sampler.close()
                except Exception as exc:
                    failure = f"telemetry cleanup {type(exc).__name__}: {exc}"
                    _write_record(raw, {
                        "record_type": "cleanup_failure", "run_id": run_id,
                        "error": failure,
                    })

    raw_sha256 = _sha256(raw_path.read_bytes())
    passed = failure is None and _request_passed(row)
    summary = {
        "schema": "edge-agent-single-visual-summary-v1",
        "run_id": run_id, "case_id": CASE_ID,
        "route_id": route.route_id, "batch_size": 1, "concurrency": 1,
        "measured_agent_requests": 1 if row is not None else 0,
        "success": passed, "status": "functional_smoke_pass" if passed else "failed_or_blocked",
        "error": failure or (row.get("error") if row is not None else None),
        "answer_latency_s": row.get("answer_latency_s") if row is not None else None,
        "ttft_s": row.get("ttft_s") if row is not None else None,
        "evaluation": row.get("evaluation") if row is not None else None,
        "raw_jsonl_sha256": raw_sha256,
        "protocol_compliant": False, "release_qualified": False,
        "limits": "One cold-loaded, fixed browser-image Agent request only; no warmups, latency distribution, endurance, independent expert-placement proof, or release qualification.",
    }
    summary_sha256 = _write_new_json(work_root / "summary.json", summary)
    index = {
        "schema": "edge-agent-single-visual-index-v1",
        "run_id": run_id, "summary": "summary.json",
        "summary_sha256": summary_sha256,
        "raw_jsonl": "samples.jsonl", "raw_jsonl_sha256": raw_sha256,
        "source_config_sha256": _sha256(config_bytes),
        "lineage_sha256": _sha256(lineage_bytes),
        "download_manifest_sha256": _sha256(download_bytes),
        "success": passed, "protocol_compliant": False, "release_qualified": False,
    }
    index_path = work_root / "index.json"
    index_sha256 = _write_new_json(index_path, index)
    with (work_root / "index.sha256").open("x", encoding="ascii") as handle:
        handle.write(index_sha256 + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return index_path, passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    index, passed = asyncio.run(run_one(
        config_path=args.config, lineage_path=args.lineage,
        output_dir=args.output_dir,
    ))
    print(index)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
