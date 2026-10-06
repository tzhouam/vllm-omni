# SPDX-License-Identifier: Apache-2.0
"""Probe one native Windows Omni Agent route's memory claim at batch size one.

This is an independent, private diagnostic. It records system-wide RAM and
NVML GPU-0 VRAM before load, throughout cold load, and throughout one complete
Agent turn. Sampled global increments are compared with the route's declared
reservations; they are *not* model-process allocation or an upper bound on an
unsampled transient. An actual ResourceLedger over-ceiling declaration is also
refused without allocating memory. No result from this script signs or sets an
Agent qualification gate.

Run from native Windows Python with ``-X utf8`` and this checkout on PYTHONPATH.
Raw JSONL and the exact temporary config are written only to the current
Windows user's LocalAppData/OmniEdgeAgent/memory-probes directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from benchmarks.edge_agent.experiments.profile_binding import bind_profile_to_live
from vllm_omni.engine.resource_ledger import ResourceLedger, ResourceUnavailable


DEFAULT_PROMPT = "Reply with the single word ready."


def private_record_path(*, root: Path, name: str | None = None) -> Path:
    """Choose a new basename under the private directory, never a caller path."""
    if name is None:
        name = f"memory-probe-{uuid.uuid4().hex}.jsonl"
    if not name.endswith(".jsonl") or Path(name).name != name or name in {".", ".."}:
        raise ValueError("record name must be a simple .jsonl basename")
    return root / name


def over_ceiling_ledger_refusal(capacities: Mapping[str, int], pool: str) -> dict[str, Any]:
    """Exercise Omni's real reservation gate with a declaration, not malloc."""
    if pool not in capacities:
        raise ValueError(f"unknown physical memory pool: {pool}")
    ceiling = capacities[pool]
    if type(ceiling) is not int or ceiling < 0:
        raise ValueError("ledger capacity must be a nonnegative integer")
    ledger = ResourceLedger(capacities)
    demand = ceiling + 1
    try:
        ledger.reserve("probe-over-ceiling", {pool: demand})
    except ResourceUnavailable as exc:
        snapshot = ledger.snapshot()
        if snapshot["owners"] or snapshot["quarantined"] or any(snapshot["reserved"].values()):
            raise AssertionError("failed ledger reservation retained a claim") from exc
        return {
            "pool": pool, "ceiling_bytes": ceiling, "declared_bytes": demand,
            "refusal": str(exc), "ledger_after_refusal": snapshot,
            "physical_allocation_attempted": False,
        }
    raise AssertionError("Omni ResourceLedger admitted a declaration over its ceiling")


def compare_samples(
    samples: list[Mapping[str, Any]], demands: Mapping[str, int],
) -> dict[str, Any]:
    """Report observed whole-host increments from the pre-load baseline.

    Include both load and request phases in each peak. Negative movement from
    unrelated processes is clipped at zero, but positive interference remains
    included; a sampled low peak cannot certify a safe admission ceiling.
    """
    if not samples or samples[0].get("phase") != "pre_load":
        raise ValueError("a pre-load baseline sample is required")
    if not any(sample.get("phase") == "cold_load" for sample in samples):
        raise ValueError("cold-load samples are required")
    if not any(sample.get("phase") == "complete_request" for sample in samples):
        raise ValueError("complete-request samples are required")
    fields = {"host_ram": "ram_used_bytes", "vram": "vram_used_bytes"}
    comparison: dict[str, Any] = {}
    for pool, declared in demands.items():
        if pool not in fields or type(declared) is not int or declared < 0:
            raise ValueError(f"unsupported or invalid route claim: {pool}")
        field = fields[pool]
        values = [sample.get("values", {}).get(field) for sample in samples]
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError(f"{pool} measurement unavailable or malformed")
        baseline = values[0]
        peak = max(values)
        comparison[pool] = {
            "declared_reservation_bytes": declared,
            "pre_load_used_bytes": baseline,
            "sampled_global_used_peak_bytes": peak,
            "sampled_global_incremental_peak_bytes": max(0, peak - baseline),
            "reservation_minus_sampled_incremental_bytes": declared - max(0, peak - baseline),
            "sample_count": len(values),
        }
    return comparison


def verify_resident_claim(
    snapshot: Mapping[str, Any], route_id: str, demands: Mapping[str, int],
) -> None:
    """Bind the sampled resident interval to the coordinator's real claim."""
    ledger = snapshot.get("ledger", {})
    if (snapshot.get("resident_route") != route_id or
        ledger.get("owners") != [route_id] or ledger.get("quarantined") or
        any(ledger.get("reserved", {}).get(pool) != amount
            for pool, amount in demands.items())):
        raise AssertionError("host coordinator does not own the declared resident route")


def sample_operation(
    operation: Future[Any], *, phase: str, telemetry: Any,
    write_sample: Any, interval_s: float, timeout_s: float,
    write_deadline: Any = None,
) -> Any:
    """Sample while one already-submitted blocking operation is in flight."""
    if not 0.01 <= interval_s <= 10 or timeout_s <= 0:
        raise ValueError("invalid sampling interval or operation timeout")
    deadline = time.monotonic() + timeout_s
    first_error: BaseException | None = None
    while True:
        try:
            write_sample(phase, telemetry.sample())
        except BaseException as exc:
            first_error = first_error or exc
        if operation.done():
            break
        if time.monotonic() >= deadline:
            if write_deadline is not None:
                # A native loader may ignore cancellation. Persist the missed
                # deadline before executor teardown can wait on that thread.
                write_deadline(phase)
            raise TimeoutError(f"{phase} exceeded its memory-probe deadline")
        time.sleep(min(interval_s, max(0, deadline - time.monotonic())))
    result = operation.result()
    if first_error is not None:
        raise RuntimeError(f"{phase} telemetry failed") from first_error
    return result


def run_probe(
    *, config_path: Path, route_id: str, record_root: Path,
    record_name: str | None = None, prompt: str = DEFAULT_PROMPT,
    interval_s: float = 0.1, profile_index: Path | None = None,
) -> Path:
    if sys.platform != "win32":
        raise RuntimeError("the physical memory probe requires native Windows Python")
    from benchmarks.edge_agent.native_profile import WindowsTelemetry
    from vllm_omni.edge.agent.native_app import build_controller

    original_bytes = config_path.read_bytes()
    original = json.loads(original_bytes)
    matches = [entry for entry in original["routes"] if entry["route_id"] == route_id]
    if len(matches) != 1:
        raise ValueError("the selected route must occur exactly once in the native config")
    selected = dict(matches[0])
    demands = selected["memory_demands"]
    record_root = record_root.resolve()
    record_root.mkdir(parents=True, exist_ok=True)
    record = private_record_path(root=record_root, name=record_name)
    config_copy = record.with_suffix(".config.json")
    # This controller is isolated from the user's persistent Agent memory and
    # from any reviewed qualifications. It still uses the exact route backend,
    # physical memory coordinator, and StageRuntime ledger.
    temporary = dict(original)
    temporary["routes"] = [selected]
    temporary.pop("qualification_bundles", None)
    temporary.pop("trusted_review_keys", None)
    temporary.pop("qualification_file", None)
    temporary["experimental_bootstrap_route_id"] = route_id
    temporary["memory_file"] = str(record.with_suffix(".memory.sqlite"))
    selected["log_file"] = str(record.with_suffix(".server.log"))
    config_bytes = json.dumps(temporary, indent=2, ensure_ascii=False).encode("utf-8")
    # Exclusive creation protects earlier evidence; private root is on local
    # NTFS, avoiding both git tracking and SQLite locking on the WSL share.
    with config_copy.open("xb") as output:
        output.write(config_bytes)
    samples: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    controller = None
    coordinator = None
    telemetry = None
    answer: str | None = None
    result = "failed"
    error: str | None = None
    failure: BaseException | None = None
    with record.open("x", encoding="utf-8") as output:
        def write(row: Mapping[str, Any]) -> None:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()

        def write_sample(phase: str, values: Mapping[str, Any]) -> None:
            row = {
                "record_type": "sample", "phase": phase,
                "monotonic_ns": time.monotonic_ns(),
                "utc": datetime.now(timezone.utc).isoformat(),
                "values": dict(values),
            }
            samples.append(row)
            write(row)
            for pool, field in (("host_ram", "ram_used_bytes"), ("vram", "vram_used_bytes")):
                if pool in demands and (
                    type(values.get(field)) is not int or values[field] < 0
                ):
                    raise RuntimeError(f"{pool} sensor did not return a valid used-byte reading")

        manifest = {
            "record_type": "manifest", "scope": "independent_memory_admission_probe",
            "batch_size": 1, "concurrency": 1, "route_id": route_id,
            "source_config_sha256": hashlib.sha256(original_bytes).hexdigest(),
            "probe_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "probe_config": str(config_copy),
            "model_sha256": selected["model_sha256"],
            "server_sha256": selected["server_sha256"],
            "mmproj_sha256": selected.get("mmproj_sha256"),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_utf8_bytes": len(prompt.encode("utf-8")),
            "sampling_interval_s": interval_s,
            "profile_binding_requested": profile_index is not None,
            "profile_binding": None,
            "signed_qualification_gate": False,
        }
        manifest_written = False
        try:
            controller, hardware = build_controller(config_copy)
            if profile_index is not None:
                manifest["profile_binding"] = bind_profile_to_live(
                    profile_index, source_config_bytes=original_bytes,
                    route_id=route_id, hardware=hardware,
                )
            write(manifest)
            manifest_written = True
            write({"record_type": "hardware", "hardware": hardware})
            capacities = {"host_ram": hardware["host_ram_available_bytes"]}
            if hardware["vram_available_bytes"] is not None:
                capacities["vram"] = hardware["vram_available_bytes"]
            for pool in demands:
                write({"record_type": "over_ceiling_refusal",
                       **over_ceiling_ledger_refusal(capacities, pool)})
            route = next(route for route in controller.routes if route.route_id == route_id)
            admission = controller.admit(route)
            write({"record_type": "pre_load_admission", "admitted": admission.admitted,
                   "reason": admission.reason})
            if not admission.admitted:
                raise ResourceUnavailable(admission.reason)
            backend = controller.backends[route_id]
            coordinator = backend._coordinator
            if backend.execution_plan is not None:
                raise RuntimeError("probe route was already resident before cold load")
            telemetry = WindowsTelemetry(hardware["power_condition"])
            write_sample("pre_load", telemetry.sample())
            controller.add_listener(lambda event: events.append({
                "kind": event.get("kind"), "epoch": event.get("epoch"),
                "seq": event.get("seq"), "request_id": event.get("request_id"),
            }))
            with ThreadPoolExecutor(max_workers=1) as pool:
                load = pool.submit(backend.start)
                sample_operation(
                    load, phase="cold_load", telemetry=telemetry,
                    write_sample=write_sample, interval_s=interval_s,
                    timeout_s=float(selected.get("start_timeout_s", 300)) + 30,
                    write_deadline=lambda phase: write({
                        "record_type": "operation_deadline_exceeded",
                        "phase": phase, "monotonic_ns": time.monotonic_ns(),
                        "worker_still_running": not load.done(),
                    }),
                )
            plan = backend.execution_plan
            if not isinstance(plan, Mapping) or (
                plan.get("requested_device") != route.placement or
                dict(plan.get("reserved_bytes", {})) != dict(demands)
            ):
                raise AssertionError("loaded Omni placement or stage ledger differs from declared route")
            write({"record_type": "loaded_plan", "plan": dict(plan)})
            loaded_host_claim = coordinator.snapshot()
            verify_resident_claim(loaded_host_claim, route_id, demands)
            write({"record_type": "host_ledger_after_load", "snapshot": loaded_host_claim})
            task = controller.submit(prompt)
            answer = sample_operation(
                task, phase="complete_request", telemetry=telemetry,
                write_sample=write_sample, interval_s=interval_s,
                timeout_s=float(selected.get("request_timeout_s", 300)) *
                          int(original.get("limits", {}).get("max_model_steps", 6)) + 30,
                write_deadline=lambda phase: write({
                    "record_type": "operation_deadline_exceeded",
                    "phase": phase, "monotonic_ns": time.monotonic_ns(),
                    "worker_still_running": not task.done(),
                }),
            )
            kinds = [event["kind"] for event in events]
            if (kinds.count("final") != 1 or any(kind in kinds for kind in
                    ("error", "refusal", "cancelled", "approval_required"))):
                raise AssertionError("Agent turn did not complete with one final answer")
            if not isinstance(answer, str) or not answer:
                raise AssertionError("complete Agent turn returned no final text")
            request_host_claim = coordinator.snapshot()
            verify_resident_claim(request_host_claim, route_id, demands)
            write({"record_type": "host_ledger_after_request", "snapshot": request_host_claim})
            write_sample("post_request", telemetry.sample())
            result = "one_complete_agent_turn_measured"
        except BaseException as exc:
            if not manifest_written:
                write(manifest)
            failure = exc
            error = f"{type(exc).__name__}: {exc}"
            if controller is not None:
                try:
                    controller.cancel()
                except Exception:
                    pass
        finally:
            try:
                if controller is not None:
                    controller.close()
                if coordinator is not None:
                    released = coordinator.snapshot()
                    write({"record_type": "host_ledger_after_release", "snapshot": released})
                    ledger = released["ledger"]
                    if (released["resident_route"] is not None or ledger["owners"] or
                        ledger["quarantined"] or any(ledger["reserved"].values())):
                        raise AssertionError("host coordinator retained a memory claim after close")
                if telemetry is not None:
                    write_sample("post_release", telemetry.sample())
            except BaseException as exc:
                if failure is None:
                    failure = exc
                    error = f"{type(exc).__name__}: {exc}"
            finally:
                if telemetry is not None:
                    try:
                        telemetry.close()
                    except BaseException as exc:
                        if failure is None:
                            failure = exc
                            error = f"{type(exc).__name__}: {exc}"
                comparison = None
                if samples and failure is None:
                    try:
                        comparison = compare_samples(samples, demands)
                    except BaseException as exc:
                        failure = exc
                        error = f"{type(exc).__name__}: {exc}"
                if samples:
                    write({"record_type": "sampled_comparison",
                           "caveat": "global sampled increments; external processes may interfere and transients may be missed",
                           "comparison": comparison})
                if failure is not None:
                    result = "failed"
                write({
                    "record_type": "outcome", "result": result, "error": error,
                    "sample_count": len(samples), "event_kinds": [event["kind"] for event in events],
                    "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest()
                    if isinstance(answer, str) else None,
                    "signed_qualification_gate": False,
                })
    if failure is not None:
        raise failure
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--record-name", help="New .jsonl basename under private LocalAppData")
    parser.add_argument("--sample-interval-s", type=float, default=0.1)
    parser.add_argument("--profile-index", type=Path,
                        help="Audited full-protocol profile to bind this later diagnostic")
    args = parser.parse_args()
    local_data = Path(os.environ.get("LOCALAPPDATA", ""))
    if not str(local_data) or not local_data.is_absolute():
        raise RuntimeError("LOCALAPPDATA must identify native Windows private storage")
    record = run_probe(
        config_path=args.config, route_id=args.route_id,
        record_root=local_data / "OmniEdgeAgent" / "memory-probes",
        record_name=args.record_name, interval_s=args.sample_interval_s,
        profile_index=args.profile_index,
    )
    print(json.dumps({"private_record": str(record),
                      "scope": "memory_admission_diagnostic_not_qualification"}))


if __name__ == "__main__":
    main()
