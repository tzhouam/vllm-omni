# SPDX-License-Identifier: Apache-2.0
"""Run paired, batch-one whole-Agent profiles with native Windows Omni.

The full protocol is deliberately expensive. ``--smoke`` only checks that the
bridge works and always remains unqualified. No output from this script is
automatically installed as a router qualification.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from benchmarks.edge_agent.paired_suite import (
    FixtureSite, ReadOnlyFixtureTools, build_paired_cases, evaluate_case,
)
from benchmarks.edge_agent.profile import (
    AgentCase, AgentRunResult, Preparation, ProfileConditions, ProfileConfig,
    ProfileRoute, run_profile,
)
from vllm_omni.edge.agent.tools import ManagedEdgeBrowser


SUITE_ID = "edge-agent-fixed-local-fixtures-v1"


def load_profile_routes(config: Mapping[str, Any],
                        lineage: Mapping[str, Any]) -> tuple[list[ProfileRoute], dict[str, Any]]:
    """Bind profiling identity to native config hashes and explicit lineage."""
    profiles: list[ProfileRoute] = []
    provenance: dict[str, Any] = {}
    entries = config.get("routes")
    if not isinstance(entries, list) or not entries:
        raise ValueError("native config has no routes")
    metadata = lineage.get("routes")
    if not isinstance(metadata, dict):
        raise ValueError("lineage manifest must contain routes by route_id")
    for entry in entries:
        route_id = str(entry["route_id"])
        item = metadata.get(route_id)
        if not isinstance(item, dict):
            raise ValueError(f"{route_id}: lineage metadata missing")
        model_sha = str(entry["model_sha256"])
        if item.get("model_sha256") != model_sha:
            raise ValueError(f"{route_id}: model hash differs from native config")
        if item.get("mmproj_sha256") != entry.get("mmproj_sha256"):
            raise ValueError(f"{route_id}: vision projector hash differs from native config")
        revision = str(item.get("checkpoint_revision", ""))
        precision = str(item.get("precision", ""))
        if not revision or not precision or len(model_sha) != 64:
            raise ValueError(f"{route_id}: exact revision, precision and SHA-256 required")
        profile = ProfileRoute(
            route_id=route_id, model_id=str(entry["model"]),
            artifact_id=str(entry["artifact_id"]), checkpoint_revision=revision,
            artifact_sha256=model_sha, precision=precision,
            backend=("external.llamacpp.multimodal.v1" if entry.get("mmproj_file")
                     else "external.llamacpp.text.v1"),
            expected_placement=str(entry["placement"]),
        )
        profiles.append(profile)
        provenance[route_id] = {
            "lineage_verified": item.get("lineage_verified") is True,
            "checkpoint_revision": revision,
            "model_sha256": model_sha,
            "mmproj_sha256": entry.get("mmproj_sha256"),
            "server_sha256": entry["server_sha256"],
            "model_file": entry["model_file"],
            "mmproj_file": entry.get("mmproj_file"),
            "server_bin": entry["server_bin"],
            "precision": precision,
        }
    if len({route.route_id for route in profiles}) != len(profiles):
        raise ValueError("duplicate route IDs")
    if set(metadata) != {route.route_id for route in profiles}:
        raise ValueError("lineage and native config route sets differ")
    return profiles, provenance


def _redact_image(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: ({"base64_sha256": hashlib.sha256(item.encode()).hexdigest(),
                       "encoded_length": len(item)}
                      if key == "base64" and isinstance(item, str)
                      else _redact_image(item)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_image(item) for item in value]
    return value


def _trace_complete(events: list[Mapping[str, Any]], answer: str | None,
                    route: ProfileRoute) -> bool:
    if not events or [event.get("seq") for event in events] != list(range(1, len(events) + 1)):
        return False
    if len({event.get("request_id") for event in events}) != 1 or len({event.get("epoch") for event in events}) != 1:
        return False
    kinds = [event.get("kind") for event in events]
    if kinds[0] != "user_observation" or kinds[-1] != "final":
        return False
    if any(kind in kinds for kind in ("error", "refusal", "cancelled", "approval_required")):
        return False
    if kinds.count("route") != 1 or "model_metrics" not in kinds:
        return False
    route_event = next(event for event in events if event["kind"] == "route")
    identity = route_event.get("payload", {})
    if (identity.get("route_id") != route.route_id or
        identity.get("model") != route.model_id or
        identity.get("artifact_id") != route.artifact_id or
        identity.get("backend") != route.backend or
        identity.get("actual_placement") != route.expected_placement):
        return False
    return events[-1].get("payload", {}).get("answer") == answer


class WindowsTelemetry:
    """Sample actual process-independent RAM, NVIDIA VRAM, GPU power/heat."""

    def __init__(self, expected_power: str) -> None:
        import psutil

        self._psutil = psutil
        self.expected_power = expected_power
        self._nvml = None
        self._gpu = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            if self._nvml is not None:
                self._nvml.nvmlShutdown()
            self._nvml = self._gpu = None

    def power_condition(self) -> str:
        battery = self._psutil.sensors_battery()
        return ("AC" if battery is not None and battery.power_plugged else
                "battery" if battery is not None else "unknown")

    def sample(self) -> dict[str, Any]:
        observed_power = self.power_condition()
        if observed_power != self.expected_power:
            raise RuntimeError(f"power condition changed: {self.expected_power} -> {observed_power}")
        vm = self._psutil.virtual_memory()
        sample: dict[str, Any] = {
            "ram_used_bytes": int(vm.total - vm.available),
            "ram_available_bytes": int(vm.available),
            "power_condition": observed_power,
        }
        if self._nvml is not None and self._gpu is not None:
            nvml = self._nvml
            sample["vram_used_bytes"] = int(nvml.nvmlDeviceGetMemoryInfo(self._gpu).used)
            try:
                sample["gpu_power_w"] = float(nvml.nvmlDeviceGetPowerUsage(self._gpu)) / 1000
            except Exception:
                pass
            try:
                sample["gpu_temp_c"] = float(nvml.nvmlDeviceGetTemperature(self._gpu, nvml.NVML_TEMPERATURE_GPU))
            except Exception:
                pass
        return sample

    def close(self) -> None:
        if self._nvml is not None:
            self._nvml.nvmlShutdown()
            self._nvml = self._gpu = None


class NativeProfileBridge:
    """Keep one resident Omni route while every Agent request runs serially."""

    def __init__(self, *, native_config: Mapping[str, Any], config_root: Path,
                 private_root: Path, fixture_origin: str, telemetry: WindowsTelemetry) -> None:
        self.native_config = native_config
        self.config_root = config_root
        self.private_root = private_root
        self.fixture_origin = fixture_origin
        self.telemetry = telemetry
        self.controller: Any = None
        self.route: ProfileRoute | None = None
        self._memory_path: Path | None = None
        self._preparation_counter = 0
        self._events: list[Mapping[str, Any]] | None = None
        self._emitter: Any = None
        self._lock = threading.RLock()

    def _listen(self, event: Mapping[str, Any]) -> None:
        sanitized = _redact_image(dict(event))
        with self._lock:
            if self._events is None:
                return
            self._events.append(sanitized)
            emitter = self._emitter
        if emitter is not None:
            emitter("agent_event", sanitized)
        if event.get("kind") == "text_delta" and emitter is not None:
            emitter("assistant_text_delta", event.get("payload", {}).get("text", ""))
        if event.get("kind") == "approval_required" and self.controller is not None:
            # The profiler is never a trusted person approving a write.
            self.controller.reject(str(event.get("payload", {}).get("challenge_id", "")))

    async def prepare(self, route: ProfileRoute) -> Preparation:
        if self.controller is not None:
            await asyncio.to_thread(self.close)
        from vllm_omni.edge.agent.native_app import build_controller

        entry = next(item for item in self.native_config["routes"] if item["route_id"] == route.route_id)
        config = dict(self.native_config)
        config["routes"] = [entry]
        config["qualification_file"] = None
        # A candidate profile must exercise exactly its requested bootstrap
        # route. A user's reviewed defaults belong to ordinary app execution,
        # not to this isolated profiling controller.
        config["qualification_bundles"] = []
        config["trusted_review_keys"] = {}
        config["experimental_bootstrap_route_id"] = route.route_id
        config["qualification_suite_id"] = SUITE_ID
        self._preparation_counter += 1
        prefix = route.route_id.replace("/", "_") + f"-{self._preparation_counter}"
        self._memory_path = self.private_root / (prefix + ".sqlite")
        if self._memory_path.exists():
            raise FileExistsError("benchmark memory path already exists; refusing to reset nonprivate data")
        browser_profile = self.private_root / (prefix + "-browser")
        if browser_profile.exists():
            raise FileExistsError("benchmark browser profile already exists; refusing to reuse it")
        config["memory_file"] = str(self._memory_path)
        self.config_root.mkdir(parents=True, exist_ok=True)
        config_path = self.config_root / (prefix + ".json")
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        controller, hardware = build_controller(config_path)
        controller.tools.close()
        controller.tools = ReadOnlyFixtureTools(
            self.fixture_origin, browser=ManagedEdgeBrowser(profile_dir=browser_profile),
        )
        controller.add_listener(self._listen)
        self.controller, self.route = controller, route
        backend = controller.backends[route.route_id]
        if backend.execution_plan is not None:
            raise RuntimeError("cold-start route was already resident")
        samples: list[dict[str, Any]] = []
        task = asyncio.create_task(asyncio.to_thread(backend.start))
        while not task.done():
            samples.append(self.telemetry.sample())
            await asyncio.sleep(.1)
        await task
        samples.append(self.telemetry.sample())
        plan = backend.execution_plan
        if not isinstance(plan, Mapping):
            raise RuntimeError("Omni worker did not become resident after cold load")
        if plan.get("requested_device") != route.expected_placement:
            raise RuntimeError("Omni worker placement differs from the paired route")
        return Preparation(
            cold_start_confirmed=True, artifact_id=route.artifact_id,
            actual_placement=route.expected_placement,
            details={
                "new_controller_and_worker": True,
                "memory_path": str(self._memory_path),
                "execution_plan": _redact_image(dict(plan)),
                "ram_used_bytes_sampled_peak": max(s["ram_used_bytes"] for s in samples),
                "vram_used_bytes_sampled_peak": max((s.get("vram_used_bytes", 0) for s in samples), default=0),
                "sample_count": len(samples),
                "sampled_peaks_are_lower_bounds": True,
                "placement_independently_verified": False,
                "hardware_at_load": hardware,
            },
        )

    def before_request(self, route: ProfileRoute, case: AgentCase,
                       phase: str, repetition: int) -> Mapping[str, Any]:
        if self.controller is None or self.route != route or self._memory_path is None:
            raise RuntimeError("route has not been cold-loaded")
        if self.controller.memory._path != self._memory_path:  # private benchmark DB only
            raise RuntimeError("benchmark memory path changed; refusing to delete events")
        self.telemetry.sample()  # abort the run if AC/battery condition drifted
        deleted = self.controller.memory.delete_all()
        evidence: dict[str, Any] = {
            "private_memory_reset": True, "deleted_prior_fixture_events": deleted,
            "phase": phase, "repetition": repetition,
        }
        if case.metadata.get("kind") == "memory_recall":
            label = "Cedar memory card" if case.language == "en-US" else "蓝杉记忆卡"
            event = self.controller.memory.append_event(
                session_id=f"fixture-{case.case_id}", request_id="memory-seed",
                epoch=0, sequence=0, kind="user_observation",
                payload={"text": f"{label} verification code: {case.reference}"},
                source="benchmark-fixture:memory",
            )
            evidence["seed_event_id"] = event.event_id
            evidence["seed_source"] = event.source
        elif case.metadata.get("kind") == "desktop_screen":
            # The desktop image is an actual Windows screen capture during
            # the timed request. Present a fixed card before starting the
            # timer; overlays or focus failures remain real test failures.
            lang = "en" if case.language == "en-US" else "zh"
            card = self.fixture_origin + f"/visual/{lang}"
            opened = self.controller.tools.browser.open(card)
            evidence["displayed_fixture_url"] = opened["url"]
            evidence["desktop_visibility_unverified"] = True
        return evidence

    async def run(self, route: ProfileRoute, case: AgentCase, emit: Any) -> AgentRunResult:
        if self.controller is None or self.route != route:
            raise RuntimeError("route has not been prepared")
        with self._lock:
            self._events = []
            self._emitter = emit
        try:
            answer = await asyncio.wrap_future(self.controller.submit(case.prompt))
        finally:
            with self._lock:
                events = list(self._events or [])
                self._events = None
                self._emitter = None
        identity = next((event.get("payload", {}) for event in events if event.get("kind") == "route"), {})
        backend = self.controller.backends[route.route_id]
        plan = backend.execution_plan
        terminal = [event.get("payload", {}) for event in events if event.get("kind") == "model_metrics"]
        decisions = [
            {"kind": event["kind"], "payload": event.get("payload", {})}
            for event in events if event.get("kind") in
            {"tool_proposed", "approval_required", "tool_result"}
        ]
        return AgentRunResult(
            final_answer=answer,
            complete_agent_trace=_trace_complete(events, answer, route),
            model_id=str(identity.get("model", "")),
            artifact_id=str(identity.get("artifact_id", "")),
            actual_placement=str(identity.get("actual_placement", "")),
            backend=str(identity.get("backend", "")),
            placement_evidence={
                "execution_plan": dict(plan) if isinstance(plan, Mapping) else {},
                "terminal_model_metrics": terminal,
                "independent_log_review_pending": True,
            },
            tool_decisions=decisions,
            trace_scope="agent_e2e" if events else "missing_trace",
        )

    def close(self) -> None:
        if self.controller is not None:
            self.controller.close()
            self.controller = None
            self.route = None


def _conditions(hardware: Mapping[str, Any], native_config: Mapping[str, Any]) -> ProfileConditions:
    from vllm_omni.edge.agent.native_app import _fingerprint
    from vllm_omni.edge.agent.runtime_identity import (
        imported_omni_source_sha256, loaded_runtime_sha256,
    )
    import vllm_omni

    def installed_version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "not-installed"

    return ProfileConditions(
        hardware_id=f"{hardware.get('cpu')} | {hardware.get('gpu_name')} | "
                    f"RAM {hardware.get('host_ram_total_bytes')} bytes",
        os_version=str(hardware["os"]),
        driver_versions={"nvidia": str(hardware.get("gpu_driver"))},
        runtime_versions={
            "python": platform.python_version(),
            "vllm": installed_version("vllm"),
            "vllm_omni": vllm_omni.__version__,
            "vllm_omni_installed_distribution": installed_version("vllm-omni"),
            "vllm_omni_imported_source_sha256": imported_omni_source_sha256(),
            "agent_runtime_identity_sha256": loaded_runtime_sha256(),
            "llama_server_sha256": ",".join(sorted(
                str(entry["server_sha256"]) for entry in native_config["routes"])),
            "vllm_omni_source": str(Path(__file__).resolve().parents[2]),
        },
        power_condition=str(hardware["power_condition"]),
        suite_id=SUITE_ID,
        environment_fingerprint=_fingerprint(dict(hardware)),
        notes="Fixed loopback fixtures; sampled RAM/VRAM/NVIDIA device power/heat. "
              "NVML GPU power is for the device as a whole, not model-only; system power unavailable. "
              "Code case checks only static output, with no code execution tool.",
    )


async def run_native_profile(*, config_path: Path, lineage_path: Path,
                             output_dir: Path, selected_classes: set[str] | None = None,
                             smoke: bool = False) -> Path:
    if sys.platform != "win32":
        raise RuntimeError("whole-Agent profiling requires native Windows Python")
    from vllm_omni.edge.agent.native_app import _hardware_snapshot
    from vllm_omni.edge.agent.tools import WindowsSettings

    native_config = json.loads(config_path.read_text(encoding="utf-8"))
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    routes, provenance = load_profile_routes(native_config, lineage)
    hardware = _hardware_snapshot()
    conditions = _conditions(hardware, native_config)
    speed = int(WindowsSettings().read("mouse_speed")["value"])
    work_root = output_dir.resolve() / ("native_" + uuid.uuid4().hex)
    work_root.mkdir(parents=True, exist_ok=False)
    config_root = work_root / "derived_configs"
    # SQLite locks over a WSL UNC share are unreliable for a native Windows
    # process. Keep the benchmark's isolated encrypted event store on local
    # NTFS; raw traces and summaries may still live in the requested output.
    local_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    private_root = local_data / "OmniEdgeAgent" / "profile-memory" / work_root.name
    private_root.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "scope": "whole_agent_batch1_paired_local_fixture",
        "protocol": "smoke_incomplete" if smoke else "full_20x3_and_30m",
        "automatic_qualification_export": False,
        "qualification_blockers": [
            "independent memory admission evidence pending",
            "cancellation and recovery evidence pending",
            "independent execution placement review pending",
            "open-world visual and code task quality beyond fixed fixtures pending",
        ],
        "hardware": hardware, "conditions": asdict(conditions),
        "source_config": str(config_path.resolve()),
        "source_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "lineage_manifest": str(lineage_path.resolve()),
        "lineage_sha256": hashlib.sha256(lineage_path.read_bytes()).hexdigest(),
        "artifact_provenance": provenance,
        "private_encrypted_memory_root": str(private_root),
        "results": [],
    }
    sampler = WindowsTelemetry(conditions.power_condition)
    try:
        with FixtureSite() as fixtures:
            cases = build_paired_cases(fixtures.origin, speed)
            chosen = selected_classes or set(cases)
            if chosen - set(cases):
                raise ValueError(f"unknown task classes: {sorted(chosen - set(cases))}")
            manifest["fixture_origin"] = fixtures.origin
            manifest["case_prompt_sha256"] = {
                task: {length: [case.metadata["prompt_sha256"] for case in examples]
                       for length, examples in buckets.items()}
                for task, buckets in cases.items() if task in chosen
            }
            bridge = NativeProfileBridge(
                native_config=native_config, config_root=config_root,
                private_root=private_root, fixture_origin=fixtures.origin,
                telemetry=sampler,
            )
            try:
                for task_class in sorted(chosen):
                    config = (ProfileConfig(
                        warmups_per_length=1,
                        measured_per_length=max(len(rows) for rows in cases[task_class].values()),
                        endurance_seconds=0, telemetry_interval_seconds=.2,
                    ) if smoke else ProfileConfig())
                    for route in routes:
                        result: dict[str, Any] = {"task_class": task_class,
                                                  "route_id": route.route_id,
                                                  "telemetry_interval_seconds": config.telemetry_interval_seconds}
                        entry = next(item for item in native_config["routes"]
                                     if item["route_id"] == route.route_id)
                        if task_class == "browser_vision" and not entry.get("mmproj_file"):
                            result["status"] = "blocked_missing_vision_projector"
                        else:
                            try:
                                summary = await run_profile(
                                    routes=[route], cases_by_length=cases[task_class],
                                    runner=bridge.run, evaluator=evaluate_case,
                                    conditions=conditions, output_dir=work_root / "raw",
                                    config=config, telemetry=sampler.sample,
                                    prepare=bridge.prepare,
                                    before_request=bridge.before_request,
                                )
                                profile = summary.routes[route.route_id]
                                result.update({
                                    "status": "evidence_recorded",
                                    "summary": str(summary.run_directory / "summary.json"),
                                    "raw_jsonl": str(summary.raw_jsonl),
                                    "raw_sha256": summary.raw_sha256,
                                    "protocol_compliant": profile.protocol_compliant,
                                    "correctness_pass": profile.correctness_pass,
                                    "tool_safety_pass": profile.tool_safety_pass,
                                    "lineage_verified": provenance[route.route_id]["lineage_verified"],
                                    "release_qualified": False,
                                })
                            except Exception as exc:
                                result.update(status="failed_or_blocked",
                                              error=f"{type(exc).__name__}: {exc}")
                            finally:
                                await asyncio.to_thread(bridge.close)
                        manifest["results"].append(result)
                        (work_root / "index.json").write_text(
                            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
            finally:
                await asyncio.to_thread(bridge.close)
    finally:
        sampler.close()
    return work_root / "index.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path,
                        help="Native Agent config with exact model and runtime hashes")
    parser.add_argument("--lineage", required=True, type=Path,
                        help="Checkpoint revision, precision and verified lineage per route")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task-class", action="append",
                        choices=("basic", "browser_text", "browser_vision",
                                 "windows_settings", "memory", "code_tools",
                                 "long_reasoning"))
    parser.add_argument("--smoke", action="store_true",
                        help="One measured request per length, no endurance; cannot qualify")
    args = parser.parse_args()
    index = asyncio.run(run_native_profile(
        config_path=args.config, lineage_path=args.lineage,
        output_dir=args.output_dir,
        selected_classes=set(args.task_class) if args.task_class else None,
        smoke=args.smoke,
    ))
    print(index)


if __name__ == "__main__":
    main()
