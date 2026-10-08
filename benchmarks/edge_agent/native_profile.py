# SPDX-License-Identifier: Apache-2.0
"""Run paired, batch-one whole-Agent profiles with native Windows Omni.

The full protocol is deliberately expensive. ``--smoke`` only checks that the
bridge works and always remains unqualified. No output from this script is
automatically installed as a router qualification.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import threading
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from benchmarks.edge_agent.paired_suite import (
    FixtureSite,
    ReadOnlyFixtureTools,
    build_paired_cases,
    evaluate_case,
)
from benchmarks.edge_agent.profile import (
    AgentCase,
    AgentRunResult,
    Preparation,
    ProfileConditions,
    ProfileConfig,
    ProfileRoute,
    run_profile,
)
from vllm_omni.edge.agent.consumer_trace import (
    consumer_trace_requested,
    model_step_capture_policy,
    validate_consumer_trace,
    validated_consumer_final,
)
from vllm_omni.edge.agent.placement import (
    STRATA_BACKEND,
    evidence_sha256,
    strata_route_binding,
    validate_strata_profile_plan,
    validate_strata_request_evidence,
)
from vllm_omni.edge.agent.qualification import STRUCTURED_READ_URL_SUITE_ID, SUITE_ID
from vllm_omni.edge.agent.tools import ManagedEdgeBrowser, WindowsScreen
from vllm_omni.engine.resource_ledger import ResourceUnavailable

ORDINARY_SUBMISSION_MODE = "model_selected_tools_v1"
STRUCTURED_READ_URL_MODE = "explicit_read_url_v1"
MODEL_PROMPT_IDENTITY_CAPTURE = "backend_generate_sha256_v1"


def _profile_classes(selected_classes: set[str] | None,
                     structured_read_url: bool) -> set[str] | None:
    """Keep the explicit URL workflow isolated to its fixed browser cases."""
    if not structured_read_url:
        return selected_classes
    if selected_classes is None:
        return {"browser_text"}
    if selected_classes != {"browser_text"}:
        raise ValueError("--structured-read-url requires only --task-class browser_text")
    return selected_classes


def _structured_input(case: AgentCase, fixture_origin: str) -> tuple[str, str]:
    """Accept only exact generated fixtures, never parse a free-form prompt."""
    if case.task_class != "browser_text" or case.metadata.get("kind") != "browser_text":
        raise ValueError("structured Read URL accepts browser_text fixtures only")
    canonical = [row for rows in build_paired_cases(fixture_origin, 10)["browser_text"].values()
                 for row in rows if row.case_id == case.case_id]
    if len(canonical) != 1 or case != canonical[0]:
        raise ValueError("structured Read URL case differs from canonical fixed fixture")
    return str(canonical[0].metadata["source"]), canonical[0].prompt


def _input_contract(case: AgentCase, *, structured_read_url: bool,
                    fixture_origin: str) -> dict[str, Any]:
    url, instruction = (_structured_input(case, fixture_origin) if structured_read_url
                        else (None, case.prompt))
    payload = {
        "submission_mode": (STRUCTURED_READ_URL_MODE if structured_read_url
                            else ORDINARY_SUBMISSION_MODE),
        "case_id": case.case_id,
        "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        "instruction_utf8_bytes": len(instruction.encode("utf-8")),
        "explicit_read_url": url,
    }
    payload["contract_sha256"] = hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return payload


def _foreground_fixture_window(title_marker: str) -> Mapping[str, Any]:
    """Raise one fixture Edge window and verify it is actually foreground.

    The profiler never calls this for a user's normal browser session.  A
    failed focus check aborts setup instead of recording a model vision miss.
    """
    if sys.platform != "win32":
        raise RuntimeError("desktop fixture visibility requires native Windows")
    import psutil

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    wndproc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_ssize_t)
    user32.EnumWindows.argtypes = [wndproc, ctypes.c_ssize_t]
    user32.GetWindowTextLengthW.argtypes = [ctypes.c_void_p]
    user32.GetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
    user32.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    user32.GetForegroundWindow.restype = ctypes.c_void_p
    user32.IsWindowVisible.argtypes = [ctypes.c_void_p]
    user32.IsIconic.argtypes = [ctypes.c_void_p]
    user32.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
    user32.SetForegroundWindow.argtypes = [ctypes.c_void_p]
    user32.AttachThreadInput.argtypes = [ctypes.c_ulong, ctypes.c_ulong, ctypes.c_bool]
    matches: list[tuple[int, str, int]] = []

    @wndproc
    def collect(hwnd: int, _unused: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length < len(title_marker):
            return True
        text_buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, text_buffer, length + 1)
        title = text_buffer.value
        if title_marker not in title:
            return True
        pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        try:
            executable = psutil.Process(pid.value).name().casefold()
        except (psutil.Error, OSError):
            return True
        if executable == "msedge.exe":
            matches.append((int(hwnd), title, pid.value))
        return True

    if not user32.EnumWindows(collect, 0):
        raise OSError(ctypes.get_last_error(), "cannot enumerate desktop windows")
    if len(matches) != 1:
        raise RuntimeError(f"expected one visible Edge fixture window; found {len(matches)}")
    hwnd, title, pid = matches[0]
    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    if user32.GetForegroundWindow() != hwnd:
        user32.SetForegroundWindow(hwnd)
    if user32.GetForegroundWindow() != hwnd:
        foreground = user32.GetForegroundWindow()
        foreground_thread = user32.GetWindowThreadProcessId(foreground, None)
        own_thread = kernel32.GetCurrentThreadId()
        if foreground_thread and foreground_thread != own_thread:
            attached = bool(user32.AttachThreadInput(own_thread, foreground_thread, True))
            try:
                user32.SetForegroundWindow(hwnd)
            finally:
                if attached:
                    user32.AttachThreadInput(own_thread, foreground_thread, False)
    if user32.GetForegroundWindow() != hwnd or user32.IsIconic(hwnd):
        raise RuntimeError("Edge fixture window could not be verified in foreground")
    return {"foreground_window_verified": True, "window_title": title,
            "window_pid": pid, "window_handle": hwnd}


def _verify_fixture_is_foreground(title_marker: str) -> None:
    """Fail closed if another window took focus before the actual capture."""
    import psutil

    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.restype = ctypes.c_void_p
    hwnd = user32.GetForegroundWindow()
    if not hwnd or user32.IsIconic(hwnd):
        raise RuntimeError("fixture window lost foreground before screen capture")
    length = user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    pid = ctypes.c_ulong()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    try:
        executable = psutil.Process(pid.value).name().casefold()
    except (psutil.Error, OSError) as exc:
        raise RuntimeError("cannot verify foreground fixture process") from exc
    if title_marker not in buffer.value or executable != "msedge.exe":
        raise RuntimeError("fixture window lost foreground before screen capture")


class _FixtureForegroundScreen:
    """Benchmark-only screen backend; every capture checks visible target."""

    def __init__(self) -> None:
        self.expected_title: str | None = None
        self._screen = WindowsScreen()

    def capture(self) -> Mapping[str, Any]:
        title = self.expected_title
        if title is None:
            raise RuntimeError("desktop fixture was not prepared for screen capture")
        _verify_fixture_is_foreground(title)
        result = self._screen.capture()
        _verify_fixture_is_foreground(title)
        return {**result, "fixture_foreground_verified": True}


def load_profile_routes(
    config: Mapping[str, Any], lineage: Mapping[str, Any]
) -> tuple[list[ProfileRoute], dict[str, Any]]:
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
        if entry.get("model_output_contract") is not None and entry.get("backend") != STRATA_BACKEND:
            raise ValueError("explicit output consumers require the reviewed Strata text profile binding")
        item = metadata.get(route_id)
        if not isinstance(item, dict):
            raise ValueError(f"{route_id}: lineage metadata missing")
        if entry.get("backend") == "external.strata.multimodal.v1":
            raise ValueError("Strata image functional receipts require a dedicated reviewed image profile adapter")
        if entry.get("backend") == STRATA_BACKEND:
            binding = strata_route_binding(entry)
            if (
                item.get("artifact_manifest_sha256") != binding["artifact_manifest_sha256"]
                or item.get("checkpoint_revision") != binding["checkpoint_revision"]
                or not isinstance(item.get("precision"), str)
                or not item["precision"]
            ):
                raise ValueError(f"{route_id}: Strata all-shard lineage identity differs or is missing")
            profiles.append(
                ProfileRoute(
                    route_id=route_id,
                    model_id=str(entry["model"]),
                    artifact_id=str(entry["artifact_id"]),
                    checkpoint_revision=binding["checkpoint_revision"],
                    artifact_sha256=binding["artifact_manifest_sha256"],
                    precision=item["precision"],
                    backend=STRATA_BACKEND,
                    expected_placement=str(entry["placement"]),
                    backend_identity=binding,
                )
            )
            provenance[route_id] = {
                **binding,
                "lineage_verified": item.get("lineage_verified") is True,
                "precision": item["precision"],
                "artifact_hash_scope": "complete_source_manifest_all_shards",
            }
            continue
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
            route_id=route_id,
            model_id=str(entry["model"]),
            artifact_id=str(entry["artifact_id"]),
            checkpoint_revision=revision,
            artifact_sha256=model_sha,
            precision=precision,
            backend=("external.llamacpp.multimodal.v1" if entry.get("mmproj_file") else "external.llamacpp.text.v1"),
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


def _trace_complete(
    events: list[Mapping[str, Any]],
    answer: str | None,
    route: ProfileRoute,
    placement_evidence: Mapping[str, Any] | None = None,
    *,
    trusted_task_binding: Mapping[str, Any] | None = None,
) -> bool:
    route_map = asdict(route)
    try:
        consumer_requested = consumer_trace_requested(route_map)
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError):
        return False
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
    if (
        identity.get("route_id") != route.route_id
        or identity.get("model") != route.model_id
        or identity.get("artifact_id") != route.artifact_id
        or identity.get("backend") != route.backend
    ):
        return False
    if route.backend == "external.strata.multimodal.v1":
        return False  # text trace and nullable placement cannot qualify an image chain
    if consumer_requested:
        if route.backend != STRATA_BACKEND:
            return False
        try:
            validate_consumer_trace(
                events,
                answer,
                route_map,
                placement_evidence,
                trusted_task_binding=trusted_task_binding,
            )
        except (KeyError, TypeError, ValueError, RuntimeError, AttributeError, StopIteration):
            return False
    elif route.backend == STRATA_BACKEND:
        try:
            validate_strata_request_evidence(placement_evidence, route_map, events=events)
        except (KeyError, TypeError, ValueError, RuntimeError, AttributeError, StopIteration):
            return False
    elif identity.get("actual_placement") != route.expected_placement:
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
        self._native_gpu_observer = None
        self._native_gpu_enabled = False
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
        return (
            "AC" if battery is not None and battery.power_plugged else "battery" if battery is not None else "unknown"
        )

    def bind_native_process(self, identity: Mapping[str, Any] | None) -> None:
        """Observe the backend-owned native generation; never scan by name/PID."""
        from vllm_omni.edge.windows_gpu_memory import WindowsProcessGpuObserver

        self._native_gpu_enabled = True
        self._native_gpu_observer = WindowsProcessGpuObserver(dict(identity)) if identity is not None else None

    def clear_native_process(self) -> None:
        self._native_gpu_enabled = False
        self._native_gpu_observer = None

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
        if self._native_gpu_enabled:
            from vllm_omni.edge.windows_gpu_memory import unknown_observation

            sample["native_process_gpu_memory"] = (
                self._native_gpu_observer.sample()
                if self._native_gpu_observer is not None
                else unknown_observation("backend-owned native process identity unavailable during startup")
            )
        return sample

    def close(self) -> None:
        if self._nvml is not None:
            self._nvml.nvmlShutdown()
            self._nvml = self._gpu = None


class _PromptIdentityBackend:
    """Borrow model input and retain bounded per-turn metadata, never its text.

    The policy bounds stored identity records. It does not describe whole
    profiler memory, the borrowed prompt or the backend's request buffers.
    """

    def __init__(self, backend: Any, *, capture_policy: Mapping[str, Any] | None = None) -> None:
        self._backend = backend
        self._identities: list[dict[str, Any]] = []
        self._retired_identities: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._capture_policy = dict(capture_policy) if capture_policy is not None else None
        if self._capture_policy is not None:
            policy = self._capture_policy
            if (
                set(policy)
                != {"schema", "max_model_steps", "max_record_bytes", "transient_copies", "declared_metadata_bytes"}
                or policy["schema"] != "omni-agent-model-step-identities-v1"
                or type(policy["max_model_steps"]) is not int
                or policy["max_model_steps"] < 1
                or type(policy["max_record_bytes"]) is not int
                or policy["max_record_bytes"] != 16384
                or type(policy["transient_copies"]) is not int
                or policy["transient_copies"] != 2
                or type(policy["declared_metadata_bytes"]) is not int
                or policy["declared_metadata_bytes"] != 32768 * policy["max_model_steps"]
            ):
                raise ValueError("invalid bounded model identity capture policy")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def reset(self) -> None:
        with self._lock:
            self._identities.clear()
            self._retired_identities.clear()

    def identities(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._identities]

    def snapshot_and_reset(self) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """Transfer this turn's records before any later turn or resource release."""
        with self._lock:
            identities = self._identities or self._retired_identities
            self._identities, self._retired_identities = [], []
            policy = dict(self._capture_policy) if self._capture_policy is not None else None
            return identities, policy

    async def cancel(self, request_id: str) -> None:
        if self._capture_policy is not None:
            with self._lock:
                if self._identities:
                    self._retired_identities, self._identities = self._identities, []
        await self._backend.cancel(request_id)

    def close(self) -> Any:
        if self._capture_policy is not None:
            self.reset()
        return self._backend.close()

    def _check_record_capacity(self, request_id: str) -> None:
        """Check before encoding the borrowed prompt or dispatching the model."""
        policy = self._capture_policy
        if policy is None:
            return
        if self._retired_identities:
            raise ValueError("cancelled model identity capture must be reset before dispatch")
        if len(self._identities) >= policy["max_model_steps"]:
            raise ValueError("model identity capture step budget exhausted")
        # This matches the shared trace validator. Even escaped control
        # characters leave room for every other field within a 16 KiB record.
        if (
            type(request_id) is not str
            or not request_id
            or len(request_id) > 256
            or any(0xD800 <= ord(char) <= 0xDFFF for char in request_id)
        ):
            raise ValueError("model request identity exceeds capture record budget")

    async def generate(
        self, prompt: str, *, request_id: str, max_tokens: int, image_data_url: str | None = None
    ) -> Any:
        with self._lock:
            self._check_record_capacity(request_id)
            encoded = prompt.encode("utf-8")
            identity = {
                "step": len(self._identities),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "utf8_bytes": len(encoded),
                "chars": len(prompt),
            }
            del encoded  # No encoded prompt survives backend dispatch or a yield.
            if self._capture_policy is not None:
                identity["model_request_id"] = request_id
                record = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                if len(record) > self._capture_policy["max_record_bytes"]:
                    raise ValueError("model identity exceeds capture record budget")
                del record
            self._identities.append(identity)
        chunks = self._backend.generate(
            prompt,
            request_id=request_id,
            max_tokens=max_tokens,
            image_data_url=image_data_url,
        )
        try:
            async for chunk in chunks:
                yield chunk
        finally:
            closer = getattr(chunks, "aclose", None)
            if callable(closer):
                await closer()


def _validated_consumer_final(
    event: Mapping[str, Any],
    route: ProfileRoute | None,
    events: list[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """Keep the visibility boundary shared with offline trace validation."""
    if route is None:
        return None
    return validated_consumer_final(event, asdict(route), events)


async def _observe_startup_terminal(task: asyncio.Task[Any]) -> Any:
    """Drain the owned loader despite repeated cancellation of its caller.

    Cancelling an asyncio.to_thread wrapper does not stop its native thread.
    Every await therefore shields this task; no cancellation is sent to it.
    This helper is used only after the original preparation failure is saved.
    """
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()


class NativeProfileBridge:
    """Keep one resident Omni route while every Agent request runs serially."""

    def __init__(
        self,
        *,
        native_config: Mapping[str, Any],
        config_root: Path,
        private_root: Path,
        fixture_origin: str,
        telemetry: WindowsTelemetry,
        structured_read_url: bool = False,
    ) -> None:
        self.native_config = native_config
        self.config_root = config_root
        self.private_root = private_root
        self.fixture_origin = fixture_origin
        self.telemetry = telemetry
        self.structured_read_url = structured_read_url
        self.suite_id = STRUCTURED_READ_URL_SUITE_ID if structured_read_url else SUITE_ID
        self.controller: Any = None
        self.route: ProfileRoute | None = None
        self._memory_path: Path | None = None
        self._preparation_counter = 0
        self._events: list[Mapping[str, Any]] | None = None
        self._emitter: Any = None
        self._lock = threading.RLock()
        self._fixture_screen: _FixtureForegroundScreen | None = None
        self._prompt_backend: _PromptIdentityBackend | None = None

    def _listen(self, event: Mapping[str, Any]) -> None:
        sanitized = _redact_image(dict(event))
        with self._lock:
            if self._events is None:
                return
            self._events.append(sanitized)
            emitter = self._emitter
            events = list(self._events)
        if emitter is not None:
            emitter("agent_event", sanitized)
        try:
            consumer = self.route is not None and consumer_trace_requested(asdict(self.route))
        except (KeyError, TypeError, ValueError, RuntimeError, AttributeError):
            consumer = True  # An invalid partial consumer must never expose raw chunks.
        if event.get("kind") == "text_delta" and emitter is not None and not consumer:
            emitter("assistant_text_delta", event.get("payload", {}).get("text", ""))
        elif emitter is not None and consumer:
            visible = _validated_consumer_final(event, self.route, events)
            if visible is not None:
                emitter("assistant_final", visible)
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
        config["qualification_suite_id"] = self.suite_id
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
        self._fixture_screen = _FixtureForegroundScreen()
        controller.tools = ReadOnlyFixtureTools(
            self.fixture_origin, browser=ManagedEdgeBrowser(profile_dir=browser_profile),
            screen=self._fixture_screen,
        )
        controller.add_listener(self._listen)
        self.controller, self.route = controller, route
        capture_policy = None
        if consumer_trace_requested(asdict(route)):
            capture_policy = model_step_capture_policy(
                route.backend_identity["model_output_consumer_identity"]["contract"],
                controller.limits.max_model_steps,
            )
        native_route = next((item for item in controller.routes
                             if item.route_id == route.route_id), None)
        if native_route is None:
            raise RuntimeError(f"profile route {route.route_id} is absent from the native controller")
        admission = controller.admit(native_route)
        if not admission.admitted:
            raise ResourceUnavailable(
                f"profile route {route.route_id} refused before model load: {admission.reason}"
            )
        backend = controller.backends.get(route.route_id)
        if backend is None:
            raise RuntimeError(f"profile route {route.route_id} has no admitted native backend")
        if backend.execution_plan is not None:
            raise RuntimeError("cold-start route was already resident")
        clear_gpu = getattr(self.telemetry, "clear_native_process", None)
        if callable(clear_gpu):
            clear_gpu()
        bind_gpu = getattr(self.telemetry, "bind_native_process", None)
        if route.backend == STRATA_BACKEND and callable(bind_gpu):
            bind_gpu(None)
        samples: list[dict[str, Any]] = []
        task = asyncio.create_task(asyncio.to_thread(backend.start))
        try:
            while not task.done():
                samples.append(self.telemetry.sample())
                await asyncio.sleep(.1)
            await asyncio.shield(task)
        except BaseException as failure:
            # A power/sampler failure or caller cancellation must not unwind
            # into controller.close() while backend.start() is still loading.
            try:
                await _observe_startup_terminal(task)
            except BaseException as startup_failure:
                if startup_failure is not failure:
                    raise failure from startup_failure
            raise
        samples.append(self.telemetry.sample())
        plan = backend.execution_plan
        if not isinstance(plan, Mapping):
            raise RuntimeError("Omni worker did not become resident after cold load")
        if plan.get("requested_device") != route.expected_placement:
            raise RuntimeError("Omni worker placement differs from the paired route")
        if route.backend == STRATA_BACKEND:
            validate_strata_profile_plan(plan, asdict(route))
            if callable(bind_gpu):
                bind_gpu(plan.get("gpu_observer_identity"))
                samples.append(self.telemetry.sample())
        self._prompt_backend = _PromptIdentityBackend(backend, capture_policy=capture_policy)
        controller.backends[route.route_id] = self._prompt_backend
        return Preparation(
            cold_start_confirmed=True, artifact_id=route.artifact_id,
            actual_placement=(None if route.backend == STRATA_BACKEND else route.expected_placement),
            details={
                "new_controller_and_worker": True,
                "memory_path": str(self._memory_path),
                "execution_plan": _redact_image(dict(plan)),
                "loaded_plan_sha256": evidence_sha256(dict(plan)),
                "placement_verification_scope": (
                    "loaded_configuration_and_per_request_routed_decode_experts"
                    if route.backend == STRATA_BACKEND else "reported_whole_model_placement"),
                "ram_used_bytes_sampled_peak": max(s["ram_used_bytes"] for s in samples),
                "vram_used_bytes_sampled_peak": max((s.get("vram_used_bytes", 0) for s in samples), default=0),
                "sample_count": len(samples),
                "sampled_peaks_are_lower_bounds": True,
                "native_gpu_startup_peak_covered": False,
                "native_process_gpu_load_samples": [s["native_process_gpu_memory"] for s in samples
                                                   if "native_process_gpu_memory" in s],
                "placement_independently_verified": False,
                "hardware_at_load": hardware,
            },
        )

    def before_request(self, route: ProfileRoute, case: AgentCase,
                       phase: str, repetition: int) -> Mapping[str, Any]:
        if self.controller is None or self.route != route or self._memory_path is None:
            raise RuntimeError("route has not been cold-loaded")
        input_contract = _input_contract(
            case, structured_read_url=self.structured_read_url,
            fixture_origin=self.fixture_origin,
        )
        if self.controller.memory._path != self._memory_path:  # private benchmark DB only
            raise RuntimeError("benchmark memory path changed; refusing to delete events")
        self.telemetry.sample()  # abort the run if AC/battery condition drifted
        deleted = self.controller.memory.delete_all()
        evidence: dict[str, Any] = {
            "private_memory_reset": True, "deleted_prior_fixture_events": deleted,
            "phase": phase, "repetition": repetition,
            "input_contract": input_contract,
        }
        if self._fixture_screen is not None:
            self._fixture_screen.expected_title = None
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
            browser = self.controller.tools.browser
            selected = browser.bring_to_front()
            marker = f"Omni visual fixture {urlparse(self.fixture_origin).port} {lang}"
            if (opened.get("url") != card or selected.get("url") != card or
                    opened.get("title") != marker or selected.get("title") != marker):
                raise RuntimeError("desktop fixture page identity changed before capture")
            focused = _foreground_fixture_window(marker)
            if self._fixture_screen is None:
                raise RuntimeError("desktop screen backend was not prepared")
            self._fixture_screen.expected_title = marker
            evidence["displayed_fixture_url"] = opened["url"]
            evidence["fixture_foreground_window"] = focused
            evidence["fixture_visibility_checked_before_request"] = True
        return evidence

    async def run(self, route: ProfileRoute, case: AgentCase, emit: Any) -> AgentRunResult:
        if self.controller is None or self.route != route:
            raise RuntimeError("route has not been prepared")
        url, instruction = (
            _structured_input(case, self.fixture_origin) if self.structured_read_url else (None, case.prompt)
        )
        trusted_task_binding = {
            "text": instruction,
            "read_url": url,
            "mode": "read_url" if self.structured_read_url else "ordinary",
        }
        with self._lock:
            self._events = []
            self._emitter = emit
        if self._prompt_backend is not None:
            self._prompt_backend.reset()
        try:
            if self.structured_read_url:
                answer = await asyncio.wrap_future(self.controller.submit_read_url(url, instruction))
            else:
                answer = await asyncio.wrap_future(self.controller.submit(case.prompt))
        finally:
            model_prompt_identities, capture_policy = (
                self._prompt_backend.snapshot_and_reset() if self._prompt_backend is not None else ([], None)
            )
            first_identity = (
                {key: model_prompt_identities[0][key] for key in ("step", "sha256", "utf8_bytes", "chars")}
                if model_prompt_identities
                else None
            )
            prompt_evidence = {"first": first_identity, "model_steps": len(model_prompt_identities)}
            if capture_policy is not None:
                prompt_evidence.update(
                    steps=[dict(row) for row in model_prompt_identities], policy=dict(capture_policy)
                )
            try:
                emit("model_prompt_identity", prompt_evidence)
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
            for event in events
            if event.get("kind") in {"tool_proposed", "approval_required", "tool_result"}
        ]
        placement_evidence = {
            "execution_plan": dict(plan) if isinstance(plan, Mapping) else {},
            "loaded_plan_sha256": evidence_sha256(dict(plan)) if isinstance(plan, Mapping) else None,
            "terminal_model_metrics": terminal,
            "first_model_prompt_identity": first_identity,
            "model_prompt_step_count": len(model_prompt_identities),
            "placement_verification_scope": (
                "loaded_configuration_and_per_request_routed_decode_experts"
                if route.backend == STRATA_BACKEND
                else "reported_whole_model_placement"
            ),
            "independent_log_review_pending": True,
        }
        if capture_policy is not None:
            placement_evidence.update(
                model_step_identities=model_prompt_identities, model_step_identity_policy=capture_policy
            )
        return AgentRunResult(
            final_answer=answer,
            complete_agent_trace=_trace_complete(
                events,
                answer,
                route,
                placement_evidence,
                trusted_task_binding=trusted_task_binding,
            ),
            model_id=str(identity.get("model", "")),
            artifact_id=str(identity.get("artifact_id", "")),
            actual_placement=identity.get("actual_placement"),
            backend=str(identity.get("backend", "")),
            placement_evidence=placement_evidence,
            tool_decisions=decisions,
            trace_scope="agent_e2e" if events else "missing_trace",
        )

    def close(self) -> None:
        if self._prompt_backend is not None:
            self._prompt_backend.reset()
        with self._lock:
            self._events = None
            self._emitter = None
        if self.controller is not None:
            self.controller.close()
            self.controller = None
            self.route = None
            self._prompt_backend = None


def _conditions(hardware: Mapping[str, Any], native_config: Mapping[str, Any],
                *, structured_read_url: bool = False) -> ProfileConditions:
    import vllm_omni
    from vllm_omni.edge.agent.native_app import _fingerprint
    from vllm_omni.edge.agent.runtime_identity import (
        imported_omni_source_sha256,
        loaded_runtime_sha256,
    )

    def installed_version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "not-installed"

    mode = STRUCTURED_READ_URL_MODE if structured_read_url else ORDINARY_SUBMISSION_MODE
    notes = (
        "Fixed loopback fixtures; sampled RAM/VRAM/NVIDIA device power/heat. "
        "NVML GPU power is for the device as a whole, not model-only; system power unavailable. "
        "Verified native WDDM process local/nonlocal samples stay separate; startup peak coverage is unavailable. "
        "Code case checks only static output, with no code execution tool. "
        f"Submission mode: {mode}."
    )
    if structured_read_url:
        notes += (" The canonical prompt is retained verbatim, so the URL also "
                  "appears in the separate explicit URL field.")
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
                str(entry["server_sha256"]) for entry in native_config["routes"] if "server_sha256" in entry)),
            "strata_runtime_manifest_sha256": ",".join(sorted(
                strata_route_binding(entry)["runtime_manifest_sha256"]
                for entry in native_config["routes"] if entry.get("backend") == STRATA_BACKEND)),
            "vllm_omni_source": str(Path(__file__).resolve().parents[2]),
        },
        power_condition=str(hardware["power_condition"]),
        suite_id=(STRUCTURED_READ_URL_SUITE_ID if structured_read_url else SUITE_ID),
        environment_fingerprint=_fingerprint(dict(hardware)),
        notes=notes,
    )


async def run_native_profile(*, config_path: Path, lineage_path: Path,
                             output_dir: Path, selected_classes: set[str] | None = None,
                             smoke: bool = False,
                             structured_read_url: bool = False) -> Path:
    selected_classes = _profile_classes(selected_classes, structured_read_url)
    if sys.platform != "win32":
        raise RuntimeError("whole-Agent profiling requires native Windows Python")
    from vllm_omni.edge.agent.native_app import _hardware_snapshot
    from vllm_omni.edge.agent.tools import WindowsSettings

    native_config = json.loads(config_path.read_text(encoding="utf-8"))
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    routes, provenance = load_profile_routes(native_config, lineage)
    hardware = _hardware_snapshot()
    conditions = _conditions(hardware, native_config,
                             structured_read_url=structured_read_url)
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
        "submission_mode": (STRUCTURED_READ_URL_MODE if structured_read_url
                            else ORDINARY_SUBMISSION_MODE),
        "model_prompt_identity_capture": MODEL_PROMPT_IDENTITY_CAPTURE,
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
    consumer_routes = [route.route_id for route in routes if consumer_trace_requested(asdict(route))]
    if consumer_routes:
        manifest["model_step_identity_capture"] = {
            "schema": "omni-agent-model-step-identities-v1",
            "route_ids": consumer_routes,
            "scope": "bounded per-turn metadata; each request binds the actual controller limit and policy",
            "prompt_preimages_retained": False,
            "whole_profiler_memory_bound": False,
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
            manifest["case_input_contracts"] = {
                task: {length: [_input_contract(
                    case, structured_read_url=structured_read_url,
                    fixture_origin=fixtures.origin,
                ) for case in examples] for length, examples in buckets.items()}
                for task, buckets in cases.items() if task in chosen
            }
            bridge = NativeProfileBridge(
                native_config=native_config, config_root=config_root,
                private_root=private_root, fixture_origin=fixtures.origin,
                telemetry=sampler, structured_read_url=structured_read_url,
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
                                                  "submission_mode": manifest["submission_mode"],
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
    parser.add_argument("--structured-read-url", action="store_true",
                        help="Profile only browser_text with explicit URL and canonical prompt instruction")
    args = parser.parse_args()
    index = asyncio.run(run_native_profile(
        config_path=args.config, lineage_path=args.lineage,
        output_dir=args.output_dir,
        selected_classes=set(args.task_class) if args.task_class else None,
        smoke=args.smoke,
        structured_read_url=args.structured_read_url,
    ))
    print(index)


if __name__ == "__main__":
    main()
