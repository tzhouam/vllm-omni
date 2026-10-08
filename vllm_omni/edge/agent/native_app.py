# SPDX-License-Identifier: Apache-2.0
"""Native Windows entry point for the local Omni Agent.

Run with a Windows Python that can import this checkout's vLLM-Omni and with
``-X utf8`` so the installed Torch templates are decoded consistently. A
configuration pins *local* GGUF/server hashes and an explicit physical-pool
budget. It cannot silently download, requantize, swap models or overcommit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from vllm_omni.edge.agent.admission import HostMemoryCoordinator
from vllm_omni.edge.agent.controller import AgentController, AgentLimits
from vllm_omni.edge.agent.memory import EncryptedMemoryStore
from vllm_omni.edge.agent.omni_backend import (
    OmniLlamaBackend,
    OmniLlamaConfig,
    OmniStrataBackend,
    OmniStrataConfig,
    OmniStrataImageBackend,
    OmniStrataImageConfig,
)
from vllm_omni.edge.agent.router import Qualification, Route
from vllm_omni.edge.agent.tools import WindowsToolBoundary


def _dxgi_adapter_inventory() -> list[dict[str, Any]]:
    """Read physical video-memory topology from native DXGI, fail closed.

    A route's ``integrated_gpu`` JSON flag is only a declaration. DXGI adapter
    identity and D3D12's UMA bit provide an independent admission check.
    Devices with ambiguous or unavailable topology remain unadmitted.
    """
    if sys.platform != "win32":
        return []
    try:
        import ctypes

        class GUID(ctypes.Structure):
            _fields_ = [
                ("data1", ctypes.c_uint32),
                ("data2", ctypes.c_uint16),
                ("data3", ctypes.c_uint16),
                ("data4", ctypes.c_ubyte * 8),
            ]

        class LUID(ctypes.Structure):
            _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_int32)]

        class AdapterDesc1(ctypes.Structure):
            _fields_ = [
                ("description", ctypes.c_wchar * 128),
                ("vendor_id", ctypes.c_uint32),
                ("device_id", ctypes.c_uint32),
                ("subsystem_id", ctypes.c_uint32),
                ("revision", ctypes.c_uint32),
                ("dedicated_video_memory_bytes", ctypes.c_size_t),
                ("dedicated_system_memory_bytes", ctypes.c_size_t),
                ("shared_system_memory_bytes", ctypes.c_size_t),
                ("luid", LUID),
                ("flags", ctypes.c_uint32),
            ]

        class Architecture1(ctypes.Structure):
            _fields_ = [
                ("node_index", ctypes.c_uint32),
                ("tile_based_renderer", ctypes.c_int32),
                ("uma", ctypes.c_int32),
                ("cache_coherent_uma", ctypes.c_int32),
                ("isolated_mmu", ctypes.c_int32),
            ]

        iid = GUID(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
        factory = ctypes.c_void_p()
        create = ctypes.WinDLL("dxgi").CreateDXGIFactory1
        create.argtypes = [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
        create.restype = ctypes.c_long
        if create(ctypes.byref(iid), ctypes.byref(factory)) != 0 or not factory.value:
            return []

        def com_method(pointer: ctypes.c_void_p, index: int, result: Any, *args: Any) -> Any:
            table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            return ctypes.WINFUNCTYPE(result, ctypes.c_void_p, *args)(table[index])

        adapters: list[dict[str, Any]] = []
        create_d3d12 = ctypes.WinDLL("d3d12").D3D12CreateDevice
        create_d3d12.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(GUID),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        create_d3d12.restype = ctypes.c_long
        device_iid = GUID(
            0x189819F1, 0x1DB6, 0x4B57, (ctypes.c_ubyte * 8)(0xBE, 0x54, 0x18, 0x21, 0x33, 0x9B, 0x85, 0xF7)
        )
        try:
            enum = com_method(factory, 12, ctypes.c_long, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p))
            for index in range(64):
                pointer = ctypes.c_void_p()
                result = enum(factory, index, ctypes.byref(pointer))
                if result & 0xFFFFFFFF == 0x887A0002:  # DXGI_ERROR_NOT_FOUND
                    break
                if result != 0 or not pointer.value:
                    return []
                try:
                    description = AdapterDesc1()
                    get_desc = com_method(pointer, 10, ctypes.c_long, ctypes.POINTER(AdapterDesc1))
                    if get_desc(pointer, ctypes.byref(description)) != 0:
                        return []
                    if not description.flags & 2:  # DXGI_ADAPTER_FLAG_SOFTWARE
                        device = ctypes.c_void_p()
                        uma: bool | None = None
                        if (
                            create_d3d12(pointer, 0xB000, ctypes.byref(device_iid), ctypes.byref(device)) == 0
                            and device.value
                        ):
                            try:
                                features = Architecture1()
                                check = com_method(
                                    device, 13, ctypes.c_long, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32
                                )
                                if check(device, 16, ctypes.byref(features), ctypes.sizeof(features)) == 0:
                                    uma = features.uma == 1
                            finally:
                                com_method(device, 2, ctypes.c_ulong)(device)
                        adapters.append(
                            {
                                "description": description.description.rstrip("\x00"),
                                "vendor_id": int(description.vendor_id),
                                "device_id": int(description.device_id),
                                "dedicated_video_memory_bytes": int(description.dedicated_video_memory_bytes),
                                "shared_system_memory_bytes": int(description.shared_system_memory_bytes),
                                "uma": uma,
                            }
                        )
                finally:
                    com_method(pointer, 2, ctypes.c_ulong)(pointer)
            else:
                return []  # truncated inventory cannot establish uniqueness
        finally:
            com_method(factory, 2, ctypes.c_ulong)(factory)
        return adapters
    except Exception:
        return []


def _power_condition() -> str:
    import psutil

    battery = psutil.sensors_battery()
    return "AC" if battery is not None and battery.power_plugged else "battery" if battery is not None else "unknown"


def _windows_commit_available() -> int | None:
    """Windows commit is an additional ceiling, never an extra RAM pool."""
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in (
                    "total_physical",
                    "available_physical",
                    "total_commit",
                    "available_commit",
                    "total_virtual",
                    "available_virtual",
                    "available_extended_virtual",
                )
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.available_commit)
    except (AttributeError, OSError):
        return None


def _allocated_file_bytes(path: Path) -> int:
    """Stored bytes, excluding sparse holes; unknown allocation is not capacity."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        compressed_size = kernel.GetCompressedFileSizeW
        compressed_size.argtypes = (wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD))
        compressed_size.restype = wintypes.DWORD
        high = wintypes.DWORD()
        ctypes.set_last_error(0)
        low = compressed_size(str(path), ctypes.byref(high))
        error = ctypes.get_last_error()
        if low == 0xFFFFFFFF and error:
            raise ValueError(f"SSD file allocation could not be measured: {path}")
        return (high.value << 32) | low
    blocks = getattr(path.stat(), "st_blocks", None)
    if type(blocks) is not int or blocks < 0:
        raise ValueError(f"SSD file allocation could not be measured: {path}")
    return blocks * 512


def _artifact_disk_capacity(routes: list[dict[str, Any]]) -> int | None:
    """Free space plus already present artifacts on one explicit SSD volume."""
    import shutil

    roots, files = [], set()
    for entry in routes:
        if "ssd" not in entry["memory_demands"]:
            continue
        cfg = entry.get("backend_config", entry)
        try:
            root = Path(cfg["artifact_root"]).resolve(strict=True)
        except OSError as exc:
            raise ValueError("SSD artifact root is missing or unavailable") from exc
        if not root.is_dir():
            raise ValueError("SSD artifact root must be a directory")
        roots.append(root)
        for item in cfg["artifact_manifest"]["files"]:
            path = (root / item["path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("SSD artifact path escapes its root")
            files.add(path)
        pack = cfg.get("prepared_model_dir")
        if pack:
            try:
                pack_root = Path(pack).resolve(strict=True)
            except OSError as exc:
                raise ValueError("SSD prepared pack root is missing or unavailable") from exc
            if not pack_root.is_dir():
                raise ValueError("SSD prepared pack root must be a directory")
            if pack_root.anchor.casefold() != root.anchor.casefold():
                raise ValueError("prepared pack and GGUF files need one explicit SSD volume")
            for path in pack_root.rglob("*"):
                if path.is_file():
                    resolved = path.resolve()
                    if not resolved.is_relative_to(pack_root):
                        raise ValueError("SSD prepared file escapes its root")
                    files.add(resolved)
    if not roots:
        return None
    if len({root.anchor.casefold() for root in roots}) != 1:
        raise ValueError("one SSD pool cannot represent several physical volumes")
    return shutil.disk_usage(roots[0]).free + sum(_allocated_file_bytes(path) for path in files if path.is_file())


def _hardware_snapshot(*, include_topology: bool = True) -> dict[str, Any]:
    import psutil

    vm = psutil.virtual_memory()
    vram_available = None
    vram_total = None
    gpu_name = None
    driver = None
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode("utf-8", "replace")
            gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_available = int(gpu_memory.free)
            vram_total = getattr(gpu_memory, "total", None)
            driver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver, bytes):
                driver = driver.decode("ascii", "replace")
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor(),
        "host_ram_total_bytes": int(vm.total),
        "host_ram_available_bytes": int(vm.available),
        "vram_available_bytes": vram_available,
        "vram_total_bytes": vram_total,
        "windows_commit_available_bytes": _windows_commit_available(),
        "gpu_name": gpu_name,
        "gpu_driver": driver,
        "dxgi_adapters": _dxgi_adapter_inventory() if include_topology else [],
        "power_condition": _power_condition(),
    }


def _fingerprint(hardware: dict[str, Any]) -> str:
    from vllm_omni.edge.agent.runtime_identity import loaded_runtime_sha256

    stable = {key: hardware[key] for key in ("os", "machine", "cpu", "gpu_name", "gpu_driver")}
    stable["dxgi_adapters"] = sorted(
        hardware.get("dxgi_adapters", []),
        key=lambda row: (row["vendor_id"], row["device_id"], row["description"]),
    )
    stable["loaded_agent_runtime_sha256"] = loaded_runtime_sha256()
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def _gpu_pool_refusal(entry: dict[str, Any], hardware: dict[str, Any]) -> str | None:
    """Bind dedicated VRAM to its measured GPU; charge iGPU against shared RAM."""
    if entry["placement"] == "cpu":
        return None
    gpu_pool = entry.get("gpu_memory_pool", "vram")
    if gpu_pool == "vram" and (not hardware["gpu_name"] or entry.get("expected_device_name") != hardware["gpu_name"]):
        return (
            "dedicated VRAM can only be admitted for the exact NVML GPU 0 "
            "reported by this controller; choose an explicit shared-RAM "
            "iGPU route or provide a matching NVIDIA device"
        )
    if gpu_pool == "host_ram":
        expected = entry.get("expected_dxgi_adapter_name", entry.get("expected_device_name"))
        matches = [adapter for adapter in hardware.get("dxgi_adapters", []) if adapter.get("description") == expected]
        if (
            entry.get("integrated_gpu") is not True
            or not expected
            or expected == hardware["gpu_name"]
            or len(matches) != 1
            or matches[0].get("uma") is not True
            or type(matches[0].get("shared_system_memory_bytes")) is not int
            or matches[0]["shared_system_memory_bytes"] <= 0
        ):
            return (
                "shared-RAM Vulkan route requires one independently detected "
                "DXGI/D3D12 integrated GPU adapter with UMA and positive shared "
                "RAM, distinct from NVML GPU 0"
            )
    return None


def _qualifications(config: dict[str, Any], *, config_dir: Path | None = None) -> list[Qualification]:
    # A plain Qualification JSON can assert every boolean gate without any
    # underlying requests. Native defaults accept only separately reviewed,
    # signed bundles whose raw requests and gate files are re-audited at load.
    if config.get("qualification_file") is not None:
        raise ValueError("claim-only qualification_file is unsupported; use signed qualification_bundles")
    bundles = config.get("qualification_bundles", [])
    keys = config.get("trusted_review_keys", {})
    if not isinstance(bundles, list) or not isinstance(keys, dict):
        raise ValueError("qualification_bundles and trusted_review_keys must be collections")
    if bundles and not keys:
        raise ValueError("reviewed qualifications require a trusted Ed25519 public key")
    if not bundles:
        return []
    from vllm_omni.edge.agent.qualification import load_reviewed_qualification

    def bundle_path(value: str) -> Path:
        path = Path(value)
        return config_dir / path if config_dir is not None and not path.is_absolute() else path

    profiles = [
        load_reviewed_qualification(bundle_path(path), trusted_keys=keys, native_config=config) for path in bundles
    ]
    identities = [
        (p.route_id, p.task_class, p.suite_id, p.environment_fingerprint, p.power_condition) for p in profiles
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate reviewed qualification for the same route and condition")
    return profiles


def build_controller(config_path: str | Path) -> tuple[AgentController, dict[str, Any]]:
    if sys.platform != "win32":
        raise RuntimeError("this application requires native Windows Python")
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    hardware = _hardware_snapshot()
    routes: list[Route] = []
    stage_backends: dict[str, OmniLlamaBackend] = {}
    capacity_refusals: dict[str, str] = {}
    capacities = {"host_ram": hardware["host_ram_available_bytes"]}
    if hardware["vram_available_bytes"] is not None:
        capacities["vram"] = hardware["vram_available_bytes"]
    if hardware.get("windows_commit_available_bytes") is not None:
        capacities["windows_commit"] = hardware["windows_commit_available_bytes"]
    disk_capacity = _artifact_disk_capacity(data["routes"])
    if disk_capacity is not None:
        capacities["ssd"] = disk_capacity
    from vllm_omni.edge.agent.model_output import validate_output_contract_entry

    for entry in data["routes"]:
        output_contract = validate_output_contract_entry(entry)
        route = Route(
            route_id=entry["route_id"],
            artifact_id=entry["artifact_id"],
            model=entry["model"],
            backend=entry.get(
                "backend",
                "external.llamacpp.multimodal.v1" if entry.get("mmproj_file") else "external.llamacpp.text.v1",
            ),
            modalities=frozenset(entry.get("modalities", ["text"])),
            placement=entry["placement"],
            memory_demands=entry["memory_demands"],
            requires_nvidia=bool(entry.get("requires_nvidia", False)),
            model_output_contract=output_contract,
            base_artifact_id=entry.get("base_artifact_id"),
        )
        routes.append(route)
        if route.requires_nvidia and not hardware["gpu_name"]:
            capacity_refusals[route.route_id] = "required NVIDIA GPU is not detected"
            continue
        gpu_pool = entry.get("gpu_memory_pool", "vram")
        pool_refusal = _gpu_pool_refusal(entry, hardware)
        if pool_refusal is not None:
            capacity_refusals[route.route_id] = pool_refusal
            continue
        over_budget = next(
            (
                (pool, amount, capacities.get(pool))
                for pool, amount in route.memory_demands.items()
                if pool not in capacities or amount > capacities[pool]
            ),
            None,
        )
        if over_budget is not None:
            pool, demand, ceiling = over_budget
            capacity_refusals[route.route_id] = (
                f"{pool}: declared demand {demand} bytes exceeds the native controller ceiling {ceiling} bytes"
            )
            continue
        # The engine manager lends the exact same lease to StageRuntime.
        if route.backend in {"external.strata.text.v1", "external.strata.multimodal.v1"}:
            backend_config = dict(entry["backend_config"])
            total = hardware.get("vram_total_bytes")
            if type(total) is not int or total <= 0 or backend_config.get("gpu_total_bytes", total) != total:
                capacity_refusals[route.route_id] = "Strata GPU total does not match the measured native GPU 0"
                continue
            backend_config["gpu_total_bytes"] = total
            image = route.backend == "external.strata.multimodal.v1"
            if route.modalities != frozenset({"text", "image"} if image else {"text"}):
                raise ValueError("Strata modalities differ from its explicit backend capability")
            if image and data.get("experimental_bootstrap_route_id") != route.route_id:
                raise ValueError("image Strata requires explicit experimental bootstrap opt-in")
            extra = ({"mmproj_file": entry.get("mmproj_file"), "image_capability": entry.get("image_capability")}
                     if image else {})
            backend_type = OmniStrataImageBackend if image else OmniStrataBackend
            config_type = OmniStrataImageConfig if image else OmniStrataConfig
            stage_backends[route.route_id] = backend_type(
                config_type(
                    route_id=route.route_id,
                    backend_config=backend_config,
                    placement=route.placement,
                    capacities=capacities,
                    demands=route.memory_demands,
                    context_tokens=entry.get("context_tokens", 4096),
                    max_new_tokens=entry.get("max_new_tokens", 128),
                    max_io_bytes=entry.get("max_io_bytes", 1 << 20),
                    request_timeout_s=entry.get("request_timeout_s", 900),
                    start_timeout_s=entry.get("start_timeout_s", 900),
                    **extra,
                )
            )
            if output_contract is not None:
                stage_backends[route.route_id].bind_output_contract(output_contract,
                                                                  base_artifact_id=route.base_artifact_id)
            continue
        if route.backend not in {"external.llamacpp.text.v1", "external.llamacpp.multimodal.v1"}:
            raise ValueError(f"unsupported complete Agent route backend: {route.backend}")
        backend = OmniLlamaBackend(
            OmniLlamaConfig(
                route_id=route.route_id,
                model_file=entry["model_file"],
                model_sha256=entry["model_sha256"],
                server_bin=entry["server_bin"],
                server_sha256=entry["server_sha256"],
                log_file=entry["log_file"],
                placement=route.placement,
                capacities=capacities,
                demands=route.memory_demands,
                memory_overhead_bytes=entry["memory_overhead_bytes"],
                context_tokens=entry.get("context_tokens", 4096),
                max_new_tokens=entry.get("max_new_tokens", 512),
                max_io_bytes=entry.get("max_io_bytes", 1 << 20),
                request_timeout_s=entry.get("request_timeout_s", 300),
                start_timeout_s=entry.get("start_timeout_s", 300),
                expected_device_name=entry.get("expected_device_name"),
                ggml_vk_visible_devices=entry.get("ggml_vk_visible_devices"),
                mmproj_file=entry.get("mmproj_file"),
                mmproj_sha256=entry.get("mmproj_sha256"),
                max_image_bytes=entry.get("max_image_bytes", 0),
                image_token_reserve=entry.get("image_token_reserve", 0),
                disable_repack=entry.get("disable_repack", False),
                gpu_layers=entry.get("gpu_layers"),
                cpu_moe_layers=entry.get("cpu_moe_layers", 0),
                host_mapped_expert_layers=entry.get("host_mapped_expert_layers", 0),
                cpu_weight_budget_bytes=entry.get("cpu_weight_budget_bytes"),
                gpu_weight_budget_bytes=entry.get("gpu_weight_budget_bytes"),
                vram_overhead_bytes=entry.get("vram_overhead_bytes", 0),
                gpu_memory_pool=gpu_pool,
                artifact_root=entry.get("artifact_root"),
                artifact_manifest=entry.get("artifact_manifest"),
            )
        )
        stage_backends[route.route_id] = backend

    def live_free() -> dict[str, int | None]:
        live = _hardware_snapshot(include_topology=False)
        return {
            "host_ram": live["host_ram_available_bytes"],
            "vram": live["vram_available_bytes"],
            "windows_commit": live.get("windows_commit_available_bytes"),
            "ssd": _artifact_disk_capacity(data["routes"]),
        }

    coordinator = HostMemoryCoordinator(
        routes=routes,
        backends=stage_backends,
        capacities=capacities,
        free_bytes=live_free,
        blocked_reasons=capacity_refusals,
    )

    local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    memory_path = Path(data.get("memory_file") or local_app_data / "OmniEdgeAgent" / "memory.sqlite")
    controller = AgentController(
        routes=routes,
        qualifications=_qualifications(data, config_dir=Path(config_path).resolve().parent),
        backends=coordinator.wrappers(),
        memory=EncryptedMemoryStore(memory_path),
        tools=WindowsToolBoundary(),
        admit=coordinator.admit,
        environment_fingerprint=_fingerprint(hardware),
        power_condition=hardware["power_condition"],
        power_condition_provider=_power_condition,
        qualification_suite_id=data.get("qualification_suite_id", "edge-agent-paired-v1"),
        bootstrap_route_id=data.get("experimental_bootstrap_route_id"),
        limits=AgentLimits(**data.get("limits", {})),
    )
    return controller, hardware


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--once", help="Run one headless local task, preserving Agent events")
    parser.add_argument(
        "--record", help="Optional raw JSONL event record for --once; may contain sensitive observations"
    )
    args = parser.parse_args()
    controller, hardware = build_controller(args.config)
    print(
        json.dumps({"hardware": hardware, "qualification_suite": controller.qualification_suite_id}, ensure_ascii=False)
    )
    if args.once is not None:
        recorded: list[dict[str, Any]] = []

        def show(event: dict[str, Any]) -> None:
            recorded.append(event)
            if event["kind"] in {"route", "refusal", "final", "error", "cancelled", "approval_required"}:
                print(json.dumps(event, ensure_ascii=False))

        controller.add_listener(show)
        try:
            controller.submit(args.once).result()
        finally:
            try:
                controller.close()
            finally:
                if args.record:
                    path = Path(args.record)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    config_bytes = Path(args.config).read_bytes()
                    config_copy = path.with_name(path.stem + ".config.json")
                    # Raw evidence is immutable: a later run needs a new name.
                    # Keep the exact configuration beside it so an experimental
                    # budget failure remains reproducible after adjustment.
                    with config_copy.open("xb") as snapshot:
                        snapshot.write(config_bytes)
                    with path.open("x", encoding="utf-8") as output:
                        output.write(
                            json.dumps(
                                {
                                    "record_type": "manifest",
                                    "hardware": hardware,
                                    "config": str(Path(args.config).resolve()),
                                    "config_snapshot": str(config_copy.resolve()),
                                    "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                                    "batch_size": 1,
                                    "concurrency": 1,
                                    "scope": "one_complete_agent_request_smoke",
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        for event in recorded:
                            output.write(json.dumps(event, ensure_ascii=False) + "\n")
        return
    from PySide6.QtWidgets import QApplication

    from vllm_omni.edge.agent.desktop import AgentWindow

    app = QApplication(sys.argv)
    window = AgentWindow(controller)
    window.show()
    try:
        app.exec()
    finally:
        controller.close()


if __name__ == "__main__":
    main()
