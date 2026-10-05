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
from vllm_omni.edge.agent.omni_backend import OmniLlamaBackend, OmniLlamaConfig
from vllm_omni.edge.agent.router import Qualification, Route
from vllm_omni.edge.agent.tools import WindowsToolBoundary


def _hardware_snapshot() -> dict[str, Any]:
    import psutil

    vm = psutil.virtual_memory()
    battery = psutil.sensors_battery()
    vram_available = None
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
            vram_available = int(pynvml.nvmlDeviceGetMemoryInfo(handle).free)
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
        "gpu_name": gpu_name,
        "gpu_driver": driver,
        "power_condition": (
            "AC" if battery is not None and battery.power_plugged else
            "battery" if battery is not None else "unknown"
        ),
    }


def _fingerprint(hardware: dict[str, Any]) -> str:
    from vllm_omni.edge.agent.runtime_identity import loaded_runtime_sha256

    stable = {key: hardware[key] for key in ("os", "machine", "cpu", "gpu_name", "gpu_driver")}
    stable["loaded_agent_runtime_sha256"] = loaded_runtime_sha256()
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def _gpu_pool_refusal(entry: dict[str, Any], hardware: dict[str, Any]) -> str | None:
    """Bind dedicated VRAM to its measured GPU; charge iGPU against shared RAM."""
    if entry["placement"] == "cpu":
        return None
    gpu_pool = entry.get("gpu_memory_pool", "vram")
    if gpu_pool == "vram" and (
        not hardware["gpu_name"] or
        entry.get("expected_device_name") != hardware["gpu_name"]
    ):
        return (
            "dedicated VRAM can only be admitted for the exact NVML GPU 0 "
            "reported by this controller; choose an explicit shared-RAM "
            "iGPU route or provide a matching NVIDIA device"
        )
    if gpu_pool == "host_ram" and (
        entry.get("integrated_gpu") is not True or
        entry.get("expected_device_name") == hardware["gpu_name"]
    ):
        return (
            "shared-RAM Vulkan route requires an explicitly identified "
            "integrated GPU distinct from NVML GPU 0"
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
        return (config_dir / path if config_dir is not None and not path.is_absolute()
                else path)

    profiles = [load_reviewed_qualification(bundle_path(path), trusted_keys=keys,
                                            native_config=config)
                for path in bundles]
    identities = [(p.route_id, p.task_class, p.suite_id,
                   p.environment_fingerprint, p.power_condition) for p in profiles]
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
    for entry in data["routes"]:
        route = Route(
            route_id=entry["route_id"], artifact_id=entry["artifact_id"],
            model=entry["model"], backend="external.llamacpp.multimodal.v1" if entry.get("mmproj_file") else "external.llamacpp.text.v1",
            modalities=frozenset(entry.get("modalities", ["text"])),
            placement=entry["placement"], memory_demands=entry["memory_demands"],
            requires_nvidia=bool(entry.get("requires_nvidia", False)),
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
        over_budget = next((
            (pool, amount, capacities.get(pool))
            for pool, amount in route.memory_demands.items()
            if pool not in capacities or amount > capacities[pool]
        ), None)
        if over_budget is not None:
            pool, demand, ceiling = over_budget
            capacity_refusals[route.route_id] = (
                f"{pool}: declared demand {demand} bytes exceeds the native "
                f"controller ceiling {ceiling} bytes"
            )
            continue
        # Each StageRuntime checks its stage claim against the same fixed
        # physical-pool ceilings. The coordinator below holds a second,
        # host-wide claim so two separately constructed runtimes cannot each
        # spend the entire budget. Native Windows does not inherit WSL quota.
        backend = OmniLlamaBackend(OmniLlamaConfig(
            route_id=route.route_id,
            model_file=entry["model_file"], model_sha256=entry["model_sha256"],
            server_bin=entry["server_bin"], server_sha256=entry["server_sha256"],
            log_file=entry["log_file"], placement=route.placement,
            capacities=capacities, demands=route.memory_demands,
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
        ))
        stage_backends[route.route_id] = backend

    def live_free() -> dict[str, int | None]:
        live = _hardware_snapshot()
        return {"host_ram": live["host_ram_available_bytes"],
                "vram": live["vram_available_bytes"]}

    coordinator = HostMemoryCoordinator(
        routes=routes, backends=stage_backends,
        capacities=capacities, free_bytes=live_free,
        blocked_reasons=capacity_refusals,
    )

    local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    memory_path = Path(data.get("memory_file") or local_app_data / "OmniEdgeAgent" / "memory.sqlite")
    controller = AgentController(
        routes=routes, qualifications=_qualifications(data, config_dir=Path(config_path).resolve().parent),
        backends=coordinator.wrappers(), memory=EncryptedMemoryStore(memory_path),
        tools=WindowsToolBoundary(), admit=coordinator.admit,
        environment_fingerprint=_fingerprint(hardware),
        power_condition=hardware["power_condition"],
        qualification_suite_id=data.get("qualification_suite_id", "edge-agent-paired-v1"),
        bootstrap_route_id=data.get("experimental_bootstrap_route_id"),
        limits=AgentLimits(**data.get("limits", {})),
    )
    return controller, hardware


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--once", help="Run one headless local task, preserving Agent events")
    parser.add_argument("--record", help="Optional raw JSONL event record for --once; may contain sensitive observations")
    args = parser.parse_args()
    controller, hardware = build_controller(args.config)
    print(json.dumps({"hardware": hardware, "qualification_suite": controller.qualification_suite_id}, ensure_ascii=False))
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
                        output.write(json.dumps({
                            "record_type": "manifest", "hardware": hardware,
                            "config": str(Path(args.config).resolve()),
                            "config_snapshot": str(config_copy.resolve()),
                            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                            "batch_size": 1, "concurrency": 1,
                            "scope": "one_complete_agent_request_smoke",
                        }, ensure_ascii=False) + "\n")
                        for event in recorded:
                            output.write(json.dumps(event, ensure_ascii=False) + "\n")
        return
    from vllm_omni.edge.agent.desktop import AgentWindow
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = AgentWindow(controller)
    window.show()
    try:
        app.exec()
    finally:
        controller.close()


if __name__ == "__main__":
    main()
