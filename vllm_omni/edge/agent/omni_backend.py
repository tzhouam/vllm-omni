# SPDX-License-Identifier: Apache-2.0
"""Agent-facing adapters to existing Omni/vLLM execution paths.

This module owns no model implementation. The GGUF route is an Omni graph
stage backed by the existing llama.cpp StageClient; native vLLM routes use the
existing LocalTextEngine. Both keep weights and KV with their backends.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class BackendChunk:
    text: str
    terminal: bool = False
    ttft_s: float | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OmniLlamaConfig:
    """Pinned GGUF and binary plus an explicit whole-stage ledger claim."""

    route_id: str
    model_file: str
    model_sha256: str
    server_bin: str
    server_sha256: str
    log_file: str
    placement: str
    capacities: Mapping[str, int]
    demands: Mapping[str, int]
    memory_overhead_bytes: int
    context_tokens: int = 4096
    max_new_tokens: int = 256
    max_io_bytes: int = 1 << 20
    request_timeout_s: int = 300
    start_timeout_s: int = 300
    expected_device_name: str | None = None
    ggml_vk_visible_devices: str | None = None
    mmproj_file: str | None = None
    mmproj_sha256: str | None = None
    max_image_bytes: int = 0
    image_token_reserve: int = 0
    disable_repack: bool = False
    gpu_layers: int | None = None
    cpu_moe_layers: int = 0
    host_mapped_expert_layers: int = 0
    cpu_weight_budget_bytes: int | None = None
    gpu_weight_budget_bytes: int | None = None
    vram_overhead_bytes: int = 0
    gpu_memory_pool: str = "vram"
    artifact_root: str | None = None
    artifact_manifest: Mapping[str, Any] | None = None
    launch_controls: Mapping[str, Any] | None = None
    launch_controls_runtime_manifest: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if (self.launch_controls is None) != (self.launch_controls_runtime_manifest is None):
            raise ValueError("llama.cpp launch controls and runtime manifest must be paired")
        if self.launch_controls is not None:
            from vllm_omni.engine.backends.llamacpp_controls import normalized_launch_controls_bundle

            controls, runtime = normalized_launch_controls_bundle(
                self.launch_controls, self.launch_controls_runtime_manifest
            )
            if (any(type(value) is not int or value <= 0 for value in
                    (self.context_tokens, self.max_new_tokens, self.max_io_bytes))
                    or self.context_tokens <= self.max_new_tokens
                    or type(self.disable_repack) is not bool):
                raise ValueError("controlled llama.cpp context/output/I/O/repack values must be explicit typed bounds")
            object.__setattr__(self, "launch_controls", MappingProxyType(controls))
            object.__setattr__(self, "launch_controls_runtime_manifest", runtime)
        if not self.route_id or not self.model_sha256 or not self.server_sha256:
            raise ValueError("route and pinned GGUF/binary hashes are required")
        if not self.demands or any(type(n) is not int or n < 0 for n in self.demands.values()):
            raise ValueError("positive, explicit memory demands are required")
        if any(type(n) is not int or n < 0 for n in self.capacities.values()):
            raise ValueError("explicit nonnegative memory ceilings are required")
        if any(pool not in self.capacities or demand > self.capacities[pool] for pool, demand in self.demands.items()):
            raise ValueError("stage demand exceeds a memory-pool ceiling")
        cpu_hybrid = re.fullmatch(r"cpu\+Vulkan\d+", self.placement) is not None
        host_hybrid = re.fullmatch(r"Vulkan_Host\+Vulkan\d+", self.placement) is not None
        hybrid = cpu_hybrid or host_hybrid
        if self.placement != "cpu" and not re.fullmatch(r"Vulkan\d+", self.placement) and not hybrid:
            raise ValueError("llama.cpp stage placement must be cpu, VulkanN, cpu+VulkanN, or Vulkan_Host+VulkanN")
        if self.gpu_memory_pool not in {"host_ram", "vram"}:
            raise ValueError("gpu_memory_pool must be host_ram or vram")
        if self.demands.get("host_ram", 0) <= 0:
            raise ValueError("llama.cpp stage must reserve positive host RAM")
        if self.placement == "cpu" and "vram" in self.demands:
            raise ValueError("CPU-only stage cannot reserve a GPU VRAM pool")
        if self.placement != "cpu" and (self.demands.get(self.gpu_memory_pool, 0) <= 0):
            raise ValueError("GPU route must reserve a positive physical GPU memory pool")
        if self.placement != "cpu" and self.gpu_memory_pool == "host_ram" and "vram" in self.demands:
            raise ValueError("shared-memory iGPU route cannot reserve a separate VRAM pool")
        if self.gpu_layers is not None and (type(self.gpu_layers) is not int or self.gpu_layers < 1):
            raise ValueError("gpu_layers must be a positive exact layer count")
        if type(self.cpu_moe_layers) is not int or self.cpu_moe_layers < 0:
            raise ValueError("cpu_moe_layers must be nonnegative")
        if type(self.host_mapped_expert_layers) is not int or self.host_mapped_expert_layers < 0:
            raise ValueError("host_mapped_expert_layers must be nonnegative")
        if hybrid:
            if cpu_hybrid and self.host_mapped_expert_layers:
                raise ValueError("CPU-expert route cannot claim Vulkan_Host expert layers")
            if host_hybrid and (
                self.cpu_moe_layers or self.gpu_layers is not None or self.host_mapped_expert_layers == 0
            ):
                raise ValueError(
                    "Vulkan_Host route requires an exact host-mapped expert count and no CPU or GPU layer split"
                )
            if cpu_hybrid and self.gpu_layers is None and self.cpu_moe_layers == 0:
                raise ValueError("hybrid route requires an explicit layer or CPU-expert split")
            if any(
                type(value) is not int or value <= 0
                for value in (
                    self.cpu_weight_budget_bytes,
                    self.gpu_weight_budget_bytes,
                    self.vram_overhead_bytes,
                    self.memory_overhead_bytes,
                )
            ):
                raise ValueError("hybrid route requires positive CPU/GPU weight and overhead budgets")
            if self.gpu_memory_pool == "host_ram":
                if self.demands["host_ram"] < (
                    self.cpu_weight_budget_bytes
                    + self.gpu_weight_budget_bytes
                    + self.memory_overhead_bytes
                    + self.vram_overhead_bytes
                ):
                    raise ValueError("shared-memory hybrid claim omits combined CPU/iGPU weights or overhead")
            else:
                if self.demands["host_ram"] < self.cpu_weight_budget_bytes + self.memory_overhead_bytes:
                    raise ValueError("hybrid host RAM claim omits CPU weights or loading overhead")
                if self.demands["vram"] < self.gpu_weight_budget_bytes + self.vram_overhead_bytes:
                    raise ValueError("hybrid VRAM claim omits GPU weights or workspace overhead")
        elif (
            self.gpu_layers is not None
            or self.cpu_moe_layers
            or self.host_mapped_expert_layers
            or self.cpu_weight_budget_bytes is not None
            or self.gpu_weight_budget_bytes is not None
            or self.vram_overhead_bytes
        ):
            raise ValueError("split controls are valid only for explicit hybrid routes")
        if self.mmproj_file is None and (self.mmproj_sha256 or self.max_image_bytes or self.image_token_reserve):
            raise ValueError("image bounds require a pinned vision projector")
        if self.mmproj_file is not None and not all(
            (self.mmproj_sha256, self.max_image_bytes, self.image_token_reserve)
        ):
            raise ValueError("multimodal stage needs projector hash and image bounds")


def llama_config_from_entry(entry: Mapping[str, Any], *, capacities: Mapping[str, int]) -> OmniLlamaConfig:
    """Keep native config translation shared with reproducible route identity."""
    return OmniLlamaConfig(
        route_id=entry["route_id"],
        model_file=entry["model_file"],
        model_sha256=entry["model_sha256"],
        server_bin=entry["server_bin"],
        server_sha256=entry["server_sha256"],
        log_file=entry["log_file"],
        placement=entry["placement"],
        capacities=capacities,
        demands=entry["memory_demands"],
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
        gpu_memory_pool=entry.get("gpu_memory_pool", "vram"),
        artifact_root=entry.get("artifact_root"),
        artifact_manifest=entry.get("artifact_manifest"),
        launch_controls=entry.get("launch_controls"),
        launch_controls_runtime_manifest=entry.get("launch_controls_runtime_manifest"),
    )


class OmniCompleteModelBackend:
    """Agent adapter for a complete model stage with one engine-owned lease.

    This class owns no weights, cache or tools. A concrete adapter supplies
    only pinned backend configuration; Omni owns event delivery and lifecycle.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self._shared_ledger: Any = None
        self._shared_reservation: Any = None
        self._runtime: Any = None
        self._pool: Any = None
        self._active: str | None = None
        self._active_task: asyncio.Task[Any] | None = None
        self._recovery_blocked_reason: str | None = None
        self._last_turn_request_id: str | None = None
        self.release_evidence: Mapping[str, Any] | None = None
        self.execution_plan: Mapping[str, Any] | None = None
        self.model_output_contract_identity: Mapping[str, Any] | None = None
        self._last_model_output: Mapping[str, Any] | None = None

    def bind_output_contract(self, contract: Any, *, base_artifact_id: str) -> None:
        from vllm_omni.edge.agent.model_output import AgentOutputContract, _hash

        if self._runtime is not None or self._active is not None or not isinstance(contract, AgentOutputContract):
            raise ValueError("output consumer must be explicitly bound before stage loading")
        if base_artifact_id != "strata:" + _hash(self.config.backend_config):
            raise ValueError("output consumer base identity differs from actual backend config")
        budget = self.config.backend_config["weight_tier_plan"]["budget"]
        contract.admit(max_io_bytes=self.config.max_io_bytes,
                       workspace_reserved_bytes=budget["host_workspace_bytes"])
        self.model_output_contract_identity = contract.consumer_identity(base_artifact_id)

    def last_model_output(self) -> Mapping[str, Any] | None:
        # Strings are immutable: retain one reference, never another full text copy.
        if self._last_model_output is None:
            return None
        return {**self._last_model_output, "stage_event": dict(self._last_model_output["stage_event"])}

    def bind_resource_lease(self, ledger: Any, reservation: Any) -> None:
        if self._runtime is not None:
            raise RuntimeError("cannot rebind a resident route's engine lease")
        if not ledger.owns(reservation) or dict(reservation.demands) != dict(self.config.demands):
            raise ValueError("engine lease is stale or differs from the complete route claim")
        self._shared_ledger, self._shared_reservation = ledger, reservation

    @property
    def resident(self) -> bool:
        """Whether a verified, healthy worker still owns this route's weights."""
        if self._runtime is None or self._pool is None or self.execution_plan is None:
            return False
        client = self._pool.stage_client
        check_health = getattr(client, "check_health", None)
        if not callable(check_health):
            return False
        try:
            check_health()
        except Exception:
            return False
        return True

    def _stage_backend_config(self) -> dict[str, Any]:
        raise NotImplementedError

    def _validate_loaded_plan(self, plan: Mapping[str, Any]) -> None:
        cfg = self.config
        if plan.get("requested_device") != cfg.placement:
            raise RuntimeError("backend did not verify the requested placement")
        observed = plan.get("observed_model_placement")
        if observed != cfg.placement and not (
            cfg.placement.startswith("Vulkan_Host+")
            and observed is None
            and plan.get("placement_evidence_level") == "override_selection_only"
        ):
            raise RuntimeError("backend did not verify model load placement")

    def start(self) -> None:
        if self._runtime is not None:
            if self.resident:
                return
            if not self.close():
                raise RuntimeError(
                    self._recovery_blocked_reason or "previous Omni worker has not released its memory reservation"
                )
        # The heavy native-Windows vLLM/Omni import is intentionally deferred.
        from vllm_omni.config.stage_config import (
            DeployConfig,
            PipelineConfig,
            StageDeployConfig,
            StageExecutionType,
            StagePipelineConfig,
            merge_pipeline_deploy,
        )
        from vllm_omni.engine.stage_runtime import StageRuntime

        cfg = self.config
        backend = self._stage_backend_config()
        pipeline = PipelineConfig(
            model_type=f"agent_{cfg.route_id}",
            stages=(
                StagePipelineConfig(
                    stage_id=0,
                    model_stage="agent",
                    execution_type=StageExecutionType.GRAPH,
                    final_output=True,
                    final_output_type="text",
                ),
            ),
        )
        deploy = DeployConfig(
            async_chunk=False,
            stages=[
                StageDeployConfig(
                    stage_id=0,
                    backend=backend,
                    resource_budget={"capacities": dict(cfg.capacities), "demands": dict(cfg.demands)},
                )
            ],
        )
        configs = [stage.to_omegaconf() for stage in merge_pipeline_deploy(pipeline, deploy)]
        runtime = StageRuntime(
            configs,
            f"agent-{cfg.route_id}",
            "",
            stage_init_timeout=cfg.start_timeout_s,
            async_chunk=False,
            resource_ledger=self._shared_ledger,
            resource_reservations=(
                {(0, 0): self._shared_reservation} if self._shared_reservation is not None else None
            ),
        )
        try:
            runtime.initialize()
            pool = runtime.stage_pools[0]
            plan = pool.stage_client.execution_plan
            self._validate_loaded_plan(plan)
        except BaseException:
            # Retain a failed loader until its internal ledger proves that
            # every claim drained. The app's outer ledger must not release a
            # claim merely because initialize raised.
            self._runtime = runtime
            self._pool = None
            self.execution_plan = None
            self.close()
            raise
        self._runtime = runtime
        self._pool = pool
        self.execution_plan = plan
        self._recovery_blocked_reason = None

    async def generate(
        self,
        prompt: str,
        *,
        request_id: str,
        max_tokens: int,
        image_data_url: str | None = None,
    ) -> AsyncIterator[BackendChunk]:
        if self._pool is None:
            raise RuntimeError("Omni stage has not started")
        if self._active is not None:
            raise RuntimeError("batch=1 stage already has an active request")
        if max_tokens > self.config.max_new_tokens:
            raise ValueError("requested generation exceeds the admitted token bound")
        if image_data_url is not None and self.config.mmproj_file is None:
            raise ValueError("this text route has no vision projector")
        self._last_model_output = None
        self._active = request_id
        self._last_turn_request_id = request_id.rsplit("-step-", 1)[0]
        self.release_evidence = None
        self._active_task = asyncio.current_task()
        state = type("_GraphState", (), {"sampling_params_list": [None]})()
        payload: dict[str, Any] = {"text": prompt, "max_tokens": max_tokens, "stream_agent": True}
        if image_data_url is not None:
            payload["image_data_url"] = image_data_url
        started = time.perf_counter()
        output = None
        failed = False
        try:
            await self._pool.submit_initial(request_id, state, payload)
            client = self._pool.stage_client
            if not callable(getattr(client, "receive_agent_delta", None)):
                raise RuntimeError("loaded complete-model StageClient lacks credited Agent streaming")
            emitted: list[str] = []
            first_delta_s: float | None = None
            deadline = time.monotonic() + self.config.request_timeout_s
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Omni Agent stream exceeded its deadline")
                delta = await asyncio.wait_for(client.receive_agent_delta(request_id), remaining)
                if delta is None:
                    break
                text, emitted_at = delta
                emitted.append(text)
                if first_delta_s is None:
                    first_delta_s = emitted_at - started
                yield BackendChunk(text=text, ttft_s=first_delta_s)
            while output is None:
                output = self._pool.poll_graph_output(0)
                if output is not None:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("whole Omni Agent model request exceeded its deadline")
                await asyncio.sleep(0.01)
            if output.request_id != request_id:
                raise RuntimeError("Omni stage returned another request's output")
            if output.error:
                raise RuntimeError(str(output.error))
            if not output.outputs or not isinstance(output.outputs[0].text, str):
                raise RuntimeError("Omni stage returned no text")
            if output.outputs[0].text != "".join(emitted):
                raise RuntimeError("Omni SSE deltas differ from the backend's final text")
            stage_event = output.custom_output.get("stage_event")
            if not stage_event or not stage_event.get("terminal"):
                raise RuntimeError("Omni stage did not return a terminal event")
            if stage_event.get("request_id") != request_id:
                raise RuntimeError("Omni terminal event belongs to another request")
            extra = self._validate_terminal_output(output, image_data_url=image_data_url)
            if self.model_output_contract_identity is not None:
                import hashlib
                self._last_model_output = {
                    "request_id": request_id,
                    "text": output.outputs[0].text,
                    "raw_output_sha256": hashlib.sha256(output.outputs[0].text.encode()).hexdigest(),
                    "finish_reason": output.outputs[0].finish_reason,
                    "stage_event": dict(stage_event),
                }
            yield BackendChunk(
                text="",
                terminal=True,
                ttft_s=first_delta_s,
                metrics={
                    **({"finish_reason": output.outputs[0].finish_reason,
                        "raw_model_output_sha256": self._last_model_output["raw_output_sha256"],
                        "raw_model_ttft_s": first_delta_s}
                       if self.model_output_contract_identity is not None else {}),
                    "whole_model_wall_s": time.perf_counter() - started,
                    "backend_metrics": output.metrics,
                    "stage_event": stage_event,
                    "ttft_note": "first visible SSE delta after Omni submission",
                    **extra,
                },
            )
        except BaseException:
            failed = True
            await self._pool.abort_requests([request_id])
            raise
        finally:
            try:
                if output is not None:
                    output.release_stage_buffers()
                    await asyncio.sleep(0)
            finally:
                self._active = None
                self._active_task = None
                if failed:
                    # Abort terminates the llama-server. A cached StageRuntime
                    # must never be reused after its client has closed.
                    self.close()

    def _validate_terminal_output(self, output: Any, *, image_data_url: str | None) -> dict[str, Any]:
        return {}

    async def cancel(self, request_id: str) -> None:
        if self._pool is not None and self._active == request_id:
            task = self._active_task
            if task is not None and task is not asyncio.current_task():
                # The waiting generator owns the abort and worker reset. Merely
                # aborting the StageClient would leave receive_agent_delta
                # blocked on a queue whose producer has exited.
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                return
            await self._pool.abort_requests([request_id])

    def request_state_released(self, request_id: str) -> bool:
        """Drain the worker so a cancelled request cannot retain server KV state."""
        if self._last_turn_request_id != request_id:
            return False
        if self._active == request_id or self._active_task is not None:
            return False
        if self._recovery_blocked_reason is not None:
            return False
        # A model-step cancellation already closes its worker in generate().
        # A cancellation during a subsequent tool call can leave llama-server
        # healthy, but health does not prove its slot/KV state was discarded.
        # Shutdown plus an empty StageRuntime ledger is the release proof.
        if not self.close():
            return False
        proof = self.release_evidence
        return bool(
            proof
            and proof.get("request_id") == request_id
            and proof.get("worker_exit_confirmed") is True
            and proof.get("stage_ledger_empty") is True
        )

    def close(self) -> bool:
        """Return true only after the old stage's memory claim is drained."""
        self._last_model_output = None
        runtime = self._runtime
        if runtime is None:
            self._pool = None
            self.execution_plan = None
            return True
        stage_client = self._pool.stage_client if self._pool is not None else None
        worker_process = getattr(stage_client, "_proc", None)
        worker_pid = getattr(worker_process, "pid", None)
        self.release_evidence = None
        if self._active is not None:
            self._recovery_blocked_reason = "cannot unload an active Omni request"
            return False
        try:
            runtime.shutdown()
        except Exception as exc:
            self._recovery_blocked_reason = f"Omni worker shutdown failed: {exc}"
            self.execution_plan = None
            return False
        ledger = getattr(runtime, "resource_ledger", None)
        if ledger is None:
            self._recovery_blocked_reason = "Omni worker has no release-verifiable memory ledger"
            self.execution_plan = None
            return False
        try:
            snapshot = ledger.snapshot()
        except Exception as exc:
            self._recovery_blocked_reason = f"Omni worker release state could not be verified: {exc}"
            self.execution_plan = None
            return False
        owned = getattr(runtime, "_resource_reservations", {}).values()
        relevant = {token.owner for token in owned}
        if not relevant and self._shared_reservation is not None:
            relevant = {self._shared_reservation.owner}
        remaining = set(snapshot["owners"]) & relevant if relevant else set(snapshot["owners"])
        quarantined = set(snapshot["quarantined"]) & relevant if relevant else set(snapshot["quarantined"])
        if remaining or quarantined:
            self._recovery_blocked_reason = (
                "Omni worker memory remains reserved or quarantined after shutdown: "
                f"{snapshot['owners']!r}, {snapshot['quarantined']!r}"
            )
            self.execution_plan = None
            return False
        worker_exit_code = worker_process.poll() if worker_process is not None else None
        if worker_process is not None and worker_exit_code is None:
            self._recovery_blocked_reason = "Omni worker process remains alive after shutdown"
            self.execution_plan = None
            return False
        resource_owner = self._shared_reservation.owner if self._shared_reservation is not None else None
        shared_ledger = ledger is self._shared_ledger
        self._runtime = None
        self._pool = None
        self._shared_ledger = None
        self._shared_reservation = None
        self.execution_plan = None
        self._recovery_blocked_reason = None
        if self._last_turn_request_id is not None and type(worker_pid) is int and type(worker_exit_code) is int:
            self.release_evidence = {
                "request_id": self._last_turn_request_id,
                "release_mode": "worker_shutdown",
                "worker_pid_before": worker_pid,
                "worker_exit_code": worker_exit_code,
                "worker_exit_confirmed": True,
                "stage_ledger_empty": True,
                "schema": "omni-resource-release-v2",
                "resource_claim_released": True,
                "shared_ledger": shared_ledger,
                "resource_owner": resource_owner,
            }
        return True


class OmniLlamaBackend(OmniCompleteModelBackend):
    """Pinned llama.cpp stage using the shared complete-model lifecycle."""

    def _validate_loaded_plan(self, plan: Mapping[str, Any]) -> None:
        super()._validate_loaded_plan(plan)
        if self.config.launch_controls is not None:
            from vllm_omni.edge.agent.llamacpp_route import (
                llamacpp_config_binding,
                validate_llamacpp_launch_plan,
            )

            validate_llamacpp_launch_plan(plan, llamacpp_config_binding(self.config))

    def _stage_backend_config(self) -> dict[str, Any]:
        cfg = self.config
        multimodal = cfg.mmproj_file is not None
        backend: dict[str, Any] = {
            "name": "external.llamacpp.multimodal.v1" if multimodal else "external.llamacpp.text.v1",
            "model_file": cfg.model_file,
            "model_sha256": cfg.model_sha256,
            "server_bin": cfg.server_bin,
            "server_sha256": cfg.server_sha256,
            "log_file": cfg.log_file,
            "device": cfg.placement,
            "memory_pool": "host_ram" if cfg.placement == "cpu" else cfg.gpu_memory_pool,
            "memory_overhead_bytes": cfg.memory_overhead_bytes,
            "context_tokens": cfg.context_tokens,
            "max_new_tokens": cfg.max_new_tokens,
            "max_io_bytes": cfg.max_io_bytes,
            "request_timeout_s": cfg.request_timeout_s,
            "start_timeout_s": cfg.start_timeout_s,
            "expected_device_name": cfg.expected_device_name,
            "ggml_vk_visible_devices": cfg.ggml_vk_visible_devices,
            "disable_repack": cfg.disable_repack,
            "gpu_layers": cfg.gpu_layers,
            "cpu_moe_layers": cfg.cpu_moe_layers,
            "host_mapped_expert_layers": cfg.host_mapped_expert_layers,
            "cpu_weight_budget_bytes": cfg.cpu_weight_budget_bytes,
            "gpu_weight_budget_bytes": cfg.gpu_weight_budget_bytes,
            "vram_overhead_bytes": cfg.vram_overhead_bytes,
        }
        if cfg.launch_controls is not None:
            from vllm_omni.engine.backends.llamacpp_controls import normalized_launch_controls_bundle

            controls, runtime = normalized_launch_controls_bundle(
                cfg.launch_controls, cfg.launch_controls_runtime_manifest
            )
            backend.update(launch_controls=controls, launch_controls_runtime_manifest=runtime)
        if cfg.artifact_manifest is not None:
            backend.update(artifact_root=cfg.artifact_root, artifact_manifest=dict(cfg.artifact_manifest))
        if multimodal:
            backend.update(
                mmproj_file=cfg.mmproj_file,
                mmproj_sha256=cfg.mmproj_sha256,
                max_image_bytes=cfg.max_image_bytes,
                image_token_reserve=cfg.image_token_reserve,
            )
        return backend


@dataclass(frozen=True)
class OmniStrataConfig:
    """Strata runtime/pack manifest plus the same whole-stage engine lease."""

    route_id: str
    backend_config: Mapping[str, Any]
    placement: str
    capacities: Mapping[str, int]
    demands: Mapping[str, int]
    context_tokens: int = 4096
    max_new_tokens: int = 128
    max_io_bytes: int = 1 << 20
    request_timeout_s: int = 900
    start_timeout_s: int = 900
    mmproj_file: str | None = None

    def __post_init__(self) -> None:
        if not self.route_id or not re.fullmatch(r"cpu\+cuda:[0-9]+", self.placement):
            raise ValueError("Strata route needs an explicit GPU index and route identity")
        gpu = int(self.placement.rsplit(":", 1)[1])
        if gpu != 0 or self.backend_config.get("gpu_index", 0) != gpu:
            raise ValueError("native Agent Strata route must bind the measured GPU 0 pool")
        if self.mmproj_file is not None:
            raise ValueError("this Strata adapter has no qualified image stage")
        if self.backend_config.get("image_route") is not None:
            raise ValueError("text Strata adapter cannot carry an image route")
        if self.backend_config.get("name") != "external.strata.text.v1":
            raise ValueError("Strata route must select its complete-model StageClient")
        if any(
            pool not in self.capacities or type(amount) is not int or amount < 0 or amount > self.capacities[pool]
            for pool, amount in self.demands.items()
        ):
            raise ValueError("Strata demand exceeds a physical-pool ceiling")
        if self.demands.get("host_ram", 0) <= 0 or self.demands.get("vram", 0) <= 0:
            raise ValueError("Strata requires explicit host RAM and dedicated VRAM claims")
        protected = {
            "context_tokens": self.context_tokens,
            "max_new_tokens": self.max_new_tokens,
            "max_io_bytes": self.max_io_bytes,
            "gpu_pool": "vram",
        }
        for key, value in protected.items():
            if key in self.backend_config and self.backend_config[key] != value:
                raise ValueError(f"Strata stage and Agent {key} declarations differ")


class OmniStrataBackend(OmniCompleteModelBackend):
    """Reuse Omni's streaming/lifecycle adapter; weights stay inside Strata.

    Agent loading still requires verified placement. A functional experimental
    harness can exercise the backend before it is eligible for Agent routing.
    """

    def __init__(self, config: OmniStrataConfig) -> None:
        super().__init__(config)

    def _validate_loaded_plan(self, plan: dict[str, Any]) -> None:
        # A verified native GPU/CPU-pool configuration permits explicit
        # experimental execution. It is distinct from the per-request neural
        # counters and never creates a route qualification or default.
        from vllm_omni.engine.backends.strata import validate_strata_load_plan

        validate_strata_load_plan(plan, self.config.placement)

    def _stage_backend_config(self) -> dict[str, Any]:
        cfg = self.config
        return {
            **cfg.backend_config,
            "context_tokens": cfg.context_tokens,
            "max_new_tokens": cfg.max_new_tokens,
            "max_io_bytes": cfg.max_io_bytes,
            "request_timeout_s": cfg.request_timeout_s,
            "start_timeout_s": cfg.start_timeout_s,
            "gpu_pool": "vram",
        }


@dataclass(frozen=True)
class OmniStrataImageConfig(OmniStrataConfig):
    """Explicit CPU image capability; never re-label a text config."""

    image_capability: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from vllm_omni.edge.agent.strata_image_evidence import IMAGE_BACKEND, image_capability_from_config

        if self.backend_config.get("name") != IMAGE_BACKEND:
            raise ValueError("image config requires the distinct image stage")
        # Reuse the text config's physical-pool and token validation, using a
        # detached text-only view. This never reaches StageRuntime or loading.
        text_view = dict(self.backend_config)
        text_view.pop("image_route", None)
        text_view["name"] = "external.strata.text.v1"
        OmniStrataConfig(self.route_id, text_view, self.placement, self.capacities, self.demands,
                        self.context_tokens, self.max_new_tokens, self.max_io_bytes,
                        self.request_timeout_s, self.start_timeout_s)
        capability = image_capability_from_config(self.backend_config)
        if self.image_capability != capability or self.mmproj_file != capability["projector_file"]:
            raise ValueError("Agent projector capability differs from its bound image stage")


class OmniStrataImageBackend(OmniStrataBackend):
    """Same Omni lease, streaming and cancellation, with a strict image gate."""

    def __init__(self, config: OmniStrataImageConfig) -> None:
        if not isinstance(config, OmniStrataImageConfig):
            raise TypeError("image backend requires an explicit image configuration")
        super().__init__(config)
        self._image_seen_states: set[tuple[Any, ...]] = set()

    def _validate_loaded_plan(self, plan: dict[str, Any]) -> None:
        from vllm_omni.edge.agent.strata_image_evidence import validate_image_load_plan_for_config

        validate_image_load_plan_for_config(plan, self.config.backend_config, self.config.placement)
        if plan.get("reserved_bytes") != dict(self.config.demands):
            raise RuntimeError("image stage does not hold the Agent's exact shared resource lease")

    def _validate_terminal_output(self, output: Any, *, image_data_url: str | None) -> dict[str, Any]:
        from vllm_omni.edge.agent.strata_image_evidence import image_input_identity, validate_image_terminal

        identity = (image_input_identity(image_data_url, self.config.image_capability)
                    if image_data_url is not None else None)
        proof = validate_image_terminal(self.execution_plan, output.metrics, output.custom_output["stage_event"],
                                        request_id=output.request_id, input_identity=identity,
                                        seen_states=self._image_seen_states)
        return {"image_chain_evidence": proof}


class OmniVllmTextBackend:
    """Adapter to the existing vLLM local plan and its credited token stream."""

    def __init__(self, execution_plan: Any) -> None:
        self.execution_plan = execution_plan
        self._engine: Any = None
        self._session: Any = None
        self._active: str | None = None

    async def start(self) -> None:
        from vllm_omni.edge.local.engine import LocalTextEngine

        self._engine = LocalTextEngine(self.execution_plan)
        await self._engine.start()
        self._session = self._engine.open_session()

    async def generate(
        self,
        prompt: str,
        *,
        request_id: str,
        max_tokens: int,
        image_data_url: str | None = None,
    ) -> AsyncIterator[BackendChunk]:
        if image_data_url is not None:
            raise ValueError("this vLLM text plan does not admit images")
        if self._engine is None or self._session is None:
            raise RuntimeError("vLLM route has not started")
        self._active = request_id
        _, stream = await self._engine.submit(
            self._session,
            prompt,
            request_id=request_id,
            max_tokens=max_tokens,
        )
        try:
            while True:
                event = await stream.get()
                if event is None:
                    break
                try:
                    if event.error:
                        raise RuntimeError(event.error)
                    if event.kind == "token":
                        yield BackendChunk(text=event.payload["text"])
                    elif event.kind == "done":
                        yield BackendChunk(text="", terminal=True, metrics=self._engine.records[request_id].to_dict())
                finally:
                    await stream.acknowledge(event)
        except BaseException:
            await self._engine.cancel(request_id)
            raise
        finally:
            self._active = None

    async def cancel(self, request_id: str) -> None:
        if self._engine is not None and self._active == request_id:
            await self._engine.cancel(request_id)

    def report_placement(self) -> str | None:
        if self._engine is None:
            return None
        report = self._engine.report_placement()
        if not isinstance(report, dict):
            return None
        planned = report.get("planned_device")
        configured = str(report.get("configured", {}).get("device", ""))
        if planned == "cpu" and configured.startswith("cpu"):
            return planned
        if isinstance(planned, str) and planned.startswith("cuda") and configured.startswith("cuda"):
            return planned
        return None

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.close()
            self._engine = None
