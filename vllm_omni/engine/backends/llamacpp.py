# SPDX-License-Identifier: Apache-2.0
"""Bounded llama.cpp text and optional image stage owned by StageRuntime.

The llama-server subprocess owns its weights and KV. Omni owns the admission
reservation, request identity, output acknowledgement and process lifetime.
This v1 backend has one slot and streams bounded text deltas within a complete request.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import dataclasses
import hashlib
import io
import json
import math
import os
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable

from omni_stage_contracts import StageEvent, StageRequest
from vllm.outputs import CompletionOutput

from vllm_omni.engine.resource_ledger import ResourceUnavailable
from vllm_omni.engine.stage_client import StageClientBase
from vllm_omni.outputs import OmniRequestOutput

_LAYER_ASSIGNMENT = re.compile(r"load_tensors: layer\s+\d+ assigned to device ([^,\s]+)")
_OFFLOADED_LAYERS = re.compile(r"load_tensors: offloaded (\d+)/(\d+) layers to GPU")
_MODEL_BUFFER = re.compile(r"load_tensors:\s+(Vulkan\d+) model buffer size")
_ANY_MODEL_BUFFER = re.compile(r"load_tensors:\s+(\S+) model buffer size")
_CLIP_BACKEND = re.compile(r"clip_ctx: CLIP using (\S+) backend")
_MODEL_BUFFER_SIZE = re.compile(r"load_tensors:\s+(\S+) model buffer size\s*=\s*([0-9]+(?:\.[0-9]+)?) MiB")
_EXPERT_BUFFER_OVERRIDE = re.compile(
    r"tensor blk\.(\d+)\.(ffn_(?:up|down|gate|gate_up)_(?:ch)?exps\.weight) "
    r"\([^)]*\) buffer type overridden to (CPU|Vulkan_Host|Vulkan\d{1,3})(?:\s|$)"
)
_LOG_LINE_LIMIT = 4096
_SANITIZED_LOG_LIMIT = 8 << 20
_SAFE_DEVICE = r"(?:CPU|Vulkan_Host|Vulkan\d{1,3})"
_LOG_DEVICE = re.compile(r"using device (Vulkan\d{1,3}) \([^\r\n]*\)")
_LOG_LAYER = re.compile(
    rf"load_tensors: layer\s+(\d{{1,4}}) assigned to device ({_SAFE_DEVICE})(?:[,\s]|$)"
)
_LOG_OFFLOAD = re.compile(r"load_tensors: offloaded (\d{1,4})/(\d{1,4}) layers to GPU")
_LOG_BUFFER = re.compile(
    r"load_tensors:\s+(CPU(?:_[A-Za-z0-9_]{1,24})?|Vulkan_Host|Vulkan\d{1,3}) "
    r"model buffer size\s*=\s*(\d{1,9}(?:\.\d{1,2})?) MiB"
)
_LOG_CAPACITY_BUFFER = re.compile(
    rf"({_SAFE_DEVICE}) (KV|compute) buffer size\s*=\s*(\d{{1,9}}(?:\.\d{{1,2}})?) MiB"
)
_LOG_EXPERT = re.compile(
    rf"tensor blk\.(\d{{1,4}})\.(ffn_(?:up|down|gate|gate_up)_(?:ch)?exps)\.weight "
    rf"\([^)]*\) buffer type overridden to ({_SAFE_DEVICE})(?:\s|$)"
)
_LOG_CLIP = re.compile(rf"clip_ctx: CLIP using ({_SAFE_DEVICE}) backend")
_LOG_MODEL_LOADED = re.compile(
    r"\d{1,4}(?:\.\d{1,4}){3}\s+I\s+srv\s+llama_server:\s+model loaded"
)
_LOG_LISTENING = re.compile(r"\bllama_server: listening on http://127\.0\.0\.1:(\d{1,5})(?:\s|$)")
_LOG_ERROR_SEVERITY = re.compile(r"^(?:\d+\.){3}\d+\s+[EW]\s+")
_LOG_ERROR_SOURCE = re.compile(r"^(?:ggml\w*|llama\w*|load_tensors|error|fatal|failed)\s*:", re.I)


def _sanitize_llama_log_line(line: str, *, expected_device_name: str | None) -> str | None:
    """Allow only canonical placement fields or content-free diagnostic codes.

    llama-server verbosity 5 echoes HTTP request and response bodies, including
    Agent prompts and SSE tokens. Never copy a raw server line to persistent
    storage, even when it looks like a warning or an error.
    """
    if expected_device_name:
        match = _LOG_DEVICE.search(line)
        if match and f"using device {match[1]} ({expected_device_name})" in line:
            return f"llama_model_load: using device {match[1]} ({expected_device_name})"
    match = _LOG_LAYER.search(line)
    if match:
        return f"load_tensors: layer {match[1]} assigned to device {match[2]}"
    match = _LOG_OFFLOAD.search(line)
    if match:
        return f"load_tensors: offloaded {match[1]}/{match[2]} layers to GPU"
    match = _LOG_BUFFER.search(line)
    if match:
        # Only CPU-vs-GPU matters for admission. A backend-specific CPU suffix
        # is not needed for placement and may contain uncontrolled text.
        device = "CPU" if match[1].startswith("CPU") else match[1]
        return f"load_tensors: {device} model buffer size = {match[2]} MiB"
    match = _LOG_CAPACITY_BUFFER.search(line)
    if match:
        return f"llama.cpp capacity: {match[1]} {match[2]} buffer size = {match[3]} MiB"
    match = _LOG_EXPERT.search(line)
    if match:
        return (f"tensor blk.{match[1]}.{match[2]}.weight (redacted) "
                f"buffer type overridden to {match[3]}")
    if ("buffer type overridden to" in line and
            re.search(r"tensor blk\.\d{1,4}\.ffn_[A-Za-z0-9_]{1,48}exps", line)):
        # An unexpected expert override must remain visible to the verifier
        # without copying untrusted tensor metadata into the persistent log.
        return "llama.cpp diagnostic: unexpected_expert_override"
    match = _LOG_CLIP.search(line)
    if match:
        return f"clip_ctx: CLIP using {match[1]} backend"
    if _LOG_MODEL_LOADED.fullmatch(line.strip()):
        # A content-free, anchored server startup marker is needed by the
        # independent placement gate. Request/response debug lines cannot
        # masquerade as this exact logger record.
        return "llama.cpp status: model loaded"
    if "warning: no usable GPU found" in line:
        return "warning: no usable GPU found"

    # Debug-level request/response dumps can contain words such as "error" and
    # "out of memory" as ordinary user content. Require a backend diagnostic
    # severity or a known error source before classifying those words.
    if not (_LOG_ERROR_SEVERITY.match(line) or _LOG_ERROR_SOURCE.match(line)):
        return None
    lowered = line.lower()
    if "out of memory" in lowered or "not enough memory" in lowered:
        return "llama.cpp diagnostic: out_of_memory"
    if "failed to allocate" in lowered or "allocation failed" in lowered:
        return "llama.cpp diagnostic: allocation_failed"
    if "device lost" in lowered:
        return "llama.cpp diagnostic: device_lost"
    if "failed to load model" in lowered or "error loading model" in lowered:
        return "llama.cpp diagnostic: model_load_failed"
    if "context" in lowered and ("exceed" in lowered or "too large" in lowered):
        return "llama.cpp diagnostic: context_capacity"
    if re.search(r"\b(?:error|fatal|failed)\b", lowered):
        return "llama.cpp diagnostic: backend_error"
    return None


class _FilteredLlamaLog:
    """Drain server stdout without retaining raw content in memory or on disk."""

    def __init__(self, pipe, path: Path, *, expected_device_name: str | None,
                 port: int) -> None:
        self._pipe = pipe
        self._file = path.open("w", encoding="utf-8", buffering=1)
        self._expected_device_name = expected_device_name
        self._port = port
        self._condition = threading.Condition()
        self._startup_lines: list[str] = []
        self._bytes_written = 0
        self._ready = False
        self._eof = False
        self._error = False
        self._diagnostics: set[str] = set()
        self._thread = threading.Thread(target=self._drain, name="llamacpp-log-filter", daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        skipping_long_line = False
        try:
            while raw := self._pipe.readline(_LOG_LINE_LIMIT + 1):
                # A verbose HTTP body can be arbitrarily long and may not end
                # with a newline. Read it in bounded chunks and discard all of it.
                complete = raw.endswith(b"\n")
                if skipping_long_line:
                    skipping_long_line = not complete
                    continue
                if len(raw) > _LOG_LINE_LIMIT or not complete:
                    skipping_long_line = not complete
                    continue
                line = raw.decode("utf-8", errors="replace")
                safe = _sanitize_llama_log_line(
                    line, expected_device_name=self._expected_device_name,
                )
                if self._ready and safe is not None and not safe.startswith("llama.cpp diagnostic:"):
                    # Placement is immutable after startup. A later prompt may
                    # echo a placement-looking phrase; it is never evidence.
                    safe = None
                if safe is not None:
                    if safe.startswith("llama.cpp diagnostic:"):
                        if safe in self._diagnostics:
                            safe = None
                        else:
                            self._diagnostics.add(safe)
                    if safe is not None:
                        size = len(safe.encode("utf-8")) + 1
                        if self._bytes_written + size <= _SANITIZED_LOG_LIMIT:
                            try:
                                self._file.write(safe + "\n")
                            except Exception:
                                with self._condition:
                                    self._error = True
                                    self._condition.notify_all()
                            else:
                                self._bytes_written += size
                                if not self._ready:
                                    self._startup_lines.append(safe)
                        else:
                            with self._condition:
                                self._error = True
                                self._condition.notify_all()
                listening = _LOG_LISTENING.search(line)
                if listening and int(listening[1]) == self._port:
                    with self._condition:
                        self._ready = True
                        self._condition.notify_all()
        except Exception:
            with self._condition:
                self._error = True
                self._condition.notify_all()
        finally:
            try:
                self._file.close()
            except Exception:
                with self._condition:
                    self._error = True
            with self._condition:
                self._eof = True
                self._condition.notify_all()

    def wait_for_startup(self, deadline: float) -> str:
        """Wait until the worker consumed the server's post-load listen line."""
        with self._condition:
            while not self._ready and not self._eof and not self._error:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("llama.cpp filtered startup log did not reach listen marker")
                self._condition.wait(remaining)
            if self._error:
                raise RuntimeError("llama.cpp filtered startup log could not be recorded")
            if not self._ready:
                raise RuntimeError("llama.cpp server exited before complete placement evidence")
            result = "\n".join(self._startup_lines)
            self._startup_lines.clear()
            return result

    def join_after_exit(self) -> bool:
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            try:
                self._pipe.close()
            except OSError:
                pass
            self._thread.join(timeout=1)
        try:
            self._pipe.close()
        except OSError:
            return False
        return not self._thread.is_alive() and not self._error


def _hybrid_placement_evidence(
    log_text: str, *, gpu_device: str, expected_device_name: str,
    gpu_layers: int | None, cpu_moe_layers: int,
    cpu_weight_budget_bytes: int, gpu_weight_budget_bytes: int,
    host_mapped_expert_layers: int = 0,
) -> dict[str, Any]:
    """Verify logged buffer bounds and the requested override selection.

    llama.cpp prints model buffer sizes rounded to 0.01 MiB; include that
    rounding interval when checking the declared per-pool upper bounds. A
    Vulkan_Host override is selected before mmap/pinned-allocation fallback,
    so this log cannot prove final expert storage or compute location.
    """
    if cpu_moe_layers and host_mapped_expert_layers:
        raise ValueError("CPU and Vulkan_Host expert controls are mutually exclusive")
    if "llama.cpp diagnostic: unexpected_expert_override" in log_text:
        raise RuntimeError("REFUSE_DEVICE_PLACEMENT: unexpected expert tensor override")
    expected_line = f"using device {gpu_device} ({expected_device_name})"
    offloaded = _OFFLOADED_LAYERS.findall(log_text)
    if expected_line not in log_text or not offloaded:
        raise RuntimeError("REFUSE_DEVICE_PLACEMENT: hybrid GPU identity or offload count absent")
    offloaded_count, total_layers = map(int, offloaded[-1])
    if not 0 < offloaded_count <= total_layers:
        raise RuntimeError("REFUSE_DEVICE_PLACEMENT: hybrid route did not use the requested GPU")
    assignments = _LAYER_ASSIGNMENT.findall(log_text)
    numbered_assignments = re.findall(
        r"load_tensors: layer\s+(\d+) assigned to device ([^,\s]+)", log_text,
    )
    if not assignments or gpu_device not in assignments or any(
        device not in {"CPU", gpu_device} for device in assignments
    ):
        raise RuntimeError("REFUSE_DEVICE_PLACEMENT: hybrid layer assignments are missing or unexpected")
    if gpu_layers is None:
        if (offloaded_count != total_layers or
                max(cpu_moe_layers, host_mapped_expert_layers) < 1):
            raise RuntimeError("REFUSE_DEVICE_PLACEMENT: expert route did not offload all layers")
    elif offloaded_count != gpu_layers or (
        cpu_moe_layers == 0 and "CPU" not in assignments
    ):
        raise RuntimeError("REFUSE_DEVICE_PLACEMENT: actual GPU/CPU layer count differs from split")

    buffers = _MODEL_BUFFER_SIZE.findall(log_text)
    if not buffers:
        raise RuntimeError("REFUSE_DEVICE_PLACEMENT: hybrid model buffer sizes are absent")
    by_pool = {"host_ram": 0, "vram": 0}
    for name, size_mib in buffers:
        pool = ("host_ram" if name.startswith("CPU") or name == "Vulkan_Host"
                else "vram" if name == gpu_device else None)
        if pool is None:
            raise RuntimeError(f"REFUSE_DEVICE_PLACEMENT: unexpected model buffer device {name}")
        # The printed value is rounded to two decimals. Using the upper edge
        # avoids accepting a weight buffer that may exceed its budget.
        by_pool[pool] += math.ceil((float(size_mib) + .01) * (1 << 20))
    if not by_pool["host_ram"] or not by_pool["vram"]:
        raise RuntimeError("REFUSE_DEVICE_PLACEMENT: hybrid route lacks CPU or GPU weights")
    if (by_pool["host_ram"] > cpu_weight_budget_bytes or
            by_pool["vram"] > gpu_weight_budget_bytes):
        raise ResourceUnavailable("actual hybrid model buffers exceed the declared per-pool weight budgets")

    expert_overrides = _EXPERT_BUFFER_OVERRIDE.findall(log_text)
    if cpu_moe_layers:
        expected_overrides = {
            (str(layer), f"ffn_{tensor}_exps.weight", "CPU")
            for layer in range(cpu_moe_layers)
            for tensor in ("up", "down", "gate")
        }
        if (len(expert_overrides) != len(expected_overrides) or
                set(expert_overrides) != expected_overrides):
            raise RuntimeError(
                "REFUSE_DEVICE_PLACEMENT: exact CPU-expert tensor overrides were not verified"
            )
    if host_mapped_expert_layers:
        expected_overrides = {
            (str(layer), f"ffn_{tensor}_exps.weight", "Vulkan_Host")
            for layer in range(host_mapped_expert_layers)
            for tensor in ("up", "down", "gate")
        }
        if (gpu_layers is not None or host_mapped_expert_layers >= total_layers or
                len(expert_overrides) != len(expected_overrides) or
                set(expert_overrides) != expected_overrides or
                len(numbered_assignments) != total_layers or
                set(numbered_assignments) != {
                    (str(layer), gpu_device) for layer in range(total_layers)
                }):
            raise RuntimeError(
                "REFUSE_DEVICE_PLACEMENT: exact Vulkan_Host expert overrides and GPU layers were not verified"
            )
    return {
        "offloaded_layers": (offloaded_count, total_layers),
        "model_buffer_bytes_by_pool": by_pool,
        "expert_cpu_tensors_verified": len(expert_overrides) if cpu_moe_layers else 0,
        "expert_host_mapped_tensors_selected": len(expert_overrides) if host_mapped_expert_layers else 0,
        "placement_evidence_level": "override_selection_only" if host_mapped_expert_layers else "startup_log",
        "expert_final_storage_verified": False if host_mapped_expert_layers else None,
        "expert_compute_verified": False if host_mapped_expert_layers else None,
        "layer_assignments": len(assignments),
    }


def _verify_vision_backend(backends: list[str], *, expected_device: str) -> None:
    if backends != [expected_device]:
        raise RuntimeError("REFUSE_DEVICE_PLACEMENT: vision projector backend differs from requested device")


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _json_request(url: str, body: dict | None, *, timeout: float, limit: int) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read(limit + 1)
    if len(payload) > limit:
        raise ResourceUnavailable("llama.cpp response exceeds admitted I/O bound")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("llama.cpp response must be a JSON object")
    return value


def _stream_json_request(
    url: str, body: dict, *, timeout: float, limit: int,
    on_delta: Callable[[str], None],
) -> dict:
    """Read bounded OpenAI SSE from the already admitted local llama-server.

    The callback blocks when the Agent consumer has no event credit, so a slow
    UI cannot make this worker buffer an unbounded answer. This remains inside
    the existing Omni-owned stage process and its one-request reservation.
    """
    data = json.dumps({**body, "stream": True}).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
    )
    pieces: list[str] = []
    used = 0
    finish_reason = None
    usage = None
    done = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            used += len(raw)
            if used > limit:
                raise ResourceUnavailable("llama.cpp SSE response exceeds admitted I/O bound")
            line = raw.strip()
            if not line or not line.startswith(b"data: "):
                continue
            if line == b"data: [DONE]":
                done = True
                break
            chunk = json.loads(line[6:])
            if not isinstance(chunk, dict):
                raise ValueError("llama.cpp SSE chunk must be a JSON object")
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            for choice in choices:
                delta = choice.get("delta") or {}
                part = delta.get("content")
                if part is not None:
                    if not isinstance(part, str):
                        raise ValueError("llama.cpp SSE text delta is not a string")
                    pieces.append(part)
                    for start in range(0, len(part), 8192):
                        on_delta(part[start:start + 8192])
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
    if not done or finish_reason != "stop":
        raise RuntimeError(f"llama.cpp SSE incomplete: done={done}, finish_reason={finish_reason!r}")
    return {"content": "".join(pieces), "finish_reason": finish_reason, "usage": usage}


class LlamaCppTextStageClient(StageClientBase):
    def __init__(self, metadata, config: dict, ledger, reservation) -> None:
        for name, value in vars(metadata).items():
            setattr(self, name, value)
        # The complete-request scheduler is reused; model execution is not an ONNX graph.
        self.stage_type = "graph"
        self._ledger, self._reservation = ledger, reservation
        self._generation = uuid.uuid4().hex
        self._config = dict(config)
        self._proc: subprocess.Popen | None = None
        self._log_filter: _FilteredLlamaLog | None = None
        self._closed = False
        self._active: str | None = None
        self._task: asyncio.Task | None = None
        self._output: OmniRequestOutput | None = None
        self._agent_stream: asyncio.Queue[tuple[str, float] | None] | None = None
        self._agent_cancel = threading.Event()
        self._epoch = 0
        self._max_io_bytes = int(config.get("max_io_bytes", 1 << 20))
        self._max_image_bytes = int(config.get("max_image_bytes", 0))
        self._image_token_reserve = int(config.get("image_token_reserve", 0))
        self._max_new_tokens = int(config.get("max_new_tokens", 96))
        self._context_tokens = int(config.get("context_tokens", 4096))
        self._request_timeout_s = float(config.get("request_timeout_s", 120))
        self._placement = str(config["device"])
        self._host_mapped = re.fullmatch(r"Vulkan_Host\+Vulkan\d+", self._placement) is not None
        self._hybrid = self._host_mapped or re.fullmatch(r"cpu\+Vulkan\d+", self._placement) is not None
        self._gpu_device = self._placement.split("+", 1)[1] if self._hybrid else self._placement
        self._gpu_layers = config.get("gpu_layers")
        self._cpu_moe_layers = int(config.get("cpu_moe_layers") or 0)
        self._host_mapped_expert_layers = int(config.get("host_mapped_expert_layers") or 0)
        self._cpu_weight_budget_bytes = config.get("cpu_weight_budget_bytes")
        self._gpu_weight_budget_bytes = config.get("gpu_weight_budget_bytes")
        self._vram_overhead_bytes = int(config.get("vram_overhead_bytes") or 0)
        self._memory_pool = str(config.get("memory_pool", "host_ram"))
        self._port = int(config.get("port") or _free_local_port())
        if (
            self._max_io_bytes <= 0
            or self._max_new_tokens <= 0
            or self._context_tokens <= self._max_new_tokens
            or self._request_timeout_s <= 0
            or not 0 < self._port < 65536
        ):
            raise ValueError("invalid llama.cpp request, context, timeout or port bound")
        try:
            if self._memory_pool not in reservation.demands:
                raise ResourceUnavailable("llama.cpp memory pool absent from stage reservation")
            if ("host_ram" not in reservation.demands or
                    self._max_io_bytes > reservation.demands["host_ram"]):
                raise ResourceUnavailable("llama.cpp host I/O bound exceeds host RAM reservation")
            self._model = Path(config["model_file"]).resolve(strict=True)
            self._binary = Path(config["server_bin"]).resolve(strict=True)
            self._log_path = Path(config["log_file"]).resolve()
            self._expected_model_sha = str(config["model_sha256"]).lower()
            self._expected_binary_sha = str(config["server_sha256"]).lower()
            if _digest(self._model) != self._expected_model_sha or _digest(self._binary) != self._expected_binary_sha:
                raise ValueError("llama.cpp executable or GGUF differs from the declared artifact hash")
            self._mmproj = None
            self._expected_mmproj_sha = None
            if config.get("mmproj_file") is not None:
                if config.get("name") != "external.llamacpp.multimodal.v1":
                    raise ValueError("a vision projector requires the multimodal llama.cpp backend")
                self._mmproj = Path(config["mmproj_file"]).resolve(strict=True)
                self._expected_mmproj_sha = str(config["mmproj_sha256"]).lower()
                if _digest(self._mmproj) != self._expected_mmproj_sha:
                    raise ValueError("llama.cpp vision projector differs from the declared artifact hash")
                if self._max_image_bytes <= 0 or self._image_token_reserve <= 0:
                    raise ValueError("multimodal llama.cpp stage requires image byte and token bounds")
            elif self._max_image_bytes or self._image_token_reserve:
                raise ValueError("image bounds require a pinned multimodal projector")
            overhead = int(config["memory_overhead_bytes"])
            artifact_bytes = self._model.stat().st_size + (
                self._mmproj.stat().st_size if self._mmproj is not None else 0
            )
            if overhead <= 0:
                raise ValueError("llama.cpp stage requires positive load/KV/workspace headroom")
            if self._hybrid:
                cpu_budget = self._cpu_weight_budget_bytes
                gpu_budget = self._gpu_weight_budget_bytes
                if (self._memory_pool not in {"host_ram", "vram"} or
                        type(cpu_budget) is not int or cpu_budget <= 0
                        or type(gpu_budget) is not int or gpu_budget <= 0
                        or self._vram_overhead_bytes <= 0):
                    raise ValueError("hybrid route requires explicit CPU/GPU budgets and GPU workspace overhead")
                if cpu_budget + gpu_budget < artifact_bytes:
                    raise ResourceUnavailable("hybrid weight budgets are below the pinned GGUF and projector bytes")
                if self._memory_pool == "host_ram":
                    if (cpu_budget + gpu_budget + overhead + self._vram_overhead_bytes >
                            reservation.demands["host_ram"]):
                        raise ResourceUnavailable("shared-memory hybrid weights and workspaces exceed RAM reservation")
                elif (cpu_budget + overhead > reservation.demands["host_ram"] or
                      gpu_budget + self._vram_overhead_bytes > reservation.demands["vram"]):
                    raise ResourceUnavailable("hybrid weight, loading and workspace claims exceed reservation")
                if self._host_mapped:
                    if (self._gpu_layers is not None or self._cpu_moe_layers or
                            self._host_mapped_expert_layers <= 0):
                        raise ValueError("Vulkan_Host route requires exact host-mapped expert layers only")
                elif self._host_mapped_expert_layers or (
                    self._gpu_layers is None and self._cpu_moe_layers <= 0
                ):
                    raise ValueError("CPU hybrid route requires a pinned layer or CPU-expert split only")
            elif self._host_mapped_expert_layers:
                raise ValueError("host_mapped_expert_layers require Vulkan_Host+VulkanN placement")
            elif artifact_bytes + overhead > reservation.demands[self._memory_pool]:
                raise ResourceUnavailable("GGUF and projector plus declared KV/workspace/transfer/headroom exceed reservation")
            if self._placement != "cpu" and not re.fullmatch(r"Vulkan\d+", self._gpu_device):
                raise ValueError("llama.cpp device must be cpu or an explicit Vulkan index")
            expected_device_name = config.get("expected_device_name")
            if self._placement != "cpu" and not expected_device_name:
                raise ValueError("Vulkan stage requires expected_device_name")

            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            if config.get("ggml_vk_visible_devices") is not None:
                env["GGML_VK_VISIBLE_DEVICES"] = str(config["ggml_vk_visible_devices"])
            command = [
                str(self._binary), "-m", str(self._model),
                "-dev", "none" if self._placement == "cpu" else self._gpu_device,
                "-ngl", ("0" if self._placement == "cpu" else
                         str(self._gpu_layers) if self._gpu_layers is not None else
                         "all" if self._hybrid else "99"),
                "-c", str(self._context_tokens), "-np", "1",
                "--host", "127.0.0.1", "--port", str(self._port),
                "--reasoning", "off", "--no-webui", "--fit", "off",
                "--cache-ram", "0", "-lv", "5" if self._hybrid else "4",
            ]
            if config.get("disable_repack", False):
                command.append("--no-repack")
            expert_layers = self._host_mapped_expert_layers if self._host_mapped else self._cpu_moe_layers
            if expert_layers:
                command.extend(["--n-cpu-moe", str(expert_layers)])
            if self._mmproj is not None:
                command.extend(["--mmproj", str(self._mmproj)])
                if self._placement == "cpu":
                    command.append("--no-mmproj-offload")
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self._proc = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env=env, creationflags=flags,
            )
            assert self._proc.stdout is not None
            self._log_filter = _FilteredLlamaLog(
                self._proc.stdout, self._log_path,
                expected_device_name=str(expected_device_name) if expected_device_name else None,
                port=self._port,
            )
            base = f"http://127.0.0.1:{self._port}"
            deadline = time.monotonic() + float(config.get("start_timeout_s", 120))
            while True:
                if self._proc.poll() is not None:
                    raise RuntimeError(f"llama.cpp server exited during load; see {self._log_path}")
                try:
                    health = _json_request(base + "/health", None, timeout=1, limit=4096)
                    if health.get("status") == "ok":
                        break
                except (urllib.error.URLError, TimeoutError, OSError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"llama.cpp server did not become ready; see {self._log_path}")
                time.sleep(0.2)
            props = _json_request(base + "/props", None, timeout=5, limit=64 << 10)
            if Path(props.get("model_path", "")).resolve() != self._model or props.get("total_slots") != 1:
                raise RuntimeError("llama.cpp did not load the pinned GGUF into one slot")
            self._model_alias = props["model_alias"]
            self._base_url = base
            # /health and /props can complete before the pipe reader catches up.
            # The listen line follows all tensor placement/buffer records.
            log_text = self._log_filter.wait_for_startup(deadline)
            assignments = _LAYER_ASSIGNMENT.findall(log_text)
            offloaded = _OFFLOADED_LAYERS.findall(log_text)
            device_buffers = _MODEL_BUFFER.findall(log_text)
            all_model_buffers = _ANY_MODEL_BUFFER.findall(log_text)
            vision_backends = _CLIP_BACKEND.findall(log_text)
            hybrid_evidence: dict[str, Any] | None = None
            if self._placement == "cpu":
                # CPU-only llama.cpp builds can omit the offload-count line.
                # Require positive CPU tensor-buffer evidence in that layout;
                # a missing GPU line alone must never imply CPU placement.
                cpu_only_layout = (
                    not offloaded
                    and "warning: no usable GPU found" in log_text
                    and bool(all_model_buffers)
                    and all(name.startswith("CPU") for name in all_model_buffers)
                )
                if (
                    (not cpu_only_layout and (not offloaded or offloaded[-1][0] != "0"))
                    or device_buffers
                    or (all_model_buffers and any(not name.startswith("CPU") for name in all_model_buffers))
                    or (assignments and any(name != "CPU" for name in assignments))
                ):
                    raise RuntimeError("CPU stage assigned model layers to an accelerator")
            elif self._hybrid:
                hybrid_evidence = _hybrid_placement_evidence(
                    log_text, gpu_device=self._gpu_device,
                    expected_device_name=str(expected_device_name),
                    gpu_layers=self._gpu_layers, cpu_moe_layers=self._cpu_moe_layers,
                    host_mapped_expert_layers=self._host_mapped_expert_layers,
                    cpu_weight_budget_bytes=self._cpu_weight_budget_bytes,
                    gpu_weight_budget_bytes=(
                        self._gpu_weight_budget_bytes - self._mmproj.stat().st_size
                        if self._mmproj is not None else self._gpu_weight_budget_bytes
                    ),
                )
            else:
                expected_line = f"using device {self._gpu_device} ({expected_device_name})"
                fully_offloaded = bool(offloaded) and offloaded[-1][0] == offloaded[-1][1]
                if (
                    expected_line not in log_text or not fully_offloaded
                    or device_buffers != [self._gpu_device]
                    or (assignments and any(name != self._gpu_device for name in assignments))
                ):
                    raise RuntimeError("REFUSE_DEVICE_PLACEMENT: llama.cpp layer assignment differs from requested GPU")
            if self._mmproj is not None:
                expected_vision_backend = "CPU" if self._placement == "cpu" else self._gpu_device
                _verify_vision_backend(vision_backends, expected_device=expected_vision_backend)
            self._loaded_rss_bytes = self._process_rss_bytes()
            if self._loaded_rss_bytes > reservation.demands["host_ram"]:
                raise ResourceUnavailable("loaded llama.cpp process RSS exceeds host RAM reservation")
            self.execution_plan = {
                "backend": "external.llamacpp.multimodal.v1" if self._mmproj is not None else "external.llamacpp.text.v1",
                "stage_id": self.stage_id,
                "worker_generation": self._generation,
                "model_sha256": self._expected_model_sha,
                "mmproj_sha256": self._expected_mmproj_sha,
                "server_sha256": self._expected_binary_sha,
                "model_alias": self._model_alias,
                "requested_device": self._placement,
                # The loader report independently verifies ordinary CPU/GPU
                # layer placement. Vulkan_Host only proves override selection:
                # final expert storage and compute may differ after fallback.
                "observed_model_placement": None if self._host_mapped else self._placement,
                "expected_device_name": expected_device_name,
                "layer_assignments": len(assignments),
                "offloaded_layers": offloaded[-1] if offloaded else None,
                "model_buffer_devices": device_buffers,
                "all_model_buffers": all_model_buffers,
                "hybrid_placement_evidence": hybrid_evidence,
                "gpu_buffer_physical_pool": self._memory_pool if self._placement != "cpu" else None,
                "cpu_moe_layers_requested": self._cpu_moe_layers,
                "host_mapped_expert_layers_requested": self._host_mapped_expert_layers,
                "gpu_layers_requested": self._gpu_layers,
                "placement_evidence_level": (
                    "override_selection_only" if self._host_mapped else "startup_log"
                ),
                "expert_final_storage_verified": False if self._host_mapped else None,
                "expert_compute_verified": False if self._host_mapped else None,
                "vision_backends": vision_backends,
                "cpu_only_layout": cpu_only_layout if self._placement == "cpu" else False,
                "worker_pid": self._proc.pid,
                "loaded_rss_bytes": self._loaded_rss_bytes,
                "reserved_bytes": dict(reservation.demands),
                "memory_pool": self._memory_pool,
                "memory_overhead_bytes": overhead,
                "vram_overhead_bytes": self._vram_overhead_bytes,
                "memory_evidence": (
                    "declared dual-pool admission plus loaded RSS and model-buffer log; "
                    "Vulkan_Host override selection does not prove final expert storage or compute; "
                    "loading/whole-request RAM and VRAM peaks require telemetry"
                    if self._host_mapped else
                    "declared dual-pool admission plus loaded RSS and model-buffer log; "
                    "loading/whole-request RAM and VRAM peaks require telemetry"
                ),
                "disable_repack": bool(config.get("disable_repack", False)),
                "context_tokens": self._context_tokens,
                "max_new_tokens": self._max_new_tokens,
                "max_io_bytes": self._max_io_bytes,
                "max_image_bytes": self._max_image_bytes,
                "image_token_reserve": self._image_token_reserve,
                "request_capacity": 1,
                "stateful_session": "llama.cpp-owned; reset for each complete request",
                "evidence": "B",
            }
        except BaseException:
            self.shutdown()
            raise

    def _process_rss_bytes(self) -> int:
        import psutil

        return int(psutil.Process(self._proc.pid).memory_info().rss)

    async def add_request_async(self, request_id: str, prompt: Any, params: Any = None) -> None:
        self.check_health()
        if self._active is not None:
            raise ResourceUnavailable("llama.cpp stage has an unacknowledged request; capacity is one")
        if not isinstance(prompt, dict) or not isinstance(prompt.get("text"), str) or not prompt["text"]:
            raise ValueError("llama.cpp prompt must contain nonempty text from its model adapter")
        text = prompt["text"]
        image_data_url = prompt.get("image_data_url")
        if image_data_url is not None:
            if self._mmproj is None:
                raise ValueError("llama.cpp text stage does not accept images")
            prefix = "data:image/png;base64,"
            if not isinstance(image_data_url, str) or not image_data_url.startswith(prefix):
                raise ValueError("llama.cpp multimodal stage requires a PNG data URL")
            if len(image_data_url.encode("utf-8")) > self._max_io_bytes:
                raise ResourceUnavailable("PNG data URL exceeds admitted I/O bound")
            try:
                image_bytes = base64.b64decode(image_data_url[len(prefix):], validate=True)
                from PIL import Image

                with Image.open(io.BytesIO(image_bytes)) as image:
                    if image.format != "PNG" or image.width * image.height > 1024 * 1024:
                        raise ValueError("PNG image exceeds admitted format or pixel bound")
                    image.verify()
            except (OSError, ValueError, SyntaxError, binascii.Error) as exc:
                raise ValueError("invalid or oversized PNG image") from exc
            if len(image_bytes) > self._max_image_bytes:
                raise ResourceUnavailable("PNG image exceeds admitted image byte bound")
        if len(text.encode("utf-8")) + (len(image_data_url.encode("utf-8")) if image_data_url else 0) > self._max_io_bytes:
            raise ResourceUnavailable("llama.cpp prompt exceeds admitted I/O bound")
        max_tokens = int(prompt.get("max_tokens", self._max_new_tokens))
        if not 0 < max_tokens <= self._max_new_tokens:
            raise ValueError("requested token limit exceeds admitted llama.cpp maximum")
        temperature = float(prompt.get("temperature", 0))
        if temperature != 0:
            raise ValueError("llama.cpp text v1 accepts deterministic temperature=0 only")
        tokens = await asyncio.to_thread(
            _json_request,
            self._base_url + "/tokenize",
            {"content": text},
            timeout=self._request_timeout_s,
            limit=self._max_io_bytes,
        )
        token_ids = tokens.get("tokens")
        if not isinstance(token_ids, list) or not all(isinstance(item, int) for item in token_ids):
            raise ValueError("llama.cpp returned invalid prompt tokenization")
        # The projector owns actual image patching; reserve a conservative bound
        # before submission rather than relying on server-side truncation.
        image_tokens = self._image_token_reserve if image_data_url is not None else 0
        if len(token_ids) + max_tokens + image_tokens + 64 > self._context_tokens:
            raise ResourceUnavailable("llama.cpp prompt, image and generation bounds exceed context; refusing truncation")
        self._epoch += 1
        epoch = self._epoch
        request = StageRequest(request_id, self.stage_id, epoch, self._generation)
        agent_stream = prompt.get("stream_agent", False)
        if type(agent_stream) is not bool:
            raise ValueError("stream_agent must be a boolean")
        self._active = request_id
        self._agent_cancel.clear()
        self._agent_stream = asyncio.Queue(maxsize=64) if agent_stream else None
        self._task = asyncio.create_task(
            self._run(request, text, max_tokens, image_data_url, self._agent_stream),
            name=f"llamacpp-{self.stage_id}-{request_id}"
        )

    async def _run(
        self, request: StageRequest, text: str, max_tokens: int,
        image_data_url: str | None,
        agent_stream: asyncio.Queue[tuple[str, float] | None] | None,
    ) -> None:
        try:
            started = time.perf_counter()
            content: str | list[dict] = text
            if image_data_url is not None:
                content = [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ]
            body = {
                "model": self._model_alias,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "max_tokens": max_tokens,
                "stream": False,
                "cache_prompt": False,
            }
            if agent_stream is None:
                result = await asyncio.to_thread(
                    _json_request, self._base_url + "/v1/chat/completions", body,
                    timeout=self._request_timeout_s, limit=self._max_io_bytes,
                )
                choice = result["choices"][0]
                content = choice["message"]["content"]
                finish_reason = choice["finish_reason"]
                usage = result.get("usage")
            else:
                loop = asyncio.get_running_loop()

                def on_delta(part: str) -> None:
                    if self._agent_cancel.is_set():
                        raise RuntimeError("Agent SSE stream was cancelled")
                    ticket = asyncio.run_coroutine_threadsafe(
                        agent_stream.put((part, time.perf_counter())), loop,
                    )
                    ticket.result(timeout=self._request_timeout_s)

                result = await asyncio.to_thread(
                    _stream_json_request,
                    self._base_url + "/v1/chat/completions", body,
                    timeout=self._request_timeout_s, limit=self._max_io_bytes,
                    on_delta=on_delta,
                )
                content = result["content"]
                finish_reason = result["finish_reason"]
                usage = result.get("usage")
            wall_s = time.perf_counter() - started
            if not isinstance(content, str) or finish_reason != "stop":
                raise RuntimeError(f"llama.cpp response incomplete: finish_reason={finish_reason!r}")
            event = StageEvent(
                request.request_id, self.stage_id, request.epoch, 1, "text",
                self._generation, terminal=True,
            )
            output = OmniRequestOutput(
                request_id=request.request_id,
                prompt=text,
                stage_id=self.stage_id,
                final_output_type="text",
                outputs=[CompletionOutput(0, content, [], None, None, finish_reason=finish_reason)],
                _custom_output={"stage_event": dataclasses.asdict(event)},
                metrics={"llamacpp_wall_s": wall_s, "usage": usage,
                         "agent_sse": agent_stream is not None},
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            output = OmniRequestOutput.from_error(request.request_id, str(exc))
            output.stage_id = self.stage_id
        if not self._closed and self._epoch == request.epoch:
            loop = asyncio.get_running_loop()
            output._stage_release = lambda: loop.call_soon_threadsafe(
                self.acknowledge, request.request_id, request.epoch, request.worker_generation
            )
            self._output = output
        if agent_stream is not None and self._epoch == request.epoch and not self._closed:
            await agent_stream.put(None)

    async def receive_agent_delta(self, request_id: str) -> tuple[str, float] | None:
        """Receive one credited local-model delta for this active request."""
        if self._active != request_id or self._agent_stream is None:
            raise ValueError("no active Agent SSE stream for this request")
        return await self._agent_stream.get()

    def get_graph_output_nowait(self) -> OmniRequestOutput | None:
        output, self._output = self._output, None
        return output

    def acknowledge(self, request_id: str, epoch: int, generation: str) -> None:
        if (
            self._active == request_id
            and epoch == self._epoch
            and generation == self._generation
            and (self._task is None or self._task.done())
        ):
            self._active = None
            self._task = None
            self._agent_stream = None

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        if self._active not in request_ids:
            return
        self._agent_cancel.set()
        if self._agent_stream is not None:
            while not self._agent_stream.empty():
                self._agent_stream.get_nowait()
        self._epoch += 1
        self._output = None
        if self._task is not None and not self._task.done():
            drained = await asyncio.to_thread(self._terminate)
            self._closed = True
            done, _ = await asyncio.wait({self._task}, timeout=5)
            if done:
                self._ledger.release(self._reservation, drained=drained)
            else:
                self._ledger.release(self._reservation, drained=False)
                self._task.add_done_callback(
                    lambda task: self._ledger.release(
                        self._reservation, drained=drained and not task.cancelled()
                    )
                )
        self._active = None
        self._agent_stream = None

    def _terminate(self) -> bool:
        proc = self._proc
        if proc is None:
            return True
        exited = False
        try:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        except OSError:
            exited = False
        else:
            exited = proc.poll() is not None
        finally:
            if self._log_filter is not None:
                exited = self._log_filter.join_after_exit() and exited
                self._log_filter = None
            elif proc.stdout is not None:
                proc.stdout.close()
        return exited

    def check_health(self) -> None:
        if self._closed or self._proc is None or self._proc.poll() is not None:
            from vllm.v1.engine.exceptions import EngineDeadError

            raise EngineDeadError()

    async def collective_rpc_async(self, method, timeout=None, args=(), kwargs=None):
        raise NotImplementedError(f"llama.cpp backend does not implement collective RPC {method}")

    def shutdown(self) -> None:
        self._closed = True
        self._agent_cancel.set()
        self._output = None
        drained = self._terminate()
        task = self._task
        if task is not None and not task.done():
            self._ledger.release(self._reservation, drained=False)
            task.add_done_callback(
                lambda done: self._ledger.release(self._reservation, drained=drained and not done.cancelled())
            )
        else:
            self._ledger.release(self._reservation, drained=drained)


class LlamaCppMultimodalStageClient(LlamaCppTextStageClient):
    """The same bounded whole-session controller with a pinned vision projector."""

    def __init__(self, metadata, config: dict, ledger, reservation) -> None:
        if config.get("mmproj_file") is None:
            raise ValueError("multimodal llama.cpp stage requires a pinned mmproj_file")
        super().__init__(metadata, config, ledger, reservation)
