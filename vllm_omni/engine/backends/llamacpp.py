# SPDX-License-Identifier: Apache-2.0
"""Bounded, complete-request llama.cpp text and optional image stage owned by StageRuntime.

The llama-server subprocess owns its weights and KV. Omni owns the admission
reservation, request identity, output acknowledgement and process lifetime.
This v1 backend deliberately has one slot and no incremental output contract.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import dataclasses
import hashlib
import io
import json
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from omni_stage_contracts import StageEvent, StageRequest
from vllm.outputs import CompletionOutput

from vllm_omni.engine.resource_ledger import ResourceUnavailable
from vllm_omni.engine.stage_client import StageClientBase
from vllm_omni.outputs import OmniRequestOutput

_LAYER_ASSIGNMENT = re.compile(r"load_tensors: layer\s+\d+ assigned to device (\S+)")
_OFFLOADED_LAYERS = re.compile(r"load_tensors: offloaded (\d+)/(\d+) layers to GPU")
_MODEL_BUFFER = re.compile(r"load_tensors:\s+(Vulkan\d+) model buffer size")
_ANY_MODEL_BUFFER = re.compile(r"load_tensors:\s+(\S+) model buffer size")
_CLIP_BACKEND = re.compile(r"clip_ctx: CLIP using (\S+) backend")


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
        self._log_stream = None
        self._closed = False
        self._active: str | None = None
        self._task: asyncio.Task | None = None
        self._output: OmniRequestOutput | None = None
        self._epoch = 0
        self._max_io_bytes = int(config.get("max_io_bytes", 1 << 20))
        self._max_image_bytes = int(config.get("max_image_bytes", 0))
        self._image_token_reserve = int(config.get("image_token_reserve", 0))
        self._max_new_tokens = int(config.get("max_new_tokens", 96))
        self._context_tokens = int(config.get("context_tokens", 4096))
        self._request_timeout_s = float(config.get("request_timeout_s", 120))
        self._placement = str(config["device"])
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
            if self._max_io_bytes > reservation.demands[self._memory_pool]:
                raise ResourceUnavailable("llama.cpp I/O bound exceeds stage reservation")
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
            if overhead <= 0 or artifact_bytes + overhead > reservation.demands[self._memory_pool]:
                raise ResourceUnavailable("GGUF and projector plus declared KV/workspace/transfer/headroom exceed reservation")
            if self._placement != "cpu" and not re.fullmatch(r"Vulkan\d+", self._placement):
                raise ValueError("llama.cpp device must be cpu or an explicit Vulkan index")
            expected_device_name = config.get("expected_device_name")
            if self._placement != "cpu" and not expected_device_name:
                raise ValueError("Vulkan stage requires expected_device_name")

            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_stream = self._log_path.open("wb")
            env = os.environ.copy()
            if config.get("ggml_vk_visible_devices") is not None:
                env["GGML_VK_VISIBLE_DEVICES"] = str(config["ggml_vk_visible_devices"])
            command = [
                str(self._binary), "-m", str(self._model),
                "-dev", "none" if self._placement == "cpu" else self._placement,
                "-ngl", "0" if self._placement == "cpu" else "99",
                "-c", str(self._context_tokens), "-np", "1",
                "--host", "127.0.0.1", "--port", str(self._port),
                "--reasoning", "off", "--no-webui", "--fit", "off",
                "--cache-ram", "0", "-lv", "4",
            ]
            if config.get("disable_repack", False):
                command.append("--no-repack")
            if self._mmproj is not None:
                command.extend(["--mmproj", str(self._mmproj)])
                if self._placement == "cpu":
                    command.append("--no-mmproj-offload")
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self._proc = subprocess.Popen(
                command, stdout=self._log_stream, stderr=subprocess.STDOUT,
                env=env, creationflags=flags,
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
            log_text = self._log_path.read_text(encoding="utf-8", errors="replace")
            assignments = _LAYER_ASSIGNMENT.findall(log_text)
            offloaded = _OFFLOADED_LAYERS.findall(log_text)
            device_buffers = _MODEL_BUFFER.findall(log_text)
            all_model_buffers = _ANY_MODEL_BUFFER.findall(log_text)
            vision_backends = _CLIP_BACKEND.findall(log_text)
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
            else:
                expected_line = f"using device {self._placement} ({expected_device_name})"
                fully_offloaded = bool(offloaded) and offloaded[-1][0] == offloaded[-1][1]
                if (
                    expected_line not in log_text or not fully_offloaded
                    or device_buffers != [self._placement]
                    or (assignments and any(name != self._placement for name in assignments))
                ):
                    raise RuntimeError("REFUSE_DEVICE_PLACEMENT: llama.cpp layer assignment differs from requested GPU")
            if self._mmproj is not None:
                expected_vision_backend = "CPU" if self._placement == "cpu" else self._placement
                if vision_backends != [expected_vision_backend]:
                    raise RuntimeError("REFUSE_DEVICE_PLACEMENT: vision projector backend differs from requested device")
            self._loaded_rss_bytes = self._process_rss_bytes()
            if self._loaded_rss_bytes > reservation.demands[self._memory_pool]:
                raise ResourceUnavailable("loaded llama.cpp process RSS exceeds stage reservation")
            self.execution_plan = {
                "backend": "external.llamacpp.multimodal.v1" if self._mmproj is not None else "external.llamacpp.text.v1",
                "stage_id": self.stage_id,
                "worker_generation": self._generation,
                "model_sha256": self._expected_model_sha,
                "mmproj_sha256": self._expected_mmproj_sha,
                "server_sha256": self._expected_binary_sha,
                "model_alias": self._model_alias,
                "requested_device": self._placement,
                "expected_device_name": expected_device_name,
                "layer_assignments": len(assignments),
                "offloaded_layers": offloaded[-1] if offloaded else None,
                "model_buffer_devices": device_buffers,
                "all_model_buffers": all_model_buffers,
                "vision_backends": vision_backends,
                "cpu_only_layout": cpu_only_layout if self._placement == "cpu" else False,
                "worker_pid": self._proc.pid,
                "loaded_rss_bytes": self._loaded_rss_bytes,
                "reserved_bytes": dict(reservation.demands),
                "memory_pool": self._memory_pool,
                "memory_overhead_bytes": overhead,
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
        self._active = request_id
        self._task = asyncio.create_task(
            self._run(request, text, max_tokens, image_data_url), name=f"llamacpp-{self.stage_id}-{request_id}"
        )

    async def _run(self, request: StageRequest, text: str, max_tokens: int, image_data_url: str | None) -> None:
        try:
            started = time.perf_counter()
            content: str | list[dict] = text
            if image_data_url is not None:
                content = [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ]
            result = await asyncio.to_thread(
                _json_request,
                self._base_url + "/v1/chat/completions",
                {
                    "model": self._model_alias,
                    "messages": [{"role": "user", "content": content}],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "stream": False,
                    "cache_prompt": False,
                },
                timeout=self._request_timeout_s,
                limit=self._max_io_bytes,
            )
            wall_s = time.perf_counter() - started
            choice = result["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice["finish_reason"]
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
                metrics={"llamacpp_wall_s": wall_s, "usage": result.get("usage")},
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

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        if self._active not in request_ids:
            return
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

    def _terminate(self) -> bool:
        proc = self._proc
        if proc is None:
            if self._log_stream is not None:
                self._log_stream.close()
                self._log_stream = None
            return True
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
            return False
        finally:
            if self._log_stream is not None:
                self._log_stream.close()
                self._log_stream = None
        return proc.poll() is not None

    def check_health(self) -> None:
        if self._closed or self._proc is None or self._proc.poll() is not None:
            from vllm.v1.engine.exceptions import EngineDeadError

            raise EngineDeadError()

    async def collective_rpc_async(self, method, timeout=None, args=(), kwargs=None):
        raise NotImplementedError(f"llama.cpp backend does not implement collective RPC {method}")

    def shutdown(self) -> None:
        self._closed = True
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
