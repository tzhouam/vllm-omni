# SPDX-License-Identifier: Apache-2.0
"""Bounded complete-request Qwen3-TTS stage backed by local CrispASR.

CrispASR owns the talker, code predictor and codec state. Omni owns the
reservation, process, request identity, terminal audio and output acknowledgement.
This backend does not advertise playable incremental audio or post-cancel restart.
"""

from __future__ import annotations

import asyncio
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
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omni_stage_contracts import StageEvent, StageRequest
from vllm.outputs import CompletionOutput

from vllm_omni.engine.resource_ledger import ResourceUnavailable
from vllm_omni.engine.stage_client import StageClientBase
from vllm_omni.outputs import OmniRequestOutput


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _post_wav(url: str, text: str, seed: int, *, timeout: float, max_bytes: int) -> bytes:
    payload = json.dumps({"input": text, "response_format": "wav", "seed": seed}).encode()
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"CrispASR returned HTTP {response.status}")
        wav_bytes = response.read(max_bytes + 1)
    if len(wav_bytes) > max_bytes:
        raise ResourceUnavailable("CrispASR WAV exceeds admitted output bound")
    return wav_bytes


def _decode_wav(wav_bytes: bytes, max_audio_s: float) -> tuple[torch.Tensor, bytes, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        channels, width, sample_rate, frames = (
            wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
        )
        if (channels, width, sample_rate) != (1, 2, 24000) or not 0 < frames <= max_audio_s * sample_rate:
            raise ValueError("CrispASR WAV violates 24 kHz mono PCM16 or duration contract")
        pcm = wav.readframes(frames)
    if len(pcm) != frames * 2:
        raise ValueError("CrispASR returned truncated PCM")
    audio = torch.from_numpy(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0)
    if not torch.isfinite(audio).all() or float(torch.sqrt(torch.mean(audio.square()))) <= 1e-5:
        raise ValueError("CrispASR returned silent or invalid audio")
    return audio, pcm, frames


class CrispTTSStageClient(StageClientBase):
    def __init__(self, metadata, config: dict, ledger, reservation) -> None:
        for name, value in vars(metadata).items():
            setattr(self, name, value)
        self.stage_type = "graph"  # Omni's complete-request control path.
        self._ledger, self._reservation = ledger, reservation
        self._generation = uuid.uuid4().hex
        self._proc: subprocess.Popen | None = None
        self._log_stream = None
        self._closed = False
        self._active: str | None = None
        self._task: asyncio.Task | None = None
        self._output: OmniRequestOutput | None = None
        self._epoch = 0
        self._memory_pool = str(config.get("memory_pool", "host_ram"))
        self._max_text_bytes = int(config.get("max_text_bytes", 4096))
        self._max_wav_bytes = int(config.get("max_wav_bytes", 4 << 20))
        self._max_audio_s = float(config.get("max_audio_s", 10))
        self._request_timeout_s = float(config.get("request_timeout_s", 180))
        self._voice = str(config.get("voice", "ryan"))
        self._seed = int(config.get("seed", 42))
        self._metric_name = "crisp_tts_wall_s"
        self._port = int(config.get("port") or _local_port())
        try:
            if (
                self._max_text_bytes <= 0 or self._max_wav_bytes <= 0 or self._max_audio_s <= 0
                or self._request_timeout_s <= 0 or not 0 < self._port < 65536
                or self._seed < 1 or not re.fullmatch(r"[a-z_]+", self._voice)
            ):
                raise ValueError("invalid CrispASR text, WAV, voice, seed, timeout or port bound")
            if self._memory_pool not in reservation.demands:
                raise ResourceUnavailable("CrispASR memory pool absent from stage reservation")
            demand = reservation.demands[self._memory_pool]
            if self._max_wav_bytes > demand:
                raise ResourceUnavailable("CrispASR WAV bound exceeds stage reservation")
            self._binary = Path(config["server_bin"]).resolve(strict=True)
            self._talker = Path(config["talker_file"]).resolve(strict=True)
            self._codec = Path(config["codec_file"]).resolve(strict=True)
            self._punc = Path(config["punc_file"]).resolve(strict=True)
            expected = {
                "server_sha256": (self._binary, str(config["server_sha256"]).lower()),
                "talker_sha256": (self._talker, str(config["talker_sha256"]).lower()),
                "codec_sha256": (self._codec, str(config["codec_sha256"]).lower()),
                "punc_sha256": (self._punc, str(config["punc_sha256"]).lower()),
            }
            for label, (path, digest) in expected.items():
                if _sha256(path) != digest:
                    raise ValueError(f"CrispASR {label} differs from the declared artifact hash")
            overhead = int(config["memory_overhead_bytes"])
            size = self._talker.stat().st_size + self._codec.stat().st_size + self._punc.stat().st_size
            if overhead <= 0 or size + overhead > demand:
                raise ResourceUnavailable("TTS weights plus declared state/workspace/headroom exceed reservation")
            self._expected_gpu_name = str(config["expected_gpu_name"])
            if not self._expected_gpu_name:
                raise ValueError("hybrid TTS stage requires expected_gpu_name")
            visible_device = str(config.get("ggml_vk_visible_devices", "1"))
            if self._expected_gpu_name == "AMD Radeon(TM) 890M Graphics" and visible_device == "1":
                gpu_label = "Radeon"
                gpu_memory_overhead = None  # iGPU shares the admitted host RAM.
            elif self._expected_gpu_name == "NVIDIA GeForce RTX 5090 Laptop GPU" and visible_device == "0":
                gpu_label = "RTX 5090 Laptop"
                gpu_pool = str(config.get("gpu_memory_pool", "gpu_vram"))
                gpu_demand = reservation.demands.get(gpu_pool)
                gpu_memory_overhead = int(config.get("gpu_memory_overhead_bytes", 0))
                if (gpu_pool == self._memory_pool or gpu_demand is None
                        or gpu_memory_overhead < (2 << 30)):
                    raise ResourceUnavailable("discrete TTS route requires a distinct GPU-VRAM reservation and 2 GiB overhead")
                if size + gpu_memory_overhead > gpu_demand:
                    raise ResourceUnavailable("TTS weights plus GPU state/workspace/headroom exceed GPU-VRAM reservation")
            else:
                raise ValueError("CrispASR GPU name and visible Vulkan device do not match a qualified route")
            self._log_path = Path(config["log_file"]).resolve()
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_stream = self._log_path.open("wb")
            env = os.environ.copy()
            env["GGML_VK_VISIBLE_DEVICES"] = visible_device
            env["CRISPASR_QWEN3_TTS_VULKAN_NATIVE"] = "1"
            env["CRISPASR_QWEN3_TTS_CP_BACKEND"] = "cpu-f32"
            command = [
                str(self._binary), "--server", "--host", "127.0.0.1", "--port", str(self._port),
                "--backend", "qwen3-tts-customvoice", "-m", str(self._talker),
                "--codec-model", str(self._codec), "--voice", self._voice,
                "--punc-model", str(self._punc),
                "--gpu-backend", "vulkan", "-dev", visible_device, "--no-flash-attn",
                "-n", "96", "--seed", str(self._seed), "--verbose",
            ]
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self._proc = subprocess.Popen(
                command, stdout=self._log_stream, stderr=subprocess.STDOUT,
                env=env, creationflags=flags,
            )
            base = f"http://127.0.0.1:{self._port}"
            deadline = time.monotonic() + float(config.get("start_timeout_s", 180))
            while True:
                if self._proc.poll() is not None:
                    raise RuntimeError(f"CrispASR exited during load; see {self._log_path}")
                try:
                    with urllib.request.urlopen(base + "/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError, OSError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"CrispASR did not become ready; see {self._log_path}")
                time.sleep(0.2)
            log_text = self._log_path.read_text(encoding="utf-8", errors="replace")
            placement = (
                f"0 = {self._expected_gpu_name}" in log_text
                and "using preferred GPU backend: Vulkan0" in log_text
                and "code_pred CPU-pinned" in log_text
                and "codec: GPU default - loading weights onto Vulkan0" in log_text
                and "qwen3_tts: loaded " in log_text
                and f"loaded punctuation model '{self._punc}'" in log_text
                and "falling back to CPU" not in log_text
            )
            if not placement:
                raise RuntimeError("REFUSE_DEVICE_PLACEMENT: CrispASR hybrid route differs from declared CPU+GPU")
            import psutil

            self._loaded_rss_bytes = int(psutil.Process(self._proc.pid).memory_info().rss)
            if self._loaded_rss_bytes > demand:
                raise ResourceUnavailable("loaded CrispASR RSS exceeds stage reservation")
            self._base_url = base
            self.execution_plan = {
                "backend": "external.crisp.tts.v1",
                "stage_id": self.stage_id,
                "worker_generation": self._generation,
                "worker_pid": self._proc.pid,
                "artifact_sha256": {label: digest for label, (_, digest) in expected.items()},
                "placement": f"{gpu_label} Vulkan0 talker/codec + CPU FP32 code predictor",
                "expected_gpu_name": self._expected_gpu_name,
                "ggml_vk_visible_devices": env["GGML_VK_VISIBLE_DEVICES"],
                "gpu_memory_pool": gpu_pool if gpu_memory_overhead is not None else None,
                "gpu_memory_overhead_bytes": gpu_memory_overhead,
                "loaded_rss_bytes": self._loaded_rss_bytes,
                "reserved_bytes": dict(reservation.demands),
                "memory_overhead_bytes": overhead,
                "max_text_bytes": self._max_text_bytes,
                "max_wav_bytes": self._max_wav_bytes,
                "max_audio_s": self._max_audio_s,
                "voice": self._voice,
                "seed": self._seed,
                "request_capacity": 1,
                "stateful_session": "CrispASR-owned; complete WAV per request",
                "evidence": "B",
            }
        except BaseException:
            self.shutdown()
            raise

    async def add_request_async(self, request_id: str, prompt: Any, params: Any = None) -> None:
        self.check_health()
        if self._active is not None:
            raise ResourceUnavailable("audio stage has an unacknowledged request; capacity is one")
        if not isinstance(prompt, dict) or not isinstance(prompt.get("text"), str) or not prompt["text"]:
            raise ValueError("TTS prompt must contain nonempty text")
        text = prompt["text"]
        if len(text.encode("utf-8")) > self._max_text_bytes:
            raise ResourceUnavailable("TTS text exceeds admitted request bound")
        if prompt.get("voice", self._voice) != self._voice or int(prompt.get("seed", self._seed)) != self._seed:
            raise ValueError("TTS voice and seed must match the pinned stage configuration")
        self._epoch += 1
        request = StageRequest(request_id, self.stage_id, self._epoch, self._generation)
        self._active = request_id
        self._task = asyncio.create_task(self._run(request, text), name=f"crisp-tts-{request_id}")

    async def _run(self, request: StageRequest, text: str) -> None:
        try:
            started = time.perf_counter()
            wav = await asyncio.to_thread(
                _post_wav, self._base_url + "/v1/audio/speech", text, self._seed,
                timeout=self._request_timeout_s, max_bytes=self._max_wav_bytes,
            )
            audio, pcm, frames = _decode_wav(wav, self._max_audio_s)
            wall_s = time.perf_counter() - started
            completion = CompletionOutput(0, "", [], None, None, finish_reason="stop")
            completion.multimodal_output = {"audio": audio, "sr": 24000}
            event = StageEvent(
                request.request_id, self.stage_id, request.epoch, 1, "audio",
                self._generation, terminal=True,
            )
            output = OmniRequestOutput(
                request_id=request.request_id,
                prompt=text,
                stage_id=self.stage_id,
                final_output_type="audio",
                outputs=[completion],
                _custom_output={
                    "stage_event": dataclasses.asdict(event),
                    "audio_metadata": {
                        "sample_rate_hz": 24000,
                        "channels": 1,
                        "pcm_frames": frames,
                        "duration_s": frames / 24000,
                        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                    },
                },
                metrics={self._metric_name: wall_s},
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
            self._active == request_id and epoch == self._epoch
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
        try:
            if proc is not None and proc.poll() is None:
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
        return proc is None or proc.poll() is not None

    def check_health(self) -> None:
        if self._closed or self._proc is None or self._proc.poll() is not None:
            from vllm.v1.engine.exceptions import EngineDeadError

            raise EngineDeadError()

    async def collective_rpc_async(self, method, timeout=None, args=(), kwargs=None):
        raise NotImplementedError(f"complete audio backend does not implement collective RPC {method}")

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
