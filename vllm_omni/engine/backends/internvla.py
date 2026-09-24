# SPDX-License-Identifier: Apache-2.0
"""Bounded whole-policy InternVLA stage with explicit local placements.

The existing Omni diffusion policy owns action semantics. StageRuntime owns
shared-RAM admission, one in-flight request, terminal output and cancellation.
The isolated worker is retired on an in-flight abort; no state is replayed.
"""

from __future__ import annotations

import asyncio
import dataclasses
import io
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from omni_stage_contracts import BufferRef, StageEvent, StageRequest

from vllm_omni.engine.resource_ledger import ResourceUnavailable
from vllm_omni.outputs import OmniRequestOutput

from .crisp_tts import CrispTTSStageClient, _local_port, _sha256


def _post_actions(url: str, body: bytes, *, timeout: float, max_bytes: int) -> bytes:
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/octet-stream"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"InternVLA worker returned HTTP {response.status}")
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ResourceUnavailable("InternVLA action response exceeds admitted bound")
    return payload


class InternVLAStageClient(CrispTTSStageClient):
    """One complete action request under Omni's graph-stage control path."""

    def __init__(self, metadata, config: dict, ledger, reservation) -> None:
        for name, value in vars(metadata).items():
            setattr(self, name, value)
        self.stage_type = "graph"
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
        self._max_input_bytes = int(config.get("max_input_bytes", 8 << 20))
        self._max_action_bytes = int(config.get("max_action_bytes", 1 << 20))
        self._request_timeout_s = float(config.get("request_timeout_s", 60))
        self._port = int(config.get("port") or _local_port())
        self._placement = str(config.get("placement", ""))
        self._cuda_demand_bytes = reservation.demands.get("cuda:0", 0)
        try:
            if (
                self._placement not in {"cpu", "cuda", "radeon-cosmos", "amd-npu-conv13"} or self._max_input_bytes <= 0
                or self._max_action_bytes <= 0 or self._request_timeout_s <= 0
                or not 0 < self._port < 65536
            ):
                raise ValueError("invalid InternVLA placement, request bounds, timeout or port")
            if self._memory_pool not in reservation.demands:
                raise ResourceUnavailable("InternVLA shared-RAM pool absent from stage reservation")
            demand = reservation.demands[self._memory_pool]
            if self._placement == "cuda" and reservation.demands.get("cuda:0", 0) <= 0:
                raise ResourceUnavailable("InternVLA CUDA placement requires an explicit cuda:0 reservation")
            if self._max_input_bytes + self._max_action_bytes > demand:
                raise ResourceUnavailable("InternVLA I/O bound exceeds stage reservation")

            self._python = Path(config["python_bin"]).absolute()
            if not self._python.is_file():
                raise FileNotFoundError(f"InternVLA interpreter absent: {self._python}")
            self._model_dir = Path(config["model_dir"]).resolve(strict=True)
            self._processor_dir = Path(config["processor_dir"]).resolve(strict=True)
            self._cosmos_dir = Path(config["cosmos_dir"]).resolve(strict=True)
            self._graph = Path(config["graph_file"]).resolve(strict=True) if self._placement == "radeon-cosmos" else None
            if self._placement == "amd-npu-conv13":
                self._graph = Path(config["graph_file"]).resolve(strict=True)
                self._prefix = Path(config["prefix_file"]).resolve(strict=True)
                self._suffix = Path(config["suffix_file"]).resolve(strict=True)
                self._ep_dll = Path(config["ep_dir"]).resolve(strict=True) / "onnxruntime_vitisai_ep.dll"
            else:
                self._prefix = self._suffix = self._ep_dll = None
            self._worker = Path(__file__).with_name("internvla_worker.py").resolve(strict=True)
            artifacts = {
                "python": self._python,
                "model": self._model_dir / "model.safetensors",
                "model_config": self._model_dir / "config.json",
                "train_config": self._model_dir / "train_config.json",
                "stats": self._model_dir / "stats.json",
                "cosmos_encoder": self._cosmos_dir / "encoder.safetensors",
                "cosmos_decoder": self._cosmos_dir / "decoder.safetensors",
                "processor_tokenizer": self._processor_dir / "tokenizer.json",
                "processor_config": self._processor_dir / "preprocessor_config.json",
            }
            if self._graph is not None:
                artifacts["graph"] = self._graph
            if self._prefix is not None:
                artifacts.update(prefix=self._prefix, suffix=self._suffix, ep_dll=self._ep_dll)
            hashes = dict(config["artifact_sha256"])
            if set(hashes) != set(artifacts):
                raise ValueError("InternVLA artifact manifest has missing or unexpected entries")
            for name, path in artifacts.items():
                if _sha256(path) != str(hashes[name]).lower():
                    raise ValueError(f"InternVLA {name} differs from declared artifact hash")
            overhead = int(config["memory_overhead_bytes"])
            artifact_bytes = sum(path.stat().st_size for name, path in artifacts.items() if name != "python")
            if overhead <= 0 or artifact_bytes + overhead > demand:
                raise ResourceUnavailable("InternVLA weights, graph and state/workspace/headroom exceed reservation")
            cuda_min_bytes = 0
            if self._placement == "cuda":
                cuda_min_bytes = (artifacts["model"].stat().st_size
                                  + artifacts["cosmos_encoder"].stat().st_size + overhead)
                if cuda_min_bytes > self._cuda_demand_bytes:
                    raise ResourceUnavailable("InternVLA CUDA weights and workspace/headroom exceed reservation")

            self._log_path = Path(config["log_file"]).resolve()
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_stream = self._log_path.open("wb")
            env = os.environ.copy()
            repo_root = Path(__file__).resolve().parents[3]
            env.update({
                "PYTHONPATH": str(repo_root) + os.pathsep + env.get("PYTHONPATH", ""),
                "CUDA_VISIBLE_DEVICES": "0" if self._placement == "cuda" else "",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "PYTHONUTF8": "1",
                "TOKENIZERS_PARALLELISM": "false",
            })
            command = [
                str(self._python), "-u", str(self._worker),
                "--model-dir", str(self._model_dir),
                "--processor-dir", str(self._processor_dir),
                "--cosmos-dir", str(self._cosmos_dir),
                "--placement", self._placement,
                "--port", str(self._port), "--threads", "8",
                "--max-input-bytes", str(self._max_input_bytes),
            ]
            if self._graph is not None:
                command += ["--graph", str(self._graph), "--graph-sha256", hashes["graph"]]
            if self._prefix is not None:
                command += [
                    "--prefix", str(self._prefix), "--prefix-sha256", hashes["prefix"],
                    "--suffix", str(self._suffix), "--suffix-sha256", hashes["suffix"],
                    "--ep-dir", str(self._ep_dll.parent), "--ep-dll-sha256", hashes["ep_dll"],
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
                    raise RuntimeError(f"InternVLA worker exited during load; see {self._log_path}")
                try:
                    with urllib.request.urlopen(base + "/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError, OSError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"InternVLA worker did not become ready; see {self._log_path}")
                time.sleep(.2)
            with urllib.request.urlopen(base + "/props", timeout=5) as response:
                props = json.load(response)
            expected_external = self._placement == "radeon-cosmos"
            observed_external = props.get("external_load")
            if (
                props.get("placement") != self._placement
                or Path(props.get("model_dir", "")).resolve() != self._model_dir
                or Path(props.get("processor_dir", "")).resolve() != self._processor_dir
                or Path(props.get("cosmos_dir", "")).resolve() != self._cosmos_dir
                or props.get("policy_device") != ("cuda" if self._placement == "cuda" else "cpu")
                or props.get("runtime_mode") != "real_checkpoint_loaded"
                or props.get("policy_dtype") != "bfloat16"
                or props.get("cosmos_dtype") != ("float32" if expected_external or self._placement == "amd-npu-conv13" else "bfloat16")
                or props.get("action_shape") != [1, 50, 32]
                or props.get("action_mode") != "delta"
                or props.get("control_ready") is not False
                or props.get("torch") != str(config["expected_torch"])
                or bool(observed_external) != expected_external
                or (expected_external and observed_external.get("device_name") != "AMD Radeon(TM) 890M Graphics")
                or bool(props.get("npu_load")) != (self._placement == "amd-npu-conv13")
                or (self._placement == "amd-npu-conv13" and (
                    props["npu_load"].get("provider") != "vitisai"
                    or props["npu_load"].get("warmup_npu_node_events", 0) < 1
                ))
                or (self._placement == "cuda" and (
                    props.get("cuda_device_name") != config.get("expected_cuda_device_name")
                    or props.get("cuda_device_index") != 0
                    or not 0 < props.get("cuda_reserved_after_load_bytes", 0) <= reservation.demands["cuda:0"]
                ))
            ):
                raise RuntimeError("InternVLA worker model, precision or device differs from plan")
            import psutil

            root = psutil.Process(self._proc.pid)
            descendants = [root, *root.children(recursive=True)]
            self._loaded_process_tree = []
            for process in descendants:
                try:
                    memory = process.memory_full_info()
                    self._loaded_process_tree.append({
                        "pid": process.pid, "rss_bytes": int(memory.rss),
                        "private_bytes": int(getattr(memory, "private", memory.rss)),
                    })
                except psutil.NoSuchProcess:
                    pass
            self._loaded_rss_bytes = sum(row["rss_bytes"] for row in self._loaded_process_tree)
            if self._loaded_rss_bytes > demand:
                raise ResourceUnavailable("loaded InternVLA process tree exceeds stage reservation")
            self._base_url = base
            self.execution_plan = {
                "backend": "external.internvla.policy.v1", "stage_id": self.stage_id,
                "worker_generation": self._generation, "worker_pid": self._proc.pid,
                "worker_script_sha256": _sha256(self._worker),
                "artifact_sha256": hashes, "placement": self._placement,
                "worker_props": props, "loaded_rss_bytes": self._loaded_rss_bytes,
                "loaded_process_tree": self._loaded_process_tree,
                "reserved_bytes": dict(reservation.demands),
                "memory_overhead_bytes": overhead,
                "cuda_weights_and_headroom_min_bytes": cuda_min_bytes,
                "max_input_bytes": self._max_input_bytes,
                "max_action_bytes": self._max_action_bytes,
                "request_capacity": 1,
                "stateful_session": "InternVLA policy worker-owned; complete action chunk per request",
                "evidence": "B",
            }
        except BaseException:
            self.shutdown()
            raise

    @staticmethod
    def _encode_prompt(request_id: str, prompt: Any) -> bytes:
        if not isinstance(prompt, dict):
            raise ValueError("InternVLA prompt must contain camera histories and state")
        expected = {"image0", "image1", "image2", "mask0", "mask1", "mask2",
                    "state", "task", "noise", "observation_timestamp_ns"}
        if set(prompt) != expected:
            raise ValueError("InternVLA observation fields differ from the declared contract")
        arrays: dict[str, Any] = {}
        for i in range(3):
            image = np.asarray(prompt[f"image{i}"])
            mask = np.asarray(prompt[f"mask{i}"])
            if (
                image.dtype != np.float32 or image.shape != (1, 2, 3, 224, 224)
                or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1
            ):
                raise ValueError("camera history must be finite normalized float32 [1,2,3,224,224]")
            if mask.dtype != np.bool_ or mask.shape != (1,):
                raise ValueError("camera mask must be bool [1]")
            arrays[f"image{i}"] = image
            arrays[f"mask{i}"] = mask
        for name, shape in (("state", (1, 32)), ("noise", (1, 50, 32))):
            value = np.asarray(prompt[name])
            if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite float32 {shape}")
            arrays[name] = value
        task = prompt["task"]
        timestamp = prompt["observation_timestamp_ns"]
        if not isinstance(task, str) or len(task.encode("utf-8")) > 4096:
            raise ValueError("task must be a bounded string")
        if type(timestamp) is not int or timestamp <= 0:
            raise ValueError("observation timestamp must be positive Unix nanoseconds")
        arrays.update(task=np.array(task), observation_timestamp_ns=np.int64(timestamp),
                      request_id=np.array(request_id))
        with io.BytesIO() as output:
            np.savez(output, **arrays)
            return output.getvalue()

    async def add_request_async(self, request_id: str, prompt: Any, params: Any = None) -> None:
        self.check_health()
        if self._active is not None:
            raise ResourceUnavailable("InternVLA stage has an unacknowledged request; capacity is one")
        body = self._encode_prompt(request_id, prompt)
        if len(body) > self._max_input_bytes:
            raise ResourceUnavailable("InternVLA observation exceeds admitted input bound")
        self._epoch += 1
        request = StageRequest(request_id, self.stage_id, self._epoch, self._generation)
        self._active = request_id
        self._task = asyncio.create_task(self._run(request, body), name=f"internvla-{request_id}")

    async def _run(self, request: StageRequest, body: bytes) -> None:
        try:
            started = time.perf_counter()
            data = await asyncio.to_thread(
                _post_actions, self._base_url + "/v1/actions", body,
                timeout=self._request_timeout_s, max_bytes=self._max_action_bytes,
            )
            with np.load(io.BytesIO(data), allow_pickle=False) as response:
                if set(response.files) != {"actions", "observation_timestamp_ns",
                                          "generation_timestamp_ns", "request_id", "worker_wall_s",
                                          "cuda_peak_reserved_bytes"}:
                    raise ValueError("InternVLA response fields differ from declared action contract")
                actions = np.asarray(response["actions"])
                observation_ns = int(response["observation_timestamp_ns"].item())
                generation_ns = int(response["generation_timestamp_ns"].item())
                returned_id = str(response["request_id"].item())
                worker_wall_s = float(response["worker_wall_s"].item())
                cuda_peak_reserved = int(response["cuda_peak_reserved_bytes"].item())
            if (returned_id != request.request_id or actions.dtype != np.float32
                or actions.shape != (1, 50, 32) or not np.isfinite(actions).all()
                or generation_ns < observation_ns):
                raise ValueError("InternVLA worker returned invalid, stale or misrouted actions")
            if self._placement == "cuda" and not 0 < cuda_peak_reserved <= self._cuda_demand_bytes:
                raise ResourceUnavailable("InternVLA CUDA peak reservation exceeded the admitted VRAM budget")
            if self._placement != "cuda" and cuda_peak_reserved != 0:
                raise ValueError("InternVLA non-CUDA worker reported CUDA allocation")
            ref = BufferRef(
                "actions", str(self.stage_id), self._generation, "float32", tuple(actions.shape), int(actions.nbytes)
            )
            event = StageEvent(
                request.request_id, self.stage_id, request.epoch, 1, "action", self._generation,
                (ref,), terminal=True,
            )
            output = OmniRequestOutput(
                request_id=request.request_id, stage_id=self.stage_id,
                final_output_type="action",
                _custom_output={
                    "actions": actions.copy(), "stage_event": dataclasses.asdict(event),
                    "action_metadata": {
                        "observation_timestamp_ns": observation_ns,
                        "generation_timestamp_ns": generation_ns,
                        "observation_age_s": (generation_ns - observation_ns) / 1e9,
                        "action_mode": "delta", "action_shape": [1, 50, 32],
                        "action_units": "unverified", "joint_order": "unverified",
                        "action_step_s": None, "control_ready": False,
                    },
                },
                metrics={"policy_wall_s": time.perf_counter() - started,
                         "worker_wall_s": worker_wall_s,
                         "cuda_peak_reserved_bytes": cuda_peak_reserved},
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

    def _terminate(self) -> bool:
        """Retire both the policy server and its possible DirectML child."""
        import psutil

        children = []
        try:
            if self._proc is not None:
                children = psutil.Process(self._proc.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            pass
        except psutil.Error:
            return False
        for child in reversed(children):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
            except psutil.Error:
                return False
        # The policy process is the worker's parent and may be the one that
        # reaps it. Retire the parent before deciding a transient zombie means
        # the shared-RAM reservation must remain quarantined.
        parent_drained = super()._terminate()
        _, alive = psutil.wait_procs(children, timeout=5)
        for child in alive:
            try:
                if child.status() != psutil.STATUS_ZOMBIE:
                    child.kill()
            except psutil.NoSuchProcess:
                pass
            except psutil.Error:
                return False
        _, alive = psutil.wait_procs(alive, timeout=5)
        for child in alive:
            try:
                if child.status() != psutil.STATUS_ZOMBIE:
                    return False
            except psutil.NoSuchProcess:
                pass
            except psutil.Error:
                return False
        return parent_drained
