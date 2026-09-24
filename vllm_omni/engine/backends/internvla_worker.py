# SPDX-License-Identifier: Apache-2.0
"""InternVLA action worker on CPU/CUDA with optional local encoder backends.

The policy remains the existing Omni diffusion implementation. Requests carry
explicit normalized camera histories and state; the worker owns policy state
and, for hybrid routes, DirectML or a pinned ORT/VitisAI encoder component.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
import time
import traceback
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--cosmos-dir", type=Path, required=True)
    parser.add_argument("--placement", choices=("cpu", "cuda", "radeon-cosmos", "amd-npu-conv13", "amd-npu-radeon-cosmos"), required=True)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--graph-sha256")
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--prefix-sha256")
    parser.add_argument("--suffix", type=Path)
    parser.add_argument("--suffix-sha256")
    parser.add_argument("--ep-dir", type=Path)
    parser.add_argument("--ep-dll-sha256")
    parser.add_argument("--npu-profile-prefix", type=Path)
    parser.add_argument("--dml-python", type=Path)
    parser.add_argument("--dml-python-sha256")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-input-bytes", type=int, default=8 << 20)
    args = parser.parse_args()
    if not 0 < args.port < 65536 or not 0 < args.threads <= 24 or args.max_input_bytes <= 0:
        parser.error("invalid port, threads or input bound")
    if args.placement == "radeon-cosmos" and (args.graph is None or not args.graph_sha256):
        parser.error("Radeon route requires a graph and hash")
    if args.placement in {"amd-npu-conv13", "amd-npu-radeon-cosmos"} and not all((
        args.graph, args.graph_sha256, args.prefix, args.prefix_sha256,
        args.suffix, args.suffix_sha256, args.ep_dir, args.ep_dll_sha256,
    )):
        parser.error("AMD NPU route requires pinned graph, prefix, suffix and EP directory")
    if args.placement == "amd-npu-radeon-cosmos" and not all((
        args.dml_python, args.dml_python_sha256,
    )):
        parser.error("joint AMD route requires a pinned DirectML interpreter")
    expected_visible = "0" if args.placement == "cuda" else ""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visible:
        raise RuntimeError("InternVLA CUDA visibility differs from the placement plan")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("InternVLA worker requires offline checkpoint loading")

    model_dir = args.model_dir.resolve(strict=True)
    processor_dir = args.processor_dir.resolve(strict=True)
    cosmos_dir = args.cosmos_dir.resolve(strict=True)
    os.environ["INTERNVLA_A1_COSMOS_DIR"] = str(cosmos_dir)
    os.environ["INTERNVLA_A1_PROCESSOR_DIR"] = str(processor_dir)

    import numpy as np
    import torch

    if args.placement == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("InternVLA CUDA placement requested but CUDA is unavailable")

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.registry import initialize_model
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    torch.set_num_threads(args.threads)
    policy_device = "cuda" if args.placement == "cuda" else "cpu"
    config = OmniDiffusionConfig(
        model=str(model_dir), model_class_name="InternVLAA1Pipeline", dtype=torch.bfloat16,
        custom_pipeline_args={
            "device": policy_device, "dtype": "bfloat16", "compile_model": False,
            "enable_regional_compile": False, "enable_warmup": False,
            "strict_load": True, "processor_model_name": str(processor_dir),
        },
    )
    started = time.perf_counter()
    pipeline = initialize_model(config)
    policy_load_s = time.perf_counter() - started
    if pipeline.runtime_mode() != "real_checkpoint_loaded":
        raise RuntimeError("InternVLA real checkpoint did not load")
    if {parameter.device.type for parameter in pipeline.policy.parameters()} != {policy_device}:
        raise RuntimeError("InternVLA policy parameters differ from the placement plan")

    external = None
    external_report = None
    npu_report = None
    if args.placement == "radeon-cosmos":
        from vllm_omni.edge.local.external.client import ExternalWorker
        from vllm_omni.edge.local.external.launch import ROUTE_TORCH_DML, resolve

        graph = args.graph.resolve(strict=True)
        if _sha256(graph) != args.graph_sha256.lower():
            raise RuntimeError("Radeon Cosmos graph hash differs from plan")
        route = resolve(ROUTE_TORCH_DML)
        if not route.available:
            raise RuntimeError(f"DirectML worker route unavailable: {route.reason}")
        external = ExternalWorker(route, max_payload_bytes=args.max_input_bytes)
        try:
            external.start()
            external_report = external.load(
                graph, example_inputs={"pixels": np.zeros((6, 3, 256, 256), dtype=np.float32)}
            )
            if (
                external_report.device_name != "AMD Radeon(TM) 890M Graphics"
                or external_report.placement_granularity != "output_device"
                or external_report.fraction_on_target != 1.0
                or tuple(external_report.outputs[0]["shape"]) != (6, 16, 32, 32)
            ):
                raise RuntimeError("Radeon Cosmos output placement/shape differs from plan")

            class RemoteEncoder(torch.nn.Module):
                def forward(self, pixels: torch.Tensor) -> torch.Tensor:
                    if pixels.device.type != "cpu" or tuple(pixels.shape) != (6, 3, 256, 256):
                        raise ValueError("Cosmos input differs from fixed six-frame CPU contract")
                    output, _ = external.run({"pixels": pixels.detach().float().contiguous().numpy()})
                    latent = output["out_0"]
                    if latent.shape != (6, 16, 32, 32) or not np.isfinite(latent).all():
                        raise RuntimeError("Radeon Cosmos returned invalid latent")
                    return torch.from_numpy(latent).to(pixels.dtype)

            pipeline.policy.model.cosmos._enc_model = RemoteEncoder()
        except BaseException:
            external.close()
            raise

    if args.placement in {"amd-npu-conv13", "amd-npu-radeon-cosmos"}:
        import onnxruntime as ort

        paths = {
            "graph": (args.graph, args.graph_sha256),
            "prefix": (args.prefix, args.prefix_sha256),
            "suffix": (args.suffix, args.suffix_sha256),
        }
        for name, (path, digest) in paths.items():
            if _sha256(path.resolve(strict=True)) != digest.lower():
                raise RuntimeError(f"AMD NPU {name} graph hash differs from plan")
        ep_dir = args.ep_dir.resolve(strict=True)
        ep_dll = ep_dir / "onnxruntime_vitisai_ep.dll"
        if not ep_dll.is_file():
            raise FileNotFoundError(ep_dll)
        if _sha256(ep_dll) != args.ep_dll_sha256.lower():
            raise RuntimeError("VitisAI EP DLL hash differs from plan")
        os.environ["PATH"] = str(ep_dir) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(ep_dir))
        ort.register_execution_provider_library("vitisai", str(ep_dll))
        devices = [device for device in ort.get_ep_devices()
                   if device.ep_name == "vitisai" and str(device.device.type).endswith("NPU")]
        if not devices:
            raise RuntimeError("VitisAI NPU device unavailable")
        prefix = ort.InferenceSession(str(args.prefix), providers=["CPUExecutionProvider"])
        suffix = ort.InferenceSession(str(args.suffix), providers=["CPUExecutionProvider"])
        if ([item.name for item in prefix.get_inputs()] != ["pixels"]
                or [item.name for item in prefix.get_outputs()] != ["group_norm"]
                or {item.name for item in suffix.get_inputs()} != {"pixels", "conv2d_13"}
                or [item.name for item in suffix.get_outputs()] != ["latent"]):
            raise RuntimeError("Cosmos encoder graph boundary differs from plan")
        options = ort.SessionOptions()
        options.add_provider_for_devices(devices, {})
        options.enable_profiling = True
        options.profile_file_prefix = str(
            args.npu_profile_prefix or Path(tempfile.gettempdir()) / f"internvla_npu_placement_{os.getpid()}"
        )
        npu = ort.InferenceSession(str(args.graph), sess_options=options)
        if ([item.name for item in npu.get_inputs()] != ["group_norm"]
                or [item.name for item in npu.get_outputs()] != ["conv2d_13"]):
            raise RuntimeError("NPU Conv13 graph boundary differs from plan")
        warm = np.zeros((1, 128, 64, 64), dtype=np.float32)
        npu.run(None, {"group_norm": warm})
        profile = Path(npu.end_profiling())
        events = json.loads(profile.read_text(encoding="utf-8"))
        placed = sum(event.get("cat") == "Node" and
                     (event.get("args") or {}).get("provider") == "vitisai"
                     for event in events)
        if placed < 1:
            raise RuntimeError("AMD NPU Conv13 has no VitisAI node event")
        npu_report = {"device_type": str(devices[0].device.type), "provider": "vitisai",
                      "warmup_npu_node_events": placed, "onnxruntime": ort.__version__,
                      "profile_path": str(profile) if args.npu_profile_prefix else None}
        if args.npu_profile_prefix is None:
            profile.unlink(missing_ok=True)

        if args.placement == "amd-npu-radeon-cosmos":
            from vllm_omni.edge.local.external.client import ExternalWorker
            from vllm_omni.edge.local.external.launch import ROUTE_DML, resolve

            dml_python = args.dml_python.resolve(strict=True)
            if _sha256(dml_python) != args.dml_python_sha256.lower():
                raise RuntimeError("DirectML interpreter hash differs from plan")
            os.environ["VLLM_OMNI_EXTERNAL_PYTHON_ORT_DML"] = str(dml_python)
            route = resolve(ROUTE_DML)
            if not route.available or Path(route.interpreter).resolve() != dml_python:
                raise RuntimeError(f"pinned DirectML route unavailable: {route.reason}")
            external = ExternalWorker(route, max_payload_bytes=64 << 20)
            try:
                external.start()
                external_report = external.load(
                    args.suffix.resolve(strict=True),
                    example_inputs={
                        "pixels": np.zeros((6, 3, 256, 256), dtype=np.float32),
                        "conv2d_13": np.zeros((6, 256, 64, 64), dtype=np.float32),
                    },
                    device_id=1,
                )
                if (external_report.device_id_requested != 1
                        or external_report.node_counts.get("DmlExecutionProvider", 0) < 1
                        or not external_report.fraction_on_target
                        or {item["name"] for item in external_report.inputs} != {"pixels", "conv2d_13"}
                        or tuple(external_report.outputs[0]["shape"]) != (6, 16, 32, 32)):
                    raise RuntimeError("Radeon suffix device, placement or graph contract differs")
            except BaseException:
                external.close()
                raise

        class NpuConv13Encoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.last_timing: dict[str, float] | None = None

            def forward(self, pixels: torch.Tensor) -> torch.Tensor:
                if pixels.device.type != "cpu" or tuple(pixels.shape) != (6, 3, 256, 256):
                    raise ValueError("Cosmos input differs from fixed six-frame CPU contract")
                values = np.ascontiguousarray(pixels.detach().float().numpy())
                started = time.perf_counter()
                boundary = prefix.run(None, {"pixels": values})[0]
                prefix_s = time.perf_counter() - started
                if boundary.shape != (6, 128, 64, 64):
                    raise RuntimeError("Cosmos prefix boundary shape changed")
                started = time.perf_counter()
                parts = [npu.run(None, {"group_norm": np.ascontiguousarray(boundary[i:i + 1])})[0]
                         for i in range(6)]
                conv = np.concatenate(parts, axis=0)
                npu_s = time.perf_counter() - started
                if external is not None:
                    output, timing = external.run({"pixels": values, "conv2d_13": conv})
                    latent = output["latent"]
                    self.last_timing = {
                        "cosmos_cpu_prefix_s": prefix_s,
                        "cosmos_npu_conv_s": npu_s,
                        "cosmos_dml_worker_s": timing.worker_s,
                        "cosmos_dml_round_trip_s": timing.round_trip_s,
                    }
                else:
                    latent = suffix.run(None, {"pixels": values, "conv2d_13": conv})[0]
                if latent.shape != (6, 16, 32, 32) or not np.isfinite(latent).all():
                    raise RuntimeError("AMD NPU Cosmos returned invalid latent")
                return torch.from_numpy(latent).to(pixels.dtype)

        npu_encoder = NpuConv13Encoder()
        pipeline.policy.model.cosmos._enc_model = npu_encoder

    props = {
        "model_dir": str(model_dir), "processor_dir": str(processor_dir),
        "cosmos_dir": str(cosmos_dir), "placement": args.placement,
        "torch": torch.__version__, "policy_dtype": "bfloat16",
        "cosmos_dtype": "float32" if external is not None or npu_report is not None else "bfloat16",
        "policy_device": policy_device, "runtime_mode": pipeline.runtime_mode(),
        "cuda_device_name": torch.cuda.get_device_name(0) if policy_device == "cuda" else None,
        "cuda_device_index": torch.cuda.current_device() if policy_device == "cuda" else None,
        "cuda_allocated_after_load_bytes": torch.cuda.memory_allocated(0) if policy_device == "cuda" else None,
        "cuda_reserved_after_load_bytes": torch.cuda.memory_reserved(0) if policy_device == "cuda" else None,
        "policy_load_s": policy_load_s, "action_shape": [1, 50, 32],
        "image_shape": [1, 2, 3, 224, 224], "state_shape": [1, 32],
        "action_mode": "delta", "action_units": "unverified",
        "action_joint_order": "unverified", "action_step_s": None,
        "control_ready": False,
        "external_load": external_report.to_dict() if external_report else None,
        "npu_load": npu_report,
    }

    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, value: dict) -> None:
            body = json.dumps(value).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(200, {"status": "ok"})
            elif self.path == "/props":
                self._json(200, props)
            else:
                self._json(404, {"error": "unknown endpoint"})

        def do_POST(self) -> None:
            if self.path != "/v1/actions":
                self._json(404, {"error": "unknown endpoint"})
                return
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= args.max_input_bytes:
                self._json(413, {"error": "observation exceeds admitted input bound"})
                return
            try:
                raw = self.rfile.read(size)
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    if (len(archive.infolist()) != 11 or
                        any(info.compress_type != zipfile.ZIP_STORED for info in archive.infolist()) or
                        sum(info.file_size for info in archive.infolist()) > args.max_input_bytes):
                        raise ValueError("observation archive exceeds uncompressed bound")
                with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                    expected = {
                        "image0", "image1", "image2", "mask0", "mask1", "mask2",
                        "state", "task", "noise", "observation_timestamp_ns", "request_id",
                    }
                    if set(data.files) != expected:
                        raise ValueError("observation fields differ from the declared contract")
                    images = [np.asarray(data[f"image{i}"]) for i in range(3)]
                    masks = [np.asarray(data[f"mask{i}"]) for i in range(3)]
                    state = np.asarray(data["state"])
                    noise = np.asarray(data["noise"])
                    task = str(data["task"].item())
                    request_id = str(data["request_id"].item())
                    observation_ns = int(data["observation_timestamp_ns"].item())
                if len(task.encode("utf-8")) > 4096 or not request_id or observation_ns <= 0:
                    raise ValueError("invalid task, request identity or observation time")
                for image in images:
                    if (image.dtype != np.float32 or image.shape != (1, 2, 3, 224, 224)
                        or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1):
                        raise ValueError("camera history must be finite normalized float32 [1,2,3,224,224]")
                for mask in masks:
                    if mask.dtype != np.bool_ or mask.shape != (1,):
                        raise ValueError("camera mask must be bool [1]")
                if state.dtype != np.float32 or state.shape != (1, 32) or not np.isfinite(state).all():
                    raise ValueError("state must be finite float32 [1,32]")
                if noise.dtype != np.float32 or noise.shape != (1, 50, 32) or not np.isfinite(noise).all():
                    raise ValueError("noise must be finite float32 [1,50,32]")
                batch = {"observation.state": torch.from_numpy(state.copy()).to(device=policy_device, dtype=torch.bfloat16),
                         "observation.task": [task]}
                for i in range(3):
                    batch[f"observation.images.image{i}"] = torch.from_numpy(images[i].copy()).to(device=policy_device, dtype=torch.bfloat16)
                    batch[f"observation.images.image{i}_mask"] = torch.from_numpy(masks[i].copy()).to(policy_device)
                started = time.perf_counter()
                if args.placement == "amd-npu-radeon-cosmos":
                    npu_encoder.last_timing = None
                result = pipeline.forward(DiffusionRequestBatch(requests=[
                    OmniDiffusionRequest(
                        prompt="",
                        sampling_params=OmniDiffusionSamplingParams(extra_args={
                            "batch_inputs": batch, "noise": torch.from_numpy(noise.copy()).to(policy_device),
                            "decode_image": False,
                        }),
                        request_id=request_id,
                    )
                ]))
                if result.error:
                    raise RuntimeError(result.error)
                actions = result.output["payload"]["actions"]
                if actions.device.type != policy_device or tuple(actions.shape) != (1, 50, 32):
                    raise RuntimeError("policy action device or shape differs from plan")
                values = actions.detach().float().cpu().contiguous().numpy()
                if policy_device == "cuda":
                    torch.cuda.synchronize(0)
                cuda_peak_reserved = torch.cuda.max_memory_reserved(0) if policy_device == "cuda" else 0
                if not np.isfinite(values).all():
                    raise RuntimeError("policy returned nonfinite actions")
                with io.BytesIO() as out:
                    fields = {
                        "actions": values,
                        "observation_timestamp_ns": np.int64(observation_ns),
                        "generation_timestamp_ns": np.int64(time.time_ns()),
                        "request_id": np.array(request_id),
                        "worker_wall_s": np.float64(time.perf_counter() - started),
                        "cuda_peak_reserved_bytes": np.int64(cuda_peak_reserved),
                    }
                    if args.placement == "amd-npu-radeon-cosmos":
                        if npu_encoder.last_timing is None:
                            raise RuntimeError("joint Cosmos encoder did not report its live handoff")
                        fields.update({key: np.float64(value)
                                       for key, value in npu_encoder.last_timing.items()})
                    np.savez(out, **fields)
                    body = out.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                traceback.print_exc()
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    try:
        with HTTPServer(("127.0.0.1", args.port), Handler) as server:
            print(f"internvla-worker ready port={args.port} props={json.dumps(props)}", flush=True)
            server.serve_forever(poll_interval=0.1)
    finally:
        if external is not None:
            external.close()


if __name__ == "__main__":
    main()
