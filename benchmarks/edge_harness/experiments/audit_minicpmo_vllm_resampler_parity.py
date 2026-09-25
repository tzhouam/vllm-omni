#!/usr/bin/env python3
"""Check installed vLLM and Omni MiniCPM-o resamplers against BF16 source tensors."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
from pathlib import Path
from unittest import mock

REVISION = "503e754207c94da6bb26850b4469f367c9ea3582"
SHARD_SHA = "f61addf4747c94fedcaee059e5d9918ed15543beec494404139a99f2f86c9b31"
INPUTS_SHA = "ab71cc3a6b8461c99cdf9e458b2e8b99cc092dd6adc3d49f92d643ec4b8693de"
REFERENCE_SHA = "b155827adf49819355193ca743ae03b5dee2c05c857e70bef21589094ce6d368"
CASES = ("red", "blue", "third")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_l2(reference, observed) -> float:
    import numpy as np

    a, b = reference.astype(np.float64), observed.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import torch
    import vllm
    from safetensors import safe_open
    from vllm.model_executor.models.minicpmv import Resampler4_5
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
        Resampler as OmniResampler,
    )

    if sha256(args.inputs) != INPUTS_SHA or sha256(args.reference) != REFERENCE_SHA:
        raise ValueError("source input/reference changed")
    shard = args.model_dir / "model-00004-of-00004.safetensors"
    metadata = args.model_dir / ".cache/huggingface/download" / (shard.name + ".metadata")
    if sha256(shard) != SHARD_SHA or metadata.read_text(encoding="utf-8").splitlines()[0] != REVISION:
        raise ValueError("MiniCPM-o checkpoint changed")
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    torch.set_num_threads(8)
    # ReplicatedLinear consults a TP group only at construction. Single-rank
    # stubs let this standalone audit use the unmodified vLLM layer code.
    with (mock.patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", return_value=0),
          mock.patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", return_value=1),
          mock.patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0),
          mock.patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1)):
        resampler = Resampler4_5(
            num_queries=config["query_num"], embed_dim=config["hidden_size"],
            num_heads=config["hidden_size"] // 128,
            kv_dim=config["vision_config"]["hidden_size"], prefix="resampler",
        )
    with safe_open(shard, framework="pt", device="cpu") as source:
        state = {key[len("resampler."):]: source.get_tensor(key)
                 for key in source.keys() if key.startswith("resampler.")}
    resampler.load_state_dict(state, strict=True)
    resampler.eval().to(torch.bfloat16)
    omni_resampler = OmniResampler(
        num_queries=config["query_num"], embed_dim=config["hidden_size"],
        num_heads=config["hidden_size"] // 128,
        kv_dim=config["vision_config"]["hidden_size"],
    )
    omni_resampler.load_state_dict(state, strict=True)
    omni_resampler.eval().to(torch.bfloat16)
    # vLLM's CPU model loader normally installs this after weight loading.
    resampler.kv_proj.cpu_linear = torch.nn.functional.linear
    with np.load(args.inputs, allow_pickle=False) as data:
        inputs = {name: np.ascontiguousarray(data[name]) for name in CASES}
    with np.load(args.reference, allow_pickle=False) as data:
        reference = {key: np.ascontiguousarray(data[key]) for key in data.files}
    tgt_sizes = torch.tensor([[32, 32]], dtype=torch.long)
    cases = {}
    with torch.inference_mode():
        for name in CASES:
            x = torch.from_numpy(inputs[name]).to(torch.bfloat16)
            projected, _ = resampler.kv_proj(x)
            full = resampler(x, tgt_sizes).float().numpy()
            omni_projected = omni_resampler.kv_proj(x)
            omni_full = omni_resampler(x, tgt_sizes).float().numpy()
            cases[name] = {
                "kv_projection_bitwise_equal_to_source": bool(np.array_equal(
                    projected.float().numpy(), reference[name + "_projected"])),
                "final_bitwise_equal_to_source": bool(np.array_equal(full, reference[name])),
                "final_relative_l2": relative_l2(reference[name], full),
                "omni_kv_projection_bitwise_equal_to_source": bool(np.array_equal(
                    omni_projected.float().numpy(), reference[name + "_projected"])),
                "omni_final_bitwise_equal_to_source": bool(np.array_equal(
                    omni_full, reference[name])),
                "omni_final_relative_l2": relative_l2(reference[name], omni_full),
            }
    report = {
        "scope": "installed vLLM 0.28 and source Omni MiniCPM-o 4.5 BF16 resampler parity on three fixed images",
        "model_revision": REVISION, "shard_sha256": SHARD_SHA,
        "inputs_sha256": INPUTS_SHA, "reference_sha256": REFERENCE_SHA,
        "vllm_resampler_source": inspect.getfile(Resampler4_5),
        "omni_resampler_source": inspect.getfile(OmniResampler),
        "vllm": vllm.__version__, "torch": torch.__version__,
        "platform": platform.platform(), "cases": cases,
        "status": "bitwise_pass" if all(
            case["kv_projection_bitwise_equal_to_source"]
            and case["final_bitwise_equal_to_source"]
            and case["omni_kv_projection_bitwise_equal_to_source"]
            and case["omni_final_bitwise_equal_to_source"] for case in cases.values())
        else "parity_failed",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
