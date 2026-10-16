# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Provision pinned reference code and an isolated video evaluator environment."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

from benchmarks.lingbot_world.regression import RegressionError, read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[3]


def checkout(path: Path, url: str, revision: str, *, submodules: bool = False) -> None:
    if not (path / ".git").exists():
        path.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "remote", "add", "origin", url], check=True)
        subprocess.run(["git", "-C", str(path), "fetch", "--depth=1", "origin", revision], check=True, timeout=600)
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", "FETCH_HEAD"], check=True)
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise RegressionError(f"Existing checkout is at {actual}, expected {revision}: {path}")
    if submodules:
        subprocess.run(
            ["git", "-C", str(path), "submodule", "update", "--init", "--recursive"], check=True, timeout=1200
        )


def _prepare_evaluators(root: Path) -> Path:
    lock = read_json(ROOT / "tests/dfx/lingbot_world/evaluators.lock.json")
    for name, repository in lock["repositories"].items():
        checkout(root / name, repository["url"], repository["revision"], submodules=name in {"wbench", "vbench"})
    cached_weights = os.environ.get("LINGBOT_EVALUATOR_CACHE")
    for relative, digest in lock["files"].items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            cached = Path(cached_weights) / relative if cached_weights else None
            if cached is not None and cached.is_file():
                shutil.copyfile(cached, destination)
            else:
                # Write atomically; an interrupted download must not poison a
                # shared cache or pass a later preflight.
                partial = destination.with_name(destination.name + ".partial")
                with (
                    urllib.request.urlopen(lock["urls"][relative], timeout=300) as response,
                    partial.open("wb") as handle,
                ):
                    shutil.copyfileobj(response, handle)
                if sha256(partial) != digest:
                    partial.unlink()
                    raise RegressionError(f"Downloaded asset checksum mismatch: {relative}")
                partial.replace(destination)
        if sha256(destination) != digest:
            raise RegressionError(f"Evaluator cache checksum mismatch: {relative}")
    venv = root / "venv"
    if not (venv / "bin/python").exists():
        subprocess.run(["uv", "venv", "--python", "3.12", str(venv)], check=True)
    subprocess.run(
        [
            "uv",
            "pip",
            "sync",
            "--python",
            str(venv / "bin/python"),
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu124",
            "--index-strategy",
            "unsafe-best-match",
            str(ROOT / "requirements/lingbot-quality.txt"),
        ],
        check=True,
        timeout=1800,
    )
    # Import the actual scoring stack before taking a GPU evaluator slot.
    # HPSv3 imports training helpers even during inference, so its dependencies
    # are not discovered by importing WBench's lazy evaluator alone.
    subprocess.run(
        [
            str(venv / "bin/python"),
            "-c",
            "import sys; sys.path[:0] = sys.argv[1:]; "
            "import clip, pyiqa; "
            "from vbench import VBench; "
            "from src.metrics.video_quality.evaluator import VideoQualityEvaluator; "
            "from hpsv3.inference import HPSv3RewardInferencer",
            str(root / "vbench"),
            str(root / "wbench"),
            str(root / "wbench/third_party/HPSv3"),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=True,
        timeout=180,
    )
    return venv / "bin/python"


def prepare_evaluators(root: Path) -> Path:
    from filelock import FileLock

    root.parent.mkdir(parents=True, exist_ok=True)
    # CUDA and NPU evaluation jobs can share an HF cache on the same host.
    with FileLock(str(root) + ".lock", timeout=7200):
        return _prepare_evaluators(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("reference", "evaluators"))
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-output", type=Path)
    args = parser.parse_args()
    if args.stage == "evaluators":
        print(prepare_evaluators(args.root.resolve()))
    else:
        if args.profile is None:
            parser.error("reference needs --profile")
        settings = read_json(args.profile)
        checkout(args.root.resolve(), "https://github.com/vllm-project/vllm-omni.git", settings["reference_revision"])
        model = os.environ.get("LINGBOT_MODEL_PATH")
        if model is None:
            lock = read_json(ROOT / "tests/dfx/lingbot_world/assets.lock.json")
            # Use the CLI so runtime hub access remains owned by the existing
            # model-loading stack, not another Python Hub API wrapper.
            model = (
                subprocess.check_output(
                    ["hf", "download", lock["model_id"], "--revision", lock["model_revision"], "--quiet"],
                    text=True,
                    timeout=7200,
                )
                .strip()
                .splitlines()[-1]
            )
        if args.env_output:
            write_json(args.env_output, {"model_path": str(Path(model).resolve())})


if __name__ == "__main__":
    main()
