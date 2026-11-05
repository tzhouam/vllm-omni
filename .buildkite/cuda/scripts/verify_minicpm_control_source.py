# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ephemeral matched-main source proof; never modifies source or serving code."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = "82fc4f511fe02bb933d01c5b5f24e1f8a9e5b415"
SNAPSHOT = "eb4141de24550a3a1a4082f718be58f46b761728"
IMAGE = (
    "936637512419.dkr.ecr.us-west-2.amazonaws.com/vllm-ci-pull-through-cache/"
    "q9t5s3a7/vllm-ci-test-repo:51586fdd276e5822932200dcff6769df70169846"
    "@sha256:a3577fe2bcf0caf6f39c3b07dae898fdfc283f81cd90705459741b53368f6145"
)
CHANGES = {
    ".buildkite/cuda/pipeline.yml",
    ".buildkite/cuda/scripts/verify_minicpm_control_source.py",
}


def write_report(output: Path, record: dict) -> None:
    output = output.resolve()
    if not output.is_relative_to(ROOT / "diagnostic-artifacts"):
        raise RuntimeError("Diagnostic output must stay in its own artifact directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "--no-optional-locks", "-C", str(ROOT), *arguments], text=True, timeout=30
    ).strip()


def snapshot() -> dict:
    cache = Path(os.environ.get("HF_HOME", ""))
    if cache != Path("/root/.cache/huggingface"):
        raise RuntimeError("Unmatched control: original HF_HOME is required")
    if os.environ.get("MODEL_PREFIX") or (ROOT / "openbmb/MiniCPM-o-4_5").is_dir():
        raise RuntimeError("Unmatched control: local model overrides are not part of the failed job")
    model = cache / "hub/models--openbmb--MiniCPM-o-4_5"
    revision = (model / "refs/main").read_text().strip()
    if revision != SNAPSHOT or not (model / "snapshots" / SNAPSHOT).is_dir():
        raise RuntimeError("Unmatched control: cached model snapshot differs or is unavailable")
    return {"model": "openbmb/MiniCPM-o-4_5", "snapshot": revision, "cache": str(cache)}


def prepare(output: Path) -> None:
    head = git("rev-parse", "HEAD")
    if os.environ.get("BUILDKITE_COMMIT") != head or git("rev-parse", "HEAD^") != BASE:
        raise RuntimeError("Control must check out its literal diagnostic commit directly on main82")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Control source must be clean before writing diagnostic artifacts")
    if set(git("diff", "--name-only", BASE, "HEAD").splitlines()) != CHANGES:
        raise RuntimeError("Only the two diagnostic files may differ from main82")
    if any(not git("ls-tree", "HEAD", "--", path).startswith("100644 blob ") for path in CHANGES):
        raise RuntimeError("Diagnostic changes must be ordinary regular source files")
    if Path(os.environ.get("PYTHONPATH", "").split(os.pathsep)[0]).resolve() != ROOT:
        raise RuntimeError("The actual main checkout must be first in inherited PYTHONPATH")
    origins = {}
    for name in (
        "vllm_omni",
        "vllm_omni.entrypoints.cli.main",
        "vllm_omni.core.sched.output",
        "vllm_omni.core.sched.omni_generation_scheduler",
        "vllm_omni.model_executor.models.registry",
        "tests.helpers.runtime",
    ):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(ROOT):
            raise RuntimeError(f"Control imported installed-image source instead of main82: {name}: {path}")
        origins[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    package = importlib.import_module("vllm_omni")
    if any(not Path(path).resolve().is_relative_to(ROOT) for path in package.__path__):
        raise RuntimeError("Source package search path escapes the actual main checkout")
    runtime = importlib.import_module("tests.helpers.runtime")
    if Path(runtime._omni_subprocess_cwd()).resolve() != ROOT:
        raise RuntimeError("Unmodified main server helper selects a different source checkout")
    dist = importlib.metadata.distribution("vllm-omni")
    write_report(
        output,
        {
            "scope": "Matched main source control; neither PR qualification nor a main-built-image claim",
            "head": head,
            "main_base": BASE,
            "tree": git("rev-parse", "HEAD^{tree}"),
            "source_root": str(ROOT),
            "expected_image": IMAGE,
            "changed_paths": sorted(CHANGES),
            "origins": origins,
            "pythonpath": os.environ["PYTHONPATH"],
            "model": snapshot(),
            "installed_distribution": {
                "version": dist.version,
                "direct_url": json.loads(dist.read_text("direct_url.json") or "null"),
                "note": "Image installation metadata is separate from the selected main source",
            },
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("vllm", "torch", "transformers", "pytest", "flashinfer-python", "lmcache")
            },
        },
    )


def process_identity(pid: int) -> tuple[int, str]:
    value = Path(f"/proc/{pid}/stat").read_text()
    fields = value[value.rindex(")") + 2 :].split()
    return int(fields[1]), fields[19]


def ancestors_to_native(pid: int, native_pid: int) -> list[int] | None:
    ancestors = [pid]
    parent = process_identity(pid)[0]
    while parent > 1 and parent not in ancestors and parent != native_pid:
        ancestors.append(parent)
        parent = process_identity(parent)[0]
    return [*ancestors, native_pid] if parent == native_pid else None


def observe(output: Path, stop_file: Path, native_pid: int) -> None:
    """Capture only a descendant of our own pytest, then exit before startup."""
    native_identity = process_identity(native_pid)
    while not stop_file.exists():
        if process_identity(native_pid) != native_identity:
            raise RuntimeError("Native pytest process identity changed")
        for directory in Path("/proc").iterdir():
            if not directory.name.isdecimal():
                continue
            try:
                command = (directory / "cmdline").read_bytes().split(b"\0")
                args = [part.decode() for part in command if part]
                if args[1:5] != ["-m", "vllm_omni.entrypoints.cli.main", "serve", "openbmb/MiniCPM-o-4_5"]:
                    continue
                pid = int(directory.name)
                identity = process_identity(pid)
                ancestors = ancestors_to_native(pid, native_pid)
                if ancestors is None:
                    continue
                if (directory / "exe").resolve(strict=True) != Path(sys.executable).resolve():
                    raise RuntimeError("Owned server launch uses a different Python executable")
                cwd = (directory / "cwd").resolve(strict=True)
                environment = (directory / "environ").read_bytes().split(b"\0")
                pythonpath = next(
                    value.partition(b"=")[2].decode() for value in environment if value.startswith(b"PYTHONPATH=")
                )
                if process_identity(pid) != identity or process_identity(native_pid) != native_identity:
                    continue
            except (FileNotFoundError, ProcessLookupError):
                continue
            if cwd != ROOT or Path(pythonpath.split(os.pathsep)[0]).resolve() != ROOT:
                raise RuntimeError("Actual owned server launch does not select ordinary main82 source")
            write_report(
                output,
                {
                    "scope": "Actual ordinary Python -m launch identity; not an intrusive sys.modules trace",
                    "pid": pid,
                    "start_ticks": identity[1],
                    "native_pytest_pid": native_pid,
                    "native_pytest_start_ticks": native_identity[1],
                    "ancestors_to_native_pytest": ancestors,
                    "command": args,
                    "cwd": str(cwd),
                    "pythonpath": pythonpath,
                    "source_root": str(ROOT),
                },
            )
            return
        time.sleep(0.5)
    raise RuntimeError("Native pytest returned without observing its owned main-source server launch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "observe", "snapshot"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--native-pid", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.output)
    elif args.mode == "snapshot":
        write_report(args.output, snapshot())
    elif args.stop_file is not None and args.native_pid is not None:
        observe(args.output, args.stop_file, args.native_pid)
    else:
        parser.error("observe requires --stop-file and --native-pid")


if __name__ == "__main__":
    main()
