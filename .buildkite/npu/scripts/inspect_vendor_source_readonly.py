#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One-off source/image diagnosis; no runtime imports, devices, models or writes to vendor trees."""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import stat
import subprocess
import sys
import sysconfig
from pathlib import Path


def digest_file(path):
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode):
        value = os.readlink(str(path)).encode()
        return {"type": "symlink", "target": value.decode(), "sha256": hashlib.sha256(value).hexdigest()}
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("Not a regular file: " + str(path))
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    after = path.lstat()

    def identity(s):
        return (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

    if identity(before) != identity(after):
        raise ValueError("File changed during hash: " + str(path))
    return {
        "type": "regular",
        "bytes": before.st_size,
        "mode": oct(stat.S_IMODE(before.st_mode)),
        "sha256": h.hexdigest(),
        "stable_identity": True,
    }


def inspect_source(root):
    commands = []
    env = dict(
        os.environ,
        GIT_OPTIONAL_LOCKS="0",
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_CONFIG_SYSTEM="/dev/null",
        GIT_CONFIG_COUNT="0",
    )

    def git(*arguments, binary=False):
        command = ["git", "--no-optional-locks", "-c", "safe.directory=" + str(root), "-C", str(root), *arguments]
        result = subprocess.run(command, env=env, capture_output=True, timeout=90)
        commands.append(
            {
                "arguments": list(arguments),
                "exit_code": result.returncode,
                "stderr": result.stderr.decode("utf-8", "backslashreplace"),
            }
        )
        if result.returncode:
            raise ValueError("Read-only Git failed: " + json.dumps(commands[-1]))
        return result.stdout if binary else result.stdout.decode("utf-8", "backslashreplace")

    head = git("rev-parse", "HEAD").strip()
    tree = git("rev-parse", "HEAD^{tree}").strip()
    initial_status = git("status", "--porcelain=v1", "-z", "--untracked-files=all", binary=True)
    index_bytes = git("ls-files", "--stage", "-z", binary=True)
    tracked = {}
    for record in index_bytes.split(b"\0"):
        if not record:
            continue
        metadata, name = record.split(b"\t", 1)
        mode, blob, stage = metadata.decode().split()
        relative = os.fsdecode(name)
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Unsafe tracked path")
        path = root / relative
        if mode == "160000":
            populated = (path / ".git").exists()
            worktree = {"type": "gitlink", "populated": populated}
            if populated:
                worktree["head"] = git("-C", str(path), "rev-parse", "HEAD").strip()
                worktree["status_porcelain_z"] = git(
                    "-C", str(path), "status", "--porcelain=v1", "-z", "--untracked-files=all"
                )
        else:
            worktree = digest_file(path)
        tracked[relative] = {"index_mode": mode, "index_blob": blob, "index_stage": stage, "worktree": worktree}
    head_dockerfile = git("show", "HEAD:Dockerfile", binary=True) if (root / "Dockerfile").is_file() else None
    native = {}
    for subtree in ("vllm", "vllm_ascend"):
        start = root / subtree
        if start.is_dir():
            for current, directories, files in os.walk(str(start), followlinks=False):
                directories[:] = [
                    d for d in directories if d not in ("__pycache__", ".git") and not (Path(current) / d).is_symlink()
                ]
                for name in files:
                    if name.endswith(".so") or ".so." in name:
                        path = Path(current) / name
                        native[str(path.relative_to(root))] = digest_file(path)
    final_status = git("status", "--porcelain=v1", "-z", "--untracked-files=all", binary=True)
    final_index = git("ls-files", "--stage", "-z", binary=True)
    if initial_status != final_status or index_bytes != final_index or git("rev-parse", "HEAD").strip() != head:
        raise ValueError("Vendor source metadata changed during inspection")
    return {
        "root": str(root),
        "head": head,
        "head_tree": tree,
        "status_porcelain_z": initial_status.decode("utf-8", "backslashreplace"),
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "tracked_file_count": len(tracked),
        "tracked_files": tracked,
        "native_extensions": native,
        "head_diff_name_status": git("diff", "--name-status", "HEAD"),
        "index_diff_name_status": git("diff", "--cached", "--name-status", "HEAD"),
        "dockerfile_head_sha256": hashlib.sha256(head_dockerfile).hexdigest() if head_dockerfile is not None else None,
        "dockerfile_head_bytes": len(head_dockerfile) if head_dockerfile is not None else None,
        "dockerfile_exact_diff": git("diff", "--no-ext-diff", "--no-textconv", "HEAD", "--", "Dockerfile")
        if head_dockerfile is not None
        else None,
        "stable_source_metadata": True,
        "git_commands": commands,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-image", required=True)
    args = parser.parse_args()
    result = {
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": (
            "Read-only vendor-source/image diagnosis; no model/runtime imports, "
            "no device use, no inference or calibration qualification"
        ),
        "expected_ci_image": args.expected_image,
        "python": sys.version,
        "executable": sys.executable,
        "isolated": bool(sys.flags.isolated),
        "no_site": bool(sys.flags.no_site),
        "packages": {},
        "errors": [],
    }
    paths = sorted(
        set(
            p
            for p in sysconfig.get_paths().values()
            if Path(p).is_dir() and ("site-packages" in p or "dist-packages" in p)
        )
    )
    paths += [
        p
        for p in (
            "/usr/local/lib/python3.12/dist-packages",
            "/usr/local/lib/python3.12/site-packages",
            "/usr/lib/python3/dist-packages",
        )
        if Path(p).is_dir() and p not in paths
    ]
    result["metadata_search_paths"] = paths
    for package in ("vllm", "vllm-ascend", "torch", "torch-npu"):
        try:
            matches = list(importlib.metadata.Distribution.discover(name=package, path=paths))
            if len(matches) != 1:
                raise ValueError("Expected one distribution, found " + str(len(matches)))
            dist = matches[0]
            record = {"version": dist.version, "direct_url": json.loads(dist.read_text("direct_url.json") or "null")}
            if package in ("vllm", "vllm-ascend"):
                expected_root = Path("/vllm-workspace") / package
                record["source"] = inspect_source(expected_root)
            result["packages"][package] = record
        except Exception as error:
            result["errors"].append({"package": package, "error": str(error)})
    result["runtime_modules_imported"] = [
        name
        for name in sys.modules
        if name == "torch"
        or name.startswith("torch.")
        or name == "torch_npu"
        or name.startswith("torch_npu.")
        or name == "vllm"
        or name.startswith("vllm.")
        or name == "vllm_ascend"
        or name.startswith("vllm_ascend.")
    ]
    if result["runtime_modules_imported"]:
        result["errors"].append({"error": "Unexpected runtime import"})
    result["completed"] = not result["errors"]
    result["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(
        json.dumps(
            {
                "completed": result["completed"],
                "output": str(args.output),
                "package_count": len(result["packages"]),
                "errors": result["errors"],
            }
        )
    )
    return 0 if result["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
