# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One fresh M9/main pair in one fixed SDK; default invocation is offline only.

A failed role remains failed. A second role is a controlled comparison, never a
retry of the first. This does not establish the original failed Pod's image ID,
all original checkpoint bytes, statistical repeatability, or PR qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BASE = "6e8c1097cd8c73ce799fd1d7f7cd32e656b29b28"
M9 = "80ef8e167cb70daffd7507fd20c2a03c1e268b31"
IMAGE = (
    "936637512419.dkr.ecr.us-west-2.amazonaws.com/vllm-ci-pull-through-cache/"
    "q9t5s3a7/vllm-ci-test-repo:80ef8e167cb70daffd7507fd20c2a03c1e268b31"
    "@sha256:ea6116473b124000699701056bfbdf28224761b8f26469e1f5788b060d68a937"
)
ROOT = Path(__file__).resolve().parents[3]
FILES = {".buildkite/cuda/pipeline.yml", ".buildkite/cuda/scripts/verify_minicpm_paired_control.py"}
MODEL_SNAPSHOT = "eb4141de24550a3a1a4082f718be58f46b761728"
M9_ARCHIVE = "refs/heads/codex/lingbot-world-training-m9-20261106"
M9_REPOSITORY = "https://github.com/tzhouam/vllm-omni.git"
MAX_CHECKPOINT_UNIQUE_BYTES = 128 * 1024**3
MAX_CHECKPOINT_HASH_READ_BYTES = 4 * MAX_CHECKPOINT_UNIQUE_BYTES
MAX_CHECKPOINT_FILES = 100000
MARKS = "full_model and cuda and H100 and B200 and omni and cards_1"
TESTS = (
    "test_minicpmo_4_5_duplex_recognition.py",
    "test_minicpmo_4_5_duplex_expansion.py",
    "test_minicpmo_4_5_window.py",
)
NATIVE = [
    ["pytest", "-s", "-v", "tests/e2e/online_serving/" + name, "-m", MARKS, "--run-level", "full_model"]
    for name in TESTS
]
ENV_KEYS = (
    "PATH",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "VLLM_OMNI_VERSION_OVERRIDE",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "VLLM_WORKER_MULTIPROC_METHOD",
    "HF_HOME",
    "HF_HUB_DOWNLOAD_TIMEOUT",
    "HF_HUB_ETAG_TIMEOUT",
    "LINGBOT_REFERENCE_ONLY",
    "VLLM_OMNI_TARGET_DEVICE",
    "MODEL_PREFIX",
)


def save(path: Path, value: dict) -> None:
    with path.open("x") as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")


def stable(path: Path, expected_identity: list[int] | None = None) -> dict:
    """Hash a regular file without following a final symlink; retain native identity."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        assert stat.S_ISREG(before.st_mode), str(path)
        if expected_identity is not None:
            assert [before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns] == expected_identity, str(path)
        digest = hashlib.sha256()
        with os.fdopen(fd, "rb", closefd=False) as stream:
            if expected_identity is None:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(block)
            else:
                remaining = before.st_size
                while remaining:
                    block = stream.read(min(remaining, 4 * 1024 * 1024))
                    assert block, "Checkpoint truncated during bounded content read"
                    remaining -= len(block)
                    digest.update(block)
        after = os.fstat(fd)
        assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ), str(path)
        assert path.lstat() == after, str(path)
        return {"size": after.st_size, "sha256": digest.hexdigest(), "mode": stat.S_IMODE(after.st_mode)}
    finally:
        os.close(fd)


def command(argv: list[str], cwd: Path, output: Path, env: dict | None = None) -> int:
    """No retries, no added native timeout; the unchanged 120-minute job bounds the pair."""
    print(json.dumps({"phase": output.name, "argv": argv, "cwd": str(cwd)}), flush=True)
    save(output.with_suffix(".command.json"), {"argv": argv, "cwd": str(cwd)})
    with output.with_suffix(".stdout.log").open("xb") as out, output.with_suffix(".stderr.log").open("xb") as err:
        result = subprocess.run(argv, cwd=cwd, env=env, stdout=out, stderr=err, check=False)
    save(output.with_suffix(".result.json"), {"returncode": result.returncode})
    return result.returncode


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "--no-optional-locks", "-C", str(root), *args], text=True, timeout=60
    ).strip()


def source(root: Path, commit: str) -> dict:
    assert git(root, "rev-parse", "HEAD") == commit
    assert not git(root, "status", "--porcelain", "--untracked-files=no"), str(root)
    rows = subprocess.check_output(["git", "-C", str(root), "ls-tree", "-rz", "HEAD"])
    tracked_python = set()
    result = {}
    for row in rows.split(b"\0"):
        if not row:
            continue
        header, relative = row.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        name = relative.decode()
        path = root / name
        assert kind == "blob" and mode in {"100644", "100755", "120000"}, name
        data = os.readlink(path).encode() if mode == "120000" else path.read_bytes()
        actual_oid = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert actual_oid == oid, name
        if mode != "120000":
            assert stat.S_ISREG(path.lstat().st_mode), name
            assert bool(path.lstat().st_mode & 0o111) == (mode == "100755"), name
        result[name] = {"mode": mode, "git_blob": oid, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if name.endswith((".py", ".so", ".pth", ".egg-link")):
            tracked_python.add(name)
    observed = set()
    for directory, subdirs, names in os.walk(root, followlinks=False, onerror=raise_error):
        kept = []
        for name in subdirs:
            child = Path(directory) / name
            relative = str(child.relative_to(root))
            if child.is_symlink():
                assert relative in result and result[relative]["mode"] == "120000", relative
                assert hashlib.sha256(os.readlink(child).encode()).hexdigest() == result[relative]["sha256"]
            elif name != ".git":
                kept.append(name)
        subdirs[:] = kept
        for name in names:
            child = Path(directory) / name
            relative = str(child.relative_to(root))
            if name.endswith((".py", ".so", ".pth", ".egg-link")):
                observed.add(relative)
            if name.endswith((".pyc", ".pyo")) and relative not in result:
                # Only normal current-interpreter caches of verified source are harmless extras.
                assert child.parent.name == "__pycache__" and name.endswith(".pyc"), relative
                assert "." + sys.implementation.cache_tag + "." in name, relative
                cache_source = Path(importlib.util.source_from_cache(str(child)))
                source_relative = str(cache_source.relative_to(root))
                assert source_relative in tracked_python or source_relative == "vllm_omni/_version.py", relative
                assert cache_source.is_file() and not cache_source.is_symlink(), relative
                assert stat.S_ISREG(child.lstat().st_mode), relative
    assert observed - tracked_python <= {"vllm_omni/_version.py"}
    assert tracked_python <= observed
    return {"head": commit, "tree": git(root, "rev-parse", "HEAD^{tree}"), "root": str(root), "tracked": result}


def sdk() -> dict:
    """All registered non-project distribution files, plus the private cuDNN prefix.

    Generated bytecode is excluded; no import, dependency resolution, or SDK writes.
    Symlink targets/bytes are both recorded; missing registered files fail closed.
    This is an observed filesystem fingerprint, not attestation of a loaded kernel.
    """
    distributions = {}
    files = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"].lower().replace("_", "-")
        if name == "vllm-omni":
            continue
        assert dist.files is not None, name
        key = name + "==" + dist.version
        assert key not in distributions, key
        entries = []
        for item in dist.files:
            if str(item).endswith((".pyc", ".pyo")):
                continue
            lexical = Path(os.path.abspath(dist.locate_file(item)))
            real = lexical.resolve(strict=True)
            assert not real.is_relative_to(Path("/opt/vllm-omni")), str(real)
            if str(real) not in files:
                files[str(real)] = stable(real)
            entries.append(
                {
                    "path": str(lexical),
                    "resolved": str(real),
                    "symlink": os.readlink(lexical) if lexical.is_symlink() else None,
                }
            )
        distributions[key] = sorted(entries, key=lambda row: row["path"])
    private_links = {}
    private = Path("/opt/vllm-omni-cudnn")
    assert private.is_dir() and not private.is_symlink()
    for directory, subdirs, names in os.walk(private, followlinks=False, onerror=raise_error):
        for name in subdirs + names:
            path = Path(directory) / name
            if path.is_symlink():
                private_links[str(path)] = os.readlink(path)
                target = path.resolve(strict=True)
                assert target.is_relative_to(private), str(path)
                if target.is_file():
                    files[str(target)] = stable(target)
            elif path.is_file():
                files[str(path)] = stable(path)
    return {
        "distributions": distributions,
        "regular_files": files,
        "private_symlinks": private_links,
        "loader_environment": {key: os.environ.get(key) for key in ENV_KEYS},
    }


def raise_error(error: OSError) -> None:
    raise error


def snapshot() -> dict:
    assert os.environ.get("HF_HOME") == "/root/.cache/huggingface"
    assert not os.environ.get("MODEL_PREFIX")
    model = Path(os.environ["HF_HOME"]) / "hub/models--openbmb--MiniCPM-o-4_5"
    revision = (model / "refs/main").read_text().strip()
    assert revision == MODEL_SNAPSHOT and (model / "snapshots" / revision).is_dir()
    return {
        "model": "openbmb/MiniCPM-o-4_5",
        "snapshot": revision,
        "resolved_snapshot": str((model / "snapshots" / revision).resolve(strict=True)),
        "limit": "The historical failed job did not capture the full model-file SHA population.",
    }


def checkpoint_population() -> tuple[dict, dict]:
    """Inspect sizes/identities first, before opening any model content bytes."""
    model = Path(os.environ["HF_HOME"]) / "hub/models--openbmb--MiniCPM-o-4_5"
    root = model / "snapshots" / MODEL_SNAPSHOT
    allowed = model.resolve(strict=True)
    assert root.resolve(strict=True).is_relative_to(allowed)
    rows = {}
    unique = {}
    for directory, subdirs, names in os.walk(root, followlinks=False, onerror=raise_error):
        for name in subdirs:
            assert not (Path(directory) / name).is_symlink(), "Snapshot directory symlinks are unsupported"
        for name in names:
            lexical = Path(directory) / name
            before = lexical.lstat()
            assert stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode), str(lexical)
            target = lexical.resolve(strict=True)
            assert target.is_relative_to(allowed), str(lexical)
            state = target.lstat()
            assert stat.S_ISREG(state.st_mode), str(target)
            identity = (state.st_dev, state.st_ino)
            item = {
                "path": str(lexical.relative_to(root)),
                "resolved": str(target),
                "symlink": os.readlink(lexical) if stat.S_ISLNK(before.st_mode) else None,
                "size": state.st_size,
                "mode": stat.S_IMODE(state.st_mode),
                "identity": [state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns],
            }
            rows[item["path"]] = item
            unique.setdefault(identity, item)
    assert rows and len(rows) <= MAX_CHECKPOINT_FILES
    return rows, unique


def checkpoint_manifest(output: Path, used_bytes: int) -> tuple[dict, int]:
    rows, unique = checkpoint_population()
    planned_bytes = sum(item["size"] for item in unique.values())
    plan = {
        "snapshot": MODEL_SNAPSHOT,
        "path_count": len(rows),
        "unique_file_count": len(unique),
        "unique_file_bytes": planned_bytes,
        "max_unique_file_bytes": MAX_CHECKPOINT_UNIQUE_BYTES,
        "already_read_model_bytes": used_bytes,
        "whole_pair_max_model_hash_bytes": MAX_CHECKPOINT_HASH_READ_BYTES,
        "prospective_total_read_bytes": used_bytes + planned_bytes,
        "deduplication": (
            "Hash each final regular device/inode once per pass; retain all snapshot paths and symlink targets."
        ),
        "scope": "Fresh pair complete checkpoint proof; not a reconstruction of historical failed-job weights.",
    }
    save(output.with_suffix(".read-budget.json"), plan)
    assert planned_bytes <= MAX_CHECKPOINT_UNIQUE_BYTES, (
        "Checkpoint unique-content budget exceeded before model byte reads"
    )
    assert used_bytes + planned_bytes <= MAX_CHECKPOINT_HASH_READ_BYTES, (
        "Whole-pair model-read budget exceeded before byte reads"
    )
    hashes = {}
    content = {}
    for identity, item in unique.items():
        actual = stable(Path(item["resolved"]), expected_identity=item["identity"])
        assert actual["size"] == item["size"] and actual["mode"] == item["mode"]
        hashes[identity] = actual
        content.setdefault(actual["sha256"], {"size": actual["size"], "paths": []})
        assert content[actual["sha256"]]["size"] == actual["size"]
    after_rows, after_unique = checkpoint_population()
    assert after_rows == rows and after_unique == unique, "Checkpoint population/targets/identity changed during read"
    for name, item in sorted(rows.items()):
        identity = tuple(item["identity"][:2])
        item["sha256"] = hashes[identity]["sha256"]
        content[item["sha256"]]["paths"].append(name)
    result = {
        "snapshot": MODEL_SNAPSHOT,
        "files": rows,
        "shared_contents": content,
        "unique_file_bytes": planned_bytes,
        "unique_content_bytes": sum(item["size"] for item in content.values()),
    }
    save(output, result)
    return result, used_bytes + planned_bytes


def original_parents(root: Path) -> list[str]:
    # cat-file reads the checked-out commit object's original header even in a depth-1 checkout.
    header = git(root, "cat-file", "-p", "HEAD").split("\n\n", 1)[0]
    return [line.removeprefix("parent ") for line in header.splitlines() if line.startswith("parent ")]


def preflight(root: Path, commit: str, output: Path) -> None:
    """Fresh process; real import/package metadata, never model construction or patching."""
    assert Path.cwd().resolve() == root
    proof = source(root, commit)
    dist = metadata.distribution("vllm-omni")
    direct = json.loads(dist.read_text("direct_url.json") or "null")
    assert direct == {"url": root.as_uri(), "dir_info": {"editable": True}}, direct
    for name in (
        "vllm_omni",
        "vllm_omni.entrypoints.cli.main",
        "vllm_omni.core.sched.output",
        "vllm_omni.core.sched.omni_generation_scheduler",
        "vllm_omni.model_executor.models.registry",
        "tests.helpers.runtime",
    ):
        importlib.import_module(name)
    package = sys.modules["vllm_omni"]
    assert all(Path(p).resolve().is_relative_to(root / "vllm_omni") for p in package.__path__)
    assert Path(sys.modules["tests.helpers.runtime"]._omni_subprocess_cwd()).resolve() == root
    version = importlib.import_module("vllm_omni._version")
    assert version.__version__ == dist.version == package.__version__
    assert Path(version.__file__).resolve() == root / "vllm_omni/_version.py"
    fixture = root / "tests/assets/minicpmo_4_5/duplex_recognition/inputs.tar.xz"
    # The actual unmodified helper chooses the conditioning input; do not invent an asset path.
    helper = importlib.import_module("tests.e2e.online_serving.helpers.minicpmo_4_5_duplex")
    audio = Path(helper.resolve_ref_audio()).resolve(strict=True)
    assert stable(fixture)["sha256"] == "c596dbcd409619e0c3bdbf94285ce1da146dae724f8c5771eb70b464297ff189"
    assert stable(audio)["sha256"] == "cb8f06ba5080cdf548969138881fb8ad8b04e2516108f4e08ba0363b68b613ea"
    origins = {}
    for name, module in list(sys.modules.items()):
        if name == "vllm_omni" or name.startswith("vllm_omni.") or name == "tests" or name.startswith("tests."):
            paths = getattr(module, "__path__", ())
            assert all(Path(path).resolve(strict=True).is_relative_to(root) for path in paths), (name, list(paths))
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve(strict=True)
                assert path.is_relative_to(root), (name, str(path))
                relative = str(path.relative_to(root))
                actual = stable(path)
                if relative == "vllm_omni/_version.py":
                    assert path == Path(version.__file__).resolve(strict=True)
                else:
                    assert relative in proof["tracked"], (name, relative)
                    expected = proof["tracked"][relative]
                    assert expected["mode"] in {"100644", "100755"}, (name, relative)
                    assert actual["sha256"] == expected["sha256"] and actual["size"] == expected["size"]
                origins[name] = {"path": str(path), **actual}
    save(
        output,
        {
            "source": proof,
            "distribution": {"version": dist.version, "direct_url": direct},
            "actual_loaded_origins": origins,
            "generated_version": stable(Path(version.__file__)),
            "fixture": stable(fixture),
            "reference_audio": {"path": str(audio), **stable(audio)},
            "model": snapshot(),
            "scope": "Fresh ordinary import preflight; no claim of all later worker modules.",
        },
    )


def own_children(pid: int) -> set[int]:
    result = set()
    pending = [pid]
    while pending:
        parent = pending.pop()
        try:
            children = Path(f"/proc/{parent}/task/{parent}/children").read_text().split()
        except FileNotFoundError:
            continue
        for raw in children:
            child = int(raw)
            if child not in result:
                result.add(child)
                pending.append(child)
    return result


def native_command(argv: list[str], root: Path, output: Path, env: dict) -> int:
    """Observe only our pytest descendants; never inspect unrelated user commands."""
    print(json.dumps({"phase": output.name, "argv": argv, "cwd": str(root)}), flush=True)
    save(output.with_suffix(".command.json"), {"argv": argv, "cwd": str(root)})
    observations = {}
    observation_errors = []
    with output.with_suffix(".stdout.log").open("xb") as out, output.with_suffix(".stderr.log").open("xb") as err:
        proc = subprocess.Popen(argv, cwd=root, env=env, stdout=out, stderr=err)
        while proc.poll() is None:
            for pid in own_children(proc.pid):
                try:
                    directory = Path(f"/proc/{pid}")
                    fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
                    ticks = fields[19]
                    key = str(pid) + ":" + ticks
                    if key in observations:
                        continue
                    command_line = [
                        value.decode() for value in (directory / "cmdline").read_bytes().split(b"\0") if value
                    ]
                    if command_line[1:5] != ["-m", "vllm_omni.entrypoints.cli.main", "serve", "openbmb/MiniCPM-o-4_5"]:
                        continue
                    cwd = (directory / "cwd").resolve(strict=True)
                    exe = (directory / "exe").resolve(strict=True)
                    values = (directory / "environ").read_bytes().split(b"\0")
                    pythonpath = next(
                        (value.partition(b"=")[2].decode() for value in values if value.startswith(b"PYTHONPATH=")), ""
                    )
                    second = (directory / "stat").read_text().rsplit(")", 1)[1].split()
                    if second[19] != ticks or pid not in own_children(proc.pid):
                        continue
                    observations[key] = {
                        "pid": pid,
                        "start_ticks": ticks,
                        "command": command_line,
                        "cwd": str(cwd),
                        "exe": str(exe),
                        "pythonpath": pythonpath,
                    }
                    assert cwd == root and exe == Path(sys.executable).resolve()
                    assert pythonpath == env["PYTHONPATH"]
                except (FileNotFoundError, ProcessLookupError):
                    continue
                except Exception as error:
                    observation_errors.append({"pid": pid, "type": type(error).__name__, "message": str(error)})
            time.sleep(0.25)
        rc = proc.wait()
    save(
        output.with_suffix(".result.json"),
        {
            "returncode": rc,
            "owned_server_launches": observations,
            "observation_errors": observation_errors,
            "scope": "Passive actual launch/cwd/environment, not loaded worker module tracing.",
        },
    )
    assert observations and not observation_errors, "Native return is preserved; source observation failed separately"
    return rc


def selected_gpu() -> dict:
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name,memory.total", "--format=csv,noheader,nounits"], text=True, timeout=30
    ).splitlines()
    assert len(query) == 1 and "H100" in query[0], query
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"], text=True, timeout=30
    ).strip()
    assert not processes, "Compute processes remain on the selected GPU; no cleanup/reset is attempted"
    return {"gpu": query, "compute_processes": processes}


def copy_results(source_root: Path, output: Path) -> dict:
    source_dir = source_root / "tests/e2e/online_serving/results/minicpmo_duplex_recognition"
    rows = {}
    if not source_dir.exists():
        return rows
    output.mkdir()
    for directory, subdirs, names in os.walk(source_dir, followlinks=False, onerror=raise_error):
        assert all(not (Path(directory) / name).is_symlink() for name in subdirs)
        for name in names:
            path = Path(directory) / name
            relative = path.relative_to(source_dir)
            before = stable(path)
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with path.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
            os.chmod(target, before["mode"])
            assert stable(path) == before == stable(target)
            rows[str(relative)] = before
    return rows


def run_pair(expected_self: str) -> int:
    assert stable(Path(__file__))["sha256"] == expected_self
    commit = git(ROOT, "rev-parse", "HEAD")
    assert os.environ.get("BUILDKITE_COMMIT") == commit
    assert original_parents(ROOT) == [BASE]
    assert not git(ROOT, "status", "--porcelain", "--untracked-files=all")
    output = ROOT / "diagnostic-artifacts/minicpm-paired-m9-main6e8c"
    output.mkdir(parents=True, exist_ok=False)
    # A legal depth-1 checkout has the parent header without the parent object.
    if subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", BASE + "^{commit}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode:
        assert (
            command(
                [
                    "git",
                    "-c",
                    "http.version=HTTP/1.1",
                    "fetch",
                    "--no-tags",
                    "--depth=1",
                    "https://github.com/vllm-project/vllm-omni.git",
                    BASE,
                ],
                ROOT,
                output / "control-base-fetch",
            )
            == 0
        )
        assert git(ROOT, "rev-parse", "FETCH_HEAD") == BASE
    assert set(git(ROOT, "diff", "--name-only", BASE, commit).splitlines()) == FILES
    assert all(git(ROOT, "ls-tree", "HEAD", "--", path).startswith("100644 blob ") for path in FILES)
    assert os.environ.get("VLLM_OMNI_VERSION_OVERRIDE") == "0.0.0", "Keep the SDK's original CI build setting"
    save(
        output / "launch.json",
        {
            "control_commit": commit,
            "source_roles": {"m9": M9, "main6e8c": BASE},
            "configured_image": IMAGE,
            "reported_BUILDKITE_IMAGE": os.environ.get("BUILDKITE_IMAGE"),
            "original_failed_pod_resolved_image_id": None,
            "commands": NATIVE,
            "model": snapshot(),
            "helper": stable(Path(__file__)),
            "role_order": ["m9", "main6e8c"],
            "qualification_passed": False,
        },
    )
    baseline_sdk = sdk()
    save(output / "sdk-before.json", baseline_sdk)
    app = Path("/opt/vllm-omni")
    assert app.is_dir() and not app.is_symlink() and app.resolve() == app
    temporary = Path(tempfile.mkdtemp(prefix="pr9009-minicpm-pair-", dir="/tmp"))
    quarantine = temporary / "original-image-project"
    # Preserve the SDK's sibling cuDNN prefix. Remove the original project fallback, not any SDK package.
    shutil.move(str(app), str(quarantine))
    assert not app.exists() and not app.is_symlink()
    assert quarantine.is_dir() and not quarantine.is_symlink()
    save(
        output / "project-quarantine.json",
        {"operation": "shutil.move", "original": str(app), "own_destination": str(quarantine), "original_absent": True},
    )
    results = []
    overall = 0
    checkpoint = None
    checkpoint_read_bytes = 0
    inherited = os.environ.copy()
    inherited_path = inherited.get("PYTHONPATH", "")
    kept = []
    for value in inherited_path.split(os.pathsep):
        if not value:
            continue
        resolved = Path(value).resolve()
        if resolved.is_relative_to(ROOT) or resolved.is_relative_to(app):
            continue
        assert not (resolved / "vllm_omni").exists() and not (resolved / "tests").exists(), value
        kept.append(value)
    # This removes only prior project selectors. A genuine editable install and role cwd select the full project.
    inherited["PYTHONPATH"] = os.pathsep.join(kept)
    inherited["PYTHONDONTWRITEBYTECODE"] = "1"
    save(
        output / "source-selection-environment.json",
        {
            "old_pythonpath": inherited_path,
            "new_pythonpath": inherited["PYTHONPATH"],
            "other_environment_changed": ["PYTHONDONTWRITEBYTECODE"],
        },
    )
    try:
        for role, selected in (("m9", M9), ("main6e8c", BASE)):
            role_output = output / role
            role_output.mkdir()
            role_root = temporary / role
            if subprocess.run(
                ["git", "-C", str(ROOT), "cat-file", "-e", selected + "^{commit}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode:
                repository = M9_REPOSITORY if selected == M9 else "https://github.com/vllm-project/vllm-omni.git"
                ref = M9_ARCHIVE if selected == M9 else BASE
                assert (
                    command(
                        ["git", "-c", "http.version=HTTP/1.1", "fetch", "--no-tags", "--depth=1", repository, ref],
                        ROOT,
                        role_output / "source-fetch",
                    )
                    == 0
                )
                assert git(ROOT, "rev-parse", "FETCH_HEAD") == selected
            assert (
                command(
                    ["git", "worktree", "add", "--detach", str(role_root), selected],
                    ROOT,
                    role_output / "source-worktree",
                )
                == 0
            )
            save(role_output / "source-before.json", source(role_root, selected))
            selected_gpu()
            before_model = snapshot()
            model_before, checkpoint_read_bytes = checkpoint_manifest(
                role_output / "complete-checkpoint-before.json", checkpoint_read_bytes
            )
            if checkpoint is None:
                checkpoint = model_before
            assert model_before == checkpoint, "The two fresh roles must use the exact same checkpoint files and bytes"
            # Genuine complete project installation; no deps/resolver or build-isolation packages are permitted.
            assert (
                command(
                    [
                        "uv",
                        "pip",
                        "install",
                        "--system",
                        "--no-deps",
                        "--no-build-isolation",
                        "--offline",
                        "--reinstall-package",
                        "vllm-omni",
                        "-e",
                        str(role_root),
                    ],
                    role_root,
                    role_output / "editable-install",
                    inherited,
                )
                == 0
            )
            after_install = sdk()
            save(role_output / "sdk-after-install.json", after_install)
            assert after_install == baseline_sdk, "A non-project SDK byte/metadata/loader environment changed"
            # python -c starts with the actual role cwd; no partial module overlay or old image source path.
            program = (
                "import runpy; "
                f"d=runpy.run_path({str(Path(__file__))!r}); "
                f"d['preflight'](__import__('pathlib').Path({str(role_root)!r}),{selected!r},"
                f"__import__('pathlib').Path({str(role_output / 'source-origin-preflight.json')!r}))"
            )
            assert command([sys.executable, "-c", program], role_root, role_output / "origin-preflight", inherited) == 0
            assert (
                command(
                    [sys.executable, ".buildkite/cuda/scripts/check_vllm_runtime.py"],
                    role_root,
                    role_output / "ordinary-runtime-check",
                    inherited,
                )
                == 0
            )
            native = []
            try:
                for index, argv in enumerate(NATIVE):
                    rc = native_command(argv, role_root, role_output / f"native-{index}", inherited)
                    native.append({"argv": argv, "returncode": rc})
                    if rc:
                        overall = 1
                        break
            finally:
                save(
                    role_output / "native-status.json",
                    {
                        "commands": native,
                        "native_passed": len(native) == 3 and all(x["returncode"] == 0 for x in native),
                    },
                )
                save(role_output / "copied-results.json", copy_results(role_root, role_output / "recognition-results"))
                save(role_output / "source-after.json", source(role_root, selected))
                after_model = snapshot()
                save(role_output / "model-after.json", after_model)
                assert after_model == before_model
                model_after, checkpoint_read_bytes = checkpoint_manifest(
                    role_output / "complete-checkpoint-after.json", checkpoint_read_bytes
                )
                assert model_after == checkpoint, "Checkpoint files/targets/bytes changed during the native role"
                after_native = sdk()
                save(role_output / "sdk-after-native.json", after_native)
                assert after_native == baseline_sdk
                save(role_output / "post-role-selected-gpu.json", selected_gpu())
            results.append({"role": role, "head": selected, "native": native})
    except BaseException as error:
        save(
            output / "controller-error.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "completed_roles": results,
                "qualification_passed": False,
            },
        )
        raise
    finally:
        save(
            output / "controller-state.json",
            {
                "completed_roles": results,
                "original_project_quarantine": str(quarantine),
                "qualification_passed": False,
                "note": "Private Pod only; no SDK/source restoration fallback or model retry.",
            },
        )
    save(
        output / "summary.json",
        {
            "paired_native_complete": len(results) == 2,
            "native_exit": overall,
            "roles": results,
            "checkpoint_hash_read_bytes": checkpoint_read_bytes,
            "qualification_passed": False,
            "limit": "Fresh fixed-SDK source comparison; not original Pod/whole-weight byte equality.",
        },
    )
    return overall


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-paired-control", action="store_true")
    parser.add_argument("--expected-self-sha256")
    args = parser.parse_args()
    if not args.execute_paired_control:
        print(
            json.dumps(
                {
                    "mode": "offline-source-preparation-only",
                    "execution": False,
                    "roles": {"m9": M9, "main6e8c": BASE},
                    "configured_image": IMAGE,
                }
            )
        )
        return 0
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_self_sha256 or ""):
        parser.error("Execution requires the frozen helper SHA256")
    return run_pair(args.expected_self_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
