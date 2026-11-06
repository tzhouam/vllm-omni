# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One fresh M9/main pair in one fixed SDK; default invocation is offline only.

A failed role remains failed. A second role is a controlled comparison, never a
retry of the first. This does not establish the original failed Pod's image ID,
all original checkpoint bytes, statistical repeatability, or PR qualification.
"""

from __future__ import annotations

import argparse
import csv
import email
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import io
import json
import locale
import os
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
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


MAX_SDK_PASS_BYTES = 256 * 1024**3
MAX_SDK_PASS_FILES = 1000000
MAX_SDK_WHOLE_BYTES = 5 * MAX_SDK_PASS_BYTES
MAX_SDK_WHOLE_FILES = 5 * MAX_SDK_PASS_FILES
MAX_SDK_SELECTOR_BYTES = 64 * 1024**2
SDK_PRIVATE_CUDNN_PATH = Path("/opt/vllm-omni-cudnn")
SDK_CUDA_PATH = Path("/usr/local/cuda")


def sdk_path_allowed(path: Path) -> None:
    broad = {
        "/",
        "/usr",
        "/usr/local",
        "/usr/lib",
        "/lib",
        "/opt",
        "/tmp",
        "/root",
        "/home",
        "/workspace",
        "/workspace/build",
        "/usr/local/lib",
    }
    assert str(path) not in broad, ("Overbroad SDK root", str(path))
    assert not any(path.is_relative_to(Path(prefix)) for prefix in ("/proc", "/sys", "/dev", "/root/.cache"))
    assert not path.is_relative_to(ROOT) and not path.is_relative_to(Path("/opt/vllm-omni"))


def sdk_project_exclusions(dist) -> dict[Path, str]:
    """Only the verified project install's own editable metadata, finder and caches."""
    assert dist.files is not None, "Cannot safely isolate project installation without its own file metadata"
    version = dist.version
    assert version == "0.0.0"
    prefix = Path(dist.locate_file("")).resolve(strict=True)
    metadata_prefix = prefix / ("vllm_omni-" + version + ".dist-info")
    finder = prefix / ("__editable___vllm_omni_" + version.replace(".", "_") + "_finder.py")
    pth = prefix / ("__editable__.vllm_omni-" + version + ".pth")
    recorded = {Path(os.path.abspath(dist.locate_file(item))) for item in dist.files}
    finder_caches = {
        Path(importlib.util.cache_from_source(str(finder), optimization=level)) for level in ("", "1", "2")
    }
    allowed = set()
    for path in recorded:
        if path.suffix in {".pyc", ".pyo"}:
            assert finder in recorded and path in finder_caches, ("Unrecognized project bytecode record", str(path))
        else:
            assert (
                path.is_relative_to(metadata_prefix)
                or path == finder
                or path == pth
                or path == Path(sysconfig.get_path("scripts")).resolve(strict=True) / "vllm-omni"
            ), str(path)
        assert path.is_file() and not path.is_symlink(), str(path)
        allowed.add(path)
    if finder in recorded:
        for path in finder_caches:
            if path.exists() or path.is_symlink():
                assert path.is_file() and not path.is_symlink(), str(path)
                allowed.add(path)
    assert pth in allowed and metadata_prefix / "METADATA" in allowed
    # Only ancestors of verified RECORD files inside this exact project's metadata
    # directory belong to the editable install. Unknown empty directories, siblings,
    # scripts and shared cache parents remain protected SDK entries.
    metadata_directories = set()
    for path in recorded:
        if not path.is_relative_to(metadata_prefix):
            continue
        parent = path.parent
        while parent.is_relative_to(metadata_prefix):
            observed = parent.lstat()
            assert stat.S_ISDIR(observed.st_mode), ("Project metadata ancestor is not a real directory", str(parent))
            metadata_directories.add(parent)
            if parent == metadata_prefix:
                break
            parent = parent.parent
    return {**{path: "regular" for path in allowed}, **{path: "directory" for path in metadata_directories}}


def sdk_distribution_plan(distributions: list, progress: dict | None = None) -> dict:
    project = [dist for dist in distributions if dist.metadata["Name"].lower().replace("_", "-") == "vllm-omni"]
    assert len(project) == 1, "Expected exactly one captured vllm-omni project distribution"
    excluded = sdk_project_exclusions(project[0])
    providers = {} if progress is None else progress
    registered = {}
    name_version_groups = {}
    for dist in distributions:
        name = dist.metadata["Name"].lower().replace("_", "-")
        if name == "vllm-omni":
            continue
        descriptor = dist.descriptor
        key = descriptor["location"]
        assert descriptor["header"] and descriptor["root"] and descriptor["kind"]
        assert key not in providers, ("Repeated actual metadata location", key)
        name_version_groups.setdefault(name + "==" + dist.version, []).append(key)
        root = Path(dist.locate_file("")).resolve(strict=True)
        sdk_path_allowed(root)
        paths = []
        for item in dist.files or ():
            lexical = Path(os.path.abspath(dist.locate_file(item)))
            sdk_path_allowed(lexical.resolve(strict=True))
            paths.append(str(lexical))
        providers[key] = {
            "descriptor": descriptor,
            "provider_type": dist.provider_kind,
            "root": str(root),
            "files_metadata_available": dist.files is not None,
            "registered_paths": sorted(paths),
            "coverage": "Full provider filesystem root, independent of RECORD presence",
        }
        registered[key] = sorted(paths)
    return {
        "providers": providers,
        "registered": registered,
        "project_exclusions": excluded,
        "name_version_groups": {name: sorted(instances) for name, instances in sorted(name_version_groups.items())},
    }


def sdk_metadata_locations(roots: set[Path]) -> dict:
    locations = {}
    for root in roots:
        for path in root.iterdir():
            if not path.name.endswith((".dist-info", ".egg-info")):
                continue
            assert not path.is_symlink(), ("Unsupported metadata-directory symlink", str(path))
            if path.is_file():
                assert path.name.endswith(".egg-info")
                header = path
                record = None
            else:
                assert path.is_dir(), str(path)
                candidates = [path / name for name in ("METADATA", "PKG-INFO") if (path / name).is_file()]
                assert len(candidates) == 1, ("Unavailable or ambiguous package header", str(path))
                header = candidates[0]
                record = path / "RECORD" if (path / "RECORD").is_file() else None
            locations[str(path)] = {
                "location": str(path),
                "root": str(root),
                "header": str(header),
                "record": str(record) if record else None,
                "kind": "dist-info" if path.name.endswith(".dist-info") else "egg-info",
            }
    return locations


class SDKMetadataRecord:
    """Only bounded, captured standard metadata bytes; never a replacement runtime package."""

    def __init__(self, location: dict, data: dict):
        message = email.message_from_bytes(data[location["header"]])
        assert message["Name"] and message["Version"], ("Missing package Name/Version", location["header"])
        self.metadata = {"Name": message["Name"]}
        self.version = message["Version"]
        self.root = Path(location["root"])
        self.provider_kind = location["kind"]
        self.descriptor = {
            **location,
            "name": message["Name"],
            "version": message["Version"],
            "header_sha256": hashlib.sha256(data[location["header"]]).hexdigest(),
            "record_sha256": hashlib.sha256(data[location["record"]]).hexdigest() if location["record"] else None,
        }
        self.files = None
        if location["record"]:
            self.files = []
            for row in csv.reader(io.StringIO(data[location["record"]].decode())):
                assert len(row) == 3 and row[0], ("Malformed RECORD", location["record"])
                self.files.append(row[0])

    def locate_file(self, name):
        return self.root / name


def sdk_population(
    roots: set[Path], external: set[Path], excluded: dict[Path, str], selector_stage: bool = False
) -> dict:
    """Only explicitly planned SDK roots; never follow directory links silently."""
    population = {}
    pending = [root for root in roots if not any(root != other and root.is_relative_to(other) for other in roots)]
    seen = set()
    for root in roots:
        sdk_path_allowed(root)

    def leaf(path: Path) -> None:
        state = path.lstat()
        row = {
            "mode": stat.S_IMODE(state.st_mode),
            "identity": [state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns],
        }
        if stat.S_ISDIR(state.st_mode):
            row["kind"] = "directory"
        elif stat.S_ISREG(state.st_mode):
            row.update(kind="regular", excluded_project=path in excluded, bytecode=path.suffix in {".pyc", ".pyo"})
        elif stat.S_ISLNK(state.st_mode):
            target = path.resolve(strict=True)
            sdk_path_allowed(target)
            row.update(kind="symlink", target_text=os.readlink(path), resolved=str(target))
            target_state = target.lstat()
            if stat.S_ISDIR(target_state.st_mode):
                assert any(target.is_relative_to(root) for root in roots), (str(path), str(target))
            else:
                assert stat.S_ISREG(target_state.st_mode), str(target)
                covered = any(target.is_relative_to(root) for root in roots) or path in external or target in external
                assert covered or selector_stage, (
                    "Unregistered SDK link target outside declared roots",
                    str(path),
                    str(target),
                )
                if covered:
                    external.add(target)
                else:
                    row["coverage_pending_registered_metadata"] = True
        else:
            raise RuntimeError("Special entry in SDK scope: " + str(path))
        if path in excluded:
            expected_kind = excluded[path]
            assert expected_kind in {"regular", "directory"} and row["kind"] == expected_kind, (
                "Project exclusion entry changed kind after ownership validation",
                str(path),
                expected_kind,
                row["kind"],
            )
            row["excluded_project"] = True
        population[str(path)] = row
        assert len(population) <= MAX_SDK_PASS_FILES, "SDK lstat population exceeds explicit per-pass file cap"

    for root in pending:
        if root in seen:
            continue
        seen.add(root)
        leaf(root)
        assert root.is_dir() and not root.is_symlink(), str(root)
        for directory, subdirs, names in os.walk(root, followlinks=False, onerror=raise_error):
            for name in subdirs + names:
                leaf(Path(directory) / name)
    for path in list(external):
        sdk_path_allowed(path.resolve(strict=True))
        if str(path) not in population:
            leaf(path)
        target = path.resolve(strict=True)
        if str(target) not in population:
            leaf(target)
    return population


def sdk_references(paths: set[Path]) -> dict:
    result = {}
    for path in paths:
        resolved = path.resolve(strict=True)
        sdk_path_allowed(resolved)
        links = {str(part): os.readlink(part) for part in (path, *path.parents) if part.is_symlink()}
        result[str(path)] = {"resolved": str(resolved), "ancestor_symlinks": links}
    return result


def sdk_absent_references(paths: set[Path]) -> dict:
    result = {}
    for path in paths:
        assert not path.exists() and not path.is_symlink(), ("Previously absent SDK path changed", str(path))
        parts = (path, *path.parents)
        assert not any(part.is_symlink() and not part.exists() for part in parts), (
            "Dangling SDK path ancestor",
            str(path),
        )
        result[str(path)] = {
            "resolved": str(path.resolve()),
            "ancestor_symlinks": {str(part): os.readlink(part) for part in parts if part.is_symlink()},
        }
    return result


def sdk_protected_projection(population: dict) -> dict:
    result = {}
    for path, row in population.items():
        if row.get("excluded_project"):
            continue
        result[path] = {key: value for key, value in row.items() if key != "identity"}
    return result


def sdk_population_unchanged(before: dict, after: dict) -> None:
    assert after == before, "SDK population/identity changed across content hashing"


def sdk_read_plan(population: dict, budget: dict, selector: bool = False) -> dict:
    unique = {}
    for path, row in population.items():
        if row["kind"] != "regular" or row.get("excluded_project"):
            continue
        unique.setdefault(tuple(row["identity"][:2]), (path, row))
    size = sum(row["identity"][2] for path, row in unique.values())
    return {
        "unique_regular_files": len(unique),
        "planned_content_bytes": size,
        "per_pass_bytes_cap": MAX_SDK_SELECTOR_BYTES if selector else MAX_SDK_PASS_BYTES,
        "per_pass_file_cap": MAX_SDK_PASS_FILES,
        "whole_bytes_used": budget["bytes"],
        "whole_files_read": budget["files"],
        "whole_bytes_cap": MAX_SDK_WHOLE_BYTES,
        "whole_files_cap": MAX_SDK_WHOLE_FILES,
        "selector_planning_pass": selector,
    }


def sdk_read(population: dict, budget: dict, plan: dict, retain_selector_bytes: bool = False) -> dict:
    assert plan["planned_content_bytes"] <= plan["per_pass_bytes_cap"], (
        "SDK byte budget exceeded before file content reads"
    )
    assert plan["unique_regular_files"] <= plan["per_pass_file_cap"], "SDK file budget exceeded before content reads"
    assert budget["bytes"] + plan["planned_content_bytes"] <= MAX_SDK_WHOLE_BYTES
    assert budget["files"] + plan["unique_regular_files"] <= MAX_SDK_WHOLE_FILES
    by_identity = {}
    hashes = {}
    for path, row in population.items():
        if row["kind"] != "regular" or row.get("excluded_project"):
            continue
        identity = tuple(row["identity"][:2])
        if identity not in by_identity:
            if retain_selector_bytes:
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                try:
                    before = os.fstat(fd)
                    assert stat.S_ISREG(before.st_mode)
                    assert [before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns] == row["identity"]
                    remaining = before.st_size
                    data = bytearray()
                    while remaining:
                        chunk = os.read(fd, min(remaining, 4 * 1024 * 1024))
                        assert chunk, "Selector truncated during bounded read"
                        data.extend(chunk)
                        remaining -= len(chunk)
                    after = os.fstat(fd)
                    assert [after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns] == row["identity"]
                    assert Path(path).lstat() == after
                    by_identity[identity] = {
                        "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "mode": stat.S_IMODE(after.st_mode),
                        "selector_bytes": bytes(data),
                    }
                finally:
                    os.close(fd)
            else:
                by_identity[identity] = stable(Path(path), expected_identity=row["identity"])
            budget["bytes"] += row["identity"][2]
            budget["files"] += 1
        hashes[path] = by_identity[identity]
    return hashes


def sdk_selector_kind(
    path: Path, metadata_locations: dict, site_directories: set[Path], legacy_search: set[Path]
) -> str | None:
    """Selector positions are explicit; a data-file suffix does not activate it."""
    if path.name == "direct_url.json" and str(path.parent) in metadata_locations:
        return "metadata-direct-url"
    if path.suffix == ".pth" and not path.name.startswith(".") and path.parent in site_directories:
        return "site-pth"
    if path.suffix == ".egg-link" and path.parent in legacy_search:
        return "active-search-egg-link"
    return None


def sdk_startup_customization(roots: set[Path], external: set[Path], output: Path, state: dict) -> dict:
    """Passively bind only an already-loaded hook's exact stdlib file link."""
    observation = {"module": {}, "stdlib_roots": [], "entries": {}}
    state["startup_customization"] = observation
    try:
        module = sys.modules.get("sitecustomize")
        values = vars(module) if type(module) is type(sys) else {}
        spec = values.get("__spec__")
        spec_values = vars(spec) if type(spec) is type(sys.__spec__) else {}
        observation["module"] = {
            "present": module is not None,
            "is_module": type(module) is type(sys),
            "name": values.get("__name__") if type(values.get("__name__")) is str else None,
            "file": values.get("__file__") if type(values.get("__file__")) is str else None,
            "is_module_spec": type(spec) is type(sys.__spec__),
            "spec_name": spec_values.get("name") if type(spec_values.get("name")) is str else None,
            "origin": spec_values.get("origin") if type(spec_values.get("origin")) is str else None,
        }
        stdlib_roots = {Path(sysconfig.get_path(name)).resolve(strict=True) for name in ("stdlib", "platstdlib")}
        observation["stdlib_roots"] = sorted(str(path) for path in stdlib_roots)
        for root in sorted(stdlib_roots):
            path = root / "sitecustomize.py"
            row = {}
            observation["entries"][str(path)] = row
            try:
                entry = path.lstat()
            except FileNotFoundError:
                row["kind"] = "absent"
                continue
            row.update(
                mode=stat.S_IMODE(entry.st_mode),
                identity=[entry.st_dev, entry.st_ino, entry.st_size, entry.st_mtime_ns],
            )
            if stat.S_ISREG(entry.st_mode):
                row["kind"] = "regular"
            elif stat.S_ISLNK(entry.st_mode):
                row.update(kind="symlink", target_text=os.readlink(path))
                target = Path(row["target_text"])
                target = Path(os.path.abspath(target if target.is_absolute() else path.parent / target))
                row["direct_target"] = str(target)
                row["origin_ancestor_links"] = {str(p): os.readlink(p) for p in path.parents if p.is_symlink()}
                row["target_ancestor_links"] = {str(p): os.readlink(p) for p in target.parents if p.is_symlink()}
                target_entry = target.lstat()
                row.update(
                    target_kind="regular" if stat.S_ISREG(target_entry.st_mode) else "unsupported",
                    target_mode=stat.S_IMODE(target_entry.st_mode),
                    target_identity=[
                        target_entry.st_dev,
                        target_entry.st_ino,
                        target_entry.st_size,
                        target_entry.st_mtime_ns,
                    ],
                    resolved=str(path.resolve(strict=True)),
                )
            else:
                row["kind"] = "unsupported"
    except BaseException as error:
        observation["observation_error"] = {"type": type(error).__name__, "message": str(error)}
        save(output, observation)
        raise
    # Persist missing/mismatched origins and exact file descriptors before any guard.
    save(output, observation)
    for raw, row in observation["entries"].items():
        if row["kind"] != "symlink" or any(Path(row["resolved"]).is_relative_to(root) for root in roots):
            continue
        binding = observation["module"]
        assert binding["present"] and binding["is_module"] and binding["is_module_spec"], (
            "No current loaded sitecustomize binding for external stdlib link",
            raw,
        )
        assert binding["name"] == binding["spec_name"] == "sitecustomize" and binding["file"] and binding["origin"], (
            binding
        )
        assert Path(os.path.abspath(binding["file"])) == Path(raw) == Path(os.path.abspath(binding["origin"])), (
            "Loaded sitecustomize origin does not match exact stdlib link",
            binding,
            raw,
        )
        assert any(Path(raw).is_relative_to(root) for root in roots), ("Customization origin is not covered", raw)
        assert not row["origin_ancestor_links"] and not row["target_ancestor_links"], (
            "Unsupported customization ancestor link",
            raw,
        )
        assert row["target_kind"] == "regular" and row["direct_target"] == row["resolved"], (
            "Customization target must be one direct regular file",
            raw,
        )
        target = Path(row["direct_target"])
        sdk_path_allowed(target)
        external.update({Path(raw), target})
    return observation


def sdk_exact_file_chain(path: Path) -> dict:
    """Observe a bounded file-link chain; never read or execute its contents."""
    result = {"origin": str(path), "entries": [], "ancestors": {}, "complete": False}

    def descriptor(candidate: Path) -> dict:
        entry = candidate.lstat()
        kind = (
            "regular"
            if stat.S_ISREG(entry.st_mode)
            else "symlink"
            if stat.S_ISLNK(entry.st_mode)
            else "directory"
            if stat.S_ISDIR(entry.st_mode)
            else "unsupported"
        )
        return {
            "path": str(candidate),
            "kind": kind,
            "mode": stat.S_IMODE(entry.st_mode),
            "identity": [entry.st_dev, entry.st_ino, entry.st_size, entry.st_mtime_ns, entry.st_ctime_ns],
            "uid": entry.st_uid,
            "gid": entry.st_gid,
            "nlink": entry.st_nlink,
        }

    try:
        current = path
        seen = set()
        for hop in range(17):
            assert current.is_absolute() and len(current.parts) <= 256
            sdk_path_allowed(current)
            assert str(current) not in seen, ("Configured dependency link loop", str(current))
            seen.add(str(current))
            for parent in reversed(current.parents):
                observed = descriptor(parent)
                result["ancestors"][str(parent)] = observed
                assert observed["kind"] == "directory", (
                    "Configured dependency ancestor is not a direct directory",
                    str(parent),
                )
            row = descriptor(current)
            result["entries"].append(row)
            if row["kind"] == "regular":
                result.update(complete=True, final_path=str(current))
                return result
            assert row["kind"] == "symlink" and hop < 16, (
                "Unsupported configured dependency entry or chain length",
                str(current),
            )
            raw = os.readlink(current)
            row["target_text"] = raw
            assert descriptor(current) == {key: value for key, value in row.items() if key != "target_text"}, (
                "Configured dependency link changed during observation",
                str(current),
            )
            assert raw and len(raw) <= 4096 and not raw.startswith("//")
            # Leading '..' is safe after the current ancestors were proven direct
            # directories. Do not collapse a later '..' past an unobserved link.
            normal_component_seen = False
            for component in Path(raw).parts:
                if component in {"/", "."}:
                    continue
                if component == "..":
                    assert not normal_component_seen, "Unsupported dotdot after a target component"
                else:
                    normal_component_seen = True
            target = Path(raw)
            current = Path(os.path.abspath(target if target.is_absolute() else current.parent / target))
        raise AssertionError("Configured dependency chain exceeded the bounded link count")
    except Exception as error:
        result["observation_error"] = {"type": type(error).__name__, "message": str(error)}
        return result


def sdk_configured_library_snapshot(stdlib_roots: set[Path]) -> dict:
    """Read an already initialized configuration, not a loaded-library claim."""
    values = vars(sysconfig) if type(sysconfig) is type(sys) else {}
    spec = values.get("__spec__")
    spec_values = vars(spec) if type(spec) is type(sys.__spec__) else {}
    config = values.get("_CONFIG_VARS")
    keys = ("LIBPL", "LDLIBRARY", "LIBDIR", "INSTSONAME", "LIBRARY", "MULTIARCH", "VERSION", "Py_ENABLE_SHARED")
    result = {
        "basis": "current initialized CPython sysconfig configured build dependency",
        "loaded_library_identity_verified": False,
        "module": {
            "is_module": type(sysconfig) is type(sys),
            "name": values.get("__name__") if type(values.get("__name__")) is str else None,
            "file": values.get("__file__") if type(values.get("__file__")) is str else None,
            "is_module_spec": type(spec) is type(sys.__spec__),
            "spec_name": spec_values.get("name") if type(spec_values.get("name")) is str else None,
            "origin": spec_values.get("origin") if type(spec_values.get("origin")) is str else None,
        },
        "config_initialized": type(config) is dict,
        "config": {
            key: config.get(key)
            if type(config.get(key)) in {str, int, bool, type(None)}
            else {"unsupported_type": type(config.get(key)).__name__}
            for key in keys
        }
        if type(config) is dict
        else {},
        "declared_stdlib_roots": sorted(str(root) for root in stdlib_roots),
        "candidate": None,
        "chain": None,
        "module_file_chain": None,
    }
    try:
        if result["module"]["file"]:
            module_path = Path(os.path.abspath(result["module"]["file"]))
            if any(module_path.is_relative_to(root) for root in stdlib_roots):
                result["module_file_chain"] = sdk_exact_file_chain(module_path)
        assert type(config) is dict, "Current sysconfig values are not already initialized"
        directory, name = config.get("LIBPL"), config.get("LDLIBRARY")
        assert type(directory) is str and Path(directory).is_absolute(), "Unknown absolute configured LIBPL"
        assert type(name) is str and name not in {"", ".", ".."} and Path(name).name == name
        assert "/" not in name and "\\" not in name and not any(ord(c) < 32 for c in name)
        assert not any(part in {".", ".."} for part in Path(directory).parts)
        candidate = Path(directory) / name
        result["candidate"] = str(candidate)
        result["candidate_in_declared_stdlib"] = any(candidate.is_relative_to(root) for root in stdlib_roots)
        if result["candidate_in_declared_stdlib"]:
            result["chain"] = sdk_exact_file_chain(candidate)
    except Exception as error:
        result["observation_error"] = {"type": type(error).__name__, "message": str(error)}
    return result


def sdk_register_configured_library(observed: dict, roots: set[Path], external: set[Path]) -> None:
    """Register only the saved exact configured chain, after its diagnostic receipt."""
    module = observed["module"]
    assert observed["config_initialized"] and not observed.get("observation_error"), observed
    assert module["is_module"] and module["is_module_spec"] and module["name"] == module["spec_name"] == "sysconfig"
    assert type(module["file"]) is str and type(module["origin"]) is str
    module_path = Path(os.path.abspath(module["file"]))
    assert module_path == Path(os.path.abspath(module["origin"]))
    assert any(module_path.is_relative_to(root) for root in roots), "Configuration source outside protected SDK roots"
    source_chain = observed["module_file_chain"]
    assert source_chain and source_chain["complete"] is True and len(source_chain["entries"]) == 1
    assert source_chain["entries"][0]["kind"] == "regular", (
        "Configuration source is not a direct protected regular file"
    )
    assert observed.get("candidate_in_declared_stdlib") is True
    chain = observed["chain"]
    assert chain and chain["complete"] is True and not chain.get("observation_error"), chain
    assert chain["origin"] == observed["candidate"] and chain["entries"][-1]["kind"] == "regular"
    for row in chain["entries"]:
        path = Path(row["path"])
        sdk_path_allowed(path)
        external.add(path)


def sdk_configured_library_guard_view(observed: dict, semantic: bool = False) -> dict:
    """Keep path identities; unrelated ancestor directory entries are not file bytes."""

    def project(value):
        if isinstance(value, list):
            return [project(item) for item in value]
        if not isinstance(value, dict):
            return value
        directory = value.get("kind") == "directory"
        result = {
            key: project(item)
            for key, item in value.items()
            if not (key == "identity" and semantic) and not (key == "nlink" and directory)
        }
        if directory and not semantic and "identity" in result:
            result["identity"] = result["identity"][:2]
        return result

    return project(observed)


def sdk(output: Path, budget: dict) -> dict:
    """Full relevant SDK trees even for legitimate providers without RECORD/files.

    Metadata locations are found by lstat directory names; headers/RECORD/selectors
    are parsed only after a saved bounded pre-read plan. No metadata provider API
    performs implicit METADATA/RECORD content reads inside this SDK scan.
    Only exact project install paths are excluded. All non-project SDK bytecode is protected;
    no bytecode/source equivalence or import safety is claimed.
    """
    state = {
        "phase": "provider-discovery",
        "providers": {},
        "roots": {},
        "complete": False,
        "scope": "Filesystem bytes, not a loaded-kernel or original failed-Pod attestation",
    }
    pass_start = dict(budget)
    save(
        output.with_suffix(".scan-start.json"),
        {
            "whole_budget": dict(budget),
            "per_pass_bytes_cap": MAX_SDK_PASS_BYTES,
            "per_pass_files_cap": MAX_SDK_PASS_FILES,
        },
    )
    try:
        excluded = {}
        python_roots = set()
        legacy_search = set()
        registered = {}
        roots = set()
        root_inputs = set()
        external = set()

        def add_root(path: Path, reason: str) -> None:
            root_inputs.add(Path(os.path.abspath(path)))
            path = path.resolve(strict=True)
            sdk_path_allowed(path)
            if path.is_file():
                external.add(path)
            else:
                assert path.is_dir(), str(path)
                roots.add(path)
            state["roots"].setdefault(str(path), []).append(reason)

        for name in ("purelib", "platlib", "stdlib", "platstdlib"):
            add_root(Path(sysconfig.get_path(name)), "sysconfig:" + name)
            python_roots.add(Path(sysconfig.get_path(name)).resolve(strict=True))
        for raw in sys.path:
            if not raw:
                continue
            path = Path(raw).resolve()
            if path.is_relative_to(ROOT) or path.is_relative_to(Path("/opt/vllm-omni")):
                continue
            if path.exists():
                add_root(Path(raw), "active non-project sys.path")
                if path.is_dir():
                    python_roots.add(path)
                    legacy_search.add(path)
            else:
                assert not Path(raw).is_symlink(), ("Dangling Python search-path link", raw)
                state.setdefault("absent_python_search_paths", []).append(str(Path(raw).absolute()))
        add_root(SDK_PRIVATE_CUDNN_PATH, "exact private cuDNN prefix")
        for raw in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
            if raw and Path(raw).exists():
                add_root(Path(raw), "unchanged LD_LIBRARY_PATH entry")
        for raw in os.environ.get("LD_PRELOAD", "").split():
            if raw:
                external.add(Path(raw).resolve(strict=True))
        if SDK_CUDA_PATH.exists():
            add_root(SDK_CUDA_PATH, "exact installed CUDA toolkit root")
        startup_before = sdk_startup_customization(
            roots, external, output.with_suffix(".startup-customization-before.json"), state
        )
        # Only automatic active site directories process immediate .pth files.
        # Merely appending a path does not recursively call site.addsitedir.
        site_module = sys.modules.get("site")
        state["site_selector_startup"] = {
            "python_version": sys.version,
            "no_site": bool(sys.flags.no_site),
            "no_user_site": bool(sys.flags.no_user_site),
            "site_module_loaded": site_module is not None,
        }
        assert site_module is not None and not sys.flags.no_site and sys.version_info[:2] == (3, 12)
        site_candidates = list(site_module.getsitepackages())
        if site_module.ENABLE_USER_SITE is True and not sys.flags.no_user_site:
            site_candidates.append(site_module.getusersitepackages())
        site_directories = {Path(raw).resolve() for raw in site_candidates} & legacy_search
        state["site_selector_observation"] = {
            "module_file": getattr(site_module, "__file__", None),
            "module_origin": getattr(getattr(site_module, "__spec__", None), "origin", None),
            "python_version": sys.version,
            "enable_user_site": site_module.ENABLE_USER_SITE,
            "site_candidates": site_candidates,
            "active_site_directories": sorted(str(path) for path in site_directories),
            "active_search_directories": sorted(str(path) for path in legacy_search),
            "locale_encoding": locale.getencoding(),
            "guard_path_encoding": "utf-8",
            "guard_line_rule": "str.splitlines",
            "actual_site_decoder": "unverified",
        }
        # Exact metadata/selector locations may extend protected roots, never suffix-recursive activation.
        read_selectors = set()
        captured_metadata = {}
        selector_checks = {}
        selector_bytes = 0
        for wave in range(16):
            state["phase"] = "selector-population-wave-" + str(wave)
            locations = sdk_metadata_locations(python_roots)
            state["metadata_locations"] = locations
            metadata_paths = {row[key] for row in locations.values() for key in ("header", "record") if row[key]}
            population = sdk_population(roots, external, excluded, selector_stage=True)
            selectors = {}
            selector_kinds = {}
            for raw, row in population.items():
                if raw in read_selectors or Path(raw) in excluded:
                    continue
                kind = (
                    "captured-metadata"
                    if raw in metadata_paths
                    else sdk_selector_kind(Path(raw), locations, site_directories, legacy_search)
                )
                if kind is None:
                    continue
                state["current_selector"] = {"path": raw, "classification": kind, "kind": row["kind"]}
                assert row["kind"] == "regular", ("Unsupported active selector file kind", raw, row["kind"])
                selectors[raw] = row
                selector_kinds[raw] = kind
                state.pop("current_selector", None)
            if not selectors:
                break
            plan = sdk_read_plan(selectors, budget, selector=True)
            plan["per_pass_bytes_cap"] = MAX_SDK_SELECTOR_BYTES - selector_bytes
            plan["per_pass_file_cap"] = MAX_SDK_PASS_FILES - (budget["files"] - pass_start["files"])
            save(
                output.with_suffix(f".selector-{wave}-before-read.json"),
                {
                    "providers": state["providers"],
                    "metadata_locations": locations,
                    "roots": state["roots"],
                    "population": population,
                    "selector_population": selectors,
                    "selector_classifications": selector_kinds,
                    "site_selector_observation": state["site_selector_observation"],
                    "budget": plan,
                },
            )
            hashes = sdk_read(selectors, budget, plan, retain_selector_bytes=True)
            selector_bytes += plan["planned_content_bytes"]
            captured_metadata.update({path: record["selector_bytes"] for path, record in hashes.items()})
            records = []
            descriptors = []
            state["metadata_provider_descriptors"] = descriptors
            for key in sorted(locations):
                state["metadata_parse_location"] = locations[key]
                record = SDKMetadataRecord(locations[key], captured_metadata)
                records.append(record)
                descriptors.append(record.descriptor)
            state.pop("metadata_parse_location", None)
            save(
                output.with_suffix(f".metadata-provider-descriptors-{wave}.json"),
                {"wave": wave, "instances": descriptors, "instance_count": len(descriptors)},
            )
            provider_plan = sdk_distribution_plan(records)
            state["providers"] = provider_plan["providers"]
            state["name_version_groups"] = provider_plan["name_version_groups"]
            excluded = provider_plan["project_exclusions"]
            registered = provider_plan["registered"]
            for key, provider in state["providers"].items():
                add_root(Path(provider["root"]), "distribution provider:" + key)
                external.update(Path(path) for path in provider["registered_paths"])
            for raw, row in selectors.items():
                path = Path(raw)
                data = hashes[raw]["selector_bytes"]
                state["current_selector"] = {
                    "path": raw,
                    "classification": selector_kinds[raw],
                    "sha256": hashes[raw]["sha256"],
                    "bytes": len(data),
                }
                selector_checks[raw] = {"identity": row["identity"], "sha256": hashes[raw]["sha256"]}
                assert hashlib.sha256(data).hexdigest() == hashes[raw]["sha256"]
                if path in excluded or raw in metadata_paths:
                    read_selectors.add(raw)
                    state.pop("current_selector", None)
                    continue
                if selector_kinds[raw] == "metadata-direct-url":
                    import urllib.parse

                    direct = json.loads(data)
                    if direct.get("dir_info", {}).get("editable"):
                        parsed = urllib.parse.urlparse(direct["url"])
                        assert parsed.scheme == "file" and parsed.netloc in {"", "localhost"}
                        add_root(Path(urllib.parse.unquote(parsed.path)), "non-project editable direct_url source")
                        python_roots.add(Path(urllib.parse.unquote(parsed.path)).resolve(strict=True))
                else:
                    # Retain E's strict passive closure guard; do not emulate unobserved site decoder internals.
                    for line in data.decode().splitlines():
                        value = line.rstrip()
                        if not value or value.startswith("#") or value.startswith(("import ", "import\t")):
                            continue
                        # .egg-link's second line is metadata rather than an additional source root.
                        if path.suffix == ".egg-link" and value == ".":
                            continue
                        target = Path(value)
                        target = target if target.is_absolute() else path.parent / target
                        if not target.exists():
                            assert not target.is_symlink(), ("Dangling SDK selector target", str(target))
                            state.setdefault("inert_absent_selector_targets", {})[
                                str(path) + "::" + str(target.absolute())
                            ] = str(target.absolute())
                            continue
                        add_root(target, "actual non-project selector target:" + str(path))
                        if target.is_dir():
                            python_roots.add(target.resolve(strict=True))
                            if selector_kinds[raw] == "site-pth":
                                legacy_search.add(target.resolve(strict=True))
                read_selectors.add(raw)
                state.pop("current_selector", None)
            assert sdk_population(roots, external, excluded, selector_stage=True), (
                "Empty SDK population during selector closure"
            )
        else:
            raise RuntimeError("SDK selector closure exceeded 16 bounded waves")
        state["roots"] = {path: sorted(set(reasons)) for path, reasons in state["roots"].items()}
        state["phase"] = "all-pending-external-candidates-before-strict-population"
        stdlib_roots = {
            Path(path)
            for path, reasons in state["roots"].items()
            if "sysconfig:stdlib" in reasons or "sysconfig:platstdlib" in reasons
        }
        configured_before = sdk_configured_library_snapshot(stdlib_roots)
        pending = {"population_complete": False, "candidates": {}, "configured_dependency": configured_before}
        state["pending_external_candidates"] = pending
        try:
            diagnostic_population = sdk_population(roots, set(external), excluded, selector_stage=True)
            pending["population_complete"] = True
            pending["population_count"] = len(diagnostic_population)
            pending["candidates"] = {
                raw: {
                    **row,
                    "reason": "exact configured dependency candidate"
                    if raw == configured_before["candidate"]
                    else "UNKNOWN: no exact configured dependency binding",
                }
                for raw, row in diagnostic_population.items()
                if row.get("coverage_pending_registered_metadata")
            }
            del diagnostic_population
            pending["chain_observation_cap"] = 256
            pending["chain_observation_budget_sufficient"] = len(pending["candidates"]) <= 256
            for raw, row in pending["candidates"].items():
                if pending["chain_observation_budget_sufficient"]:
                    row["chain_observation"] = sdk_exact_file_chain(Path(raw))
                else:
                    row["chain_observation"] = {"complete": False, "reason": "Pending-chain metadata cap exceeded"}
        except BaseException as error:
            pending["observation_error"] = {"type": type(error).__name__, "message": str(error)}
        # Always retain the complete known candidate set and configuration/chain
        # descriptors before deciding whether any outside file can be registered.
        save(output.with_suffix(".all-pending-external-candidates.json"), pending)
        assert pending["population_complete"], "Unknown final pending-candidate population; see saved diagnostic"
        assert pending["chain_observation_budget_sufficient"], "All candidate rows saved; chain metadata cap exceeded"
        configured_registered = configured_before["candidate"] in pending["candidates"]
        configured_chain_paths = set()
        if configured_registered:
            candidate_chain = pending["candidates"][configured_before["candidate"]]["chain_observation"]
            assert sdk_configured_library_guard_view(candidate_chain) == sdk_configured_library_guard_view(
                configured_before["chain"]
            ), "Configured chain changed during pending-candidate observation"
            sdk_register_configured_library(configured_before, roots, external)
            configured_chain_paths = {row["path"] for row in configured_before["chain"]["entries"]}
        unresolved = [raw for raw in pending["candidates"] if raw not in configured_chain_paths]
        assert not unresolved, ("Unregistered external candidates remain; full known set saved", unresolved)
        state["phase"] = "full-population-before-content-read"
        before = sdk_population(roots, external, excluded)
        if configured_registered:
            configured_source = str(Path(os.path.abspath(configured_before["module"]["file"])))
            assert before[configured_source]["kind"] == "regular"
            assert not before[configured_source].get("excluded_project")
        for path, observed in selector_checks.items():
            assert before[path]["identity"] == observed["identity"], (path, "selector changed during discovery")
        references = sdk_references(root_inputs | external)
        absent_paths = {Path(path) for path in state.get("absent_python_search_paths", [])}
        absent_paths.update(Path(path) for path in state.get("inert_absent_selector_targets", {}).values())
        absent_references = sdk_absent_references(absent_paths)
        plan = sdk_read_plan(before, budget)
        plan["per_pass_bytes_cap"] = MAX_SDK_PASS_BYTES - (budget["bytes"] - pass_start["bytes"])
        plan["per_pass_file_cap"] = MAX_SDK_PASS_FILES - (budget["files"] - pass_start["files"])
        save(
            output.with_suffix(".population-before-read.json"),
            {
                "providers": state["providers"],
                "roots": state["roots"],
                "population": before,
                "exact_project_file_exclusions": sorted(str(p) for p, kind in excluded.items() if kind == "regular"),
                "exact_project_metadata_directory_exclusions": sorted(
                    str(p) for p, kind in excluded.items() if kind == "directory"
                ),
                "budget": plan,
                "root_and_registered_path_references": references,
            },
        )
        state["phase"] = "full-content-read"
        hashes = sdk_read(before, budget, plan)
        assert all(
            hashes[path]["sha256"] == observed["sha256"]
            for path, observed in selector_checks.items()
            if Path(path) not in excluded
        )
        assert sdk_absent_references(absent_paths) == absent_references, "Absent SDK selectors/search paths changed"
        after = sdk_population(roots, external, excluded)
        sdk_population_unchanged(before, after)
        assert sdk_references(root_inputs | external) == references, "SDK selector/registered ancestor link changed"
        configured_after = sdk_configured_library_snapshot(stdlib_roots)
        save(output.with_suffix(".configured-dependency-after.json"), configured_after)
        assert sdk_configured_library_guard_view(configured_after) == sdk_configured_library_guard_view(
            configured_before
        ), "Configured dependency source or chain changed during SDK read"
        startup_after = sdk_startup_customization(
            roots, external, output.with_suffix(".startup-customization-after.json"), state
        )
        assert startup_after == startup_before, "Loaded customization binding or file identity changed during SDK read"
        result = {
            "providers": state["providers"],
            "name_version_groups": state["name_version_groups"],
            "roots": state["roots"],
            "registered_paths": registered,
            "protected_population": sdk_protected_projection(before),
            "root_and_registered_path_references": references,
            "absent_python_search_paths": sorted(set(state.get("absent_python_search_paths", []))),
            "inert_absent_selector_targets": state.get("inert_absent_selector_targets", {}),
            "absent_path_references": absent_references,
            "regular_files": hashes,
            "configured_library_dependency": {
                **sdk_configured_library_guard_view(configured_before, semantic=True),
                "exact_configured_chain_registered": configured_registered,
            },
            "startup_customization": {
                "module": startup_before["module"],
                "stdlib_roots": startup_before["stdlib_roots"],
                "entries": {
                    path: {key: value for key, value in row.items() if key not in {"identity", "target_identity"}}
                    for path, row in startup_before["entries"].items()
                },
            },
            "loader_environment": {key: os.environ.get(key) for key in ENV_KEYS},
        }
        state.update(phase="complete", complete=True)
        save(
            output.with_suffix(".population-after-read.json"),
            {
                "population": after,
                "whole_budget_after": dict(budget),
                "exact_project_file_exclusions": sorted(str(p) for p, kind in excluded.items() if kind == "regular"),
                "exact_project_metadata_directory_exclusions": sorted(
                    str(p) for p, kind in excluded.items() if kind == "directory"
                ),
            },
        )
        save(
            output.with_suffix(".scan-complete.json"),
            {
                "complete": True,
                "whole_budget_after": dict(budget),
                "protected_entries": len(result["protected_population"]),
            },
        )
        return result
    except BaseException as error:
        state.update(error={"type": type(error).__name__, "message": str(error)}, whole_budget=dict(budget))
        save(output.with_suffix(".incomplete-scan.json"), state)
        raise


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
    sdk_budget = {"bytes": 0, "files": 0}
    baseline_sdk = sdk(output / "sdk-before.json", sdk_budget)
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
            after_install = sdk(role_output / "sdk-after-install.json", sdk_budget)
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
                after_native = sdk(role_output / "sdk-after-native.json", sdk_budget)
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
