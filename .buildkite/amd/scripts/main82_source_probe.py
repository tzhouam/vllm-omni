# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Observe a diagnostic checkout without changing its source or index.

This CPU-only step is not an image publisher, model test or qualification run.
Only metadata, hashes and changed paths are retained; source content is omitted.
"""

import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BASE = "82fc4f511fe02bb933d01c5b5f24e1f8a9e5b415"
BRANCH = "codex/pr9009-amd-main82-source-probe-20261106"
HELPER = ".buildkite/amd/scripts/main82_source_probe.py"
CHANGES = {HELPER: "100644", ".buildkite/amd/scripts/bootstrap-amd-omni.sh": "100755"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def command(root, *args):
    # Optional index refresh is forbidden; no persistent Git configuration is set.
    environment = os.environ.copy()
    # The empty whitelist blocks every transport even on Git 2.34, where
    # --no-lazy-fetch is unavailable. Newer Git also honors the second guard.
    environment["GIT_ALLOW_PROTOCOL"] = ""
    environment["GIT_NO_LAZY_FETCH"] = "1"
    argv = ["git", "--no-optional-locks", "-C", str(root), *args]
    try:
        result = subprocess.run(argv, capture_output=True, check=False, timeout=10, env=environment)
        return {
            "argv": argv,
            "returncode": result.returncode,
            "stdout_hex": result.stdout.hex(),
            "stderr_hex": result.stderr.hex(),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "argv": argv,
            "returncode": None,
            "error_type": "TimeoutExpired",
            "timeout_seconds": 10,
            "stdout_hex": (exc.stdout or b"").hex(),
            "stderr_hex": (exc.stderr or b"").hex(),
        }
    except OSError as exc:
        return {
            "argv": argv,
            "returncode": None,
            "error_type": type(exc).__name__,
            "errno": exc.errno,
            "stdout_hex": "",
            "stderr_hex": "",
        }


def output(root, records, *args):
    record = command(root, *args)
    records.append(record)
    require(record["returncode"] == 0, "Git metadata command failed: " + " ".join(args))
    return bytes.fromhex(record["stdout_hex"])


def identity(root, records, verify_paths=True):
    expected = os.environ.get("EXPECTED_PROBE_COMMIT", "")
    require(len(expected) == 40 and all(c in "0123456789abcdef" for c in expected), "external probe commit required")
    require(os.environ.get("AMD_MAIN82_SOURCE_PROBE") == "1", "explicit source probe flag required")
    require(os.environ.get("BUILDKITE_SOURCE") == "api", "diagnostic API context required")
    require(os.environ.get("BUILDKITE_PULL_REQUEST") == "9009", "diagnostic PR9009 fork context required")
    require(os.environ.get("BUILDKITE_BRANCH") == "tzhouam:" + BRANCH, "exact diagnostic fork branch required")
    require(
        os.environ.get("BUILDKITE_PULL_REQUEST_REPO")
        in {"https://github.com/tzhouam/vllm-omni", "https://github.com/tzhouam/vllm-omni.git"},
        "exact diagnostic fork repository required",
    )
    require(os.environ.get("BUILDKITE_COMMIT") == expected, "Buildkite commit differs from external lock")
    require(
        output(root, records, "rev-parse", "HEAD").decode().strip() == expected, "checkout HEAD differs from probe lock"
    )
    parents = [
        row[7:]
        for row in output(root, records, "cat-file", "-p", "HEAD").decode().splitlines()
        if row.startswith("parent ")
    ]
    require(parents == [BASE], "probe must have the sole literal main82 parent")
    if verify_paths:
        changed = set(
            output(root, records, "diff", "--name-only", "-z", BASE, expected).decode().strip("\0").split("\0")
        )
        require(changed == set(CHANGES), "probe must change exactly the two diagnostic CI paths")
    for path, wanted in CHANGES.items():
        mode, kind, _ = output(root, records, "ls-tree", expected, "--", path).decode().split(None, 2)
        require(mode == wanted and kind == "blob", "diagnostic path mode or type differs")
    return {
        "expected_commit": expected,
        "actual_commit": expected,
        "sole_parent": BASE,
        "expected_changed_paths": CHANGES,
        "actual_changed_paths": sorted(CHANGES) if verify_paths else None,
        "parent_tree_comparison_completed": verify_paths,
    }


def tree_rows(raw):
    rows = []
    for entry in raw.split(b"\0"):
        if entry:
            header, path = entry.split(b"\t", 1)
            mode, kind, oid = header.decode().split()
            rows.append({"path_hex": path.hex(), "path": os.fsdecode(path), "mode": mode, "kind": kind, "blob": oid})
    return rows


def index_rows(raw):
    rows = []
    for entry in raw.split(b"\0"):
        if entry:
            header, path = entry.split(b"\t", 1)
            mode, oid, stage = header.decode().split()
            rows.append(
                {"path_hex": path.hex(), "path": os.fsdecode(path), "mode": mode, "blob": oid, "stage": int(stage)}
            )
    return rows


def worktree_row(root, expected):
    relative = bytes.fromhex(expected["path_hex"])
    parts = relative.split(b"/")
    require(parts and all(part not in {b"", b".", b".."} for part in parts), "unsafe Git path")
    current = os.fsencode(root)
    row = {"path_hex": relative.hex(), "path": os.fsdecode(relative)}
    try:
        for part in parts[:-1]:
            current = os.path.join(current, part)
            meta = os.lstat(current)
            if not stat.S_ISDIR(meta.st_mode):
                return {**row, "state": "blocked_by_non_directory_ancestor"}
        target = os.path.join(current, parts[-1])
        meta = os.lstat(target)
        row["permission_bits"] = stat.S_IMODE(meta.st_mode)
        if stat.S_ISLNK(meta.st_mode):
            payload = os.readlink(target)
            row["mode"] = "120000"
        elif stat.S_ISREG(meta.st_mode):
            fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as stream:
                before = os.fstat(stream.fileno())
                require(stat.S_ISREG(before.st_mode), "tracked file changed type while reading")
                require(
                    (meta.st_dev, meta.st_ino, meta.st_mode) == (before.st_dev, before.st_ino, before.st_mode),
                    "tracked file changed before reading",
                )
                payload = stream.read()
                after = os.fstat(stream.fileno())
                require(
                    (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                    == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                    "tracked file changed while reading",
                )
            row["mode"] = "100755" if meta.st_mode & stat.S_IXUSR else "100644"
        else:
            return {**row, "state": "unsupported_file_type"}
        row.update(
            state="present",
            bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            blob=hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest(),
        )
    except FileNotFoundError:
        row["state"] = "missing"
    except OSError as exc:
        row.update(state="read_error", error_type=type(exc).__name__, errno=exc.errno)
    except RuntimeError as exc:
        row.update(state="unstable_read", error_type=type(exc).__name__, reason=str(exc))
    return row


def snapshot(root):
    commands = {
        "version": command(root, "--version"),
        "head": command(root, "rev-parse", "HEAD"),
        "head_tree": command(root, "rev-parse", "HEAD^{tree}"),
        "status_porcelain_v2_z": command(root, "status", "--porcelain=v2", "-z", "--untracked-files=all"),
        "head_manifest": command(root, "ls-tree", "-r", "-z", "--full-tree", "HEAD"),
        "index_manifest": command(root, "ls-files", "--stage", "-z"),
    }
    # These are the original rejection condition, with their exact native errors.
    for label, args in {"unstaged": (), "cached": ("--cached",)}.items():
        commands[label + "_original_name_only"] = command(root, "diff", *args, "--name-only")
        for kind in ("--name-only", "--raw", "--numstat"):
            commands[label + "_" + kind[2:].replace("-", "_") + "_z"] = command(
                root, "diff", *args, kind, "-z", "--no-ext-diff", "--no-textconv"
            )
    configs = {}
    for name in ("core.filemode", "core.autocrlf", "core.eol"):
        configs[name] = command(root, "config", "--get", name)
    head = (
        tree_rows(bytes.fromhex(commands["head_manifest"]["stdout_hex"]))
        if commands["head_manifest"]["returncode"] == 0
        else []
    )
    index = (
        index_rows(bytes.fromhex(commands["index_manifest"]["stdout_hex"]))
        if commands["index_manifest"]["returncode"] == 0
        else []
    )
    for name in ("head_manifest", "index_manifest"):
        raw = bytes.fromhex(commands[name].pop("stdout_hex"))
        commands[name].update(
            stdout_bytes=len(raw),
            stdout_sha256=hashlib.sha256(raw).hexdigest(),
            stdout_omitted="full manifest; only changed rows are retained",
        )
    by_head = {row["path_hex"]: row for row in head}
    by_index = {}
    for row in index:
        by_index.setdefault(row["path_hex"], []).append(row)
    index_changes = []
    for path in sorted(set(by_head) | set(by_index)):
        wanted = by_head.get(path)
        actual = by_index.get(path, [])
        same = wanted is not None and len(actual) == 1 and actual[0]["stage"] == 0
        same = same and actual[0]["mode"] == wanted["mode"] and actual[0]["blob"] == wanted["blob"]
        if not same:
            index_changes.append({"path_hex": path, "head": wanted, "index_stages": actual})
    tracked = dict(by_head)
    for row in index:
        tracked.setdefault(row["path_hex"], {**row, "kind": "blob"})
    expected_work = [tracked[path] for path in sorted(tracked)]
    work = [worktree_row(root, row) for row in expected_work]
    differences = []
    for expected, actual in zip(expected_work, work):
        if (
            expected["path_hex"] not in by_head
            or actual.get("mode") != expected["mode"]
            or actual.get("blob") != expected["blob"]
        ):
            differences.append(
                {
                    "head": by_head.get(expected["path_hex"]),
                    "index_stages": by_index.get(expected["path_hex"], []),
                    "worktree": actual,
                }
            )
    original = [commands[key + "_original_name_only"] for key in ("unstaged", "cached")]
    return {
        "commands": commands,
        "safe_config": configs,
        "head_entries": len(head) if commands["head_manifest"]["returncode"] == 0 else None,
        "head_manifest_sha256": digest(head) if commands["head_manifest"]["returncode"] == 0 else None,
        "index_entries": len(index) if commands["index_manifest"]["returncode"] == 0 else None,
        "index_manifest_sha256": digest(index) if commands["index_manifest"]["returncode"] == 0 else None,
        "index_changes": index_changes,
        "worktree_entries": len(work),
        "worktree_manifest_sha256": digest(work),
        "worktree_population_complete": all(
            commands[name]["returncode"] == 0 for name in ("head_manifest", "index_manifest")
        ),
        "worktree_changes": differences,
        "original_dirty_check_commands_succeeded": all(row["returncode"] == 0 for row in original),
        "original_dirty_check_rejects": any(bool(row["stdout_hex"]) for row in original),
        "all_metadata_commands_succeeded": all(row["returncode"] == 0 for row in commands.values()),
    }


def artifact_directory(root):
    current = root
    for part in ("artifacts", "amd-main82-source-probe"):
        current = current / part
        current.mkdir(exist_ok=True)
        require(stat.S_ISDIR(current.lstat().st_mode), "artifact parent must be a real directory")
    return Path(tempfile.mkdtemp(prefix="receipt-", dir=current))


def run(root, stream=None, bootstrap_identity_commands=None):
    report = {"started_epoch": time.time(), "complete": False, "qualification_passed": False, "identity_commands": []}
    if bootstrap_identity_commands is not None:
        report["bootstrap_identity_commands"] = bootstrap_identity_commands
    print("AMD source probe started: complete=false qualification_passed=false", flush=True, file=stream)
    code = 1
    try:
        # Both snapshots finish before the first write to checkout/artifacts.
        report["before"] = snapshot(root)
        try:
            report["identity"] = identity(root, report["identity_commands"])
        finally:
            report["after"] = snapshot(root)
        for label in ("before", "after"):
            observed = report[label]
            require(observed["original_dirty_check_commands_succeeded"], "original dirty-check command failed")
            require(not observed["original_dirty_check_rejects"], "tracked checkout dirty")
            require(observed["all_metadata_commands_succeeded"], "source metadata incomplete")
            require(
                not observed["index_changes"] and not observed["worktree_changes"],
                "tracked bytes/modes/index differ from HEAD",
            )
            require(not observed["commands"]["status_porcelain_v2_z"]["stdout_hex"], "checkout status is not empty")
        report["complete"] = True
        code = 0
    except BaseException as exc:
        report["exception"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        report["finished_epoch"] = time.time()
        report["limits"] = [
            "CPU-only metadata; no source/index/config mutation, fetch, image publisher, "
            "framework, model or native tests.",
            "Same amd-cpu queue and three publisher environment entries; label/key/command/timeout "
            "differ and provider hook equivalence is not claimed.",
            "Original dirty-check failure is retained; byte/mode observations do not normalize or waive it.",
            "Probe Git child environment denies all transports and optional index writes; original publisher "
            "Git commands had no added transport denial. Missing objects are unknown/guarded failure, "
            "not a reproduced dirty cause.",
        ]
        # Preserve observations in the original log even if the artifact parent
        # itself is unavailable or a symlink. Never follow/repair that parent.
        print("AMD source probe checkpoint: " + json.dumps(report, sort_keys=True), flush=True, file=stream)
        try:
            target = artifact_directory(root) / "source-state.json"
            with target.open("x") as receipt_file:
                json.dump(report, receipt_file, indent=2, sort_keys=True)
                receipt_file.write("\n")
            print("AMD source probe receipt: " + str(target), flush=True, file=stream)
        except (OSError, RuntimeError) as exc:
            print("AMD source probe artifact failure: " + type(exc).__name__, flush=True, file=stream)
            code = 1
    return code


def pipeline():
    return {
        "steps": [
            {
                "label": "AMD: CPU-only main82 source-state probe",
                "key": "amd-main82-source-probe",
                "agents": {"queue": "amd-cpu"},
                "env": {
                    "DOCKER_BUILDKIT": "1",
                    "PYTORCH_ROCM_ARCH": "gfx942",
                    "ROCM_CI_IMAGE_REGISTRY": "rocm/vllm-omni",
                },
                "command": "python3 -I -S .buildkite/amd/scripts/main82_source_probe.py run",
                "timeout_in_minutes": 10,
                "artifact_paths": ["artifacts/amd-main82-source-probe/**/*"],
            }
        ]
    }


def main():
    require(len(sys.argv) == 2 and sys.argv[1] in {"bootstrap", "run"}, "expected bootstrap or run")
    root = Path.cwd().resolve()
    if sys.argv[1] == "bootstrap":
        records = []
        try:
            # The worker verifies the parent tree after retaining observations;
            # uploader identity needs only the local HEAD/header and CI entries.
            identity(root, records, verify_paths=False)
        except BaseException:
            # A rejected uploader still emits its observations to stderr, never
            # contaminating the JSON pipeline on stdout.
            run(root, stream=sys.stderr, bootstrap_identity_commands=records)
            raise
        print(json.dumps(pipeline()))
        return 0
    return run(root)


if __name__ == "__main__":
    sys.exit(main())
