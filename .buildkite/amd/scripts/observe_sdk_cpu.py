# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Observe one published SDK filesystem; never import it or run native tests."""

import hashlib
import importlib.machinery as machinery
import importlib.util
import json
import os
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path

BASE = "82fc4f511fe02bb933d01c5b5f24e1f8a9e5b415"
BRANCH = "codex/pr9009-amd-sdk-cpu-pod-observation-v2-20261106"
HELPER = ".buildkite/amd/scripts/observe_sdk_cpu.py"
BOOTSTRAP = ".buildkite/amd/scripts/bootstrap-amd-omni.sh"
IMAGE = (
    "docker.io/rocm/vllm-omni:edfeca8c08c7398d56a5070ef2ca612293e50d21"
    "@sha256:2f86ada64b40bccc3bc29846c0c13a1a245c186f1e986fb1600d218c24b9acfa"
)
SDK = Path("/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/share/amd_smi")
GIT_OBSERVATIONS = []
WORK_DEADLINE = None
PROTECTED = {
    "vllm_omni",
    "tests",
    "sitecustomize",
    "usercustomize",
    "vllm",
    "torch",
    "torchaudio",
    "transformers",
    "numpy",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save(path, data):
    require(not path.is_symlink(), "symlink report rejected")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def file_identity(path, content=False):
    info = path.lstat()
    row = {
        "path": str(path),
        "mode": stat.S_IMODE(info.st_mode),
        "uid": info.st_uid,
        "gid": info.st_gid,
        "device": info.st_dev,
        "inode": info.st_ino,
        "size": info.st_size,
        "type": "symlink"
        if stat.S_ISLNK(info.st_mode)
        else "directory"
        if stat.S_ISDIR(info.st_mode)
        else "regular"
        if stat.S_ISREG(info.st_mode)
        else "other",
    }
    try:
        if row["type"] == "symlink":
            row["target"] = os.readlink(path)
        row["realpath"] = str(path.resolve(strict=True))
        if content and row["type"] == "regular":
            require(info.st_size <= 64 * 1024 * 1024, "observation file exceeds 64MiB read budget")
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as stream:
                before = os.fstat(stream.fileno())
                require(stat.S_ISREG(before.st_mode), "observation file changed type")
                digest = hashlib.sha256()
                header = stream.read(16)
                digest.update(header)
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                after = os.fstat(stream.fileno())
            require(
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                "observation file changed during read",
            )
            require((before.st_dev, before.st_ino) == (info.st_dev, info.st_ino), "observation inode changed")
            row.update(sha256=digest.hexdigest(), first16_hex=header.hex())
    except (OSError, RuntimeError, ValueError) as error:
        row["observation_error"] = {"type": type(error).__name__, "message": str(error)}
    return row


def finder(path):
    return machinery.FileFinder(
        str(path),
        (machinery.SourceFileLoader, machinery.SOURCE_SUFFIXES),
        (machinery.SourcelessFileLoader, machinery.BYTECODE_SUFFIXES),
        (machinery.ExtensionFileLoader, machinery.EXTENSION_SUFFIXES),
    )


def inspect_sdk(output, sdk=SDK):
    require(output.parent.is_dir() and not output.exists(), "new evidence output required")
    report = {
        "scope": "stdlib filesystem observations, not an import approval",
        "observation_complete": False,
        "qualification_passed": False,
        "sdk_code_executed": False,
        "started_epoch": time.time(),
        "interpreter": sys.executable,
        "python_version": sys.version,
        "cache_tag": sys.implementation.cache_tag,
        "uid": os.getuid(),
        "gid": os.getgid(),
        "observer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "magic_number_hex": importlib.util.MAGIC_NUMBER.hex(),
        "isolated": sys.flags.isolated,
        "no_site": sys.flags.no_site,
        "sys_path": sys.path,
        "cwd": str(Path.cwd()),
        "selected_environment": {
            name: os.environ.get(name) for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSAFEPATH", "PYTHONNOUSERSITE")
        },
        "sdk": str(sdk),
        "top_level_inventory": [],
        "specs": {},
        "cache_children": [],
        "errors": [],
    }

    def attempt(label, operation):
        try:
            value = operation()
            if isinstance(value, dict) and value.get("observation_error"):
                report["errors"].append({"operation": label, **value["observation_error"]})
            return value
        except (OSError, RuntimeError, ValueError, ImportError) as error:
            report["errors"].append({"operation": label, "type": type(error).__name__, "message": str(error)})
            return None
        finally:
            save(output, report)

    try:
        save(output, report)
        report["phase"] = "root_identity"
        report["root_identity"] = attempt("sdk_identity", lambda: file_identity(sdk))
        report["symlink_components"] = []
        cursor = Path(sdk.anchor)
        for component in sdk.parts[1:]:
            cursor /= component
            if cursor.is_symlink():
                report["symlink_components"].append({"path": str(cursor), "target": os.readlink(cursor)})
        report["phase"] = "top_level_inventory"
        children = attempt("list_sdk", lambda: sorted(sdk.iterdir()))
        names = set(PROTECTED) | {"amdsmi"}
        for child in children or []:
            row = attempt("top_identity:" + child.name, lambda child=child: file_identity(child, content=True))
            report["top_level_inventory"].append(row or {"path": str(child), "status": "UNKNOWN"})
            if row and row["type"] == "directory" and child.name.isidentifier():
                names.add(child.name)
            for suffix in machinery.all_suffixes():
                if child.name.endswith(suffix) and child.name[: -len(suffix)].isidentifier():
                    names.add(child.name[: -len(suffix)])
        report["phase"] = "all_top_level_specs"
        filesystem = finder(sdk)
        for name in sorted(names):

            def spec_record(name=name):
                spec = filesystem.find_spec(name)  # no loader execution or ambient path hooks
                if spec is None:
                    return {"status": "absent"}
                row = {
                    "status": "offered",
                    "origin": spec.origin,
                    "loader": type(spec.loader).__name__,
                    "search_locations": list(spec.submodule_search_locations or []),
                    "protected_name": name in PROTECTED,
                    "v9_would_reject_provider": name != "amdsmi",
                }
                if spec.origin is not None:
                    row["origin_identity"] = attempt(
                        "provider_origin:" + name, lambda: file_identity(Path(spec.origin), content=True)
                    )
                row["search_location_identities"] = [
                    attempt("provider_location:" + name, lambda value=value: file_identity(Path(value)))
                    for value in row["search_locations"]
                ]
                return row

            report["specs"][name] = attempt("find_spec:" + name, spec_record) or {"status": "UNKNOWN"}
        report["phase"] = "exact_cache_children"
        cache = sdk / "__pycache__"
        report["cache_identity"] = (
            attempt("cache_identity", lambda: file_identity(cache)) if os.path.lexists(cache) else {"status": "absent"}
        )
        cache_info = report["cache_identity"]
        if cache_info and cache_info.get("type") == "directory":
            cache_children = attempt("list_cache", lambda: sorted(cache.iterdir()))
            for child in cache_children or []:
                row = attempt("cache_child:" + child.name, lambda child=child: file_identity(child, content=True))
                if row and row["type"] == "regular" and child.suffix == ".pyc":
                    try:
                        source = importlib.util.source_from_cache(str(child))
                        row["source_from_cache"] = source
                        row["matching_source"] = (
                            attempt(
                                "cache_source:" + child.name,
                                lambda source=source: file_identity(Path(source), content=True),
                            )
                            if os.path.lexists(source)
                            else {"status": "absent", "path": source}
                        )
                    except ValueError as error:
                        row["cache_filename_error"] = {"type": type(error).__name__, "message": str(error)}
                    row["header_status"] = "observed" if "first16_hex" in row else "UNKNOWN"
                    row["current_magic_number_matches"] = (
                        row["first16_hex"][:8] == report["magic_number_hex"] if "first16_hex" in row else None
                    )
                report["cache_children"].append(row or {"path": str(child), "status": "UNKNOWN"})
                save(output, report)
        report["modules_imported_from_sdk_or_framework"] = [
            name for name in sys.modules if name.split(".")[0] in PROTECTED | {"amdsmi"}
        ]
        require(not report["modules_imported_from_sdk_or_framework"], "unexpected framework or SDK module loaded")
        report["observation_complete"] = not report["errors"]
        report["phase"] = "complete" if report["observation_complete"] else "complete_with_observation_errors"
        return 0 if report["observation_complete"] else 1
    except BaseException as error:
        report["fatal_error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["finished_epoch"] = time.time()
        report["limits"] = [
            "All provider specs are observations, never authorization to import them.",
            "Image environment is not the former pod environment; -I -S ignores Python path/site startup.",
            "Cache filename/header/source observations do not certify executable cache/SDK code.",
            "No framework, model, checkpoint, tensor or native test is executed.",
        ]
        save(output, report)


def git(root, *args):
    require(WORK_DEADLINE is None or time.monotonic() < WORK_DEADLINE, "controller work deadline exceeded")
    process = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(root), *args],
        capture_output=True,
        timeout=20 if WORK_DEADLINE is None else min(20, max(0.001, WORK_DEADLINE - time.monotonic())),
        env=dict(os.environ, GIT_ALLOW_PROTOCOL="", GIT_NO_LAZY_FETCH="1"),
    )
    GIT_OBSERVATIONS.append(
        {
            "args": list(args),
            "returncode": process.returncode,
            "stdout_size": len(process.stdout),
            "stdout_sha256": hashlib.sha256(process.stdout).hexdigest(),
            "stderr_hex": process.stderr.hex(),
        }
    )
    require(process.returncode == 0, "local Git identity command failed: " + " ".join(args))
    return process.stdout


def identity(root):
    expected = os.environ.get("EXPECTED_SDK_PROBE_COMMIT", "")
    require(len(expected) == 40 and all(c in "0123456789abcdef" for c in expected), "external source commit required")
    require(os.environ.get("AMD_SDK_CPU_OBSERVATION") == "1", "explicit CPU observation flag required")
    require(os.environ.get("BUILDKITE_SOURCE") == "api", "API-only diagnostic context required")
    require(os.environ.get("BUILDKITE_PULL_REQUEST") == "9009", "exact diagnostic PR context required")
    require(os.environ.get("BUILDKITE_BRANCH") == "tzhouam:" + BRANCH, "exact diagnostic branch required")
    require(
        os.environ.get("BUILDKITE_PULL_REQUEST_REPO")
        in {"https://github.com/tzhouam/vllm-omni", "https://github.com/tzhouam/vllm-omni.git"},
        "fork required",
    )
    require(os.environ.get("BUILDKITE_COMMIT") == expected, "source differs from external lock")
    require(git(root, "rev-parse", "HEAD").decode().strip() == expected, "checkout differs from lock")
    parents = [
        line[7:] for line in git(root, "cat-file", "-p", "HEAD").decode().splitlines() if line.startswith("parent ")
    ]
    require(parents == [BASE], "sole main82 parent required")
    require(not (root / HELPER).is_symlink() and (root / HELPER).is_file(), "observer must be regular")
    require(not (root / BOOTSTRAP).is_symlink() and (root / BOOTSTRAP).is_file(), "bootstrap must be regular")
    helper_bytes = (root / HELPER).read_bytes()
    require(Path(__file__).read_bytes() == helper_bytes, "executing observer differs from source path")
    require(helper_bytes == git(root, "show", "HEAD:" + HELPER), "observer differs from Git blob")
    require(
        (root / BOOTSTRAP).read_bytes() == git(root, "show", "HEAD:" + BOOTSTRAP), "bootstrap differs from Git blob"
    )
    return {
        "commit": expected,
        "sole_parent": BASE,
        "whole_committed_tree_checked_by_worker": False,
        "whole_committed_tree_proof": "required root publication receipt; not recomputed by worker",
        "base_object_required": False,
        "helper_sha256": hashlib.sha256(helper_bytes).hexdigest(),
    }


def pipeline():
    return {
        "steps": [
            {
                "label": "AMD: CPU-only direct-pod EDF SDK filesystem observation",
                "key": "amd-sdk-cpu-pod-observation-v2",
                "agents": {"queue": "amd-cpu"},
                "command": "/usr/bin/python3 -I -S -B .buildkite/amd/scripts/observe_sdk_cpu.py pod",
                "timeout_in_minutes": 10,
                "artifact_paths": ["artifacts/amd-sdk-cpu-pod-observation/**/*"],
                "plugins": [
                    {
                        "kubernetes": {
                            "podSpecPatch": {
                                "automountServiceAccountToken": False,
                                "securityContext": {"seccompProfile": {"type": "RuntimeDefault"}},
                                "imagePullSecrets": [{"name": "docker-config"}],
                                "containers": [
                                    {
                                        "name": "container-0",
                                        "image": IMAGE,
                                        "imagePullPolicy": "Always",
                                        "securityContext": {
                                            "privileged": False,
                                            "allowPrivilegeEscalation": False,
                                            "capabilities": {"add": [], "drop": ["ALL"]},
                                        },
                                        "resources": {
                                            "requests": {"amd.com/gpu": "0", "nvidia.com/gpu": "0"},
                                            "limits": {"amd.com/gpu": "0", "nvidia.com/gpu": "0"},
                                        },
                                    }
                                ],
                            }
                        }
                    }
                ],
            }
        ]
    }


def pod(root):
    """Observe inside the requested CPU pod; never invoke the host Docker path."""
    parent = root / "artifacts"
    if parent.exists():
        require(parent.is_dir() and not parent.is_symlink(), "unsafe artifact directory")
    else:
        parent.mkdir(mode=0o755)
        parent.chmod(0o755)
    target = parent / "amd-sdk-cpu-pod-observation"
    require(not target.exists() and not target.is_symlink(), "new own evidence directory required")
    target.mkdir(mode=0o755)
    target.chmod(0o755)
    output = target / uuid.uuid4().hex
    output.mkdir(mode=0o755)
    output.chmod(0o755)
    receipt_path = output / "pod-source-receipt.json"
    sdk_path = output / "sdk-observation.json"
    receipt = {
        "phase": "source_identity",
        "observation_complete": False,
        "qualification_passed": False,
        "requested_image": IMAGE,
        "actual_resolved_image_id": None,
        "actual_provider_pod_spec": None,
        "actor_uid": os.getuid(),
        "actor_gid": os.getgid(),
        "interpreter": sys.executable,
        "python_version": sys.version,
        "isolated": sys.flags.isolated,
        "no_site": sys.flags.no_site,
        "cwd": str(root),
        "started_epoch": time.time(),
        "git_commands": GIT_OBSERVATIONS,
        "host_docker_path_invoked": False,
    }
    previous_umask = os.umask(0o022)
    result = 1
    try:
        save(receipt_path, receipt)
        receipt["source"] = identity(root)
        require(sys.flags.isolated == 1 and sys.flags.no_site == 1, "isolated no-site observer required")
        require(sys.executable == "/usr/bin/python3", "observer interpreter differs from actual preflight")
        receipt["phase"] = "stdlib_sdk_filesystem_observation"
        save(receipt_path, receipt)
        result = inspect_sdk(sdk_path)
        observed = json.loads(sdk_path.read_text())
        require(observed["observer_sha256"] == receipt["source"]["helper_sha256"], "observer byte identity differs")
        receipt["source_after"] = identity(root)
        require(receipt["source_after"] == receipt["source"], "source identity changed during observation")
        receipt["sdk_report"] = {
            "file": sdk_path.name,
            "bytes": sdk_path.stat().st_size,
            "sha256": hashlib.sha256(sdk_path.read_bytes()).hexdigest(),
            "mode": stat.S_IMODE(sdk_path.stat().st_mode),
        }
        receipt["observation_complete"] = result == 0 and observed.get("observation_complete") is True
        receipt["phase"] = "complete" if receipt["observation_complete"] else "complete_with_observation_errors"
    except BaseException as error:
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        result = 1
    finally:
        receipt["finished_epoch"] = time.time()
        receipt["limits"] = [
            "Direct CPU-pod diagnostic only; no host Docker controller, SDK import, framework or native test.",
            "Requested Kubernetes image/security/zero-GPU resources are configuration, not actual pod readback.",
            "Actual resolved image ID and provider pod spec remain unknown without separate provider evidence.",
            "No readonly-rootfs or network-none claim is copied from the unexecuted host Docker path.",
            "All provider/cache observations remain evidence, never permission to import or qualification.",
            "Only newly owned artifact directories use 0755 and nonsecret reports 0644 for uploader access; "
            "no unrelated directory or source permission is changed.",
        ]
        save(receipt_path, receipt)
        os.umask(previous_umask)
        print("AMD SDK CPU pod receipt: " + str(receipt_path), flush=True)
    return result


def run(root):
    global WORK_DEADLINE
    WORK_DEADLINE = time.monotonic() + 360
    cleanup_deadline = None
    target = root / "artifacts/amd-sdk-cpu-observation"
    for part in [root / "artifacts", target]:
        part.mkdir(mode=0o700, exist_ok=True)
        require(part.is_dir() and not part.is_symlink(), "unsafe artifact directory")
    output = target / uuid.uuid4().hex
    output.mkdir(mode=0o700)
    receipt = {
        "phase": "host_identity",
        "observation_complete": False,
        "qualification_passed": False,
        "image": IMAGE,
        "controller_work_budget_seconds": 360,
        "separate_cleanup_budget_seconds": 60,
        "job_timeout_minutes": 10,
        "started_epoch": time.time(),
        "commands": [],
        "git_commands": GIT_OBSERVATIONS,
        "selected_environment": {
            name: os.environ.get(name) for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSAFEPATH", "PYTHONNOUSERSITE")
        },
        "interpreter": sys.executable,
        "python_version": sys.version,
        "cwd": str(root),
    }
    path = output / "host-receipt.json"
    name = "amd-sdk-observe-" + output.name
    creation_attempted = False
    exit_code = 1

    def command(label, argv, timeout=30):
        row = {"label": label, "argv": argv}
        receipt["commands"].append(row)
        save(path, receipt)
        try:
            deadline = cleanup_deadline if label.startswith("cleanup-") else WORK_DEADLINE
            remaining = deadline - time.monotonic()
            require(remaining > 0, "controller phase deadline exceeded")
            row["effective_timeout_seconds"] = min(timeout, remaining)
            process = subprocess.run(argv, capture_output=True, timeout=row["effective_timeout_seconds"])
            row.update(returncode=process.returncode)
            for kind, data in [("stdout", process.stdout), ("stderr", process.stderr)]:
                file = output / (label + "." + kind + ".log")
                file.write_bytes(data)
                row[kind] = {"file": file.name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            return process
        except (OSError, subprocess.TimeoutExpired) as error:
            row["error_type"] = type(error).__name__
            if isinstance(error, subprocess.TimeoutExpired):
                for kind, data in [("stdout", error.stdout or b""), ("stderr", error.stderr or b"")]:
                    file = output / (label + ".partial." + kind + ".log")
                    file.write_bytes(data)
                    row[kind] = {"file": file.name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            raise
        finally:
            save(path, receipt)

    try:
        save(path, receipt)
        receipt["source"] = identity(root)
        receipt["phase"] = "pull_exact_published_image"
        pull = command("pull", ["docker", "pull", IMAGE], timeout=180)
        require(pull.returncode == 0, "exact published image pull failed")
        image = command(
            "image-identity",
            [
                "docker",
                "image",
                "inspect",
                "--format",
                "{{json .Id}}|{{json .RepoDigests}}|{{json .Architecture}}|{{json .Os}}",
                IMAGE,
            ],
        )
        require(image.returncode == 0, "image identity observation failed")
        image_id, repo_digests, architecture, operating_system = [
            json.loads(value) for value in image.stdout.decode().strip().split("|")
        ]
        require(
            isinstance(image_id, str)
            and image_id.startswith("sha256:")
            and len(image_id) == 71
            and all(c in "0123456789abcdef" for c in image_id[7:]),
            "invalid actual image ID",
        )
        expected_digest = IMAGE.split("@", 1)[1]
        require(
            isinstance(repo_digests, list)
            and any(
                value in {"rocm/vllm-omni@" + expected_digest, "docker.io/rocm/vllm-omni@" + expected_digest}
                for value in repo_digests
            ),
            "actual RepoDigests omit exact published manifest",
        )
        require((operating_system, architecture) == ("linux", "amd64"), "unexpected image OS/architecture")
        receipt["actual_image_identity"] = {
            "id": image_id,
            "repo_digests": repo_digests,
            "os": operating_system,
            "architecture": architecture,
        }
        receipt["phase"] = "create_readonly_device_free_container"
        frozen = output / "observer-frozen.py"
        frozen.write_bytes(Path(__file__).read_bytes())
        frozen.chmod(0o400)
        receipt["frozen_observer_sha256"] = hashlib.sha256(frozen.read_bytes()).hexdigest()
        receipt["frozen_observer_matches_source"] = (
            receipt["frozen_observer_sha256"] == receipt["source"]["helper_sha256"]
        )
        require(receipt["frozen_observer_matches_source"], "frozen observer differs from verified Git source")
        user = str(os.getuid()) + ":" + str(os.getgid())
        receipt["observer_user"] = user
        creation_attempted = True
        create = command(
            "create",
            [
                "docker",
                "create",
                "--name",
                name,
                "--runtime",
                "runc",
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--user",
                user,
                "--label",
                "pr9009.sdk-observation=" + output.name,
                "--label",
                "pr9009.source=" + receipt["source"]["commit"],
                "--mount",
                "type=bind,src=" + str(frozen.resolve()) + ",dst=/observer.py,readonly",
                "--mount",
                "type=bind,src=" + str(output.resolve()) + ",dst=/evidence",
                "--entrypoint",
                "/usr/bin/python3",
                IMAGE,
                "-I",
                "-S",
                "-B",
                "/observer.py",
                "inspect",
            ],
        )
        require(create.returncode == 0, "container creation failed")
        container_id = create.stdout.decode().strip()
        require(len(container_id) == 64 and all(c in "0123456789abcdef" for c in container_id), "invalid container ID")
        receipt["container_id"] = container_id
        check = command(
            "container-config",
            [
                "docker",
                "inspect",
                "--format",
                "{{json .HostConfig.Devices}}|{{json .HostConfig.DeviceRequests}}|"
                "{{json .HostConfig.DeviceCgroupRules}}|"
                "{{json .HostConfig.Privileged}}|{{json .HostConfig.NetworkMode}}|{{json .HostConfig.ReadonlyRootfs}}|"
                "{{json .HostConfig.Runtime}}|{{json .HostConfig.CapDrop}}|"
                "{{json .Config.User}}|{{json .Config.Image}}|{{json .Image}}|"
                "{{json .HostConfig.CapAdd}}|{{json .HostConfig.SecurityOpt}}",
                container_id,
            ],
        )
        require(check.returncode == 0, "container configuration unavailable")
        config = [json.loads(value) for value in check.stdout.decode().strip().split("|")]
        require(
            len(config) == 13
            and all(value in (None, []) for value in config[:3])
            and config[3:11] == [False, "none", True, "runc", ["ALL"], user, IMAGE, image_id]
            and config[11] in (None, [])
            and config[12] in (["no-new-privileges"], ["no-new-privileges:true"]),
            "container is not the declared device-free readonly isolation",
        )
        receipt["verified_host_config"] = config
        receipt["phase"] = "stdlib_sdk_filesystem_observation"
        result = command("observe", ["docker", "start", "--attach", container_id], timeout=120)
        require(result.returncode == 0, "SDK filesystem observation incomplete")
        observed = json.loads((output / "sdk-observation.json").read_text())
        require(
            observed.get("observation_complete") is True and observed.get("sdk_code_executed") is False,
            "incomplete observation report",
        )
        require(
            observed.get("isolated") == 1 and observed.get("no_site") == 1,
            "observer did not run in the required isolated/no-site mode",
        )
        require(observed.get("interpreter") == "/usr/bin/python3", "observer interpreter differs from actual preflight")
        require(observed.get("observer_sha256") == receipt["frozen_observer_sha256"], "observer byte identity differs")
        receipt["observation_complete"] = True
        exit_code = 0
    except BaseException as error:
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        cleanup_deadline = time.monotonic() + 60
        if creation_attempted:
            try:
                owned_reference = receipt.get("container_id", name)
                check = command(
                    "cleanup-identity",
                    [
                        "docker",
                        "inspect",
                        "--format",
                        "{{json .Config.Labels}}|{{json .State.Running}}",
                        owned_reference,
                    ],
                )
                require(check.returncode == 0, "own container cleanup identity unavailable")
                labels, running = [json.loads(value) for value in check.stdout.decode().strip().split("|")]
                require(isinstance(labels, dict) and isinstance(running, bool), "unknown cleanup identity schema")
                require(
                    labels.get("pr9009.sdk-observation") == output.name
                    and labels.get("pr9009.source") == receipt["source"]["commit"],
                    "cleanup owner identity differs",
                )
                if running:
                    command("cleanup-stop", ["docker", "stop", "--time", "10", owned_reference], timeout=20)
                remove = command("cleanup-remove", ["docker", "rm", owned_reference])
                receipt["cleanup_complete"] = remove.returncode == 0
                if remove.returncode != 0:
                    exit_code = 1
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                receipt["cleanup_error"] = {"type": type(error).__name__, "message": str(error)}
                exit_code = 1
        receipt["finished_epoch"] = time.time()
        receipt["limits"] = [
            "CPU diagnostic only; no original publisher or two native test commands execute.",
            "Device-free is the explicit Docker configuration, not a claim of identical old pod environment.",
            "Observer UID/GID match the host evidence owner; historical native container identity is not inferred.",
            "Published image filesystem observations do not identify former pods' resolved image IDs.",
            "Unknown Docker/registry/queue errors remain failures; no fallback image, GPU allocation or retry.",
            "Controller work is bounded to 360s plus a separate 60s cleanup reserve inside the new 10min CPU job; "
            "provider checkout/startup/artifact overhead is not measured or guaranteed locally.",
        ]
        save(path, receipt)
        print("AMD SDK CPU receipt: " + str(path), flush=True)
    return exit_code


def main():
    require(len(sys.argv) == 2 and sys.argv[1] in {"bootstrap", "run", "inspect", "pod"}, "expected mode")
    if sys.argv[1] == "inspect":
        return inspect_sdk(Path("/evidence/sdk-observation.json"))
    root = Path.cwd().resolve()
    if sys.argv[1] == "bootstrap":
        try:
            identity(root)
        except BaseException as error:
            report = {
                "phase": "bootstrap_identity_failed",
                "observation_complete": False,
                "git_commands": GIT_OBSERVATIONS,
                "error": {"type": type(error).__name__, "message": str(error)},
                "image_or_sdk_execution_started": False,
            }
            print("AMD SDK bootstrap observation: " + json.dumps(report, sort_keys=True), file=sys.stderr, flush=True)
            return 1
        print(json.dumps(pipeline()))
        return 0
    if sys.argv[1] == "pod":
        return pod(root)
    return run(root)


if __name__ == "__main__":
    sys.exit(main())
