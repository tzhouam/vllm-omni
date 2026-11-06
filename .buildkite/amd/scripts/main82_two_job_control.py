# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in main82 source control; no model work or automatic CI submission.

The build commit/image tag remains the CI-only child, while publisher context
and actual test source are checked against BASE. The image need not contain Git.
No claim of image, host, cache or runtime equivalence to the failed M7 jobs.
"""

import hashlib
import json
import os
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BASE = "82fc4f511fe02bb933d01c5b5f24e1f8a9e5b415"
CONTROL_BRANCH = "codex/pr9009-amd-main82-two-job-control-v9-20261106"
HELPER = ".buildkite/amd/scripts/main82_two_job_control.py"
CHANGES = {HELPER, ".buildkite/amd/scripts/bootstrap-amd-omni.sh"}
KNOWN_CI_FILES = {
    ".buildkite/amd/pipeline.yaml",
    ".buildkite/amd/test-template.j2",
}
LABELS = {"MiniCPM-o 4.5 Accuracy": "165m", "Bagel img2img · 1-GPU": "75m"}
APP = Path("/app/vllm-omni")
EGG_FILES = {
    "PKG-INFO",
    "SOURCES.txt",
    "dependency_links.txt",
    "entry_points.txt",
    "requires.txt",
    "top_level.txt",
    "not-zip-safe",
}


def git(root, *args):
    # Metadata reads must not silently fetch missing promised objects or refresh
    # the original index. The one explicit BASE fetch uses its separate path.
    env = dict(os.environ, GIT_ALLOW_PROTOCOL="", GIT_NO_LAZY_FETCH="1")
    return subprocess.check_output(["git", "--no-optional-locks", "-C", str(root), *args], env=env)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def control_identity(root):
    expected = os.environ.get("EXPECTED_CONTROL_COMMIT", "")
    require(len(expected) == 40 and all(c in "0123456789abcdef" for c in expected), "external commit lock required")
    require(os.environ.get("AMD_MAIN82_TWO_JOB_CONTROL") == "1", "explicit control flag required")
    require(os.environ.get("BUILDKITE_SOURCE") == "api", "explicit diagnostic API build required")
    pr = os.environ.get("BUILDKITE_PULL_REQUEST", "false")
    require(pr in {"false", "9009"}, "only the explicit diagnostic checkout contexts are supported")
    branch = ("tzhouam:" if pr == "9009" else "") + CONTROL_BRANCH
    require(os.environ.get("BUILDKITE_BRANCH") == branch, "diagnostic fork branch identity mismatch")
    repository = os.environ.get("BUILDKITE_PULL_REQUEST_REPO")
    require(
        not repository
        or repository in {"https://github.com/tzhouam/vllm-omni", "https://github.com/tzhouam/vllm-omni.git"},
        "diagnostic fork repository mismatch",
    )
    require(os.environ.get("BUILDKITE_COMMIT") == expected, "Buildkite commit must retain real control identity")
    require(git(root, "rev-parse", "HEAD").decode().strip() == expected, "checkout is not locked control")
    parents = [s[7:] for s in git(root, "cat-file", "-p", "HEAD").decode().splitlines() if s.startswith("parent ")]
    require(parents == [BASE], "control must have exactly one literal main82 parent")
    return expected


def fetch_base(root):
    # Exactly one pinned fetch per bootstrap/job. No retry or persistent config.
    subprocess.run(
        ["git", "-C", str(root), "-c", "http.version=HTTP/1.1", "fetch", "--no-tags", "origin", BASE],
        check=True,
    )


def verify_changes(root, expected):
    changed = set(git(root, "diff", "--name-only", BASE, expected).decode().splitlines())
    require(changed == CHANGES, f"unexpected control paths: {sorted(changed)}")
    for path in CHANGES:
        mode, kind, _ = git(root, "ls-tree", expected, path).decode().split(None, 2)
        require(kind == "blob" and mode == ("100644" if path == HELPER else "100755"), "diagnostic file mode changed")


def tracked_manifest(root, revision=BASE):
    rows = []
    for entry in git(root, "ls-tree", "-r", "-z", revision).split(b"\0"):
        if not entry:
            continue
        header, path = entry.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        require(kind == "blob", "submodule cannot be source-attested")
        rows.append({"path": path.decode(), "mode": mode, "git_blob": oid})
    return rows


def transport_snapshot(root):
    """Retain the real checkout state; it is never the canonical build source."""
    head = tracked_manifest(root, "HEAD")
    index = []
    for entry in git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if entry:
            header, path = entry.split(b"\t", 1)
            mode, blob, stage = header.decode().split()
            index.append({"path": path.decode(), "mode": mode, "git_blob": blob, "stage": int(stage)})
    actual = []
    errors = []
    allowed_modes = []
    for row in head:
        path = root / row["path"]
        observed = {"path": row["path"]}
        try:
            for ancestor in path.relative_to(root).parents:
                if str(ancestor) != ".":
                    require(not (root / ancestor).is_symlink(), "tracked path has a symlink ancestor")
            meta = path.lstat()
            observed["permission_bits"] = stat.S_IMODE(meta.st_mode)
            if stat.S_ISLNK(meta.st_mode):
                payload = os.fsencode(os.readlink(path))
                observed["mode"] = "120000"
            else:
                require(stat.S_ISREG(meta.st_mode), "tracked source is not regular or symlink")
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(fd, "rb") as stream:
                    before = os.fstat(stream.fileno())
                    payload = stream.read()
                    after = os.fstat(stream.fileno())
                require(
                    (meta.st_dev, meta.st_ino, meta.st_mode, meta.st_size, meta.st_mtime_ns, meta.st_ctime_ns)
                    == (
                        before.st_dev,
                        before.st_ino,
                        before.st_mode,
                        before.st_size,
                        before.st_mtime_ns,
                        before.st_ctime_ns,
                    )
                    == (after.st_dev, after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                    "tracked source changed during read",
                )
                observed["mode"] = "100755" if meta.st_mode & stat.S_IXUSR else "100644"
            observed.update(
                git_blob=hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest(),
                sha256=hashlib.sha256(payload).hexdigest(),
                bytes=len(payload),
            )
            require(observed["git_blob"] == row["git_blob"], "tracked source bytes differ")
            if observed["mode"] != row["mode"]:
                require(
                    row["mode"] == "100644" and observed["mode"] == "100755" and observed["permission_bits"] == 0o777,
                    "unobserved transport mode mutation",
                )
                allowed_modes.append(
                    {"path": row["path"], "head_mode": row["mode"], "actual_mode": "100755", "permission_bits": 0o777}
                )
        except (OSError, RuntimeError) as exc:
            errors.append({"path": row["path"], "error_type": type(exc).__name__, "message": str(exc)})
        actual.append(observed)
    wanted_index = [{**row, "stage": 0} for row in head]
    unstaged = git(root, "diff", "--name-only", "-z").decode().strip("\0").split("\0")
    cached = git(root, "diff", "--cached", "--name-only", "-z")
    index_path = Path(git(root, "rev-parse", "--git-path", "index").decode().strip())
    if not index_path.is_absolute():
        index_path = root / index_path
    require(stat.S_ISREG(index_path.lstat().st_mode), "Git index is not a regular non-symlink file")
    fd = os.open(index_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        index_before = os.fstat(stream.fileno())
        index_bytes = stream.read()
        index_after = os.fstat(stream.fileno())
    require(
        (
            index_before.st_dev,
            index_before.st_ino,
            index_before.st_size,
            index_before.st_mtime_ns,
            index_before.st_ctime_ns,
        )
        == (
            index_after.st_dev,
            index_after.st_ino,
            index_after.st_size,
            index_after.st_mtime_ns,
            index_after.st_ctime_ns,
        ),
        "physical Git index changed while reading",
    )
    return {
        "head": git(root, "rev-parse", "HEAD").decode().strip(),
        "head_tree": git(root, "rev-parse", "HEAD^{tree}").decode().strip(),
        "head_manifest": head,
        "index_manifest": index,
        "physical_index": {
            "path": str(index_path),
            "regular_non_symlink": True,
            "bytes": len(index_bytes),
            "sha256": hashlib.sha256(index_bytes).hexdigest(),
        },
        "worktree_manifest": actual,
        "index_equals_head": index == wanted_index,
        "original_unstaged_paths": [p for p in unstaged if p],
        "original_cached_paths_hex": cached.hex(),
        "allowed_transport_mode_differences": allowed_modes,
        "unexpected_source_differences": errors,
        "identity": "observed transport checkout; not canonical source or a clean checkout claim",
    }


def verify_transport(snapshot):
    require(snapshot["index_equals_head"], "transport index differs from HEAD")
    require(not snapshot["original_cached_paths_hex"], "transport has staged changes")
    require(not snapshot["unexpected_source_differences"], "transport source bytes/types/modes differ")
    mode_paths = {row["path"] for row in snapshot["allowed_transport_mode_differences"]}
    require(set(snapshot["original_unstaged_paths"]) <= mode_paths, "transport Git diff has an unexplained change")


def materialize_base(root, rows, canonical, report_path):
    # This adds only Git worktree bookkeeping; original HEAD/index/files are
    # independently retained before/after and are never checked out or chmodded.
    env = dict(os.environ, GIT_ALLOW_PROTOCOL="", GIT_NO_LAZY_FETCH="1")
    subprocess.run(
        ["git", "-C", str(root), "-c", "core.hooksPath=/dev/null", "worktree", "add", "--detach", str(canonical), BASE],
        check=True,
        env=env,
    )
    state = transport_snapshot(canonical)
    report_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    require(state["head"] == BASE and state["head_manifest"] == rows, "private checkout differs from BASE")
    verify_transport(state)
    source_bytes(canonical, rows)
    require(not state["allowed_transport_mode_differences"], "canonical source has modified Git modes")
    require(not state["original_unstaged_paths"], "canonical source has Git differences")
    unknown_files(canonical, rows, strict=True)
    return state


def source_bytes(root, rows):
    for row in rows:
        path = root / row["path"]
        mode = row["mode"]
        if mode == "120000":
            require(path.is_symlink(), f"missing source symlink: {path}")
            payload = os.readlink(path).encode()
        else:
            require(path.is_file() and not path.is_symlink(), f"missing regular source: {path}")
            payload = path.read_bytes()
            require(bool(path.stat().st_mode & stat.S_IXUSR) == (mode == "100755"), f"source mode mismatch: {path}")
        digest = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        require(digest == row["git_blob"], f"source bytes mismatch: {path}")


def generated_image_files(root):
    # Dockerfile.rocm creates this exact pytest exit hook before its final COPY.
    # Parse only the literal printf arguments; never execute image-side code.
    dockerfile = (root / "docker/Dockerfile.rocm").read_text()
    require("ENV VLLM_OMNI_VERSION_OVERRIDE=0.0.0\n" in dockerfile, "Dockerfile version override changed")
    prefix = "RUN printf '%s\\n' \\\n"
    suffix = "    > ${COMMON_WORKDIR}/vllm-omni/conftest.py"
    require(dockerfile.count(prefix) == 1 and dockerfile.count(suffix) == 1, "Dockerfile exit-hook recipe changed")
    body = dockerfile.split(prefix, 1)[1].split(suffix, 1)[0]
    lines = shlex.split(body.replace("\\\n", ""))
    conftest = ("\n".join(lines) + "\n").encode()
    # setup.py uses the same installed setuptools_scm with this Docker ENV.
    # Generate in a private directory, without importing the unknown version file.
    with tempfile.TemporaryDirectory(prefix="amd-main82-version-") as directory:
        env = dict(os.environ, SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0")
        subprocess.run(
            [sys.executable, "-c", "from setuptools_scm import get_version; get_version(write_to='version.py')"],
            cwd=directory,
            env=env,
            check=True,
        )
        version = (Path(directory) / "version.py").read_bytes()
    return {"conftest.py": conftest, "vllm_omni/_version.py": version}


def unknown_files(root, rows, image_generated=None, strict=False):
    tracked = {row["path"] for row in rows}
    known = []
    metadata = []
    image_generated = image_generated or {}
    allowed_paths = (
        tracked
        if strict
        else (tracked | KNOWN_CI_FILES | set(image_generated) | {"vllm_omni.egg-info/" + name for name in EGG_FILES})
    )
    allowed_dirs = {str(parent) for name in allowed_paths for parent in Path(name).parents if str(parent) != "."}
    for relative, payload in image_generated.items():
        path = root / relative
        require(
            path.is_file() and not path.is_symlink() and path.read_bytes() == payload,
            f"missing or modified generated source: {relative}",
        )
    for directory, dirs, files in os.walk(root):
        for name in list(dirs):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if relative == ".git":
                require(not path.is_symlink(), "Git metadata directory must not be a symlink")
                dirs.remove(name)
            elif path.is_symlink():
                require(relative in tracked, f"unknown source/context symlink directory: {relative}")
                dirs.remove(name)
            else:
                require(relative in allowed_dirs, f"unknown source/context directory: {relative}")
        for name in files:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if relative == ".git":
                require(path.is_file() and not path.is_symlink(), "Git metadata file must be regular, not a symlink")
                continue
            if relative in tracked:
                continue
            if not strict and relative in KNOWN_CI_FILES:
                require(path.is_file() and not path.is_symlink(), f"unexpected CI file type: {relative}")
                known.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                continue
            if not strict and relative in image_generated:
                require(path.is_file() and not path.is_symlink(), f"generated source type: {relative}")
                require(path.read_bytes() == image_generated[relative], f"generated source bytes: {relative}")
                known.append(
                    {
                        "path": relative,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "identity": "exact Dockerfile/setup generated file; not Git tracked",
                    }
                )
                continue
            # Editable-install metadata is recorded, never treated as source.
            # Unknown bytecode/native modules/symlinks are rejected too.
            if not strict and path.parent == root / "vllm_omni.egg-info" and path.name in EGG_FILES:
                require(
                    path.is_file() and not path.is_symlink() and not path.stat().st_mode & 0o111,
                    f"metadata shadow: {relative}",
                )
                metadata.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                continue
            raise RuntimeError(f"unknown source/context file (including ignored files): {relative}")
    return {"known_ci_generated_files": sorted(known, key=lambda r: r["path"]), "editable_metadata": metadata}


def leaves(data):
    for group in data["steps"]:
        yield from group.get("steps", [group])


def select(input_path, output_path):
    import yaml

    data = yaml.safe_load(Path(input_path).read_text())
    selected = [step for step in leaves(data) if step.get("label") in LABELS]
    require(len(selected) == 2 and {s["label"] for s in selected} == set(LABELS), "not exactly both original leaves")
    # Keep the complete original dictionaries, including env, commands and budgets.
    Path(output_path).write_text(yaml.safe_dump({"env": data.get("env", {}), "steps": selected}, sort_keys=False))


def wrapper(role):
    # Bash reads a fixed -c string and Python reads a private file outside Git.
    # Neither running program is overwritten by the subsequent real checkout.
    body = (
        "set -euo pipefail\n"
        "control_script=$(mktemp /tmp/amd-main82-control.XXXXXX.py)\n"
        f'cp {shlex.quote(HELPER)} "$$control_script"\n'
        f'exec python3 "$$control_script" run {shlex.quote(role)} "$$PWD"\n'
    )
    return "bash -c " + shlex.quote(body)


def wrap(path):
    import yaml

    data = yaml.safe_load(Path(path).read_text())
    jobs = list(leaves(data))
    require(len(jobs) == 3, "control must render exactly publisher plus two tests")
    publisher = [s for s in jobs if s.get("key") == "amd-build"]
    require(len(publisher) == 1, "one original image publisher required")
    require(publisher[0]["commands"] == ["bash .buildkite/amd/scripts/build-ci-image.sh"], "publisher command drift")
    publisher[0]["commands"] = [wrapper("publisher")]
    for step in jobs:
        if step is publisher[0]:
            continue
        label = step["label"].removeprefix("mi300_1: ")
        require(
            label in LABELS and step["command"] == "bash .buildkite/amd/scripts/run-amd-test.sh",
            "unexpected test entry",
        )
        step["command"] = wrapper("test")
    for step in jobs:
        step.setdefault("artifact_paths", []).append("artifacts/amd-main82-control/**/*")
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def import_proof(directory):
    # Observe the ordinary test environment before deciding whether it is safe.
    import importlib.machinery as machinery

    env = dict(os.environ)
    raw = env.get("PYTHONPATH")
    env["PYTHONPATH"] = raw or ".."  # original run-amd-test.sh fallback
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # guard only; native environment unchanged
    sdk = Path("/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/share/amd_smi")
    result_path = directory / "imports.json"
    observed = {
        "phase": "environment_observed",
        "complete": False,
        "PYTHONPATH_present": "PYTHONPATH" in os.environ,
        "PYTHONPATH_raw": raw,
        "PYTHONPATH_effective": env["PYTHONPATH"],
        "fallback_used": not raw,
        "controller_cwd": str(Path.cwd()),
        "ordinary_child_cwd": str(APP),
        "APP": str(APP),
        "APP_parent": str(APP.parent),
        "interpreter": sys.executable,
        "python_version": sys.version,
        "observed_epoch": time.time(),
        "entries": [],
        "SDK_candidate": str(sdk),
        "SDK_loaded_code_certified": False,
    }

    def persist():
        (directory / "imports-preflight.json").write_text(json.dumps(observed, indent=2, sort_keys=True) + "\n")

    try:
        persist()
        for index, entry in enumerate(env["PYTHONPATH"].split(os.pathsep)):
            row = {"index": index, "raw": entry, "accepted": False}
            observed["entries"].append(row)
            observed["phase"] = "path_policy_validation"
            persist()
            path = APP / entry
            row["absolute_interpretation"] = str(path.absolute())
            row["symlink_components"] = []
            cursor = Path(path.anchor)
            for component in path.parts[1:]:
                cursor /= component
                if cursor.is_symlink():
                    row["symlink_components"].append({"path": str(cursor), "target": os.readlink(cursor)})
            resolved = path.resolve(strict=True)
            row.update(resolved=str(resolved), exists=True, is_directory=resolved.is_dir())
            persist()
            if resolved in {APP, APP.parent}:
                row.update(accepted=True, policy="existing_APP_or_parent")
                persist()
                continue
            require(entry == str(sdk) and resolved == sdk, "unreviewed PYTHONPATH source shadow")
            observed["phase"] = "static_sdk_origin_check"
            persist()
            require(resolved.is_dir() and not row["symlink_components"], "SDK root must be an exact real directory")
            root_stat = resolved.stat()
            row["root_identity"] = {
                "uid": root_stat.st_uid,
                "mode": stat.S_IMODE(root_stat.st_mode),
                "device": root_stat.st_dev,
                "inode": root_stat.st_ino,
            }
            require(root_stat.st_uid == 0 and not root_stat.st_mode & 0o022, "untrusted SDK root ownership/mode")
            finder = machinery.FileFinder(
                str(resolved),
                (machinery.SourceFileLoader, machinery.SOURCE_SUFFIXES),
                (machinery.SourcelessFileLoader, machinery.BYTECODE_SUFFIXES),
                (machinery.ExtensionFileLoader, machinery.EXTENSION_SUFFIXES),
            )
            names = {
                "amdsmi",
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
            inventory = []
            for child in resolved.iterdir():
                inventory.append({"name": child.name, "symlink": child.is_symlink(), "directory": child.is_dir()})
                if child.is_dir() and child.name.isidentifier():
                    names.add(child.name)
                for suffix in machinery.all_suffixes():
                    if child.name.endswith(suffix):
                        name = child.name[: -len(suffix)]
                        if name.isidentifier():
                            names.add(name)
            row["top_level_inventory"] = sorted(inventory, key=lambda item: item["name"])
            row["specs"] = {}
            persist()
            for name in sorted(names):
                spec = finder.find_spec(name)  # filesystem only; no loader execution/path hooks
                if spec is None:
                    row["specs"][name] = None
                    persist()
                    continue
                locations = list(spec.submodule_search_locations or [])
                item = {"origin": spec.origin, "loader": type(spec.loader).__name__, "search_locations": locations}
                row["specs"][name] = item
                persist()
                require(name == "amdsmi", "SDK entry offers an unreviewed import provider/shadow: " + name)
                require(spec.origin is not None, "SDK provider must have a regular origin, not namespace only")
                for value in [spec.origin, *locations]:
                    origin = Path(value)
                    require(origin.resolve(strict=True).is_relative_to(sdk), "SDK provider origin escapes exact root")
                    for parent in [origin, *origin.parents]:
                        if parent == sdk:
                            break
                        require(not parent.is_symlink(), "SDK provider symlink rejected")
                        mode = parent.stat()
                        require(mode.st_uid == 0 and not mode.st_mode & 0o022, "untrusted SDK provider ownership/mode")
                require(Path(spec.origin).is_file(), "SDK provider origin must be regular")
                item["origin_sha256"] = hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
                persist()
            require(row["specs"].get("amdsmi") is not None, "exact SDK amdsmi provider not established")
            row.update(accepted=True, policy="exact_SDK_static_filesystem_origin_and_no_shadow")
            persist()
        observed["phase"] = "ordinary_child_launch"
        persist()
        code = """
import importlib, importlib.metadata, json, pathlib, sys, time
root = pathlib.Path(sys.argv[1]).resolve()
output = pathlib.Path(sys.argv[2])
sdk = pathlib.Path(sys.argv[3])
report = {'phase': 'ordinary_imports', 'complete': False, 'origins': {}, 'sys_path': sys.path}
def save():
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\\n')
try:
    save()
    for name in ['vllm_omni', 'vllm_omni.entrypoints.cli.main', 'tests.helpers.runtime']:
        report['importing'] = name
        save()
        module = importlib.import_module(name)
        path = pathlib.Path(module.__file__).resolve()
        report['origins'][name] = str(path)
        save()
        assert path.is_relative_to(root), (name, str(path))
    report['phase'] = 'origin_metadata_checks'
    save()
    runtime = importlib.import_module('tests.helpers.runtime')
    assert pathlib.Path(runtime._omni_subprocess_cwd()).resolve() == root
    report['server_cwd'] = str(root)
    dist = importlib.metadata.distribution('vllm-omni')
    direct = json.loads(dist.read_text('direct_url.json') or '{}')
    report['direct_url'] = direct
    save()
    assert direct.get('url') == root.as_uri() and direct.get('dir_info', {}).get('editable') is True, direct
    report['versions'] = {}
    for name in ['vllm-omni', 'vllm', 'torch', 'torchaudio', 'transformers', 'numpy']:
        report['versions'][name] = importlib.metadata.version(name)
        save()
    assert report['versions']['vllm-omni'] == '0.0.0', report['versions']['vllm-omni']
    report['loaded_omni_origins'] = {}
    report['loaded_tests_origins'] = {}
    report['loaded_package_search_locations'] = {}
    report['loaded_SDK_origins'] = {}
    for name, module in tuple(sys.modules.items()):
        selected = name == 'vllm_omni' or name.startswith('vllm_omni.') or name == 'tests' or name.startswith('tests.')
        sdk_selected = name == 'amdsmi' or name.startswith('amdsmi.')
        if not selected and not sdk_selected:
            continue
        expected = sdk if sdk_selected else root
        origin = getattr(module, '__file__', None)
        locations = list(getattr(module, '__path__', []) or [])
        spec = getattr(module, '__spec__', None)
        locations += list(getattr(spec, 'submodule_search_locations', []) or [])
        if origin:
            if sdk_selected:
                key = 'loaded_SDK_origins'
            else:
                key = 'loaded_tests_origins' if name.split('.')[0] == 'tests' else 'loaded_omni_origins'
            report[key][name] = origin
        report['loaded_package_search_locations'][name] = locations
        save()
        for path in ([origin] if origin else []) + locations:
            assert pathlib.Path(path).resolve(strict=True).is_relative_to(expected), (name, path)
    report.update(phase='complete', complete=True, finished_epoch=time.time())
    save()
except BaseException as error:
    report['error'] = {'type': type(error).__name__, 'message': str(error)}
    try:
        save()
    except OSError as persistence_error:
        print('Import observation persistence error: ' + type(persistence_error).__name__, file=sys.stderr)
    raise
"""
        with (
            (directory / "imports.stdout.log").open("x") as stdout,
            (directory / "imports.stderr.log").open("x") as stderr,
        ):
            process = subprocess.run(
                [sys.executable, "-c", code, str(APP), str(result_path), str(sdk)],
                cwd=APP,
                env=env,
                stdout=stdout,
                stderr=stderr,
                check=True,
            )
        observed["child_returncode"] = process.returncode
        result = json.loads(result_path.read_text())
        require(result.get("complete") is True, "ordinary import proof incomplete")
        observed.update(phase="complete", complete=True)
        persist()
        return result
    except BaseException as error:
        observed["error"] = {"type": type(error).__name__, "message": str(error)}
        if observed["entries"] and not observed["entries"][-1]["accepted"]:
            observed["entries"][-1]["error"] = observed["error"]
        if isinstance(error, subprocess.CalledProcessError):
            observed["child_returncode"] = error.returncode
        try:
            persist()
        except OSError as persistence_error:
            print("Import preflight persistence error: " + type(persistence_error).__name__, file=sys.stderr)
        raise


def run(role, root):
    started = time.time()
    root = Path(root).resolve()
    receipt_dir = root / "artifacts/amd-main82-control"
    private = Path(tempfile.mkdtemp(prefix="amd-main82-control-receipt-"))
    canonical = None
    before = None
    native_code = None
    receipt = {
        "started_epoch": started,
        "role": role,
        "base": BASE,
        "complete": False,
        "qualification_passed": False,
        "stage": "preparation_started",
        "original_checkout_mutation_observed_during_preparation": None,
        "shared_git_metadata_changes": (
            "one BASE fetch and linked-worktree registration/removal; not a clean-repository claim"
        ),
    }

    def persist():
        (private / "identity.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print("Control identity checkpoint: " + json.dumps(receipt, sort_keys=True), flush=True)

    def snapshot(name):
        observed = transport_snapshot(root)
        path = private / (name + ".json")
        path.write_text(json.dumps(observed, indent=2, sort_keys=True) + "\n")
        receipt[name] = {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "head": observed["head"],
            "index_equals_head": observed["index_equals_head"],
            "tracked_files": len(observed["head_manifest"]),
            "mode_differences": len(observed["allowed_transport_mode_differences"]),
            "unexpected_source_differences": len(observed["unexpected_source_differences"]),
            "identity": observed["identity"],
        }
        persist()
        verify_transport(observed)
        return observed

    persist()
    try:
        require(role in {"bootstrap", "publisher", "test"}, "unknown control role")
        before = snapshot("transport_before")
        expected = control_identity(root)
        receipt["buildkite_context"] = {
            name: os.environ.get(name)
            for name in (
                "BUILDKITE_SOURCE",
                "BUILDKITE_BRANCH",
                "BUILDKITE_PULL_REQUEST",
                "BUILDKITE_PULL_REQUEST_REPO",
            )
        }
        if role != "bootstrap":
            require(Path(__file__).resolve().parent == Path("/tmp"), "running helper must be frozen outside checkout")
        require(Path(__file__).read_bytes() == git(root, "show", f"{expected}:{HELPER}"), "helper differs from control")
        image_generated = generated_image_files(APP) if role == "test" else None
        receipt["control_untracked"] = unknown_files(
            root, before["head_manifest"], image_generated if root == APP else None
        )
        if role == "bootstrap":
            require(
                os.environ.get("AMD_MIRROR_HW", "amdproduction") == "amdproduction",
                "native production selection required",
            )
        fetch_base(root)
        verify_changes(root, expected)
        rows = tracked_manifest(root)
        if role != "bootstrap":
            canonical = Path(tempfile.mkdtemp(prefix="amd-main82-canonical-")) / "source"
            receipt["canonical_worktree_requested"] = str(canonical)
            persist()
            canonical_report = private / "canonical-source.json"
            receipt["canonical_snapshot_file"] = canonical_report.name
            materialize_base(root, rows, canonical, canonical_report)
            receipt["canonical_source"] = {
                "snapshot_file": canonical_report.name,
                "snapshot_sha256": hashlib.sha256(canonical_report.read_bytes()).hexdigest(),
                "root": str(canonical),
                "head": BASE,
                "source_manifest_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                "tracked_files": len(rows),
                "full_source": unknown_files(canonical, rows, strict=True),
                "identity": "strict canonical BASE bytes, Git modes, index and source population",
            }
        after = snapshot("transport_after_preparation")
        require(after == before, "original transport checkout changed during preparation")
        receipt.update(
            control_commit=expected,
            buildkite_commit=os.environ["BUILDKITE_COMMIT"],
            image_tag=f"rocm/vllm-omni:{expected}",
            agent_checkout_head=expected,
            original_tracked_index_before_after_equal=True,
            original_checkout_mutation_observed_during_preparation=False,
        )
        if role == "bootstrap":
            receipt.update(complete=True, stage="bootstrap_source_verified")
            return 0
        if role == "test":
            # Preserve the strict image-side source/import contract. The CPU
            # checkout mode observation never waives an APP mismatch.
            source_bytes(APP, rows)
            receipt["test_source"] = unknown_files(APP, rows, image_generated)
            receipt["test_source_root"] = str(APP)
            if os.path.lexists(APP / ".git"):
                app_state = transport_snapshot(APP)
                verify_transport(app_state)
                require(
                    app_state["head"] == BASE and app_state["head_manifest"] == rows,
                    "APP Git identity differs from BASE",
                )
                require(not app_state["allowed_transport_mode_differences"], "APP Git modes differ from BASE")
                app_report = private / "app-git-source.json"
                app_report.write_text(json.dumps(app_state, indent=2, sort_keys=True) + "\n")
                receipt["test_source_head"] = BASE
                receipt["test_source_index"] = {
                    "file": app_report.name,
                    "sha256": hashlib.sha256(app_report.read_bytes()).hexdigest(),
                }
            else:
                receipt["test_source_head"] = None
                receipt["test_source_index"] = {"available": False, "reason": "Dockerfile excludes Git metadata"}
            receipt["test_source_identity"] = (
                "strict BASE bytes and Git modes; Git HEAD/index asserted only when present"
            )
            receipt.update(stage="preimport_observation", original_entry_started_at_preimport=False)
            persist()
            receipt["imports"] = import_proof(private)
            command = ["bash", ".buildkite/amd/scripts/run-amd-test.sh"]
        else:
            command = ["bash", ".buildkite/amd/scripts/build-ci-image.sh"]
        receipt["preparation_seconds"] = time.time() - started
        receipt["prepared"] = True
        receipt["stage"] = "original_entry_running"
        receipt["original_entry_cwd"] = str(canonical)
        persist()
        # Original native TEST_COMMANDS still cd to APP and write artifacts via
        # the original BUILDKITE_BUILD_CHECKOUT_PATH. No timers/env are replaced.
        result = subprocess.run(command, cwd=canonical)
        native_code = result.returncode
        receipt["original_entry_exit_code"] = native_code
        try:
            final = snapshot("transport_after_original_entry")
            require(final == before, "original transport checkout changed during the native entry")
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            receipt["post_entry_audit_error"] = {"error_type": type(exc).__name__, "message": str(exc)}
            if native_code == 0:
                raise
        receipt["published_image_digest"] = {"status": "not independently inspected; retain original publisher log"}
        receipt["runtime_image_digest"] = {"status": "unknown: no Kubernetes API or guaranteed pod imageID exposed"}
        receipt["complete"] = "post_entry_audit_error" not in receipt
        receipt["stage"] = "original_entry_finished"
        return native_code
    except BaseException as exc:
        receipt["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        pending_exception = sys.exc_info()[0] is not None
        finalization_error = None
        if canonical is not None:
            try:
                cleanup = subprocess.run(
                    ["git", "-C", str(root), "worktree", "remove", str(canonical)],
                    capture_output=True,
                    timeout=30,
                    env=dict(os.environ, GIT_ALLOW_PROTOCOL="", GIT_NO_LAZY_FETCH="1"),
                )
                receipt["private_worktree_cleanup"] = {
                    "returncode": cleanup.returncode,
                    "stdout_hex": cleanup.stdout.hex(),
                    "stderr_hex": cleanup.stderr.hex(),
                    "force": False,
                }
            except (OSError, subprocess.TimeoutExpired) as exc:
                receipt["private_worktree_cleanup"] = {"error_type": type(exc).__name__, "force": False}
            if before is not None:
                try:
                    after_cleanup = snapshot("transport_after_private_cleanup")
                    require(after_cleanup == before, "original transport checkout changed through cleanup")
                    receipt["original_checkout_unchanged_through_cleanup"] = True
                except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                    receipt["post_cleanup_audit_error"] = {"error_type": type(exc).__name__, "message": str(exc)}
                    receipt["original_checkout_unchanged_through_cleanup"] = None
                    receipt["complete"] = False
                    finalization_error = exc
        receipt["finished_epoch"] = time.time()
        try:
            persist()
            receipt_dir.mkdir(parents=True, exist_ok=True)
            target = receipt_dir / private.name
            target.mkdir(exist_ok=False)
            for source in private.iterdir():
                (target / source.name).write_bytes(source.read_bytes())
            print(f"Control identity receipt: {target}", flush=True)
            if role == "bootstrap":
                # Init's original artifact glob covers pipeline.yaml only.
                # Upload errors are independent; never mask its source result.
                try:
                    upload = subprocess.run(
                        ["buildkite-agent", "artifact", "upload", str(target.relative_to(root) / "*.json")],
                        cwd=root,
                        check=False,
                        timeout=60,
                    )
                    upload_record = {"returncode": upload.returncode}
                except (OSError, subprocess.TimeoutExpired) as exc:
                    upload_record = {"error_type": type(exc).__name__}
                print("Bootstrap source receipt upload: " + json.dumps(upload_record, sort_keys=True), flush=True)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            print("Control receipt persistence error: " + type(exc).__name__, flush=True)
            finalization_error = exc
        # A failed native entry retains its original exit/signal. For a zero
        # native result, mandatory source/persistence guards still fail closed.
        if finalization_error is not None and not pending_exception and native_code in {None, 0}:
            raise finalization_error


def main():
    mode = sys.argv[1]
    if mode == "bootstrap":
        return run("bootstrap", Path.cwd())
    elif mode == "select":
        select(*sys.argv[2:])
    elif mode == "wrap":
        wrap(sys.argv[2])
    elif mode == "run":
        return run(*sys.argv[2:])
    else:
        raise RuntimeError(f"unknown mode: {mode}")
    return 0


if __name__ == "__main__":
    result = main()
    if result < 0:
        if -result not in {signal.SIGKILL, signal.SIGSTOP}:
            signal.signal(-result, signal.SIG_DFL)
        os.kill(os.getpid(), -result)
    sys.exit(result)
