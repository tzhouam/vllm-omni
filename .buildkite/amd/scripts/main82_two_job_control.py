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
CONTROL_BRANCH = "codex/pr9009-amd-main82-two-job-control-20261106"
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
    return subprocess.check_output(["git", "-C", str(root), *args])


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
    require(
        not git(root, "diff", "--name-only") and not git(root, "diff", "--cached", "--name-only"),
        "tracked checkout dirty",
    )
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


def tracked_manifest(root):
    rows = []
    for entry in git(root, "ls-tree", "-r", "-z", BASE).split(b"\0"):
        if not entry:
            continue
        header, path = entry.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        require(kind == "blob", "submodule cannot be source-attested")
        rows.append({"path": path.decode(), "mode": mode, "git_blob": oid})
    return rows


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


def unknown_files(root, rows, image_generated=None):
    tracked = {row["path"] for row in rows}
    known = []
    metadata = []
    image_generated = image_generated or {}
    allowed_paths = (
        tracked | KNOWN_CI_FILES | set(image_generated) | {"vllm_omni.egg-info/" + name for name in EGG_FILES}
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
            if relative in tracked or relative == ".git":
                continue
            if relative in KNOWN_CI_FILES:
                require(path.is_file() and not path.is_symlink(), f"unexpected CI file type: {relative}")
                known.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                continue
            if relative in image_generated:
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
            if path.parent == root / "vllm_omni.egg-info" and path.name in EGG_FILES:
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
        f'cp {shlex.quote(HELPER)} "$control_script"\n'
        f'exec python3 "$control_script" run {shlex.quote(role)} "$PWD"\n'
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
    # Match run-amd-test.sh's effective PYTHONPATH, without changing test env.
    env = dict(os.environ)
    env["PYTHONPATH"] = env.get("PYTHONPATH") or ".."
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # guard only; original test environment unchanged
    for entry in env["PYTHONPATH"].split(os.pathsep):
        require((APP / entry).resolve() in {APP, APP.parent}, "unreviewed PYTHONPATH source shadow")
    code = """
import importlib, importlib.metadata, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
names = ['vllm_omni', 'vllm_omni.entrypoints.cli.main', 'tests.helpers.runtime']
origins = {}
for name in names:
    module = importlib.import_module(name)
    path = pathlib.Path(module.__file__).resolve()
    assert path.is_relative_to(root), (name, str(path))
    origins[name] = str(path)
runtime = importlib.import_module('tests.helpers.runtime')
assert pathlib.Path(runtime._omni_subprocess_cwd()).resolve() == root
dist = importlib.metadata.distribution('vllm-omni')
direct = json.loads(dist.read_text('direct_url.json') or '{}')
assert direct.get('url') == root.as_uri() and direct.get('dir_info', {}).get('editable') is True, direct
versions = {}
for name in ['vllm-omni', 'vllm', 'torch', 'torchaudio', 'transformers', 'numpy']:
    versions[name] = importlib.metadata.version(name)
assert versions['vllm-omni'] == '0.0.0', versions['vllm-omni']
loaded = {}
for name, module in tuple(sys.modules.items()):
    if name == 'vllm_omni' or name.startswith('vllm_omni.'):
        origin = getattr(module, '__file__', None)
        if origin:
            assert pathlib.Path(origin).resolve().is_relative_to(root), (name, origin)
            loaded[name] = origin
with open(sys.argv[2], 'x') as output:
    json.dump({'origins': origins, 'server_cwd': str(root), 'direct_url': direct,
               'versions': versions, 'sys_path': sys.path, 'loaded_omni_origins': loaded}, output, sort_keys=True)
"""
    result_path = directory / "imports.json"
    with (directory / "imports.stdout.log").open("x") as stdout, (directory / "imports.stderr.log").open("x") as stderr:
        subprocess.run(
            [sys.executable, "-c", code, str(APP), str(result_path)],
            cwd=APP,
            env=env,
            stdout=stdout,
            stderr=stderr,
            check=True,
        )
    return json.loads(result_path.read_text())


def run(role, root):
    started = time.time()
    root = Path(root).resolve()
    receipt_dir = root / "artifacts/amd-main82-control"
    private = Path(tempfile.mkdtemp(prefix="amd-main82-control-receipt-"))
    receipt = {
        "started_epoch": started,
        "role": role,
        "base": BASE,
        "complete": False,
        "qualification_passed": False,
        "stage": "preparation_started",
    }

    def persist():
        (private / "identity.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        # If the pod is killed before finally, the original log still records
        # incomplete preparation. No artifact or native exit is invented.
        print("Control identity checkpoint: " + json.dumps(receipt, sort_keys=True), flush=True)

    persist()
    try:
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
        require(Path(__file__).resolve().parent == Path("/tmp"), "running helper must be frozen outside checkout")
        require(
            Path(__file__).read_bytes() == git(root, "show", f"{expected}:{HELPER}"),
            "frozen helper differs from control",
        )
        fetch_base(root)
        verify_changes(root, expected)
        rows = tracked_manifest(root)
        image_generated = generated_image_files(APP) if role == "test" else None
        # Inspect all untracked AND ignored paths before checkout; no git clean.
        control_rows = [dict(row) for row in rows if row["path"] not in CHANGES]
        control_rows += [{"path": p} for p in CHANGES]
        receipt["control_untracked"] = unknown_files(root, control_rows, image_generated if root == APP else None)
        subprocess.run(["git", "-C", str(root), "checkout", "--detach", BASE], check=True)
        require(git(root, "rev-parse", "HEAD").decode().strip() == BASE, "detach failed")
        source_bytes(root, rows)
        receipt.update(
            {
                "control_commit": expected,
                "buildkite_commit": os.environ["BUILDKITE_COMMIT"],
                "image_tag": f"rocm/vllm-omni:{expected}",
                "agent_checkout_head": BASE,
                "source_manifest_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                "tracked_files": len(rows),
                "agent_source": unknown_files(root, rows, image_generated if root == APP else None),
            }
        )
        if role == "test":
            source_bytes(APP, rows)
            receipt["test_source"] = unknown_files(APP, rows, image_generated)
            receipt["test_source_root"] = str(APP)
            receipt["test_source_head"] = BASE if APP == root else None
            receipt["test_source_identity"] = (
                "Git checkout" if APP == root else "tracked bytes equal main82; image copy has no asserted Git HEAD"
            )
            receipt["imports"] = import_proof(private)
            command = ["bash", ".buildkite/amd/scripts/run-amd-test.sh"]
        else:
            require(role == "publisher", "unknown control role")
            command = ["bash", ".buildkite/amd/scripts/build-ci-image.sh"]
        receipt["preparation_seconds"] = time.time() - started
        receipt["prepared"] = True
        receipt["stage"] = "original_entry_running"
        persist()
        # Preparation consumes the original 90m/180m Buildkite job budget.
        # The original inner 75m/165m timer and collection order are unchanged;
        # preparation does not consume that inner pytest timer.
        result = subprocess.run(command, cwd=root)
        receipt["original_entry_exit_code"] = result.returncode
        receipt["published_image_digest"] = {"status": "not independently inspected; retain original publisher log"}
        receipt["runtime_image_digest"] = {"status": "unknown: no Kubernetes API or guaranteed pod imageID exposed"}
        receipt["complete"] = True
        receipt["stage"] = "original_entry_finished"
        return result.returncode
    except BaseException as exc:
        receipt["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        receipt["finished_epoch"] = time.time()
        persist()
        receipt_dir.mkdir(parents=True, exist_ok=True)
        target = receipt_dir / private.name
        target.mkdir(exist_ok=False)
        for source in private.iterdir():
            (target / source.name).write_bytes(source.read_bytes())
        print(f"Control identity receipt: {target}", flush=True)


def main():
    mode = sys.argv[1]
    if mode == "bootstrap":
        root = Path.cwd()
        expected = control_identity(root)
        require(
            os.environ.get("AMD_MIRROR_HW", "amdproduction") == "amdproduction", "native production selection required"
        )
        fetch_base(root)
        verify_changes(root, expected)
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
