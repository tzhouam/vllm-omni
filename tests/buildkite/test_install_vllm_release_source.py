# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise the source installer's ABI fence with real Git histories."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
SCRIPT = Path(__file__).resolve().parents[2] / "tools/install_vllm_release_source.sh"


def _git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.org", *args], text=True
    ).strip()


@pytest.mark.parametrize("native_changed", [False, True])
def test_binary_reuse_requires_identical_native_inputs(tmp_path, native_changed):
    repo = tmp_path / "upstream"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    (repo / "csrc").mkdir()
    (repo / "csrc/kernel.cpp").write_text("native ABI\n")
    (repo / "scheduler.py").write_text("old Python\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "binary release")
    binary = _git(repo, "rev-parse", "HEAD")
    _git(repo, "tag", "v0.33.0rc1")
    target = repo / ("csrc/kernel.cpp" if native_changed else "scheduler.py")
    target.write_text("changed\n")
    _git(repo, "commit", "--quiet", "-am", "release branch change")
    source = _git(repo, "rev-parse", "HEAD")

    # Only the install command is stubbed: Git fetches and checks real commits.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$UV_CALL_LOG"\n')
    uv.chmod(0o755)
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "_version.py").write_text(f"__commit_id__ = 'g{source}'\n")
    call_log = tmp_path / "install.log"
    env = dict(
        os.environ,
        PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        PYTHONPATH=str(tmp_path),
        UV_CALL_LOG=str(call_log),
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0=f"url.{repo.as_uri()}.insteadOf",
        GIT_CONFIG_VALUE_0="https://github.com/vllm-project/vllm.git",
    )
    result = subprocess.run(
        ["bash", str(SCRIPT), source, binary, "0.33.0rc1", sys.executable],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if native_changed:
        assert result.returncode != 0
        assert not call_log.exists(), "Native changes must fail before installing anything"
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        args = call_log.read_text().splitlines()
        assert "--no-deps" in args
        assert "--editable" not in args
        assert not Path(args[-1]).exists(), "The installed package must survive checkout cleanup"
        assert f"source={source} native-binaries={binary}" in result.stdout
