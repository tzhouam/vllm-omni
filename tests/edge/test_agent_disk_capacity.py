# SPDX-License-Identifier: Apache-2.0
"""SSD admission must not count unallocated sparse download holes as free bytes."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_omni.edge.agent import native_app
from vllm_omni.edge.agent.native_app import _allocated_file_bytes, _artifact_disk_capacity


def _route(root: Path, *files: str) -> dict:
    return {
        "memory_demands": {"host_ram": 1, "ssd": 1},
        "artifact_root": str(root),
        "artifact_manifest": {"files": [{"path": name} for name in files]},
    }


@pytest.mark.skipif(os.name == "nt", reason="POSIX sparse fixture; native Windows uses GetCompressedFileSizeW")
def test_sparse_download_holes_do_not_enlarge_capacity(tmp_path, monkeypatch):
    path = tmp_path / "weights.gguf"
    with path.open("wb") as output:
        output.write(b"header")
        output.truncate(64 << 20)
    allocated = path.stat().st_blocks * 512
    if allocated >= path.stat().st_size:
        pytest.skip("fixture filesystem does not preserve sparse holes")
    monkeypatch.setattr("shutil.disk_usage", lambda _: SimpleNamespace(free=1000))
    assert _allocated_file_bytes(path) == allocated
    assert _artifact_disk_capacity([_route(tmp_path, path.name)]) == 1000 + allocated
    assert _artifact_disk_capacity([_route(tmp_path, path.name)]) < 1000 + path.stat().st_size


def test_missing_artifact_root_is_explicit(tmp_path):
    with pytest.raises(ValueError, match="artifact root is missing"):
        _artifact_disk_capacity([_route(tmp_path / "missing", "weights.gguf")])


def test_missing_pack_root_is_explicit(tmp_path):
    route = _route(tmp_path)
    route["prepared_model_dir"] = str(tmp_path / "missing-pack")
    with pytest.raises(ValueError, match="prepared pack root is missing"):
        _artifact_disk_capacity([route])


def test_unknown_physical_allocation_is_refused(monkeypatch):
    if os.name == "nt":
        pytest.skip("POSIX unknown stat observation")
    monkeypatch.setattr(Path, "stat", lambda _: SimpleNamespace(st_size=100))
    with pytest.raises(ValueError, match="allocation could not be measured"):
        _allocated_file_bytes(Path("weights.gguf"))


@pytest.mark.parametrize("error", [0, 5])
def test_windows_allocated_size_distinguishes_valid_low_word_from_error(monkeypatch, error):
    import ctypes

    state = {"error": 0}

    class CompressedSize:
        def __call__(self, _path, high):
            high._obj.value = 1
            state["error"] = error
            return 0xFFFFFFFF

    monkeypatch.setattr(native_app, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_a, **_kw: SimpleNamespace(
        GetCompressedFileSizeW=CompressedSize()), raising=False)
    monkeypatch.setattr(ctypes, "set_last_error", lambda value: state.update(error=value), raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: state["error"], raising=False)
    if error:
        with pytest.raises(ValueError, match="allocation could not be measured"):
            _allocated_file_bytes(Path("weights.gguf"))
    else:
        assert _allocated_file_bytes(Path("weights.gguf")) == (1 << 32) | 0xFFFFFFFF


def test_no_ssd_claim_has_no_disk_capacity():
    assert _artifact_disk_capacity([{"memory_demands": {"host_ram": 1}}]) is None
