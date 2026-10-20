# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The placement report must measure per-process GPU attribution, not assume it.

[edge-infer W4] The W1 acceptance on native Windows carried a hardcoded "under
WSL2" explanation for an empty NVML process list. These tests pin the repaired
behaviour: the reason names the host it actually ran on, and a successful
attribution (NVML lists one of our PIDs) is reported as data, not a reason.
"""

from __future__ import annotations

import builtins
import sys
import types

import pytest

from vllm_omni.edge.local import engine


def test_host_kind_is_one_of_the_three_hosts():
    assert engine._host_kind() in {"windows", "wsl2", "linux"}


def test_reason_names_the_host_when_nvml_is_unavailable(monkeypatch):
    real_import = builtins.__import__

    def no_pynvml(name, *args, **kwargs):
        if name == "pynvml":
            raise ImportError("pynvml is not installed in this venv")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pynvml)
    out = engine.gpu_attribution_for_tree()
    assert out["per_process_gpu_attribution"] is None
    assert engine._host_kind() in out["per_process_gpu_attribution_reason"]
    assert "WSL2" not in out["per_process_gpu_attribution_reason"] or engine._host_kind() == "wsl2"


def _fake_pynvml(pids):
    m = types.ModuleType("pynvml")
    m.nvmlInit = lambda: None
    m.nvmlShutdown = lambda: None
    m.nvmlDeviceGetHandleByIndex = lambda i: object()
    m.nvmlDeviceGetComputeRunningProcesses = lambda h: [
        types.SimpleNamespace(pid=p, usedGpuMemory=1234) for p in pids
    ]
    return m


def test_attribution_is_data_when_nvml_lists_our_pid(monkeypatch):
    import os

    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml([os.getpid(), 1]))
    out = engine.gpu_attribution_for_tree()
    assert out["per_process_gpu_attribution"] == [
        {"pid": os.getpid(), "used_gpu_memory": 1234}
    ]
    assert out["per_process_gpu_attribution_reason"] is None


def test_reason_states_the_measured_count_when_none_are_ours(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml([1, 2]))
    out = engine.gpu_attribution_for_tree()
    assert out["per_process_gpu_attribution"] is None
    reason = out["per_process_gpu_attribution_reason"]
    assert reason.startswith("nvmlDeviceGetComputeRunningProcesses returned 2 process(es)")
    host = engine._host_kind()
    expected = {"wsl2": "WSL2", "windows": "WDDM", "linux": "no compute process"}[host]
    assert expected in reason
