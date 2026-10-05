"""llama.cpp diagnostics must never persist prompt or generated text."""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from vllm_omni.engine.backends.llamacpp import (
    _FilteredLlamaLog,
    _hybrid_placement_evidence,
    _sanitize_llama_log_line,
    LlamaCppTextStageClient,
)


DEVICE_NAME = "NVIDIA GeForce RTX 5090 Laptop GPU"


def test_sanitizer_keeps_only_canonical_placement_and_diagnostic_fields() -> None:
    lines = [
        f"llama_model_load: using device Vulkan0 ({DEVICE_NAME})",
        "load_tensors: layer  12 assigned to device CPU, is_swa = 0",
        "load_tensors: layer  60 assigned to device Vulkan0, is_swa = 0",
        "load_tensors: offloaded 40/61 layers to GPU",
        "load_tensors: CPU_Mapped model buffer size = 6588.22 MiB",
        "load_tensors: Vulkan0 model buffer size = 4299.40 MiB",
        "llama_kv_cache: Vulkan0 KV buffer size = 768.00 MiB",
        "llama_context: CPU compute buffer size = 6.77 MiB",
        "tensor blk.0.ffn_up_exps.weight (400 MiB Q4) buffer type overridden to CPU",
        "tensor blk.0.ffn_down_exps.weight (400 MiB Q4) buffer type overridden to CPU",
        "tensor blk.0.ffn_gate_exps.weight (400 MiB Q4) buffer type overridden to CPU",
        "clip_ctx: CLIP using Vulkan0 backend",
        "error: failed to allocate secret customer content",
        'data: {"choices":[{"delta":{"content":"private answer"}}]}',
        'srv  log_server_r: request: {"messages":[{"content":"private prompt"}]}',
        '0.21.039.904 I srv log_server_r: user said out of memory and failed',
    ]
    safe = [
        _sanitize_llama_log_line(line, expected_device_name=DEVICE_NAME)
        for line in lines
    ]
    persisted = "\n".join(line for line in safe if line is not None)
    assert "private" not in persisted
    assert "secret" not in persisted
    assert "6588.22 MiB" in persisted
    assert "Vulkan0 KV buffer size = 768.00 MiB" in persisted
    assert "load_tensors: layer 12 assigned to device CPU" in persisted
    assert "tensor blk.0.ffn_up_exps.weight (redacted) buffer type overridden to CPU" in persisted
    assert "llama.cpp diagnostic: allocation_failed" in persisted
    assert _sanitize_llama_log_line(
        "warning: no usable GPU found", expected_device_name=None,
    ) == "warning: no usable GPU found"
    assert _sanitize_llama_log_line(
        '0.21.039.904 I srv log_server_r: user said out of memory and failed',
        expected_device_name=DEVICE_NAME,
    ) is None
    assert _hybrid_placement_evidence(
        persisted, gpu_device="Vulkan0", expected_device_name=DEVICE_NAME,
        gpu_layers=40, cpu_moe_layers=1,
        cpu_weight_budget_bytes=7000 << 20,
        gpu_weight_budget_bytes=5000 << 20,
    )["expert_cpu_tensors_verified"] == 3


def test_sanitizer_preserves_exact_host_override_without_tensor_metadata() -> None:
    assert _sanitize_llama_log_line(
        "load_tensors: Vulkan_Host model buffer size = 16499.72 MiB",
        expected_device_name=DEVICE_NAME,
    ) == "load_tensors: Vulkan_Host model buffer size = 16499.72 MiB"
    assert _sanitize_llama_log_line(
        "tensor blk.39.ffn_down_exps.weight (private metadata) "
        "buffer type overridden to Vulkan_Host",
        expected_device_name=DEVICE_NAME,
    ) == ("tensor blk.39.ffn_down_exps.weight (redacted) "
          "buffer type overridden to Vulkan_Host")
    assert _sanitize_llama_log_line(
        "tensor blk.39.ffn_down_exps.weight.extra (private metadata) "
        "buffer type overridden to Vulkan_Host",
        expected_device_name=DEVICE_NAME,
    ) == "llama.cpp diagnostic: unexpected_expert_override"
    assert _sanitize_llama_log_line(
        "tensor blk.39.ffn_down_exps.weight (private metadata) "
        "buffer type overridden to UnknownDevice",
        expected_device_name=DEVICE_NAME,
    ) == "llama.cpp diagnostic: unexpected_expert_override"


def test_filter_drains_verbose_output_and_waits_for_complete_placement(tmp_path) -> None:
    script = """
import sys
for _ in range(2000):
    print('srv log_server_r: request ' + 'private prompt and answer ' * 200)
print('llama_model_load: using device Vulkan0 (NVIDIA GeForce RTX 5090 Laptop GPU)')
print('load_tensors: layer   0 assigned to device CPU, is_swa = 0')
print('load_tensors: layer  60 assigned to device Vulkan0, is_swa = 0')
print('load_tensors: offloaded 40/61 layers to GPU')
print('load_tensors: CPU_Mapped model buffer size = 6588.22 MiB')
print('load_tensors: Vulkan0 model buffer size = 4299.40 MiB')
print('srv  llama_server: listening on http://127.0.0.1:29011')
print('srv log_server_r: private prompt load_tensors: layer 999 assigned to device Vulkan0')
sys.stdout.flush()
"""
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    assert proc.stdout is not None
    path = tmp_path / "server.log"
    filtered = _FilteredLlamaLog(
        proc.stdout, path, expected_device_name=DEVICE_NAME, port=29011,
    )
    startup = filtered.wait_for_startup(time.monotonic() + 10)
    assert proc.wait(timeout=5) == 0
    assert filtered.join_after_exit()
    assert "offloaded 40/61" in startup
    assert "Vulkan0 model buffer size" in startup
    assert "private" not in startup
    assert "private" not in path.read_text(encoding="utf-8")
    assert "layer 999" not in path.read_text(encoding="utf-8")
    assert path.stat().st_size < 4096


def test_filter_joins_after_server_termination(tmp_path) -> None:
    script = """
import time
print('srv  llama_server: listening on http://127.0.0.1:29012', flush=True)
while True:
    print('srv log_server_r: request private tool result', flush=True)
    time.sleep(.001)
"""
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    assert proc.stdout is not None
    path = tmp_path / "cancelled.log"
    filtered = _FilteredLlamaLog(
        proc.stdout, path, expected_device_name=None, port=29012,
    )
    filtered.wait_for_startup(time.monotonic() + 5)
    client = LlamaCppTextStageClient.__new__(LlamaCppTextStageClient)
    client._proc = proc
    client._log_filter = filtered
    assert client._terminate()
    assert proc.poll() is not None
    assert client._log_filter is None
    assert "private" not in path.read_text(encoding="utf-8")


def test_filter_refuses_startup_without_listen_marker(tmp_path) -> None:
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", "print('error: private failure detail')"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    assert proc.stdout is not None
    path = tmp_path / "failed.log"
    filtered = _FilteredLlamaLog(
        proc.stdout, path, expected_device_name=None, port=29013,
    )
    with pytest.raises(RuntimeError, match="complete placement evidence"):
        filtered.wait_for_startup(time.monotonic() + 5)
    assert proc.wait(timeout=5) == 0
    assert filtered.join_after_exit()
    assert path.read_text(encoding="utf-8") == "llama.cpp diagnostic: backend_error\n"
