"""Profile binding diagnostics use audited raw identity, not route names alone."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from benchmarks.edge_agent.experiments import profile_binding as binding


def _fixture(tmp_path, monkeypatch):
    config_bytes = b'{"routes":[{"route_id":"gemma"}]}'
    hardware = {
        "os": "Windows-11-test", "machine": "AMD64", "cpu": "Test CPU",
        "host_ram_total_bytes": 64_000_000_000, "host_ram_available_bytes": 25_000_000_000,
        "gpu_name": "Test GPU", "gpu_driver": "610.71",
        "power_condition": "AC", "vram_available_bytes": 20_000_000_000,
    }
    conditions = {
        "os_version": hardware["os"], "driver_versions": {"nvidia": hardware["gpu_driver"]},
        "power_condition": hardware["power_condition"],
        "hardware_id": "Test CPU | Test GPU | RAM 64000000000 bytes",
        "environment_fingerprint": "f" * 64,
        "runtime_versions": {
            "vllm_omni_imported_source_sha256": "s" * 64,
            "agent_runtime_identity_sha256": "r" * 64,
        },
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"conditions": conditions,
                                        "raw_sha256": "a" * 64}), encoding="utf-8")
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps({
        "protocol": "full_20x3_and_30m", "conditions": conditions,
        "hardware": hardware,
        "source_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "results": [{"route_id": "gemma", "summary": str(summary_path),
                     "raw_sha256": "a" * 64}],
    }), encoding="utf-8")
    audit = SimpleNamespace(internally_valid=True, trace_verified=True,
                            protocol_compliant=True, route_id="gemma",
                            raw_sha256="a" * 64)
    monkeypatch.setattr(binding, "audit_summary", lambda _path: audit)
    monkeypatch.setattr(binding, "imported_omni_source_sha256", lambda: "s" * 64)
    monkeypatch.setattr(binding, "loaded_runtime_sha256", lambda: "r" * 64)
    monkeypatch.setattr(binding, "_fingerprint", lambda _hardware: "f" * 64)
    return index_path, config_bytes, hardware, audit


def test_binds_exact_audited_profile_hardware_and_runtime(tmp_path, monkeypatch):
    index, config, hardware, _audit = _fixture(tmp_path, monkeypatch)
    record = binding.bind_profile_to_live(index, source_config_bytes=config,
                                          route_id="gemma", hardware=hardware)
    assert record["schema"] == "omni-agent-profile-binding-v1"
    assert record["profile_index_sha256"] == hashlib.sha256(index.read_bytes()).hexdigest()
    assert record["profile_raw_sha256"] == "a" * 64
    assert record["source_config_sha256"] == hashlib.sha256(config).hexdigest()
    assert record["environment_fingerprint"] == "f" * 64
    assert record["imported_omni_source_sha256"] == "s" * 64
    assert record["loaded_runtime_sha256"] == "r" * 64
    assert record["hardware_identity"] == {
        key: hardware[key] for key in binding.HARDWARE_IDENTITY_KEYS
    }


def test_cpu_only_profile_can_bind_without_an_nvidia_driver(tmp_path, monkeypatch):
    index, config, hardware, _audit = _fixture(tmp_path, monkeypatch)
    data = json.loads(index.read_text(encoding="utf-8"))
    hardware["gpu_name"] = None
    hardware["gpu_driver"] = None
    data["hardware"] = hardware
    data["conditions"]["driver_versions"]["nvidia"] = "None"
    data["conditions"]["hardware_id"] = "Test CPU | None | RAM 64000000000 bytes"
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"conditions": data["conditions"],
                                   "raw_sha256": "a" * 64}), encoding="utf-8")
    index.write_text(json.dumps(data), encoding="utf-8")
    record = binding.bind_profile_to_live(index, source_config_bytes=config,
                                          route_id="gemma", hardware=hardware)
    assert record["hardware_identity"]["gpu_name"] is None
    assert record["hardware_identity"]["gpu_driver"] is None


@pytest.mark.parametrize("field,value", [
    ("os", "another Windows"), ("machine", "ARM64"), ("cpu", "other CPU"),
    ("host_ram_total_bytes", 32_000_000_000), ("gpu_name", "other GPU"),
    ("gpu_driver", "other driver"), ("power_condition", "battery"),
])
def test_refuses_a_different_machine_or_power(tmp_path, monkeypatch, field, value):
    index, config, hardware, _audit = _fixture(tmp_path, monkeypatch)
    hardware[field] = value
    with pytest.raises(binding.ProfileBindingError, match="live hardware differs"):
        binding.bind_profile_to_live(index, source_config_bytes=config,
                                     route_id="gemma", hardware=hardware)


def test_refuses_changed_config_raw_or_runtime(tmp_path, monkeypatch):
    index, config, hardware, audit = _fixture(tmp_path, monkeypatch)
    with pytest.raises(binding.ProfileBindingError, match="source config differs"):
        binding.bind_profile_to_live(index, source_config_bytes=config + b" ",
                                     route_id="gemma", hardware=hardware)
    audit.raw_sha256 = "b" * 64
    with pytest.raises(binding.ProfileBindingError, match="raw requests failed audit"):
        binding.bind_profile_to_live(index, source_config_bytes=config,
                                     route_id="gemma", hardware=hardware)
    audit.raw_sha256 = "a" * 64
    monkeypatch.setattr(binding, "loaded_runtime_sha256", lambda: "new-runtime")
    with pytest.raises(binding.ProfileBindingError, match="runtime, or environment"):
        binding.bind_profile_to_live(index, source_config_bytes=config,
                                     route_id="gemma", hardware=hardware)


def test_refuses_incomplete_or_ambiguous_profile(tmp_path, monkeypatch):
    index, config, hardware, _audit = _fixture(tmp_path, monkeypatch)
    data = json.loads(index.read_text(encoding="utf-8"))
    data["protocol"] = "smoke_incomplete"
    index.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(binding.ProfileBindingError, match="full-protocol"):
        binding.bind_profile_to_live(index, source_config_bytes=config,
                                     route_id="gemma", hardware=hardware)
