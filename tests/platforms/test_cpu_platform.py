"""CPU OmniPlatform: registration, MRO, config forcing (CPU, no accelerator needed)."""

from __future__ import annotations

import pytest
import torch
from vllm.platforms.cpu import CpuPlatform
from vllm.platforms.interface import Platform

import vllm_omni.platforms as omni_platforms
from vllm_omni.platforms.cpu.platform import CPUOmniPlatform
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_registered_and_selected_when_cpu_target(monkeypatch):
    assert "cpu" in omni_platforms.builtin_omni_platform_plugins
    monkeypatch.setattr(omni_platforms, "_cpu_target_requested", lambda: True)
    assert omni_platforms.cpu_omni_platform_plugin() == omni_platforms._CPU_OMNI_PLATFORM
    assert omni_platforms.resolve_current_omni_platform_cls_qualname() == omni_platforms._CPU_OMNI_PLATFORM
    monkeypatch.setattr(omni_platforms, "_cpu_target_requested", lambda: False)
    assert omni_platforms.cpu_omni_platform_plugin() is None


def test_cpu_target_requested_reads_env(monkeypatch):
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_TARGET_DEVICE", "cpu", raising=False)
    assert omni_platforms._cpu_target_requested() is True


def test_mro_and_attributes():
    mro = CPUOmniPlatform.__mro__
    assert mro.index(OmniPlatform) < mro.index(CpuPlatform) < mro.index(Platform)
    p = CPUOmniPlatform()
    assert p.is_cpu() and not p.is_cuda() and not p.is_npu()
    assert p._omni_enum is OmniPlatformEnum.CPU
    assert CPUOmniPlatform.device_name == "cpu" and CPUOmniPlatform.device_type == "cpu"
    assert CPUOmniPlatform.dist_backend == "gloo"
    assert CPUOmniPlatform.get_device_count() == 1
    assert CPUOmniPlatform.get_torch_device() == torch.device("cpu")
    assert CPUOmniPlatform.get_device_version() is None
    assert CPUOmniPlatform.synchronize() is None
    assert CPUOmniPlatform.get_free_memory() > 0
    avail, total = CPUOmniPlatform.get_device_memory()
    assert 0 < avail <= total
    assert CPUOmniPlatform.get_device_total_memory() > 0
    assert CPUOmniPlatform.supports_talker_mtp_graph_capture() is False
    assert CPUOmniPlatform.supports_torch_inductor() is False
    assert CPUOmniPlatform.get_default_stage_config_path() == "vllm_omni/deploy"
    assert "cpu_ar_worker.CPUARWorker" in CPUOmniPlatform.get_omni_ar_worker_cls()
    assert "CPUGenerationWorker" in CPUOmniPlatform.get_omni_generation_worker_cls()
    wrapper_cls = CPUOmniPlatform.get_graph_wrapper_cls()
    assert isinstance(wrapper_cls, type) and not isinstance(object(), wrapper_cls)
    with pytest.raises(NotImplementedError):
        wrapper_cls()
    with pytest.raises(NotImplementedError):
        CPUOmniPlatform.get_diffusion_worker_cls()


def test_inductor_opt_in(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_CPU_INDUCTOR", "1")
    assert CPUOmniPlatform.supports_torch_inductor() is True


def test_check_and_update_config_forces_cpu_rules():
    from vllm.config import VllmConfig
    from vllm.config.device import DeviceConfig

    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    cfg.scheduler_config.async_scheduling = True
    CPUOmniPlatform.check_and_update_config(cfg)
    assert cfg.scheduler_config.async_scheduling is False
    assert cfg.compilation_config.cudagraph_capture_sizes == []
    # worker_cls is resolved by the Omni platform (resolve_worker_cls), not here.
    assert cfg.cache_config.block_size == 128


def test_deploy_overlay_for_cpu_platform():
    from vllm_omni.config.stage_config import _DEPLOY_DIR, _apply_platform_overrides, load_deploy_config

    deploy = load_deploy_config(_DEPLOY_DIR / "qwen3_tts.yaml")
    deploy = _apply_platform_overrides(deploy, platform="cpu")
    s0, s1 = deploy.stages
    assert s0.devices == "cpu" and s1.devices == "cpu"
    assert s0.enforce_eager is True and s1.enforce_eager is True
    assert s0.async_scheduling is False
    assert s0.engine_extras["dtype"] == "bfloat16"
    assert s0.engine_extras["kv_cache_memory_bytes"] == 4 * 2**30
    assert s0.env["VLLM_CPU_OMP_THREADS_BIND"] != s1.env["VLLM_CPU_OMP_THREADS_BIND"]
