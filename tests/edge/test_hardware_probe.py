"""Tests for vllm_omni.edge.hardware_probe using synthetic /proc and /sys snapshots (CPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_omni.edge import hardware_probe as hp

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

X86_CPUINFO = """processor\t: 0
model name\t: Intel(R) Xeon(R) Platinum 8480C
flags\t\t: fpu sse avx2 avx512f avx512_bf16 amx_bf16 amx_tile fma f16c
processor\t: 1
model name\t: Intel(R) Xeon(R) Platinum 8480C
flags\t\t: fpu sse avx2 avx512f avx512_bf16 amx_bf16 amx_tile fma f16c
"""

ARM_CPUINFO = """processor\t: 0
Features\t: fp asimd evtstrm aes asimdhp asimddp
processor\t: 1
Features\t: fp asimd evtstrm aes asimdhp asimddp
processor\t: 2
Features\t: fp asimd evtstrm aes asimdhp asimddp i8mm bf16
processor\t: 3
Features\t: fp asimd evtstrm aes asimdhp asimddp i8mm bf16
Hardware\t: Rockchip RK3588
"""

MEMINFO = "MemTotal:       16000000 kB\nMemFree:         1000000 kB\nMemAvailable:   12000000 kB\n"


def _write_sysfs(root: Path, cpus: dict[int, tuple[int, int]], numa: int = 1, thermal: int = 2, tegra: bool = False):
    for cpu, (max_khz, core_id) in cpus.items():
        d = root / "sys" / "devices" / "system" / "cpu" / f"cpu{cpu}"
        (d / "cpufreq").mkdir(parents=True, exist_ok=True)
        (d / "topology").mkdir(parents=True, exist_ok=True)
        (d / "cpufreq" / "cpuinfo_max_freq").write_text(str(max_khz))
        (d / "topology" / "core_id").write_text(str(core_id))
        (d / "topology" / "physical_package_id").write_text("0")
    for n in range(numa):
        (root / "sys" / "devices" / "system" / "node" / f"node{n}").mkdir(parents=True, exist_ok=True)
    for t in range(thermal):
        (root / "sys" / "class" / "thermal" / f"thermal_zone{t}").mkdir(parents=True, exist_ok=True)
    if tegra:
        (root / "etc").mkdir(parents=True, exist_ok=True)
        (root / "etc" / "nv_tegra_release").write_text("# R36 (release), REVISION: 4.0")


def _snapshot(tmp_path: Path, cpuinfo: str, meminfo: str = MEMINFO, **kw) -> Path:
    root = tmp_path / "root"
    (root / "proc").mkdir(parents=True)
    (root / "proc" / "cpuinfo").write_text(cpuinfo)
    (root / "proc" / "meminfo").write_text(meminfo)
    _write_sysfs(root, **kw)
    return root


def test_x86_amx_snapshot(tmp_path):
    root = _snapshot(tmp_path, X86_CPUINFO, cpus={0: (3800000, 0), 1: (3800000, 0)}, numa=2)
    p = hp.probe(root, use_torch=False)
    assert p.arch == "x86_64"
    assert p.has("avx2", "avx512", "avx512_bf16", "amx_bf16", "amx_tile")
    assert p.n_logical == 2 and p.n_physical == 1  # two hyper-threads of one core
    assert p.numa_nodes == 2 and p.thermal_zones == 2
    assert p.ram_total_bytes == 16000000 * 1024 and p.ram_available_bytes == 12000000 * 1024
    assert p.accelerator == "none" and p.supports_bf16 is True
    assert hp.hardware_class(p) == hp.HW_CLASS_X86_CPU
    assert p.clusters == [{"max_khz": 3800000, "cpus": [0, 1]}]


def test_arm_big_little_snapshot(tmp_path):
    root = _snapshot(tmp_path, ARM_CPUINFO, cpus={0: (1800000, 0), 1: (1800000, 1), 2: (2400000, 2), 3: (2400000, 3)})
    p = hp.probe(root, use_torch=False)
    assert p.arch == "aarch64"
    assert p.has("neon", "fp16_arith", "dotprod", "i8mm", "bf16")
    assert p.cpu_model == "Rockchip RK3588"
    assert p.clusters[0] == {"max_khz": 2400000, "cpus": [2, 3]}  # big cluster first
    assert p.big_cpus == [2, 3]
    assert p.n_physical == 4
    assert hp.hardware_class(p) == hp.HW_CLASS_ARM64_CPU
    assert p.supports_bf16 is True and p.supports_fp16 is True


def test_tegra_snapshot_is_jetson_with_unified_memory(tmp_path):
    root = _snapshot(tmp_path, ARM_CPUINFO, cpus={0: (2200000, 0)}, tegra=True)
    p = hp.probe(root, use_torch=False)
    assert p.accelerator == "cuda_unified"
    assert p.gpu_mem_bytes == p.ram_total_bytes
    assert hp.hardware_class(p) == hp.HW_CLASS_JETSON


def test_override_env_and_roundtrip(tmp_path, monkeypatch):
    prof = hp.HardwareProfile(
        arch="aarch64", cpu_flags=["neon"], n_logical=8, accelerator="npu:qualcomm", ram_total_bytes=8 * 2**30
    )
    path = tmp_path / "hw.json"
    path.write_text(prof.to_json())
    monkeypatch.setenv(hp.ENV_OVERRIDE, str(path))
    loaded = hp.load_profile(use_torch=False)
    assert loaded.source.startswith("override:")
    assert loaded.accelerator == "npu:qualcomm"
    assert hp.hardware_class(loaded) == hp.HW_CLASS_NPU_PHONE
    # unknown keys are ignored on load
    d = json.loads(prof.to_json())
    d["future_field"] = 1
    assert hp.HardwareProfile.from_dict(d).n_logical == 8
    assert "class=npu_phone" in hp.describe(loaded)


def test_live_probe_runs_on_this_host():
    p = hp.probe(use_torch=False)
    assert p.n_logical >= 1 and p.ram_total_bytes > 0
    assert hp.hardware_class(p) in hp.HW_CLASSES


# --------------------------------------------------------- accelerators
def _ryzen_ai_root(tmp_path: Path, *, wsl: bool, driver: bool = True) -> Path:
    """A Ryzen AI laptop: XDNA2 NPU, Radeon iGPU, and one discrete card.

    Under WSL none of it has a device node; on native Linux the NPU shows up
    at ``/dev/accel/accel0`` and the GPUs at ``/dev/dri/renderD*``.
    """
    root = _snapshot(tmp_path, X86_CPUINFO, cpus={0: (5100, 0), 1: (5100, 1)})
    (root / "proc" / "version").write_text(
        "Linux version 6.18.33.2-microsoft-standard-WSL2" if wsl
        else "Linux version 6.18.0-generic (gcc 14)"
    )
    npu = root / "sys" / "bus" / "pci" / "devices" / "0000:c5:00.1"
    npu.mkdir(parents=True)
    (npu / "vendor").write_text("0x1022\n")
    (npu / "device").write_text("0x17f0\n")
    if wsl:
        return root
    if driver:
        (npu / "driver").mkdir()
        node = root / "sys" / "class" / "accel" / "accel0" / "device"
        node.mkdir(parents=True)
        (node / "vendor").write_text("0x1022\n")
        (node / "device").write_text("0x17f0\n")
        (node / "driver").mkdir()
        (root / "dev" / "accel").mkdir(parents=True)
        (root / "dev" / "accel" / "accel0").write_text("")
    (root / "dev" / "dri").mkdir(parents=True)
    for name, vendor in (("renderD128", "0x1002"), ("renderD129", "0x10de")):
        (root / "dev" / "dri" / name).write_text("")
        d = root / "sys" / "class" / "drm" / name / "device"
        d.mkdir(parents=True)
        (d / "vendor").write_text(vendor + "\n")
        (d / "device").write_text("0x150e\n")
    return root


def test_native_linux_finds_the_npu_and_both_gpus(tmp_path):
    p = hp.probe(_ryzen_ai_root(tmp_path, wsl=False), use_torch=False)
    assert p.host_os == hp.HOST_OS_LINUX
    (npu,) = p.npus
    assert npu["usable"] and npu["name"].startswith("AMD XDNA2")
    assert [a["kind"] for a in p.accelerators].count(hp.ACCEL_GPU_INTEGRATED) == 1
    assert p.accelerators_of(hp.ACCEL_GPU_DISCRETE, usable=True)
    assert p.unusable_accelerators == []


def test_an_npu_with_no_driver_is_reported_present_and_unusable(tmp_path):
    """"No /dev/accel node" and "no such hardware" need different answers."""
    p = hp.probe(_ryzen_ai_root(tmp_path, wsl=False, driver=False), use_torch=False)
    (npu,) = p.npus
    assert npu["usable"] is False
    assert "amdxdna" in npu["reason"]


def test_wsl_is_detected_and_sees_none_of_the_host_devices(tmp_path):
    """The iGPU and the NPU are in the machine and unreachable from Linux;
    the probe must not report either as absent or as available."""
    p = hp.probe(_ryzen_ai_root(tmp_path, wsl=True), use_torch=False, probe_host=False)
    assert p.host_os == hp.HOST_OS_WSL2
    # The PCI sweep still finds the NPU; there is simply no way to open it.
    (npu,) = p.npus
    assert npu["usable"] is False
    assert p.accelerators_of(hp.ACCEL_GPU_INTEGRATED, usable=True) == []


def test_a_snapshot_root_never_shells_out_to_the_host(tmp_path):
    """probe_host defaults on, but a captured snapshot must stay hermetic."""
    p = hp.probe(_ryzen_ai_root(tmp_path, wsl=True), use_torch=False, probe_host=True)
    assert all(a["source"] != "wsl-interop" for a in p.accelerators)


def test_accelerators_survive_the_json_roundtrip(tmp_path):
    p = hp.probe(_ryzen_ai_root(tmp_path, wsl=False), use_torch=False)
    back = hp.HardwareProfile.from_json(p.to_json())
    assert back.accelerators == p.accelerators
    assert back.host_os == p.host_os


def test_describe_accelerators_says_why_each_one_is_out_of_reach(tmp_path):
    p = hp.probe(_ryzen_ai_root(tmp_path, wsl=True), use_torch=False, probe_host=False)
    text = hp.describe_accelerators(p)
    assert "not here" in text and "amdxdna" in text


# ------------------------------------------------------- reachability routes
def test_the_npu_reason_names_the_recipe_not_just_the_symptom():
    """The iGPU and the NPU are both off-limits to a default WSL process, but
    for different reasons with different fixes. A blanket "unreachable" loses
    the only part a reader can act on -- and for the NPU the actionable part is
    a three-part recipe, each part of which fails silently on its own."""
    assert "MCDM" in hp._WSL_NPU_ROUTE                     # why WSL cannot see it
    assert "onnxruntime_vitisai_ep" in hp._WSL_NPU_ROUTE   # which of the two EPs
    assert "DLL search path" in hp._WSL_NPU_ROUTE          # how to make it load
    assert "A16W8" in hp._WSL_NPU_ROUTE                    # what it will accept
    assert "torch-directml" in hp._WSL_IGPU_ROUTE          # the iGPU's other fix


def test_the_npu_recipe_module_states_the_three_preconditions():
    """Each of the three can fail on its own and the symptom is identical:
    a session that runs correctly with every node on the CPU."""
    from vllm_omni.edge import npu_ryzenai as npu

    assert npu.EP_GENERAL == "onnxruntime_vitisai_ep.dll"
    assert npu.EP_LIGHT != npu.EP_GENERAL
    assert npu.PARTITION_TARGET == "AMD_AIE2P_4x8_CMC_Overlay"
    kw = npu.quantization_kwargs()
    # A16W8: 16-bit activations, 8-bit weights. A8W8 is silently rejected.
    assert "16" in str(kw["activation_type"])
    assert "8" in str(kw["weight_type"]) and "16" not in str(kw["weight_type"])


def test_the_igpu_is_usable_exactly_when_directml_is_importable(monkeypatch):
    """torch-directml pins torch==2.4.1, so it never shares the engine's venv.
    ``usable`` must answer for *this* process, not for the machine."""
    calls = {"n": 0}

    def fake_run(*a, **k):
        calls["n"] += 1
        class R: stdout = "AMD Radeon(TM) 890M Graphics\n"
        return R()

    import subprocess
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(hp, "directml_available", lambda: False)
    (igpu,) = [a for a in hp.wsl_host_accelerators() if a["kind"] == hp.ACCEL_GPU_INTEGRATED]
    assert igpu["usable"] is False and igpu["route"] == "directml"

    monkeypatch.setattr(hp, "directml_available", lambda: True)
    (igpu,) = [a for a in hp.wsl_host_accelerators() if a["kind"] == hp.ACCEL_GPU_INTEGRATED]
    assert igpu["usable"] is True


def test_directml_available_is_false_without_the_package(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "torch_directml", None)
    assert hp.directml_available() is False
