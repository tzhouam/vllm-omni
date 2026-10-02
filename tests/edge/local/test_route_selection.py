from dataclasses import replace
from pathlib import Path

import pytest

from vllm_omni.edge.local.plan import (
    GiB,
    MeasuredRouteProfile,
    ProfileContext,
    plan_text_session,
    refuse_if_checkpoint_exceeds_ram,
    select_measured_plan,
)
from vllm_omni.edge.local.manifest import build_manifest

from .conftest import blackwell_laptop, cpu_only_box


def _candidates(checkpoint):
    gpu = plan_text_session(str(checkpoint), profile=blackwell_laptop(), platform_device_type="cuda", digest_weights=True)
    cpu = plan_text_session(str(checkpoint), profile=cpu_only_box(), platform_device_type="cpu", digest_weights=True)
    assert gpu.admitted and cpu.admitted
    return gpu, cpu


def _profile(route, context, p95, **overrides):
    fields = dict(
        route_id=route,
        context=context,
        p95_ms=p95,
        sample_count=20,
        startup_amortized_ms=0,
        evidence="P",
        whole_request=True,
        quality_passed=True,
        memory_passed=True,
        placement_verified=True,
        sustained_passed=True,
    )
    fields.update(overrides)
    return MeasuredRouteProfile(**fields)


def test_only_paired_qualified_batch_one_profile_changes_placement(dense_checkpoint):
    gpu, cpu = _candidates(dense_checkpoint)
    context = ProfileContext(
        gpu.manifest.artifact_id, gpu.manifest.weight_sha256,
        "sku-os-driver-runtime", "text:128in:128out", "AC",
    )
    baseline = _profile(gpu.route_id, context, 100)
    alternative = _profile(
        cpu.route_id, context, 80,
        paired_baseline_route_id=gpu.route_id,
        paired_gain_ci_lower_ms=5,
    )
    assert select_measured_plan([gpu, cpu], [baseline, alternative], context) is cpu
    assert select_measured_plan([gpu, cpu], [baseline, replace(alternative, evidence="C")], context) is gpu
    assert select_measured_plan([gpu, cpu], [baseline, replace(alternative, whole_request=False)], context) is gpu
    assert select_measured_plan([gpu, cpu], [baseline, replace(alternative, sample_count=19)], context) is gpu
    assert select_measured_plan([gpu, cpu], [baseline, replace(alternative, paired_gain_ci_lower_ms=0)], context) is gpu


def test_context_and_startup_cost_prevent_false_speedup(dense_checkpoint):
    gpu, cpu = _candidates(dense_checkpoint)
    context = ProfileContext(
        gpu.manifest.artifact_id, gpu.manifest.weight_sha256,
        "sku-os-driver-runtime", "text:128in:128out", "AC",
    )
    baseline = _profile(gpu.route_id, context, 100)
    faster = _profile(
        cpu.route_id, context, 80,
        paired_baseline_route_id=gpu.route_id,
        paired_gain_ci_lower_ms=5,
    )
    assert select_measured_plan([gpu, cpu], [baseline, replace(faster, startup_amortized_ms=30)], context) is gpu
    assert select_measured_plan([gpu, cpu], [baseline, faster], replace(context, concurrency=2)) is gpu
    assert select_measured_plan([gpu, cpu], [baseline, replace(faster, context=replace(context, power_condition="battery"))], context) is gpu
    assert select_measured_plan([gpu, cpu], [faster], context) is gpu
    assert select_measured_plan([gpu, cpu], [baseline, faster, replace(faster, p95_ms=20)], context) is gpu


def test_load_transient_and_request_memory_peak_in_separate_phases(dense_checkpoint):
    gpu, _ = _candidates(dense_checkpoint)
    load = sum(
        r.bytes for r in gpu.reservations
        if r.lifetime in ("load", "session_weights")
        or r.purpose in ("backend_workspace", "external_reserve", "safety_margin")
    )
    steady = sum(r.bytes for r in gpu.reservations if r.lifetime != "load")
    assert gpu.peak_bytes == max(load, steady)
    assert gpu.peak_bytes < sum(r.bytes for r in gpu.reservations)


def test_plan_text_session_uses_qualified_selection_when_two_routes_are_available(dense_checkpoint, monkeypatch):
    from vllm_omni.edge.local import plan as plan_module

    gpu, cpu = _candidates(dense_checkpoint)
    context = ProfileContext(
        gpu.manifest.artifact_id, gpu.manifest.weight_sha256,
        "sku-os-driver-runtime", "text:128in:128out", "AC",
    )
    monkeypatch.setattr(plan_module, "enumerate_devices", lambda *args, **kwargs: [gpu.selected, cpu.selected])
    selected = plan_text_session(
        str(dense_checkpoint), profile=blackwell_laptop(), digest_weights=True,
        measured_profiles=[
            _profile(gpu.route_id, context, 100),
            _profile(cpu.route_id, context, 80, paired_baseline_route_id=gpu.route_id, paired_gain_ci_lower_ms=5),
        ],
        profile_context=context,
    )
    assert selected.selected.device_id == "cpu"
    assert "qualified batch-1" in selected.notes[-1]


def test_distinct_whole_routes_on_one_primary_device_can_be_compared(dense_checkpoint):
    eager = plan_text_session(
        str(dense_checkpoint), profile=blackwell_laptop(),
        platform_device_type="cuda", enforce_eager=True, digest_weights=True,
    )
    graphed = plan_text_session(
        str(dense_checkpoint), profile=blackwell_laptop(),
        platform_device_type="cuda", enforce_eager=False, digest_weights=True,
    )
    assert eager.admitted and graphed.admitted
    assert eager.selected.device_id == graphed.selected.device_id == "cuda:0"
    assert eager.route_id != graphed.route_id
    assert eager.engine_kwargs["enforce_eager"] is True
    assert graphed.engine_kwargs["enforce_eager"] is False
    context = ProfileContext(
        eager.manifest.artifact_id, eager.manifest.weight_sha256,
        "sku-os-driver-runtime", "text:128in:128out", "AC",
    )
    baseline = _profile(eager.route_id, context, 100)
    faster = _profile(
        graphed.route_id, context, 80,
        paired_baseline_route_id=eager.route_id,
        paired_gain_ci_lower_ms=5,
    )
    assert select_measured_plan([eager, graphed], [baseline, faster], context) is graphed
    with pytest.raises(ValueError, match="distinct whole-plan route_id"):
        select_measured_plan([eager, replace(graphed, route_id=eager.route_id)], [baseline, faster], context)


def test_same_checkpoint_qwen27_weight_floor_requires_exact_target_ram():
    checkpoint = Path(__file__).resolve().parents[4] / "models/Qwen3.8-27B-FP8"
    if not checkpoint.is_dir():
        pytest.skip("local Qwen3.8-27B-FP8 checkpoint is not installed")
    manifest = build_manifest(str(checkpoint))
    assert manifest.weight_bytes == 30_866_866_928
    with pytest.raises(ValueError, match="exact positive"):
        refuse_if_checkpoint_exceeds_ram(manifest, device_id="s25:unknown-sku", usable_ram_bytes=0)
    refusal = refuse_if_checkpoint_exceeds_ram(manifest, device_id="s25:12-gib-sku", usable_ram_bytes=12 * GiB)
    assert refusal is not None
    assert manifest.artifact_id in refusal.message
    assert "different artifact" in refusal.remedy
    # A weight floor below capacity is not a claim that KV/workspace will fit.
    assert refuse_if_checkpoint_exceeds_ram(manifest, device_id="pc:64-gib", usable_ram_bytes=64 * GiB) is None
