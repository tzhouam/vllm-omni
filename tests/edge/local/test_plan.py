# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Admission: say yes with a budget, or no with a reason and a remedy."""

import pytest

from vllm_omni.edge.local.plan import (
    GiB,
    REFUSE_CAPACITY,
    REFUSE_FORMAT,
    REFUSE_NO_DEVICE,
    plan_text_session,
)

from .conftest import FP8_QUANT_CONFIG, blackwell_laptop, cpu_only_box, write_checkpoint

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _plan(model_dir, profile=None, platform="cuda", **kw):
    return plan_text_session(
        str(model_dir),
        profile=profile if profile is not None else blackwell_laptop(),
        platform_device_type=platform,
        **kw,
    )


def _codes(plan):
    return {r.code for r in plan.refusals}


# ------------------------------------------------------------------ the yes
def test_a_dense_checkpoint_is_admitted_to_the_gpu(dense_checkpoint):
    plan = _plan(dense_checkpoint)
    assert plan.admitted
    assert plan.selected.device_id == "cuda:0"
    assert plan.backend == "vllm:cuda"


def test_an_admitted_plan_prices_every_line_of_section_6_2(dense_checkpoint):
    """The budget is not 'the weights'. It is the sum the design specifies,
    and each term has to be visible so a reviewer can argue with one."""
    plan = _plan(dense_checkpoint)
    purposes = {r.purpose for r in plan.reservations}
    assert purposes == {
        "weights", "kv_cache", "activations", "backend_workspace",
        "load_transient", "external_reserve", "safety_margin",
    }
    assert all(r.pool == "vram" for r in plan.reservations)
    assert all(r.evidence in {"D", "E"} for r in plan.reservations)
    # The weights line is read off the files, not estimated.
    weights = next(r for r in plan.reservations if r.purpose == "weights")
    assert weights.evidence == "D"
    assert weights.bytes == plan.manifest.weight_bytes


def test_graph_capture_is_charged_only_when_graphs_are_on(dense_checkpoint):
    eager = _plan(dense_checkpoint, enforce_eager=True)
    graphs = _plan(dense_checkpoint, enforce_eager=False)
    assert "graph_capture" not in {r.purpose for r in eager.reservations}
    assert "graph_capture" in {r.purpose for r in graphs.reservations}
    assert graphs.peak_bytes > eager.peak_bytes


def test_the_peak_is_a_max_not_a_sum(dense_checkpoint):
    """The loader's transient is gone before the KV pool is sized, so adding
    both would refuse plans that fit."""
    plan = _plan(dense_checkpoint)
    everything = sum(r.bytes for r in plan.reservations)
    assert plan.peak_bytes <= everything


def test_kv_reservation_tracks_the_context(dense_checkpoint):
    short = _plan(dense_checkpoint, max_model_len=1024)
    long = _plan(dense_checkpoint, max_model_len=8192)
    kv_short = next(r for r in short.reservations if r.purpose == "kv_cache").bytes
    kv_long = next(r for r in long.reservations if r.purpose == "kv_cache").bytes
    assert kv_long > kv_short


def test_sliding_layers_are_not_charged_full_length(dense_checkpoint):
    """21 of 28 layers hold a 512 window, not the whole context. Charging them
    all at full length is the arithmetic the kv_budget module exists to fix."""
    plan = _plan(dense_checkpoint, max_model_len=8192)
    assert plan.kv_budget.sliding_layers == 21
    assert plan.kv_budget.full_layers == 7
    assert plan.kv_budget.bytes_total_hybrid < plan.kv_budget.bytes_total


# ------------------------------------------------------------------- the no
def test_int8_is_refused_on_blackwell_although_it_fits(int8_checkpoint):
    """4 MiB of weights against 24 GiB of VRAM: every capacity check passes and
    the answer is still no."""
    plan = _plan(int8_checkpoint)
    assert not plan.admitted
    assert REFUSE_FORMAT in _codes(plan)
    refusal = next(r for r in plan.refusals if r.code == REFUSE_FORMAT)
    assert refusal.device_id == "cuda:0"
    assert "int8" in refusal.message
    assert "omni-cpu" in refusal.remedy


def test_the_same_int8_checkpoint_is_admitted_on_the_cpu(int8_checkpoint):
    """The remedy the refusal names has to actually be a plan."""
    plan = _plan(int8_checkpoint, profile=cpu_only_box(), platform="cpu")
    assert plan.admitted
    assert plan.selected.device_id == "cpu"
    assert plan.backend == "vllm:cpu"


def test_fp8_is_admitted_where_int8_is_not(tmp_path):
    fp8 = write_checkpoint(tmp_path / "fp8", quantization=FP8_QUANT_CONFIG)
    assert _plan(fp8).admitted


def test_a_checkpoint_too_large_for_the_pool_is_refused_with_a_number(tmp_path):
    huge = write_checkpoint(tmp_path / "huge", weight_bytes=40 * GiB, shards=8)
    plan = _plan(huge)
    assert not plan.admitted
    assert REFUSE_CAPACITY in _codes(plan)
    refusal = next(r for r in plan.refusals if r.code == REFUSE_CAPACITY)
    assert "GiB" in refusal.message


def test_a_capacity_refusal_never_silently_shortens_the_context(tmp_path):
    """Trimming max_model_len is offered as a remedy the operator applies, not
    something the planner does on its behalf."""
    # Sized so the overrun is smaller than the KV cache it asked for: that is
    # the case where a shorter context would in fact fit, which is exactly when
    # an engine that "helpfully" truncates would do damage.
    big = write_checkpoint(tmp_path / "big", weight_bytes=16 * GiB, shards=8)
    plan = _plan(big, max_model_len=131072)
    assert not plan.admitted
    refusal = next(r for r in plan.refusals if r.code == REFUSE_CAPACITY)
    assert "max-model-len" in refusal.remedy
    assert "forbids" in refusal.remedy
    # And the plan still reports the context that was asked for, not a
    # quietly reduced one.
    assert plan.request_limits["max_model_len"] == 131072


def test_masking_the_gpu_leaves_nothing_runnable_in_a_cuda_process(dense_checkpoint):
    """The capability-masked startup of the acceptance run: a refusal here is
    the correct outcome, not a failure."""
    plan = _plan(dense_checkpoint, mask=frozenset({"gpu_discrete"}))
    assert not plan.admitted
    assert REFUSE_NO_DEVICE in _codes(plan)
    assert any("mask" in r.message for r in plan.refusals)


def test_every_unusable_device_is_explained(dense_checkpoint):
    plan = _plan(dense_checkpoint)
    explained = {r.device_id for r in plan.refusals}
    assert {"cpu", "npu:amd", "gpu_integrated:amd"} <= explained
    assert all(r.message and r.remedy for r in plan.refusals)


# ------------------------------------------------------------ engine kwargs
def test_the_memory_fraction_is_derived_not_defaulted(dense_checkpoint):
    """vLLM reads this field as host RAM on the CPU backend, so leaving it at
    its default there asks for 92% of the machine. It is always derived."""
    for platform, profile in (("cuda", blackwell_laptop()), ("cpu", cpu_only_box())):
        plan = _plan(dense_checkpoint, profile=profile, platform=platform)
        assert plan.admitted
        fraction = plan.engine_kwargs["gpu_memory_utilization"]
        assert 0 < fraction <= 0.95
        assert fraction >= plan.peak_bytes / plan.selected.memory_bytes * 0.5


def test_the_kv_pool_is_absolute_not_leftover(dense_checkpoint):
    plan = _plan(dense_checkpoint)
    assert plan.engine_kwargs["kv_cache_memory_bytes"] == plan.kv_budget.bytes_total


def test_request_limits_reach_the_backend(dense_checkpoint):
    plan = _plan(dense_checkpoint, max_model_len=2048, max_num_seqs=2)
    assert plan.engine_kwargs["max_model_len"] == 2048
    assert plan.engine_kwargs["max_num_seqs"] == 2
    assert plan.engine_kwargs["enforce_eager"] is True


# ---------------------------------------------------------------- the stage
def test_the_plan_declares_exactly_one_text_stage(dense_checkpoint):
    plan = _plan(dense_checkpoint)
    assert plan.stage["stage_id"] == 0
    assert plan.stage["execution_type"] == "LLM_AR"
    assert plan.stage["final_output_type"] == "text"
    assert plan.stage["input_sources"] == []


def test_the_record_round_trips_to_json(dense_checkpoint):
    import json

    plan = _plan(dense_checkpoint)
    text = json.dumps(plan.to_dict(), default=str)
    back = json.loads(text)
    assert back["admitted"]
    assert back["selected_device"]["device_id"] == "cuda:0"
    assert back["manifest"]["artifact_id"] == plan.manifest.artifact_id
