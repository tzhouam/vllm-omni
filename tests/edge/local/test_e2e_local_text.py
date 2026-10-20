# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End to end: the M0 acceptance criteria, executed against a real checkpoint.

This is the test that says the local text mode works, so it tests the thing the
roadmap actually asks for rather than a proxy for it:

* both startups -- device-visible and capability-masked -- reach a decision,
  and a refusal is a pass, because "选择兼容方案或明确拒绝" is an either/or;
* the same fixed set of twelve prompts, each generating at least 128 tokens;
* TTFT and decode rate recorded per request;
* the budgeted peak is an upper bound on the measured one, and nothing OOMs;
* cancellation releases, and no event from the retired epoch is delivered.

It needs real weights and a real backend, so it skips rather than fails when
there is no checkpoint. Which one it runs is whatever this interpreter can
drive: the CUDA venv gets the bf16 build, the CPU venv gets the int8 one --
which is itself the point, since the int8 build is the one the CUDA plan
refuses. Override with ``VLLM_OMNI_LOCAL_TEST_MODEL``.

Roughly a minute on the reference laptop's GPU and three on its CPU. Marked
``local_model`` because it is not a unit test and should not run in a PR gate.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import pytest_asyncio

from vllm_omni.edge.local import (
    LocalTextEngine,
    PlanNotAdmitted,
    acceptance_prompts,
    plan_text_session,
)
from vllm_omni.edge.local.capabilities import FORMAT_INT8

# One event loop for the whole module. The engine fixture is module-scoped --
# loading 7.7 GB per test would be absurd -- and the asyncio primitives inside
# AsyncOmni belong to whichever loop created them. Leaving the tests on
# pytest-asyncio's default per-function loop means a cancel awaits a task whose
# backend cleanup lives on a different, no-longer-running loop, and it hangs
# there rather than failing.
pytestmark = [pytest.mark.local_model, pytest.mark.asyncio(loop_scope="module")]

MIN_TOKENS = 128
MAX_MODEL_LEN = 4096

# Checkpoints under the repo's models/ directory, by the platform that can run
# them. The int8 build is deliberately the CPU entry: Blackwell refuses it.
_CANDIDATES = {
    "cuda": ("Spark-X2.5-4B", "Spark-X2.5-1.7B-int8"),
    "cpu": ("Spark-X2.5-1.7B-int8", "Spark-X2.5-4B"),
}


def _models_root() -> Path:
    # tests/edge/local/ -> vllm-omni/ -> the workspace that holds models/
    return Path(__file__).resolve().parents[3].parent / "models"


def _platform() -> str:
    from vllm.platforms import current_platform

    return str(getattr(current_platform, "device_type", "") or "")


def _find_model() -> str:
    override = os.environ.get("VLLM_OMNI_LOCAL_TEST_MODEL")
    if override:
        return override
    root = _models_root()
    for name in _CANDIDATES.get(_platform(), ()):
        if (root / name / "config.json").is_file():
            return str(root / name)
    pytest.skip(
        f"no Spark checkpoint for platform {_platform()!r} under {root}; "
        "set VLLM_OMNI_LOCAL_TEST_MODEL to run this"
    )


@pytest.fixture(scope="module")
def model_dir() -> str:
    return _find_model()


@pytest.fixture(scope="module")
def plan(model_dir: str):
    p = plan_text_session(model_dir, max_model_len=MAX_MODEL_LEN, max_num_seqs=1)
    if not p.admitted:
        pytest.skip(f"this host cannot run {model_dir}:\n{p.summary()}")
    return p


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def engine(plan):
    eng = LocalTextEngine(plan)
    await eng.start()
    try:
        yield eng
    finally:
        await eng.close()


# ------------------------------------------------------------------ planning
def test_the_masked_startup_reaches_a_decision(model_dir):
    """The no-NVIDIA class must produce a plan or an explicit refusal -- never
    a crash and never a silent substitution."""
    masked = plan_text_session(
        model_dir, max_model_len=MAX_MODEL_LEN, mask=frozenset({"gpu_discrete"})
    )
    if masked.admitted:
        assert masked.selected.memory_pool == "host_ram"
        assert masked.selected.kind != "gpu_discrete"
    else:
        assert masked.refusals
        assert all(r.message and r.remedy for r in masked.refusals)
        assert any("mask" in r.message for r in masked.refusals) or _platform() != "cuda"


def test_a_refused_plan_cannot_be_started(model_dir):
    refused = plan_text_session(
        model_dir, max_model_len=MAX_MODEL_LEN, mask=frozenset({"gpu_discrete", "cpu"})
    )
    assert not refused.admitted
    with pytest.raises(PlanNotAdmitted):
        LocalTextEngine(refused)


@pytest.mark.skipif(
    not (_models_root() / "Spark-X2.5-1.7B-int8" / "config.json").is_file(),
    reason="the int8 checkpoint is not present",
)
def test_int8_is_refused_on_cuda_and_admitted_on_cpu():
    """The refusal and its remedy, checked against the real checkpoint on
    whichever backend this interpreter has."""
    int8 = str(_models_root() / "Spark-X2.5-1.7B-int8")
    p = plan_text_session(int8, max_model_len=MAX_MODEL_LEN)
    assert p.manifest.weight_format == FORMAT_INT8
    if _platform() == "cuda":
        assert not p.admitted
        formats = [r for r in p.refusals if r.code == "weight_format_unsupported"]
        assert formats and "omni-cpu" in formats[0].remedy
    else:
        assert p.admitted
        assert p.selected.device_id == "cpu"


# ------------------------------------------------------------------- loading
async def test_the_engine_loads_and_reports_where_it_landed(engine):
    assert engine.load_seconds and engine.load_seconds > 0
    placement = engine.report_placement()
    configured = placement["configured"]
    assert "error" not in configured
    # Read back from the stage's own config, not echoed from the plan.
    assert configured["device_type"] == _platform()
    assert configured["max_model_len"] == MAX_MODEL_LEN
    assert configured["enforce_eager"] is True


async def test_the_kv_pool_is_the_size_the_plan_asked_for(engine, plan):
    """An absolute budget, honoured -- not whatever was left inside a
    utilisation fraction."""
    configured = engine.report_placement()["configured"]
    assert configured["kv_cache_memory_bytes"] == plan.kv_budget.bytes_total


# ---------------------------------------------------------- the twelve prompts
async def test_twelve_prompts_each_reach_128_tokens(engine):
    prompts = acceptance_prompts()
    assert len(prompts) == 12

    records = []
    for name, text in prompts:
        session = engine.open_session()
        request_id, stream = await engine.submit(
            session, text, max_tokens=MIN_TOKENS, temperature=0.0, ignore_eos=True
        )
        seqs = []
        async for event in stream:
            if event.kind == "token":
                seqs.append(event.seq)
            assert event.epoch == session.epoch
            assert event.error is None, f"{name}: {event.error}"
        record = engine.records[request_id]
        engine.close_session(session.session_id)

        assert record.finished, f"{name} did not finish"
        assert record.output_tokens >= MIN_TOKENS, f"{name} produced {record.output_tokens}"
        assert record.ttft_s is not None and record.ttft_s > 0
        assert record.decode_tok_per_s is not None and record.decode_tok_per_s > 0
        # Incremental, in order, no repeats: seq is what detects a lost chunk.
        assert seqs == sorted(seqs)
        assert len(seqs) == len(set(seqs))
        records.append(record)

    # The long prompt crosses the 512 sliding window many times; if window
    # rollover were broken it would be broken here and not on the short ones.
    long_record = next(r for r in records if r.prompt_tokens and r.prompt_tokens > 2000)
    assert long_record.output_tokens >= MIN_TOKENS


async def test_the_budget_bounded_the_measured_peak(engine, plan):
    """Admission's whole claim: the plan's number is an upper bound, proven
    before anything allocated."""
    measured = engine.measured_peak()
    assert measured["budgeted_peak_bytes"] == plan.peak_bytes
    attributable = measured["attributable_peak_bytes"]
    assert attributable is not None and attributable > 0
    assert attributable <= measured["budgeted_peak_bytes"], (
        f"measured {attributable / 2**30:.2f} GiB exceeded the budgeted "
        f"{measured['budgeted_peak_bytes'] / 2**30:.2f} GiB"
    )


async def test_usage_separates_what_was_measured_from_what_was_estimated(engine):
    usage = engine.report_usage()
    by_purpose = {r["purpose"]: r for r in usage["reservations"]}
    assert by_purpose["weights"]["evidence"] == "D"
    assert by_purpose["kv_cache"]["evidence"] == "E"
    # Nothing gets an actual it cannot support: only the weights line is
    # bounded by a counter on this host.
    bounded = [p for p, r in by_purpose.items() if r["actual_upper_bound_bytes"]]
    assert bounded == ["weights"]


# ------------------------------------------------------------- cancellation
async def test_cancel_releases_and_delivers_nothing_from_the_retired_epoch(engine):
    session = engine.open_session()
    request_id, stream = await engine.submit(
        session, acceptance_prompts()[0][1], max_tokens=4096, ignore_eos=True
    )
    seen = 0
    async for event in stream:
        if event.kind == "token":
            seen += 1
        if seen >= 8:
            break
    assert seen >= 8

    report = await engine.cancel(request_id)
    assert report["aborted_in_backend"]
    assert report["producer_stopped"], "the producer task did not unwind"
    assert report["inflight_after"] == 0
    # The session's state is retired, not reused.
    assert report["session_epoch_now"] == session.epoch + 1

    leaked = []
    while True:
        event = await stream.get()
        if event is None:
            break
        leaked.append(event)
    assert leaked == [], f"{len(leaked)} events escaped the retired epoch"


async def test_a_retired_handle_cannot_be_resubmitted(engine):
    session = engine.open_session()
    request_id, stream = await engine.submit(session, "Hello", max_tokens=32)
    async for _ in stream:
        break
    await engine.cancel(request_id)
    with pytest.raises(RuntimeError, match="stale"):
        await engine.submit(session, "Hello again", max_tokens=8)


async def test_the_engine_still_generates_after_a_cancel(engine):
    """A cancelled request must not poison the engine for the next one."""
    session = engine.open_session()
    request_id, stream = await engine.submit(
        session, "The capital of Australia is", max_tokens=16, ignore_eos=True
    )
    async for _ in stream:
        pass
    record = engine.records[request_id]
    assert record.finished and record.output_tokens >= 16
