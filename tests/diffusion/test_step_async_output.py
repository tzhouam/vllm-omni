# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Step-mode asynchronous media packing: placeholders in the reply, originals to the packing thread."""

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.worker.diffusion_worker import (
    STEP_ASYNC_OUTPUT_KEY,
    detach_step_media,
    step_async_output_enabled,
)
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _chunk(request_id: str, *, frames: int, chunk_index: int, finished: bool = False) -> RunnerOutput:
    result = DiffusionOutput(
        output={"payload": {"video": np.zeros((frames, 4, 4, 3), np.uint8)}, "metadata": {}},
        finished=finished,
        chunk_index=chunk_index,
        total_chunks=8,
        active_event_ids=["e1"],
    )
    return RunnerOutput(request_id=request_id, step_index=3, finished=finished, result=result)


def test_detach_step_media_replaces_results_by_placeholders() -> None:
    queued: list[tuple[DiffusionOutput, str, object]] = []
    ids = iter(["id-a", "id-b"])
    chunk_a = _chunk("a", frames=34, chunk_index=2)
    chunk_b = _chunk("b", frames=17, chunk_index=7, finished=True)
    original_a, original_b = chunk_a.result, chunk_b.result
    idle = RunnerOutput(request_id="c", step_index=1, finished=False, result=None)  # a denoise step, no chunk
    failed = RunnerOutput(request_id="d", step_index=3, finished=True, result=DiffusionOutput(error="boom"))
    envelope = {"rpc_id": "r1", "result": BatchRunnerOutput.from_list([chunk_a, idle, chunk_b, failed])}

    detached = detach_step_media(
        envelope,
        new_id=lambda: next(ids),
        record_event=lambda: "event",
        enqueue=lambda output, async_id, event: queued.append((output, async_id, event)),
    )
    assert detached == 2
    assert [(o is original_a, i, e) for o, i, e in queued] == [(True, "id-a", "event"), (False, "id-b", "event")]
    assert queued[1][0] is original_b
    # The originals keep their media for the packing thread.
    assert original_a.output is not None and original_a.output["payload"]["video"].shape[0] == 34
    # The reply carries placeholders: same bookkeeping, no media, the id to await.
    batch = envelope["result"]
    for rid, async_id, chunk_index, finished in (("a", "id-a", 2, False), ("b", "id-b", 7, True)):
        placeholder = batch.get_request_output(rid).result
        assert placeholder is not None and placeholder.output is None and placeholder.media is None
        assert placeholder.async_output_id == async_id
        assert placeholder.chunk_index == chunk_index and placeholder.finished is finished
        assert placeholder.active_event_ids == ["e1"] and placeholder.total_chunks == 8
        assert batch.get_request_output(rid).step_index == 3
    # Steps without a chunk and failed results are left alone.
    assert batch.get_request_output("c").result is None
    assert (
        batch.get_request_output("d").result.error == "boom"
        and batch.get_request_output("d").result.async_output_id is None
    )
    # Detaching again is a no-op (placeholders already carry an id and no media).
    assert detach_step_media(envelope, new_id=lambda: "x", record_event=lambda: None, enqueue=lambda *a: None) == 0


def test_step_async_output_needs_step_execution_and_the_knob() -> None:
    assert step_async_output_enabled(SimpleNamespace(step_execution=True, model_config={STEP_ASYNC_OUTPUT_KEY: True}))
    assert not step_async_output_enabled(
        SimpleNamespace(step_execution=False, model_config={STEP_ASYNC_OUTPUT_KEY: True})
    )
    assert not step_async_output_enabled(SimpleNamespace(step_execution=True, model_config={}))
    assert not step_async_output_enabled(SimpleNamespace(step_execution=True, model_config=None))
    assert detach_step_media("not a reply", new_id=lambda: "x", record_event=lambda: None, enqueue=lambda *a: None) == 0
