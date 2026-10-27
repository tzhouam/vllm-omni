# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project


import pytest
import torch
from pytest_mock import MockerFixture
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

from vllm_omni.distributed.omni_coordinator import ReplicaInfo, ReplicaStatus
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine, StageRuntimeInfo
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.model_executor.stage_input_processors.bagel import ExpandedPrompt

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _SyntheticSyncQueueShutDownError(Exception):
    pass


def test_abort_is_noop_for_empty_request_ids(mocker: MockerFixture):
    engine = object.__new__(AsyncOmniEngine)
    engine._shutdown_called = False
    engine.request_queue = mocker.Mock()

    engine.abort([])

    engine.request_queue.sync_q.put.assert_not_called()


def test_abort_is_noop_after_shutdown_starts(mocker: MockerFixture):
    engine = object.__new__(AsyncOmniEngine)
    engine._shutdown_called = True
    engine.request_queue = mocker.Mock()

    engine.abort(["req-1"])

    engine.request_queue.sync_q.put.assert_not_called()


def test_abort_tolerates_queue_close_race_during_shutdown(mocker: MockerFixture):
    engine = object.__new__(AsyncOmniEngine)
    engine._shutdown_called = False
    engine.request_queue = mocker.Mock()
    mocker.patch(
        "vllm_omni.engine.async_engine_utils._JANUS_SYNC_QUEUE_SHUTDOWN",
        _SyntheticSyncQueueShutDownError,
    )

    def close_queue_during_abort(*args, **kwargs):
        del args, kwargs
        engine._shutdown_called = True
        raise _SyntheticSyncQueueShutDownError

    engine.request_queue.sync_q.put.side_effect = close_queue_during_abort

    engine.abort(["req-1"])


def test_abort_tolerates_legacy_janus_close_error_during_shutdown(mocker: MockerFixture):
    engine = object.__new__(AsyncOmniEngine)
    engine._shutdown_called = False
    engine.request_queue = mocker.Mock()
    mocker.patch(
        "vllm_omni.engine.async_engine_utils._JANUS_SYNC_QUEUE_SHUTDOWN",
        None,
    )

    def close_queue_during_abort(*args, **kwargs):
        del args, kwargs
        engine._shutdown_called = True
        raise RuntimeError("Operation on the closed queue is forbidden")

    engine.request_queue.sync_q.put.side_effect = close_queue_during_abort

    engine.abort(["req-1"])


def test_abort_surfaces_unrelated_legacy_runtime_error_during_shutdown(
    mocker: MockerFixture,
):
    engine = object.__new__(AsyncOmniEngine)
    engine._shutdown_called = False
    engine.request_queue = mocker.Mock()
    mocker.patch(
        "vllm_omni.engine.async_engine_utils._JANUS_SYNC_QUEUE_SHUTDOWN",
        None,
    )

    def fail_during_abort(*args, **kwargs):
        del args, kwargs
        engine._shutdown_called = True
        raise RuntimeError("unrelated queue failure")

    engine.request_queue.sync_q.put.side_effect = fail_during_abort

    with pytest.raises(RuntimeError, match="unrelated queue failure"):
        engine.abort(["req-1"])


def test_abort_surfaces_unexpected_closed_queue(mocker: MockerFixture):
    engine = object.__new__(AsyncOmniEngine)
    engine._shutdown_called = False
    engine.request_queue = mocker.Mock()
    mocker.patch(
        "vllm_omni.engine.async_engine_utils._JANUS_SYNC_QUEUE_SHUTDOWN",
        _SyntheticSyncQueueShutDownError,
    )
    engine.request_queue.sync_q.put.side_effect = _SyntheticSyncQueueShutDownError

    with pytest.raises(_SyntheticSyncQueueShutDownError):
        engine.abort(["req-1"])


def _make_engine_core_request(request_id: str = "req-1") -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[1, 1, 1],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def test_build_add_request_message_preserves_additional_information(mocker: MockerFixture):
    engine = object.__new__(AsyncOmniEngine)
    params = SamplingParams(max_tokens=8)
    engine.default_sampling_params_list = [params]
    engine.stage_metadata = [StageRuntimeInfo(final_output=False, final_output_type=None, stage_type="llm")]
    engine.supported_tasks = ("speech",)

    input_processor = mocker.Mock()
    input_processor.process_inputs.return_value = _make_engine_core_request()
    engine.input_processor = input_processor

    output_processor = mocker.Mock()
    engine.output_processors = [output_processor]

    prompt = {
        "prompt_token_ids": [1, 1, 1],
        "additional_information": {
            "text": ["hello world"],
            "speaker": ["vivian"],
        },
    }

    msg = engine._build_add_request_message(
        request_id="req-1",
        prompt=prompt,
        sampling_params_list=[params],
        final_stage_id=0,
        arrival_time=0.0,
    )

    request = msg.prompt
    assert isinstance(request, OmniEngineCoreRequest)
    assert request.external_req_id == "req-1"
    assert request.additional_information is not None
    assert request.additional_information.entries["text"].list_data == ["hello world"]
    assert request.additional_information.entries["speaker"].list_data == ["vivian"]
    output_processor.add_request.assert_not_called()


def test_build_add_request_message_preserves_kv_hints_after_omni_payload_upgrade(mocker):
    from vllm.v1.engine.input_processor import InputProcessor
    from vllm.v1.kv_hints import KvHintsEnvelope

    engine = object.__new__(AsyncOmniEngine)
    params = SamplingParams(max_tokens=8)
    engine.default_sampling_params_list = [params]
    engine.stage_metadata = [StageRuntimeInfo(final_output=False, final_output_type=None, stage_type="llm")]
    engine.supported_tasks = ("generate",)
    processor = mocker.create_autospec(InputProcessor, instance=True)
    hints = KvHintsEnvelope("1", "hint-message", [])

    def process(**kwargs):
        request = _make_engine_core_request()
        request.kv_hints = kwargs["kv_hints"]
        return request

    processor.process_inputs.side_effect = process
    engine.input_processor = processor
    msg = engine._build_add_request_message(
        request_id="req-1",
        prompt={"prompt_token_ids": [1], "additional_information": {"speaker": "voice"}},
        sampling_params_list=[params],
        kv_hints=hints,
    )
    assert processor.process_inputs.call_args.kwargs["kv_hints"] is hints
    assert isinstance(msg.prompt, OmniEngineCoreRequest)
    assert msg.prompt.kv_hints is hints
    assert msg.prompt.additional_information is not None


def test_build_add_request_message_injects_global_id_before_prompt_transform(mocker: MockerFixture):
    engine = object.__new__(AsyncOmniEngine)
    params = SamplingParams(max_tokens=8)
    engine.default_sampling_params_list = [params]
    engine.stage_metadata = [StageRuntimeInfo(final_output=False, final_output_type=None, stage_type="llm")]
    engine.supported_tasks = ("speech",)

    input_processor = mocker.Mock()
    input_processor.process_inputs.return_value = _make_engine_core_request()
    engine.input_processor = input_processor
    engine.output_processors = [mocker.Mock()]

    seen_global_ids: list[list[str]] = []

    def transform(prompt, _sampling_params):
        seen_global_ids.append(prompt["additional_information"]["global_request_id"])
        return prompt

    engine.prompt_transform_func = transform
    prompt = {"prompt_token_ids": [1, 1, 1]}

    message = engine._build_add_request_message(
        request_id="req-1",
        prompt=prompt,
        sampling_params_list=[params],
        final_stage_id=0,
        arrival_time=0.0,
    )

    assert seen_global_ids == [["req-1"]]
    assert message.original_prompt["additional_information"]["global_request_id"] == ["req-1"]


def test_build_add_request_message_preserves_model_intermediate_buffer(mocker: MockerFixture):
    engine = object.__new__(AsyncOmniEngine)
    params = SamplingParams(max_tokens=8)
    engine.default_sampling_params_list = [params]
    engine.stage_metadata = [StageRuntimeInfo(final_output=False, final_output_type=None, stage_type="llm")]
    engine.supported_tasks = ("speech",)

    input_processor = mocker.Mock()
    input_processor.process_inputs.return_value = _make_engine_core_request()
    engine.input_processor = input_processor

    output_processor = mocker.Mock()
    engine.output_processors = [output_processor]

    hidden = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    prompt = {
        "prompt_token_ids": [1, 1, 1],
        "model_intermediate_buffer": {
            "ids": {"tts": [11, 12]},
            "hidden_states": {"tts": hidden},
        },
    }

    msg = engine._build_add_request_message(
        request_id="req-1",
        prompt=prompt,
        sampling_params_list=[params],
        final_stage_id=0,
        arrival_time=0.0,
    )

    request = msg.prompt
    assert isinstance(request, OmniEngineCoreRequest)
    assert request.additional_information is not None
    assert request.additional_information.entries["global_request_id"].list_data == ["req-1"]
    assert request.additional_information.entries["omni_final_stage_id"].scalar_data == 0
    assert isinstance(request.model_intermediate_buffer, dict)
    info = request.model_intermediate_buffer
    assert info["ids"]["tts"] == [11, 12]
    assert torch.equal(info["hidden_states"]["tts"], hidden)


def test_cfg_companion_suppresses_payload_but_forces_kv_transfer(mocker: MockerFixture):
    engine = object.__new__(AsyncOmniEngine)
    params = SamplingParams(max_tokens=8)
    engine.prompt_expand_func = lambda *_args: [
        ExpandedPrompt(
            prompt={"prompt": "negative"},
            role="cfg_text",
            request_id_suffix="__cfg_text",
        )
    ]
    engine.supported_tasks = ("generate",)
    engine.input_processor = mocker.Mock()
    engine.input_processor.process_inputs.return_value = _make_engine_core_request("req__cfg_text")
    engine.request_queue = mocker.Mock()

    engine._enqueue_cfg_companions(
        parent_id="req",
        original_prompt={"prompt": "positive"},
        stage0_params=params,
        sampling_params_list=[params],
    )

    message = engine.request_queue.sync_q.put.call_args.args[0]
    metadata = deserialize_additional_information(message.prompt.additional_information)
    assert metadata["omni_final_stage_id"] == 0
    assert metadata["omni_force_kv_transfer"] is True


def test_build_add_request_message_with_resumable_streaming(mocker: MockerFixture):
    engine = object.__new__(AsyncOmniEngine)
    params = SamplingParams(max_tokens=8)
    engine.default_sampling_params_list = [params]
    engine.stage_metadata = [StageRuntimeInfo(final_output=False, final_output_type=None, stage_type="llm")]
    engine.supported_tasks = ("generate",)

    input_processor = mocker.Mock()
    input_processor.process_inputs.return_value = _make_engine_core_request()
    engine.input_processor = input_processor

    output_processor = mocker.Mock()
    engine.output_processors = [output_processor]

    msg = engine._build_add_request_message(
        request_id="req-stream",
        prompt={"prompt_token_ids": [1, 2, 3]},
        sampling_params_list=[params],
        final_stage_id=0,
        resumable=True,
        message_type="streaming_update",
    )

    assert msg.type == "streaming_update"
    input_processor.process_inputs.assert_called_once()
    assert input_processor.process_inputs.call_args.kwargs["resumable"] is True


class _FakeStageClient:
    stage_type = "llm"
    final_output = False

    def __init__(self, input_address: str | None = None):
        if input_address is not None:
            self.client_addresses = {"input_address": input_address}


class _FakeHub:
    def __init__(self, replicas: list[ReplicaInfo]):
        self._replicas = replicas

    def get_replicas_for_stage(self, stage_id: int):
        return type(
            "ReplicaList",
            (),
            {"replicas": [rep for rep in self._replicas if rep.stage_id == stage_id]},
        )()


class _RoundRobinLB:
    def __init__(self):
        self._next = 0

    def select(self, task, replicas):  # noqa: ARG002
        idx = self._next % len(replicas)
        self._next += 1
        return idx


def _replica(input_addr: str) -> ReplicaInfo:
    return ReplicaInfo(
        input_addr=input_addr,
        output_addr=input_addr.replace("input", "output"),
        stage_id=0,
        status=ReplicaStatus.UP,
        queue_length=0,
        last_heartbeat=0.0,
        registered_at=0.0,
    )


def _multi_replica_engine_recording_prompts(
    mocker: MockerFixture, stage_pool: StagePool
) -> tuple[AsyncOmniEngine, list[dict]]:
    engine = object.__new__(AsyncOmniEngine)
    params = SamplingParams(max_tokens=8)
    engine.model = "test-model"
    engine.default_sampling_params_list = [params]
    engine.stage_metadata = [StageRuntimeInfo(final_output=False, final_output_type=None, stage_type="llm")]
    engine.supported_tasks = ("generate",)
    engine.stage_pools = [stage_pool]

    seen_prompts: list[dict] = []

    def process_inputs(**kwargs):
        seen_prompts.append(kwargs["prompt"])
        return _make_engine_core_request(kwargs["request_id"])

    engine.input_processor = mocker.Mock()
    engine.input_processor.process_inputs.side_effect = process_inputs
    return engine, seen_prompts


def test_build_add_request_message_processes_media_as_sent_on_replicated_stage0(mocker: MockerFixture):
    """With several stage-0 replicas the frontend no longer rewrites the prompt's cache keys
    (the stage processes media with a processor-only cache instead), and routing stays a
    submit-time decision: no replica is bound while preprocessing."""
    stage_pool = StagePool(0, [_FakeStageClient(), _FakeStageClient()])
    engine, seen_prompts = _multi_replica_engine_recording_prompts(mocker, stage_pool)
    params = SamplingParams(max_tokens=8)
    prompts = [
        {"prompt": "describe", "multi_modal_data": {"video": f"video-{i}"}, "multi_modal_uuids": {"video": [None]}}
        for i in range(3)
    ]

    for i, prompt in enumerate(prompts):
        engine._build_add_request_message(
            request_id=f"req-{i}", prompt=prompt, sampling_params_list=[params], final_stage_id=0
        )

    assert [p["multi_modal_data"] for p in seen_prompts] == [{"video": f"video-{i}"} for i in range(3)]
    assert all(p["multi_modal_uuids"] == {"video": [None]} for p in seen_prompts)
    assert all(prompt["multi_modal_uuids"] == {"video": [None]} for prompt in prompts)
    assert all(stage_pool.get_bound_replica_id(f"req-{i}") is None for i in range(3))


def test_build_add_request_message_keys_reused_prompt_dict_by_its_own_media(mocker: MockerFixture):
    """A caller that builds every request from one template dict must have each request
    processed with its own media; nothing from an earlier request may leak into the template."""
    stage_pool = StagePool(0, [_FakeStageClient(), _FakeStageClient()])
    engine, seen_prompts = _multi_replica_engine_recording_prompts(mocker, stage_pool)
    params = SamplingParams(max_tokens=8)
    template = {"prompt": "describe", "multi_modal_data": {"image": "image-a"}}

    engine._build_add_request_message(
        request_id="req-0", prompt=template, sampling_params_list=[params], final_stage_id=0
    )
    for i, image in enumerate(("image-b", "image-c", "image-d"), start=1):
        prompt = dict(template)
        prompt["multi_modal_data"] = {"image": image}
        engine._build_add_request_message(
            request_id=f"req-{i}", prompt=prompt, sampling_params_list=[params], final_stage_id=0
        )

    assert "multi_modal_uuids" not in template
    assert [p["multi_modal_data"]["image"] for p in seen_prompts] == ["image-a", "image-b", "image-c", "image-d"]
    assert all("multi_modal_uuids" not in p for p in seen_prompts)


@pytest.mark.parametrize("hub_replicas", ["all", "none"])
def test_build_add_request_message_on_distributed_stage0_needs_no_serviceable_replica(
    mocker: MockerFixture, hub_replicas: str
):
    """Preprocessing on a distributed stage 0 does not depend on the hub snapshot: media is
    processed as sent whether or not any replica is serviceable yet."""
    addr0 = "tcp://host-a:1000/input"
    addr1 = "tcp://host-b:1000/input"
    stage_pool = StagePool(0, [_FakeStageClient(addr0), _FakeStageClient(addr1)])
    stage_pool.attach_hub(_FakeHub([_replica(addr0), _replica(addr1)] if hub_replicas == "all" else []))
    stage_pool.attach_load_balancer(_RoundRobinLB())
    engine, seen_prompts = _multi_replica_engine_recording_prompts(mocker, stage_pool)
    params = SamplingParams(max_tokens=8)

    engine._build_add_request_message(
        request_id="req-1",
        prompt={"prompt": "describe", "multi_modal_data": {"image": "same-image"}},
        sampling_params_list=[params],
        final_stage_id=0,
    )

    assert seen_prompts == [
        {
            "prompt": "describe",
            "multi_modal_data": {"image": "same-image"},
            "additional_information": {"global_request_id": ["req-1"]},
        }
    ]
    assert stage_pool.get_bound_replica_id("req-1") is None


def test_build_add_request_message_preprocess_error_leaves_no_binding(mocker: MockerFixture):
    stage_pool = StagePool(0, [_FakeStageClient(), _FakeStageClient()])
    engine, _ = _multi_replica_engine_recording_prompts(mocker, stage_pool)
    params = SamplingParams(max_tokens=8)
    engine.input_processor.process_inputs.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        engine._build_add_request_message(
            request_id="req-error",
            prompt={"prompt": "describe", "multi_modal_data": {"image": "same-image"}},
            sampling_params_list=[params],
            final_stage_id=0,
        )

    assert stage_pool.get_bound_replica_id("req-error") is None


def test_cfg_companion_build_failure_admits_nothing(mocker: MockerFixture):
    """A guided request is all-or-nothing.

    Admitting the parent before its companion is built leaves an orphan: a
    model whose guidance is mandatory never completes the pair, so the request
    holds scheduler and KV capacity for the scheduler's whole hold budget and
    then produces no audio. The build has to raise before anything is enqueued.
    """
    engine = object.__new__(AsyncOmniEngine)
    params = SamplingParams(max_tokens=8)
    engine.prompt_expand_func = mocker.Mock(side_effect=ValueError("cannot build the null twin"))
    engine.supported_tasks = ("generate",)
    engine.input_processor = mocker.Mock()
    engine.request_queue = mocker.Mock()

    with pytest.raises(ValueError, match="null twin"):
        engine._build_cfg_companions(
            parent_id="req",
            original_prompt={"prompt": "positive"},
            stage0_params=params,
            sampling_params_list=[params],
        )

    engine.request_queue.sync_q.put.assert_not_called()


def test_cfg_companion_processing_failure_admits_nothing(mocker: MockerFixture):
    """The same holds when expansion succeeds but input processing does not."""
    engine = object.__new__(AsyncOmniEngine)
    params = SamplingParams(max_tokens=8)
    engine.prompt_expand_func = lambda *_args: [
        ExpandedPrompt(
            prompt={"prompt": "negative"},
            role="cfg_text",
            request_id_suffix="__cfg_text",
        )
    ]
    engine.supported_tasks = ("generate",)
    engine.input_processor = mocker.Mock()
    engine.input_processor.process_inputs.side_effect = RuntimeError("tokenizer rejected the prompt")
    engine.request_queue = mocker.Mock()

    with pytest.raises(RuntimeError, match="tokenizer rejected"):
        engine._build_cfg_companions(
            parent_id="req",
            original_prompt={"prompt": "positive"},
            stage0_params=params,
            sampling_params_list=[params],
        )

    engine.request_queue.sync_q.put.assert_not_called()


def test_cfg_companion_build_returns_messages_without_enqueueing(mocker: MockerFixture):
    """Construction and admission are separate steps."""
    engine = object.__new__(AsyncOmniEngine)
    params = SamplingParams(max_tokens=8)
    engine.prompt_expand_func = lambda *_args: [
        ExpandedPrompt(
            prompt={"prompt": "negative"},
            role="cfg_text",
            request_id_suffix="__cfg_text",
        )
    ]
    engine.supported_tasks = ("generate",)
    engine.input_processor = mocker.Mock()
    engine.input_processor.process_inputs.return_value = _make_engine_core_request("req__cfg_text")
    engine.request_queue = mocker.Mock()

    companions = engine._build_cfg_companions(
        parent_id="req",
        original_prompt={"prompt": "positive"},
        stage0_params=params,
        sampling_params_list=[params],
    )

    assert len(companions) == 1
    assert companions[0].companion_id == "req__cfg_text"
    assert companions[0].parent_id == "req"
    engine.request_queue.sync_q.put.assert_not_called()


def test_cfg_expansion_returning_nothing_is_not_an_error(mocker: MockerFixture):
    """An unguided request expands to no companions and admits normally."""
    engine = object.__new__(AsyncOmniEngine)
    params = SamplingParams(max_tokens=8)
    engine.prompt_expand_func = lambda *_args: []
    engine.supported_tasks = ("generate",)
    engine.input_processor = mocker.Mock()
    engine.request_queue = mocker.Mock()

    assert (
        engine._build_cfg_companions(
            parent_id="req",
            original_prompt={"prompt": "positive"},
            stage0_params=params,
            sampling_params_list=[params],
        )
        == []
    )
    engine.request_queue.sync_q.put.assert_not_called()
