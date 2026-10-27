# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise Omni routing through the active upstream rendering pipeline."""

import asyncio
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from vllm.config import ModelConfig
from vllm.config.multimodal import MultiModalConfig
from vllm.exceptions import VLLMValidationError
from vllm.inputs import mm_input, tokens_input
from vllm.multimodal.cache import (
    LruKeyReplicatedReceiverCache,
    LruKeyReplicatedSenderCache,
    MultiModalCacheMissError,
    MultiModalProcessorOnlyCache,
    ShmObjectStoreSenderCache,
)
from vllm.multimodal.inputs import (
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalSharedField,
    PlaceholderRange,
)
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing import BaseProcessingInfo, InputProcessingContext
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer
from vllm.renderers.params import TokenizeParams
from vllm.v1.engine.input_processor import InputProcessor

from vllm_omni.inputs import preprocess as preprocess_mod
from vllm_omni.inputs.preprocess import (
    OmniRenderer,
    build_omni_renderer,
    omni_renderer_cls,
    use_processor_only_mm_cache,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Renderer(BaseRenderer):
    def __init__(self):
        self.model_config = SimpleNamespace(is_encoder_decoder=False, enable_prompt_embeds=True)
        self.tokenizer = None
        self.default_cmpl_tok_params = TokenizeParams(max_total_tokens=128)
        self._tokenize_prompt = lambda prompt, params: {**prompt, "prompt_token_ids": [1, 2, 3]}
        self._tokenize_prompt_async = AsyncMock(side_effect=self._tokenize_prompt)
        self._process_multimodal = Mock(return_value=tokens_input([1, 2, 3, 99]))
        self._process_multimodal_async = AsyncMock(side_effect=self._process_multimodal)

    def render_messages(self, messages, params):
        return messages, {"prompt": "hello"}


@pytest.fixture
def renderer():
    return omni_renderer_cls(_Renderer)()


class _VideoProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_processor_kwargs(self) -> dict[str, set[str]]:
        return {"videos_kwargs": {"size", "fps", "num_frames"}, "images_kwargs": {"size"}}

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser()


def _video_processing_info(mocker, config_kwargs: dict[str, object] | None = None) -> _VideoProcessingInfo:
    config = mocker.Mock(spec=ModelConfig)
    config.get_multimodal_config.return_value = MultiModalConfig(mm_processor_kwargs=config_kwargs)
    return _VideoProcessingInfo(InputProcessingContext(config, tokenizer=None))


@pytest.mark.parametrize("kwargs", [{}, {"target_h": 512, "target_w": 768}])
@pytest.mark.parametrize("tokenized", [False, True])
def test_process_inputs_routes_no_media_processor_kwargs(renderer, kwargs, tokenized):
    # Keep real process_inputs/render_cmpl; stub unrelated config validation.
    processor = object.__new__(InputProcessor)
    processor.renderer = renderer
    processor.model_config = renderer.model_config
    processor.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_size_local=1,
            local_engines_only=False,
        )
    )
    processor._validate_params = Mock()
    processor._validate_lora = Mock()
    processor._validate_model_inputs = Mock()
    prompt: dict[str, object] = {"prompt_token_ids": [1, 2, 3]} if tokenized else {"prompt": "hello"}
    prompt.update(mm_processor_kwargs=kwargs, cache_salt="salt")
    request = processor.process_inputs("image-request", prompt, PoolingParams(), ("embed",))
    assert request.prompt_token_ids == [1, 2, 3, 99]
    assert request.cache_salt == "salt"
    renderer._process_multimodal.assert_called_once_with(
        [1, 2, 3],
        {},
        mm_processor_kwargs=kwargs,
        mm_uuids=None,
        skip_mm_cache=False,
    )


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("kind", ["text", "tokens", "media", "kwargs", "embeds"])
def test_renderer_preserves_routes_extras_and_cache_policy(renderer, async_mode, kind):
    prompt: dict[str, object] = {
        "prompt": "hello",
        "additional_information": {"speaker": 1},
        "model_intermediate_buffer": {"x": 2},
    }
    if kind == "tokens":
        prompt["prompt_token_ids"] = [1, 2, 3]
    elif kind == "media":
        prompt["multi_modal_data"] = {"image": "image"}
    elif kind == "kwargs":
        prompt["mm_processor_kwargs"] = {}
    elif kind == "embeds":
        prompt["prompt_embeds"] = torch.ones(3, 4)
    if async_mode:
        (result,) = asyncio.run(renderer.render_cmpl_async([prompt], skip_mm_cache=True))
    else:
        (result,) = renderer.render_cmpl([prompt], skip_mm_cache=True)
    assert result["additional_information"] == {"speaker": 1}
    assert result["model_intermediate_buffer"] == {"x": 2}
    assert result["prompt"] == "hello"
    if kind in ("media", "kwargs"):
        assert result["prompt_token_ids"] == [1, 2, 3, 99]
        assert renderer._process_multimodal.call_args.kwargs["skip_mm_cache"] is True
    else:
        renderer._process_multimodal.assert_not_called()
        if kind == "embeds":
            torch.testing.assert_close(result["prompt_embeds"], prompt["prompt_embeds"])
        else:
            assert result["prompt_token_ids"] == [1, 2, 3]


def test_omni_renderer_is_a_real_subclass_created_once():
    cls = omni_renderer_cls(_Renderer)
    assert issubclass(cls, OmniRenderer) and issubclass(cls, _Renderer)
    assert omni_renderer_cls(_Renderer) is cls
    assert omni_renderer_cls(cls) is cls
    assert cls.__name__ == "Omni_Renderer"
    instance = cls()
    assert isinstance(instance, _Renderer)
    assert not hasattr(instance, "_renderer")


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("size", [123, "bad", [], None, {"shortest_edge": 4096}])
def test_qwen_video_size_validation_precedes_multimodal_processing(renderer, mocker, async_mode, size):
    renderer.model_config.hf_config = SimpleNamespace(model_type="qwen3_vl")
    renderer.get_mm_processor = mocker.Mock(return_value=mocker.Mock(info=_video_processing_info(mocker)))
    prompt = {"prompt": "video", "multi_modal_data": {"video": ["frames"]}, "mm_processor_kwargs": {"size": size}}

    def render():
        if async_mode:
            return asyncio.run(renderer.render_cmpl_async([prompt]))
        return renderer.render_cmpl([prompt])

    if size is not None and not isinstance(size, dict):
        with pytest.raises(VLLMValidationError, match="size must be a dictionary"):
            render()
        renderer._process_multimodal.assert_not_called()
    else:
        render()
        renderer._process_multimodal.assert_called_once()


def test_other_video_processors_keep_their_own_size_contract(renderer):
    renderer.model_config.hf_config = SimpleNamespace(model_type="other")
    renderer.render_cmpl(
        [
            {
                "prompt": "video",
                "multi_modal_data": {"video": ["frames"]},
                "mm_processor_kwargs": {"size": 224, "fps": [1], "num_frames": []},
            }
        ]
    )
    renderer._process_multimodal.assert_called_once()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("model_type", ["qwen3_vl", "qwen3_vl_moe"])
@pytest.mark.parametrize("from_config", [False, True])
@pytest.mark.parametrize(
    "options,valid",
    [
        ({"fps": []}, False),
        ({"num_frames": []}, False),
        ({"fps": [1.0]}, False),
        ({"num_frames": [2]}, False),
        ({"fps": [1.0, "bad"]}, False),
        ({"num_frames": [2, 2.5]}, False),
        ({"fps": [1, 2]}, False),
        ({"fps": "bad"}, False),
        ({"num_frames": {}}, False),
        ({"fps": True}, False),
        ({"num_frames": [2, True]}, False),
        ({"fps": float("inf")}, False),
        ({"fps": [1.0, float("nan")]}, False),
        ({"num_frames": 0}, False),
        ({"fps": -1.0}, False),
        ({"fps": None, "num_frames": None}, True),
        ({"fps": 1}, True),
        ({"fps": 2.5}, True),
        ({"num_frames": 2}, True),
        ({"fps": [1.0, 2.5], "num_frames": [2, 4]}, True),
    ],
)
def test_qwen_video_sampling_options(renderer, mocker, async_mode, model_type, from_config, options, valid):
    renderer.model_config.hf_config = SimpleNamespace(model_type=model_type)
    renderer.get_mm_processor = lambda: mocker.Mock(
        info=_video_processing_info(mocker, options if from_config else None)
    )
    video = torch.zeros(2, 4, 4, 3, dtype=torch.uint8)
    prompt = {
        "prompt": "video",
        "multi_modal_data": {"video": [video, video]},
        "mm_processor_kwargs": {} if from_config else options,
    }

    def render():
        if async_mode:
            return asyncio.run(renderer.render_cmpl_async([prompt]))
        return renderer.render_cmpl([prompt])

    if valid:
        render()
        renderer._process_multimodal.assert_called_once()
    else:
        with pytest.raises(VLLMValidationError, match="Video processor"):
            render()
        renderer._process_multimodal.assert_not_called()


@pytest.mark.parametrize("video_form", ["array", "frames", "metadata", "cached"])
def test_qwen_video_sampling_lists_count_videos_not_frames(renderer, mocker, video_form):
    video = torch.zeros(3, 4, 4, 3, dtype=torch.uint8)
    data = {"array": video, "frames": list(video), "metadata": (video, {}), "cached": [None]}[video_form]
    renderer.model_config.hf_config = SimpleNamespace(model_type="qwen3_vl")
    renderer.get_mm_processor = lambda: mocker.Mock(info=_video_processing_info(mocker, {"fps": [1.0]}))
    renderer.render_cmpl([{"prompt": "video", "multi_modal_data": {"video": data}}])
    renderer._process_multimodal.assert_called_once()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("from_config", [False, True])
@pytest.mark.parametrize(
    "options,valid",
    [
        ({}, True),
        ({"videos_kwargs": None}, True),
        ({"videos_kwargs": []}, False),
        ({"videos_kwargs": 123}, False),
        ({"fps": 2.0, "videos_kwargs": None}, True),
        ({"fps": 2.0, "videos_kwargs": []}, False),
        ({"fps": 2.0, "videos_kwargs": 123}, False),
        ({"size": {"shortest_edge": 4096}, "videos_kwargs": []}, False),
        ({"size": {"shortest_edge": 4096}, "videos_kwargs": 123}, False),
        ({"size": 123, "videos_kwargs": {"size": {"shortest_edge": 4096}}}, True),
        ({"size": {"shortest_edge": 4096}, "images_kwargs": {"size": 123}}, True),
        ({"fps": "bad", "videos_kwargs": {"fps": 2.0}}, True),
        ({"videos_kwargs": {"size": 123}}, False),
        ({"fps": 2.0, "videos_kwargs": {"fps": []}}, False),
        ({"videos_kwargs": {"num_frames": []}}, False),
    ],
)
def test_qwen_video_validation_uses_resolved_video_scope(renderer, mocker, async_mode, from_config, options, valid):
    renderer.model_config.hf_config = mocker.Mock(model_type="qwen3_vl")
    renderer.get_mm_processor = lambda: mocker.Mock(
        info=_video_processing_info(mocker, options if from_config else None)
    )
    prompt = {
        "prompt": "video",
        "multi_modal_data": {"video": ["frames"]},
        "mm_processor_kwargs": {} if from_config else options,
    }

    def render():
        if async_mode:
            return asyncio.run(renderer.render_cmpl_async([prompt]))
        return renderer.render_cmpl([prompt])

    if valid:
        render()
        renderer._process_multimodal.assert_called_once()
    else:
        with pytest.raises(VLLMValidationError, match="Video processor"):
            render()
        renderer._process_multimodal.assert_not_called()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "config_scope,request_scope,valid",
    [
        ([], {"fps": 3.0}, True),
        ({"fps": 3.0}, 123, False),
        ([], None, False),
        ({"fps": 3.0}, None, True),
    ],
)
def test_qwen_video_scope_validation_keeps_merge_precedence(
    renderer, mocker, async_mode, config_scope, request_scope, valid
):
    renderer.model_config.hf_config = SimpleNamespace(model_type="qwen3_vl")
    info = _video_processing_info(mocker, {"fps": 2.0, "videos_kwargs": config_scope})
    renderer.get_mm_processor = lambda: mocker.Mock(info=info)
    prompt = {
        "prompt": "video",
        "multi_modal_data": {"video": ["frames"]},
        "mm_processor_kwargs": {"videos_kwargs": request_scope},
    }

    def render():
        if async_mode:
            return asyncio.run(renderer.render_cmpl_async([prompt]))
        return renderer.render_cmpl([prompt])

    if valid:
        render()
        renderer._process_multimodal.assert_called_once()
    else:
        with pytest.raises(VLLMValidationError, match="videos_kwargs must be a mapping"):
            render()
        renderer._process_multimodal.assert_not_called()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("no_media_kwargs", [False, True])
@pytest.mark.parametrize("limit", [2, 4])
def test_renderer_applies_expanded_prompt_bound_without_splitting_media(renderer, async_mode, no_media_kwargs, limit):
    renderer._process_multimodal.return_value = mm_input(
        list(range(6)), {}, {"image": ["hash"]}, {"image": [PlaceholderRange(offset=1, length=2)]}
    )
    prompt = {"prompt_token_ids": [1, 2, 3], "additional_information": {"speaker": 1}}
    if no_media_kwargs:
        prompt["mm_processor_kwargs"] = {"target_h": 512}
    else:
        prompt["multi_modal_data"] = {"image": "image"}
    params = TokenizeParams(max_total_tokens=128, truncate_prompt_tokens=limit, truncation_side="right")

    def render():
        if async_mode:
            return asyncio.run(renderer.render_cmpl_async([prompt], params))
        return renderer.render_cmpl([prompt], params)

    if limit == 2:
        with pytest.raises(VLLMValidationError, match="would split or drop"):
            render()
    else:
        (result,) = render()
        assert result["prompt_token_ids"] == [0, 1, 2, 3]
        assert result["mm_placeholders"]["image"] == [PlaceholderRange(offset=1, length=2)]
        assert result["additional_information"] == {"speaker": 1}


def test_concrete_renderer_overrides_are_not_suppressed():
    """The old proxy bound BaseRenderer.render_cmpl itself and hid subclass overrides."""

    class _Custom(_Renderer):
        def render_cmpl(self, prompts, *args, **kwargs):
            return ["custom"]

    assert omni_renderer_cls(_Custom)().render_cmpl([{"prompt": "x"}]) == ["custom"]


class _CtorRenderer(BaseRenderer):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def render_messages(self, messages, params):
        raise NotImplementedError


def test_build_omni_renderer_with_explicit_class_skips_registry(monkeypatch):
    monkeypatch.setattr(preprocess_mod, "cached_tokenizer_from_config", Mock(side_effect=AssertionError("resolved")))
    config = SimpleNamespace(model_config=SimpleNamespace())
    built = build_omni_renderer(config, renderer_cls=_CtorRenderer, tokenizer=None)
    assert isinstance(built, OmniRenderer) and isinstance(built, _CtorRenderer)
    assert built.config is config and built.tokenizer is None


def test_build_omni_renderer_resolves_like_upstream(monkeypatch):
    config = SimpleNamespace(model_config=SimpleNamespace())
    tokenizer = object()
    monkeypatch.setattr(
        preprocess_mod, "cached_tokenizer_from_config", lambda mc: tokenizer if mc is config.model_config else None
    )
    monkeypatch.setattr(preprocess_mod, "tokenizer_args_from_config", lambda mc: ("fake-mode", None))
    load_cls = Mock(return_value=_CtorRenderer)
    monkeypatch.setattr(preprocess_mod.RENDERER_REGISTRY, "load_renderer_cls", load_cls)
    built = build_omni_renderer(config)
    load_cls.assert_called_once_with("fake-mode")
    assert isinstance(built, OmniRenderer) and isinstance(built, _CtorRenderer)
    assert built.tokenizer is tokenizer


# ---- multimodal cache mode for replicated stages ----


def _mm_model_config(cache_gb: float = 1.0):
    mm_config = SimpleNamespace(mm_processor_cache_gb=cache_gb)
    return SimpleNamespace(get_multimodal_config=lambda: mm_config)


class _CacheRenderer(BaseRenderer):
    """A renderer carrying only what ``use_processor_only_mm_cache`` touches."""

    def __init__(self, cache):
        self.model_config = _mm_model_config()
        self._mm_processor_cache = cache
        self._resources = ExitStack()

    def render_messages(self, messages, params):
        raise NotImplementedError


def _mm_item(value: int) -> MultiModalKwargsItem:
    return MultiModalKwargsItem(
        {"pixel_values": MultiModalFieldElem(data=torch.tensor([value]), field=MultiModalSharedField(batch_size=1))}
    )


def test_use_processor_only_mm_cache_replaces_lru_sender_cache():
    lru_sender = LruKeyReplicatedSenderCache(_mm_model_config())
    renderer = _CacheRenderer(lru_sender)
    closed = []
    lru_sender.close = lambda: closed.append("lru")

    assert use_processor_only_mm_cache(renderer) is renderer

    assert isinstance(renderer.mm_processor_cache, MultiModalProcessorOnlyCache)
    assert closed == ["lru"]
    renderer._resources.close()  # the replacement cache is owned by the renderer's resources


@pytest.mark.parametrize("cache_factory", [lambda: None, lambda: MultiModalProcessorOnlyCache(_mm_model_config())])
def test_use_processor_only_mm_cache_leaves_non_ipc_caches_alone(cache_factory):
    cache = cache_factory()
    renderer = _CacheRenderer(cache)
    use_processor_only_mm_cache(renderer)
    assert renderer.mm_processor_cache is cache


def test_use_processor_only_mm_cache_leaves_shm_sender_cache_alone():
    # The shm ring is shared by every reader on the node and the workers attach to
    # the ring the sender created, so it must not be swapped out behind them.
    shm_sender = object.__new__(ShmObjectStoreSenderCache)
    renderer = _CacheRenderer(shm_sender)
    use_processor_only_mm_cache(renderer)
    assert renderer.mm_processor_cache is shm_sender


def test_build_omni_renderer_switches_cache_mode_only_when_asked(monkeypatch):
    swapped = []

    def _record(renderer):
        swapped.append(renderer)
        return renderer

    monkeypatch.setattr(preprocess_mod, "use_processor_only_mm_cache", _record)
    config = SimpleNamespace(model_config=SimpleNamespace())

    kept = build_omni_renderer(config, renderer_cls=_CtorRenderer, tokenizer=None)
    assert swapped == []
    switched = build_omni_renderer(config, renderer_cls=_CtorRenderer, tokenizer=None, mm_ipc_cache=False)
    assert swapped == [switched] and kept is not switched


def test_processor_only_sender_never_omits_payload_across_replica_receivers():
    """The contract behind #8039 / #7295 / #7255.

    One frontend sender cache feeds several EngineCore receiver caches, each of which
    only knows what it was sent. The key-replicated LRU sender omits a payload it has
    sent once, so the second replica to see that media gets nothing. The processor-only
    sender always returns the full item, so every replica stores exactly what it is
    sent and no payload can be filled in from another request.
    """
    receivers = [LruKeyReplicatedReceiverCache(_mm_model_config()) for _ in range(2)]
    # Same media hashed once ("h-a"), routed to replica 0 then replica 1; a different
    # video ("h-b") routed to replica 1 in between.
    route = [("h-a", 0, 1), ("h-b", 1, 2), ("h-a", 1, 1)]

    def run(sender):
        delivered = []
        for mm_hash, replica, value in route:
            sent_item, _updates = sender.get_and_update_item((_mm_item(value), []), mm_hash)
            received = receivers[replica].get_and_update_item(sent_item, mm_hash)
            delivered.append(int(received["pixel_values"].data.item()))
        return delivered

    processor_only = MultiModalProcessorOnlyCache(_mm_model_config())
    assert run(processor_only) == [1, 2, 1]

    receivers = [LruKeyReplicatedReceiverCache(_mm_model_config()) for _ in range(2)]
    lru_sender = LruKeyReplicatedSenderCache(_mm_model_config())
    with pytest.raises(MultiModalCacheMissError):
        run(lru_sender)
