from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
pytest.importorskip("vllm")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.api_server import build_app, omni_init_app_state
from vllm_omni.outputs import OmniRequestOutput

_MODEL_NAME = "Qwen/Qwen2.5-Omni-3B"
_CHAT_TEMPLATE = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}assistant:"


class _FakeTokenizer:
    name_or_path = "fake-tokenizer"

    def __call__(self, text, **_):
        """Tokenize text, returning object like transformers tokenizers."""
        tokens = self.encode(text)
        return SimpleNamespace(input_ids=tokens, attention_mask=[1] * len(tokens))

    def apply_chat_template(self, conversation, tools=None, chat_template=None, tokenize=False, **_):
        lines = []
        for msg in conversation:
            content = msg.get("content") or ""
            lines.append(f"{msg.get('role')}: {content}")
        return "\n".join(lines) + "\nassistant:"

    def get_chat_template(self, chat_template=None, tools=None):
        return chat_template or _CHAT_TEMPLATE

    def encode(self, text: str, **_) -> list[int]:
        return [len(text) % 7 or 1, (len(text) + 1) % 11 or 2]

    def decode(self, token_ids: list[int], **_) -> str:
        return f"decoded({len(token_ids)})"


class _FakeModelConfig:
    """Fake ModelConfig that returns sensible defaults for unknown attributes."""

    def __init__(self):
        self.model = _MODEL_NAME
        self.max_model_len = 4096
        self.allowed_local_media_path = "."
        self.allowed_media_domains = ["*"]
        self.multimodal_config = SimpleNamespace(
            interleave_mm_strings=False,
            media_io_kwargs={},
            get_limit_per_prompt=lambda modality: 0,
        )
        self.is_multimodal_model = False
        self.tokenizer = "fake-tokenizer"
        self.hf_config = SimpleNamespace(model_type="qwen2_5_omni")
        self.lora_config = SimpleNamespace(default_mm_loras={})
        self.trust_remote_code = False
        self.encoder_config = None
        self.is_tracing_enabled = False
        self.revision = None
        self.dtype = "float16"
        self.quantization = None
        self.served_model_name = _MODEL_NAME

    def __getattr__(self, name):
        # Return sensible defaults for any missing attributes
        return None

    def get_multimodal_config(self):
        return self.multimodal_config

    def get_diff_sampling_param(self):
        return {}


class _FakeVllmConfig:
    """Fake VllmConfig with sensible defaults for unknown attributes."""

    def __init__(self):
        self.model_config = _FakeModelConfig()
        self.lora_config = SimpleNamespace(default_mm_loras={})
        self.parallel_config = SimpleNamespace(_api_process_rank=0)
        self.decoding_config = SimpleNamespace(guided_decoding_backend="outlines")
        self.observability_config = SimpleNamespace(
            is_tracing_enabled=False,
            otlp_traces_endpoint=None,
        )

    def __getattr__(self, name):
        return None


def _make_vllm_config():
    return _FakeVllmConfig()


def _fake_stage_configs():
    return [
        SimpleNamespace(
            stage_id=0,
            engine_args=SimpleNamespace(model_stage="thinker", engine_output_type="text"),
            final_output=True,
            final_output_type="text",
            default_sampling_params={"temperature": 0.0},
            is_comprehension=True,
        )
    ]


def _mock_initialize_stages(self, *_, **__):
    tokenizer = _FakeTokenizer()
    vllm_config = _make_vllm_config()
    stages = []
    sampling = []
    for cfg in self.stage_configs:
        params = SamplingParams(**cfg.default_sampling_params)
        stage = SimpleNamespace(
            stage_id=cfg.stage_id,
            final_output=getattr(cfg, "final_output", False),
            final_output_type=getattr(cfg, "final_output_type", "text"),
            default_sampling_params=params,
            is_comprehension=getattr(cfg, "is_comprehension", False),
            tokenizer=tokenizer,
            vllm_config=vllm_config,
            is_tracing_enabled=False,
        )
        stages.append(stage)
        sampling.append(params)
    self.stage_list = stages
    self.default_sampling_params_list = sampling
    self._stage_in_queues = [object()]
    self._stage_out_queues = [object()]
    self._stages_ready = {stage.stage_id for stage in stages}


async def _mock_generate(self, prompt, request_id, sampling_params_list=None, **_):
    if sampling_params_list is None:
        sampling_params_list = self.default_sampling_params_list
    if len(sampling_params_list) != len(self.stage_list):
        raise ValueError("sampling_params_list length mismatch with stages")
    text = "omni server smoke response"
    completion = CompletionOutput(
        index=0,
        text=text,
        token_ids=[1, 2, 3],
        cumulative_logprob=0.0,
        logprobs=None,
        finish_reason="stop",
    )
    request_output = RequestOutput(
        request_id=request_id,
        prompt="synthetic-prompt",
        prompt_token_ids=[0, 1],
        prompt_logprobs=None,
        outputs=[completion],
        finished=True,
    )
    yield OmniRequestOutput(stage_id=0, final_output_type="text", request_output=request_output)


def _install_async_omni_stubs(monkeypatch: pytest.MonkeyPatch):
    stage_cfgs = _fake_stage_configs()
    monkeypatch.setattr(
        "vllm_omni.entrypoints.async_omni.load_stage_configs_from_model",
        lambda model: stage_cfgs,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.async_omni.AsyncOmni._initialize_stages",
        _mock_initialize_stages,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.async_omni.AsyncOmni.generate", _mock_generate, raising=False
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.async_omni.AsyncOmni.close", lambda self: None, raising=False
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.async_omni.AsyncOmni.shutdown", lambda self: None, raising=False
    )


def _build_server_args():
    return SimpleNamespace(
        model=_MODEL_NAME,
        served_model_name=None,
        enable_log_requests=False,
        max_log_len=None,
        disable_log_stats=True,
        lora_modules=None,
        tool_server=None,
        response_role="assistant",
        chat_template=_CHAT_TEMPLATE,
        chat_template_content_format="string",
        trust_request_chat_template=False,
        return_tokens_as_token_ids=False,
        enable_auto_tool_choice=False,
        exclude_tools_when_tool_choice_none=False,
        tool_call_parser=None,
        structured_outputs_config=SimpleNamespace(reasoning_parser=None),
        enable_prompt_tokens_details=False,
        enable_force_include_usage=False,
        enable_log_outputs=False,
        log_error_stack=False,
        disable_fastapi_docs=True,
        root_path="",
        allowed_origins=["*"],
        allow_credentials=False,
        allowed_methods=["*"],
        allowed_headers=["*"],
        api_key=None,
        enable_request_id_headers=False,
        middleware=[],
        enable_server_load_tracking=False,
        stage_configs_path=None,
        chat_template_kwargs=None,
    )


@pytest.fixture
def omni_test_app(monkeypatch: pytest.MonkeyPatch):
    _install_async_omni_stubs(monkeypatch)
    args = _build_server_args()
    engine = AsyncOmni(model=_MODEL_NAME, init_sleep_seconds=0)
    app = build_app(args)

    async def _init_state():
        vllm_config = await engine.get_vllm_config()
        await omni_init_app_state(engine, vllm_config, app.state, args)

    asyncio.run(_init_state())
    yield app
    engine.shutdown()


def test_qwen2_5_online_server_smoke(omni_test_app):
    payload = {
        "model": _MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are an omni assistant."},
            {
                "role": "user",
                "content": "Summarize the abilities of Qwen 2.5 Omni in one sentence.",
            },
        ],
        "sampling_params_list": [
            {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 32,
            }
        ],
    }

    with TestClient(omni_test_app) as client:
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200, f"Request failed: {response.text}"
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "omni server smoke response"
    assert body["usage"]["total_tokens"] > 0

