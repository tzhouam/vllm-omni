"""Codec-token streaming: wire format, row alignment, and handler behaviour (CPU)."""

import numpy as np
import pytest
import torch
from fastapi import FastAPI, WebSocket

from vllm_omni.entrypoints.openai import codec_stream as cs
from vllm_omni.entrypoints.openai.protocol.audio import StreamingSpeechSessionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_pack_unpack_roundtrip_and_flags():
    codes = np.random.randint(0, 2048, size=(25, 16)).astype(np.int64)
    buf = cs.pack_codec_frame(7, torch.from_numpy(codes), eos=False, segment_end=True)
    assert len(buf) == cs.HEADER_SIZE + 25 * 16 * 2
    seq, n, flags, arr = cs.unpack_codec_frame(buf, 16)
    assert seq == 7 and n == 25 and flags == cs.FLAG_SEGMENT_END
    assert arr.dtype == np.int16 and np.array_equal(arr.astype(np.int64), codes)
    # EOS with zero frames
    buf = cs.pack_codec_frame(8, None, eos=True, codebooks=16)
    seq, n, flags, arr = cs.unpack_codec_frame(buf, 16)
    assert (seq, n, flags) == (8, 0, cs.FLAG_EOS) and arr.shape == (0, 16)
    buf = cs.pack_codec_frame(2**32 + 5, np.zeros((1, 16), dtype=np.int64), reset=True)
    assert cs.unpack_codec_frame(buf, 16)[0] == 5
    assert cs.unpack_codec_frame(buf, 16)[2] == cs.FLAG_RESET


def test_pack_rejects_bad_shapes_and_ranges():
    with pytest.raises(ValueError):
        cs.pack_codec_frame(0, np.zeros((2, 2, 2), dtype=np.int64))
    with pytest.raises(ValueError):
        cs.pack_codec_frame(0, np.full((1, 16), 40000, dtype=np.int64))
    with pytest.raises(ValueError):
        cs.unpack_codec_frame(b"\x00\x01", 16)
    with pytest.raises(ValueError):
        cs.unpack_codec_frame(cs.pack_codec_frame(0, np.zeros((1, 16), dtype=np.int64))[:-2], 16)


def test_align_codec_rows_drops_prefill_placeholder_and_eos():
    # 1 prefill placeholder row (zeros, emitted before any token) + 3 real frames + EOS row
    codes = torch.tensor([[0] * 16, [5] * 16, [6] * 16, [7] * 16, [0] * 16])
    token_ids = [5, 6, 7, 2150]  # codebook-0 ids then EOS
    rows = cs.align_codec_rows(codes, token_ids, 2048)
    assert rows.shape == (3, 16) and rows[:, 0].tolist() == [5, 6, 7]
    # trailing -1 placeholders in token ids are ignored
    rows = cs.align_codec_rows(codes, token_ids + [-1, -1], 2048)
    assert rows.shape == (3, 16)
    # out-of-range rows are dropped
    bad = torch.tensor([[5] * 16, [3000] * 16])
    assert cs.align_codec_rows(bad, [5, 6], 2048).shape == (1, 16)
    assert cs.align_codec_rows(None, [], 2048).shape == (0, 0)
    assert cs.align_codec_rows(torch.zeros(0, 16, dtype=torch.long), [], 2048).shape == (0, 16)


def test_session_config_codec_mode_validation():
    cfg = StreamingSpeechSessionConfig(output_mode="codec_tokens", stream_audio=True, response_format="pcm")
    assert cfg.output_mode == "codec_tokens"
    with pytest.raises(ValueError):
        StreamingSpeechSessionConfig(output_mode="codec_tokens", stream_audio=False, response_format="pcm")
    with pytest.raises(ValueError):
        StreamingSpeechSessionConfig(
            output_mode="codec_tokens", stream_audio=True, response_format="pcm", word_timestamps=True
        )
    assert StreamingSpeechSessionConfig().output_mode == "pcm"


def test_codec_start_message_fields():
    spec = {"codebooks": 16, "codebook_size": 2048, "frame_rate_hz": 12.5, "sample_rate": 24000, "decoder_id": "x"}
    msg = cs.codec_start_message(utterance_index=1, sentence_index=0, sentence_text="hi", spec=spec, model="m")
    assert msg["type"] == "codec.start" and msg["codebooks"] == 16 and msg["header_format"] == cs.HEADER_FMT
    assert msg["decoder_id"] == "x" and msg["version"] == cs.CODEC_STREAM_VERSION


# --------------------------------------------------------------------- handler
def _codec_app(mocker, *, supports: bool = True, cancel_after: int | None = None):
    from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
    from vllm_omni.entrypoints.openai.serving_speech_stream import OmniStreamingSpeechHandler

    speech_service = mocker.MagicMock(spec=OmniOpenAIServingSpeech)
    speech_service.forced_aligner_enabled = False
    speech_service._prepare_speech_generation = mocker.AsyncMock(return_value=("req-1", object(), {}))
    adapter = mocker.MagicMock()
    adapter.supports_codec_stream = supports
    adapter.codec_stream_spec = {
        "codebooks": 16,
        "codebook_size": 2048,
        "frame_rate_hz": 12.5,
        "sample_rate": 24000,
        "decoder_id": "d",
    }
    speech_service._get_tts_adapter = mocker.MagicMock(return_value=adapter)
    speech_service.engine_client = mocker.MagicMock()
    speech_service.engine_client.abort = mocker.AsyncMock()

    async def mock_codec_chunks(_generator, _request_id, *, tts_params=None, min_frames=1):
        import asyncio

        for i in range(3):
            await asyncio.sleep(0.01)
            yield np.full((1 + i, 16), i, dtype=np.int64), i == 2

    speech_service._generate_codec_chunks = mock_codec_chunks
    handler = OmniStreamingSpeechHandler(speech_service=speech_service, idle_timeout=5.0, config_timeout=5.0)
    app = FastAPI()

    @app.websocket("/v1/audio/speech/stream")
    async def ws_endpoint(websocket: WebSocket):
        # NOTE: no ``from __future__ import annotations`` in this module: FastAPI
        # must resolve the WebSocket annotation, or it treats it as a query param.
        await handler.handle_session(websocket)

    return app, speech_service


def test_codec_stream_handler_frames_and_done(mocker):
    from starlette.testclient import TestClient

    app, svc = _codec_app(mocker)
    with TestClient(app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(
            {"type": "session.config", "output_mode": "codec_tokens", "stream_audio": True, "response_format": "pcm"}
        )
        ws.send_json({"type": "input.text", "text": "hello"})
        ws.send_json({"type": "input.done"})
        start = ws.receive_json()
        assert start["type"] == "codec.start" and start["codebooks"] == 16
        frames = []
        while True:
            msg = ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                frames.append(cs.unpack_codec_frame(msg["bytes"], 16))
                continue
            done = __import__("json").loads(msg["text"])
            break
        assert [f[1] for f in frames] == [1, 2, 3]
        assert [f[0] for f in frames] == [0, 1, 2]
        assert frames[-1][2] & cs.FLAG_EOS
        assert done["type"] == "codec.done" and done["total_frames"] == 6 and done["cancelled"] is False
        assert ws.receive_json()["type"] == "session.done"
    svc.engine_client.abort.assert_not_awaited()


def test_codec_stream_rejected_when_adapter_lacks_support(mocker):
    from starlette.testclient import TestClient

    app, _ = _codec_app(mocker, supports=False)
    with TestClient(app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(
            {"type": "session.config", "output_mode": "codec_tokens", "stream_audio": True, "response_format": "pcm"}
        )
        ws.send_json({"type": "input.text", "text": "hello"})
        ws.send_json({"type": "input.done"})
        err = ws.receive_json()
        assert err["type"] == "error" and "not supported" in err["message"]


def test_input_cancel_mid_stream_aborts_request(mocker):
    from starlette.testclient import TestClient

    app, svc = _codec_app(mocker)
    with TestClient(app) as client, client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(
            {"type": "session.config", "output_mode": "codec_tokens", "stream_audio": True, "response_format": "pcm"}
        )
        ws.send_json({"type": "input.text", "text": "hello"})
        ws.send_json({"type": "input.done"})
        assert ws.receive_json()["type"] == "codec.start"
        ws.send_json({"type": "input.cancel"})
        types = []
        while True:
            msg = ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                continue
            payload = __import__("json").loads(msg["text"])
            types.append(payload["type"])
            if payload["type"] == "session.done":
                break
            if payload["type"] == "codec.done":
                assert payload["cancelled"] is True
        assert "codec.done" in types and types[-1] == "session.done"
    svc.engine_client.abort.assert_awaited()


@pytest.mark.core_model
@pytest.mark.cpu
def test_talker_only_deployment_detection():
    from types import SimpleNamespace

    from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

    probe = OmniOpenAIServingSpeech.__new__(OmniOpenAIServingSpeech)
    probe._tts_stage = SimpleNamespace(final_output=True, final_output_type="latent")
    assert probe._is_talker_only_deployment() is True
    probe._tts_stage = SimpleNamespace(final_output=False, final_output_type="latent")
    assert probe._is_talker_only_deployment() is False
    probe._tts_stage = SimpleNamespace(final_output=True, final_output_type="audio")
    assert probe._is_talker_only_deployment() is False
    probe._tts_stage = None
    assert probe._is_talker_only_deployment() is False


@pytest.mark.core_model
@pytest.mark.cpu
def test_codec_snapshot_list_is_normalized():
    import numpy as np
    import torch

    from vllm_omni.entrypoints.openai.serving_speech import _coerce_codec_snapshots

    assert _coerce_codec_snapshots(None) is None
    t = torch.arange(32).reshape(2, 16)
    assert _coerce_codec_snapshots(t) is t
    # DELTA-mode list of cumulative snapshots -> last one
    snaps = [torch.arange(16).reshape(1, 16), torch.arange(32).reshape(2, 16), torch.arange(48).reshape(3, 16)]
    out = _coerce_codec_snapshots(snaps)
    assert out.shape == (3, 16) and int(out[2, 15]) == 47
    # list of single-row deltas -> concatenation
    deltas = [torch.full((1, 16), i) for i in range(4)]
    out = _coerce_codec_snapshots(deltas)
    assert out.shape == (4, 16) and list(out[:, 0]) == [0, 1, 2, 3]
    assert _coerce_codec_snapshots([None]) is None
    assert isinstance(_coerce_codec_snapshots([np.zeros(16)]), np.ndarray)


@pytest.mark.core_model
@pytest.mark.cpu
def test_accumulate_codec_rows_handles_deltas_and_snapshots():
    import numpy as np

    from vllm_omni.entrypoints.openai.codec_stream import accumulate_codec_rows

    r = lambda i: np.full((1, 16), i, dtype=np.int64)  # noqa: E731
    acc = accumulate_codec_rows(None, r(0))
    acc = accumulate_codec_rows(acc, r(1))
    acc = accumulate_codec_rows(acc, r(2))
    assert acc.shape == (3, 16) and list(acc[:, 0]) == [0, 1, 2]
    # duplicate of the last delta is not appended twice
    assert accumulate_codec_rows(acc, acc).shape == (3, 16)
    # final cumulative snapshot with a leading placeholder row replaces the matrix
    final = np.concatenate([np.full((1, 16), -1), acc, r(3)], axis=0)
    out = accumulate_codec_rows(acc, final)
    assert out.shape == (5, 16) and out[0, 0] == -1 and out[-1, 0] == 3
    # 1-D row payload
    assert accumulate_codec_rows(None, np.arange(16)).shape == (1, 16)


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.asyncio
async def test_generate_codec_chunks_yields_per_step():
    from types import SimpleNamespace

    import numpy as np

    from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

    svc = OmniOpenAIServingSpeech.__new__(OmniOpenAIServingSpeech)
    svc._get_tts_adapter = lambda: SimpleNamespace(
        codec_stream_spec={"codebook_size": 2048}, validates_generation=False
    )

    def out(token_ids, codes, finished):
        return SimpleNamespace(
            outputs=[SimpleNamespace(token_ids=token_ids, multimodal_output={"codes": {"audio": codes}})],
            finished=finished,
        )

    rows = [np.full((1, 16), i + 10, dtype=np.int64) for i in range(4)]

    async def gen():
        # step k: token k and its codes row arrive together (DELTA mode: newest row only)
        for k in range(4):
            yield out([100 + k], rows[k], False)
        # final: EOS token plus the consolidated cumulative matrix (EOS row is all zeros)
        final = np.concatenate(rows + [np.zeros((1, 16), dtype=np.int64)], axis=0)
        yield out([2150], final, True)

    got = [(None if c is None else c.shape[0], eos) async for c, eos in svc._generate_codec_chunks(gen(), "r1")]
    assert got[:4] == [(1, False), (1, False), (1, False), (1, False)]
    assert got[-1][1] is True and sum(n for n, _ in got if n) == 4
