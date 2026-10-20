# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""E2E: codec-token streaming from a talker-only Qwen3-TTS deployment.

Deploy Setting: qwen3_tts_talker_only.yaml (single LLM_AR stage)
Input Modal: text (WebSocket /v1/audio/speech/stream, output_mode=codec_tokens)
Output Modal: codec frames decoded on the client with Qwen3TTSTokenizerV2Decoder
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import json

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.entrypoints.openai.codec_stream import FLAG_EOS, unpack_codec_frame

MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
_STAGE_CONFIG = get_deploy_config_path("qwen3_tts_talker_only.yaml")

server_params = [
    pytest.param(
        OmniServerParams(model=MODEL, stage_config_path=_STAGE_CONFIG, server_args=["--trust-remote-code"]),
        id="talker_only",
    )
]


def _stream_codec(port: int, text: str) -> tuple[dict, list[tuple[int, int, int, np.ndarray]], dict]:
    from websockets.sync.client import connect

    frames = []
    with connect(f"ws://localhost:{port}/v1/audio/speech/stream", max_size=64 * 1024 * 1024) as ws:
        ws.send(
            json.dumps(
                {
                    "type": "session.config",
                    "model": MODEL,
                    "output_mode": "codec_tokens",
                    "stream_audio": True,
                    "response_format": "pcm",
                    "voice": "vivian",
                    "task_type": "CustomVoice",
                }
            )
        )
        ws.send(json.dumps({"type": "input.text", "text": text}))
        ws.send(json.dumps({"type": "input.done"}))
        start = json.loads(ws.recv())
        assert start["type"] == "codec.start", start
        done = None
        while True:
            raw = ws.recv()
            if isinstance(raw, (bytes, bytearray)):
                frames.append(unpack_codec_frame(bytes(raw), start["codebooks"]))
                continue
            msg = json.loads(raw)
            if msg["type"] == "codec.done":
                done = msg
            elif msg["type"] == "session.done":
                break
            elif msg["type"] == "error":
                raise AssertionError(msg)
        ws.send(json.dumps({"type": "session.close"}))
    return start, frames, done


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", server_params, indirect=True)
def test_codec_token_stream_and_client_decode(omni_server) -> None:
    start, frames, done = _stream_codec(omni_server.port, "Codec token streaming end to end test.")
    assert start["codebooks"] == 16 and start["frame_rate_hz"] == 12.5 and start["sample_rate"] == 24000
    assert frames, "no codec frames received"
    seqs = [f[0] for f in frames]
    assert seqs == list(range(len(seqs)))
    assert frames[-1][2] & FLAG_EOS
    total = sum(f[1] for f in frames)
    assert total >= 10, f"too few frames: {total}"
    assert done is not None and done["total_frames"] == total and done["error"] is False
    codes = np.concatenate([f[3].astype(np.int64) for f in frames if f[1]], axis=0)
    assert codes.min() >= 0 and codes.max() < 2048

    import sys

    sys.path.insert(
        0,
        str(
            os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "examples", "online_serving", "text_to_speech", "qwen3_tts"
            )
        ),
    )
    from edge_decoder_client import LocalDecoder

    dec = LocalDecoder(MODEL, device="cpu")
    pcm = np.concatenate([dec.decode(codes[i : i + 25]) for i in range(0, codes.shape[0], 25)])
    expected_s = total / 12.5
    assert abs(pcm.size / dec.sample_rate - expected_s) < 0.1 * expected_s + 0.2
    assert float(np.abs(pcm).max()) > 1e-3
