# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""M0 serving end-to-end: the OpenAI-compatible server on the local text plan.

``test_e2e_local_text.py`` drives ``LocalTextEngine`` in-process. This module
drives the *server* path instead -- ``python -m vllm_omni.entrypoints.cli.main
serve <model> --omni`` as a subprocess, launched here with a small launcher of
its own (``tests.helpers.runtime.OmniServer`` would do, but importing it pulls
the media/diffusion test helpers and therefore ``diffusers``, which the AR/text
path and a Windows venv without the media deps do not have) -- and talks to it over HTTP
with plain ``requests``: health, model listing, non-streaming chat, streaming
chat, and a client that disconnects mid-stream after which the server must
still answer. Everything runs on one local device; no network access.

Runs on the same checkpoint rule as the engine test (``VLLM_OMNI_LOCAL_TEST_MODEL``
overrides). Marked ``local_model``: it needs the real model and a real device.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import psutil
import pytest
import requests

from tests.edge.local.test_e2e_local_text import MAX_MODEL_LEN, _find_model
from vllm_omni.edge.local.prompts import acceptance_prompts

pytestmark = [pytest.mark.local_model]

MAX_TOKENS = 64
REQUEST_TIMEOUT = 600  # s; the first request also pays any lazy warmup


def _chat_prompts() -> list[tuple[str, str]]:
    """The three chat-shaped acceptance prompts as (name, user text).

    The acceptance file stores fully rendered prompts; for the chat endpoint we
    send the user turn only and let the server apply the chat template.
    """
    rendered = dict(acceptance_prompts())
    out = []
    for name, text in rendered.items():
        if not name.startswith("chat_"):
            continue
        # The rendered form is "...<|User|>{text}<...end...><...start...><|Bot|>..."
        # Strip to the user turn; if the markers are not there, send the whole thing.
        user = text.split("<|User|>", 1)[-1]
        user = user.split("<｜end▁of▁sentence｜>", 1)[0]
        out.append((name, user.strip() or text))
    return out


@pytest.fixture(scope="module")
def model_dir() -> str:
    return _find_model()


def _free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _kill_tree(proc: subprocess.Popen, timeout: float = 30) -> None:
    """Terminate the server and every child (engine core, workers), on any OS."""
    try:
        parent = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for p in [*children, parent]:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs([*children, parent], timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


@pytest.fixture(scope="module")
def server(model_dir: str) -> Iterator[tuple[str, str]]:
    """Start the Omni server on the local text plan's settings; yield (base_url, model)."""
    env = os.environ.copy()
    env.update(
        {
            "VLLM_NO_USAGE_STATS": "1",
            "DO_NOT_TRACK": "1",
            "HF_HUB_OFFLINE": "1",
            # The same sampler setting LocalTextEngine applies to itself
            # (vllm_omni/edge/local/engine.py). Without it the serve path picks the
            # flashinfer top-k/top-p sampler, whose JIT needs nvcc at runtime; a
            # venv without a CUDA toolkit (the WSL reference venv) dies in
            # profile_run with "Could not find nvcc". The local plan never wants a
            # runtime compiler on the device it admits.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
    host, port = "127.0.0.1", _free_port()
    cmd = [
        sys.executable,
        "-m",
        "vllm_omni.entrypoints.cli.main",
        "serve",
        model_dir,
        "--host",
        host,
        "--port",
        str(port),
        "--omni",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-seqs",
        "2",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.55",
        "--stage-init-timeout",
        "600",
        "--init-timeout",
        "900",
        "--disable-log-stats",
    ]
    repo_root = Path(__file__).resolve().parents[3]
    proc = subprocess.Popen(cmd, env=env, cwd=str(repo_root))
    try:
        deadline = time.time() + 1200
        while True:
            if proc.poll() is not None:
                pytest.fail(f"server exited with code {proc.returncode} before opening its port")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                if sock.connect_ex((host, port)) == 0:
                    break
            if time.time() > deadline:
                pytest.fail("server did not open its port within 1200 s")
            time.sleep(2)
        yield f"http://{host}:{port}", model_dir
    finally:
        _kill_tree(proc)


def _wait_healthy(base_url: str, timeout: float = 300) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                return
            last = r.status_code
        except requests.RequestException as e:  # server socket open but app not ready
            last = repr(e)
        time.sleep(2)
    pytest.fail(f"/health never returned 200 within {timeout}s (last: {last})")


def _chat(base_url: str, model: str, user: str, *, stream: bool, max_tokens: int = MAX_TOKENS):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }
    return requests.post(
        f"{base_url}/v1/chat/completions", json=body, stream=stream, timeout=REQUEST_TIMEOUT
    )


def _sse_chunks(resp: requests.Response) -> Iterator[dict]:
    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            return
        yield json.loads(payload)


def test_health_and_model_listing(server):
    base_url, model = server
    _wait_healthy(base_url)
    r = requests.get(f"{base_url}/v1/models", timeout=30)
    assert r.status_code == 200, r.text
    ids = [m["id"] for m in r.json()["data"]]
    assert model in ids, ids


@pytest.mark.parametrize("name,user", _chat_prompts(), ids=[n for n, _ in _chat_prompts()])
def test_chat_completion_reaches_max_tokens_or_stops(server, name, user):
    base_url, model = server
    _wait_healthy(base_url)
    r = _chat(base_url, model, user, stream=False)
    assert r.status_code == 200, r.text
    out = r.json()
    choice = out["choices"][0]
    assert choice["finish_reason"] in ("length", "stop"), choice
    assert out["usage"]["completion_tokens"] > 0, out["usage"]
    assert out["usage"]["completion_tokens"] <= MAX_TOKENS, out["usage"]
    assert choice["message"]["content"] or choice["message"].get("reasoning_content"), choice


def test_streaming_delivers_incremental_chunks_and_a_finish(server):
    base_url, model = server
    _wait_healthy(base_url)
    r = _chat(base_url, model, "Count from one to twenty in words.", stream=True)
    assert r.status_code == 200, r.text
    n_chunks, finish, text = 0, None, []
    for chunk in _sse_chunks(r):
        n_chunks += 1
        choice = chunk["choices"][0]
        delta = choice.get("delta", {})
        text.append(delta.get("content") or delta.get("reasoning_content") or "")
        if choice.get("finish_reason"):
            finish = choice["finish_reason"]
    assert n_chunks >= 4, n_chunks  # incremental, not one blob
    assert finish in ("length", "stop"), finish
    assert "".join(text).strip()


def test_a_client_disconnect_mid_stream_does_not_take_the_server_down(server):
    base_url, model = server
    _wait_healthy(base_url)
    r = _chat(base_url, model, "Write a long story about a lighthouse.", stream=True, max_tokens=256)
    assert r.status_code == 200, r.text
    seen = 0
    for _ in _sse_chunks(r):
        seen += 1
        if seen >= 5:
            break
    r.close()  # drop the connection with the request still generating
    assert seen >= 5
    # The server must still be healthy and still answer.
    _wait_healthy(base_url, timeout=60)
    r2 = _chat(base_url, model, "Say hello.", stream=False, max_tokens=16)
    assert r2.status_code == 200, r2.text
    assert r2.json()["usage"]["completion_tokens"] > 0
