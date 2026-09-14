# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for the manual spatial decoder benchmark's failure boundary."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "benchmarks/lingbot_world" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_events_reject_wrong_temporal_chunk(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({"event_id": 0, "frames": [[], []]}))
    with pytest.raises(ValueError, match="exactly three"):
        load("common")._load_events(path)


@pytest.mark.parametrize("extra", [["--epochs", "1"], ["--tensor-parallel-size", "2"], ["--port", "65530"]])
def test_cli_rejects_invalid_protocol(extra):
    with pytest.raises(SystemExit):
        load("common").parse_args(
            ["--model", "m", "--image", "i", "--prompt", "p", "--events", "e", "--output-dir", "o", *extra]
        )


def test_rank_failure_reaps_all_children(monkeypatch):
    import signal

    import torch.multiprocessing as mp

    module = load("async_vae_spatial_worker")
    children = [Mock() for _ in range(4)]
    for child in children:
        child.is_alive.side_effect = [True, False]
    context = SimpleNamespace(processes=children, join=Mock(side_effect=RuntimeError("rank failed")))
    monkeypatch.setattr(mp, "spawn", Mock(return_value=context))
    monkeypatch.setattr(signal, "signal", Mock())
    monkeypatch.setenv("CAMPAIGN_PORT_BASE", "29500")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    conn = Mock()
    with pytest.raises(RuntimeError, match="rank failed"):
        module.worker(conn, "0,1,2,3", "unused-model")
    for child in children:
        child.terminate.assert_called_once()
        child.join.assert_called_once_with(timeout=10)
    assert "rank failed" in conn.send.call_args.args[0]["error"]
