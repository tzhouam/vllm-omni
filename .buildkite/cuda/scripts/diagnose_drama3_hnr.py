# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Capture unchanged Drama3 assertions and predeclared seeds on one H100."""

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import subprocess
import threading
import wave
from pathlib import Path

_lock = threading.Lock()
_report = {}
_original = None
_nodeid = ""

CONTROLS = """

# Diagnostic-only controls, appended after the byte-unchanged original suite.
@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
@pytest.mark.parametrize(
    "seed,stream", [(0, True), (0, False), (42, True), (42, False), (0, True)],
    ids=["seed0-stream", "seed0-wav", "seed42-stream", "seed42-wav", "seed0-repeat"],
)
def test_predeclared_seed_controls_diagnostic(omni_server, openai_client, seed, stream):
    config = {
        "model": omni_server.model,
        "input": get_prompt("clone"),
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
        "stream": stream,
        "seed": seed,
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "pcm" if stream else "wav",
        "min_audio_bytes": _MIN_AUDIO_BYTES,
        "expected_sample_rate": SAMPLE_RATE,
        "transcript_pcm_sample_rate": SAMPLE_RATE,
    }
    if stream:
        config["stream_format"] = "audio"
    openai_client.send_audio_speech_request(config)
"""


def _save():
    output = Path(os.environ["DRAMA3_PROBE_OUTPUT"])
    temporary = output / "results.json.tmp"
    temporary.write_text(json.dumps(_report, indent=2))
    temporary.replace(output / "results.json")


def _capture(response, request_config, run_level=None):
    import numpy as np

    from tests.helpers.assertions import _compute_pcm_hnr_db

    raw = response.audio_bytes or b""
    output = Path(os.environ["DRAMA3_PROBE_OUTPUT"])
    suffix = request_config["response_format"]
    with _lock:
        index = len(_report["responses"])
        stem = f"response-{index:02d}"
        (output / f"{stem}.{suffix}").write_bytes(raw)
        record = {
            "index": index,
            "nodeid": _nodeid,
            "seed": request_config.get("seed"),
            "stream": bool(request_config.get("stream")),
            "input": request_config["input"],
            "format": suffix,
            "content_type": str(response.audio_format),
            "success": bool(response.success),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "request_hashes": {
                name: hashlib.sha256(str(request_config.get(name, "")).encode()).hexdigest()
                for name in ("input", "ref_audio", "ref_text")
            },
        }
        _report["responses"].append(record)
        _save()
    try:
        assert _original is not None
        _original(response, request_config, run_level=run_level)
        with _lock:
            record["original_assertions_passed"] = True
    except AssertionError as error:
        with _lock:
            record.update(original_assertions_passed=False, assertion=str(error))
        raise
    finally:
        if suffix == "pcm":
            pcm, rate = raw, 44100
            with wave.open(str(output / f"{stem}.wav"), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(rate)
                wav.writeframes(pcm)
        else:
            with wave.open(io.BytesIO(raw), "rb") as wav:
                assert wav.getnchannels() == 1 and wav.getsampwidth() == 2
                pcm, rate = wav.readframes(wav.getnframes()), wav.getframerate()
        metrics = dict(
            sample_rate=rate,
            pcm_sha256=hashlib.sha256(pcm).hexdigest(),
            hnr_db=_compute_pcm_hnr_db(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768, sr=rate),
            transcript=getattr(response, "audio_content", None),
        )
        with _lock:
            record.update(metrics)
            _save()


def pytest_sessionstart(session):
    global _original
    import vllm
    import vllm_omni

    from tests.helpers import client, runtime

    source = Path(os.environ["DRAMA3_PROBE_SOURCE"]).resolve()
    output = Path(os.environ["DRAMA3_PROBE_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    assert Path(vllm_omni.__file__).resolve().is_relative_to(source)
    assert Path(runtime._omni_subprocess_cwd()).resolve() == source
    assert not runtime.get_model_prefix(), "Original production tests use the canonical remote model identity"
    assert vllm.__version__.startswith(os.environ["DRAMA3_PROBE_VERSION"])
    _report.update(
        diagnostic_only=True,
        source=str(source),
        source_commit=os.environ["DRAMA3_PROBE_COMMIT"],
        production_commit="91ce93754f35282691ed2fcc4c491ddfa82ff49d",
        source_module=vllm_omni.__file__,
        server_subprocess_cwd=runtime._omni_subprocess_cwd(),
        model_identity="fishaudio/drama-3",
        model_revision="b2b5ee998034b2975720906c815e7fb0e9f007bd",
        model_config_sha256=hashlib.sha256(
            (Path(os.environ["DRAMA3_MODEL_SNAPSHOT"]) / "config.json").read_bytes()
        ).hexdigest(),
        original_test_sha256=os.environ["DRAMA3_ORIGINAL_TEST_HASH"],
        input_file_hashes={
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("vllm_omni/deploy/fish_omni.yaml", "tests/assets/qwen3_tts/clone_2.wav")
        },
        hardware=subprocess.check_output(["nvidia-smi", "-L"], text=True).strip(),
        packages={
            name: importlib.metadata.version(name)
            for name in ("vllm", "torch", "transformers", "tokenizers", "numpy", "flashinfer-python")
        },
        responses=[],
        tests=[],
        completed=False,
    )
    _original = client.assert_audio_speech_response
    client.assert_audio_speech_response = _capture
    _save()


def pytest_runtest_setup(item):
    global _nodeid
    _nodeid = item.nodeid


def pytest_runtest_logreport(report):
    if report.when == "call":
        with _lock:
            _report["tests"].append({"nodeid": report.nodeid, "outcome": report.outcome})
            _save()


def pytest_sessionfinish(session, exitstatus):
    _report.update(completed=True, pytest_exit=int(exitstatus))
    with _lock:
        _save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    if args.prepare:
        original = args.prepare.read_bytes()
        assert hashlib.sha256(original).hexdigest() == os.environ["DRAMA3_ORIGINAL_TEST_HASH"]
        output = Path(os.environ["DRAMA3_PROBE_OUTPUT"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "original-test.py").write_bytes(original)
        args.prepare.write_bytes(original + CONTROLS.encode())
    elif args.check:
        report = json.loads(args.check.read_text())
        assert report["completed"] and report["pytest_exit"] in (0, 1)
        assert len(report["tests"]) == 9 and len(report["responses"]) == 11
        assert all(
            response["success"] and response["bytes"] > 40000 and response["sample_rate"] == 44100
            for response in report["responses"]
        )
        print(json.dumps({"diagnostic_only": True, "pytest_exit": report["pytest_exit"], "tests": report["tests"]}))


if __name__ == "__main__":
    main()
