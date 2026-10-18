# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Collect paired Drama3 quality controls without changing production assertions."""

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source))
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    import httpx
    import numpy as np
    import soundfile as sf
    import vllm

    import vllm_omni
    from tests.helpers.assertions import _compute_pcm_hnr_db, assert_audio_speech_response
    from tests.helpers.media import convert_audio_file_to_text, get_asset_path, release_audio_transcriber
    from tests.helpers.runtime import OmniServer

    assert vllm.__version__.startswith(args.expected_version), vllm.__version__
    assert Path(vllm_omni.__file__).resolve().is_relative_to(source), vllm_omni.__file__
    deploy = source / "vllm_omni/deploy/fish_omni.yaml"
    reference = get_asset_path("qwen3_tts/clone_2.wav", as_data_url=True)
    model = "fishaudio/drama-3"
    prompt = "The quarterly report is due on Thursday, so please finish it by noon."
    reference_text = (
        "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
    )
    report = {
        "diagnostic_only": True,
        "model_snapshot": str(args.model_path.resolve()),
        "model_revision": "b2b5ee998034b2975720906c815e7fb0e9f007bd",
        "model_config_sha256": hashlib.sha256((args.model_path / "config.json").read_bytes()).hexdigest(),
        "source": str(source),
        "source_module": vllm_omni.__file__,
        "runtime_module": vllm.__file__,
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        if (source / ".git").exists()
        else "1c0bde507dfbffd25b32f6092dbe4526c945b7dd (git archive)",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("vllm", "torch", "transformers", "tokenizers", "numpy", "flashinfer-python")
        },
        "hardware": subprocess.check_output(["nvidia-smi", "-L"], text=True).strip(),
        "input_hashes": {
            str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (deploy, get_asset_path("qwen3_tts/clone_2.wav"))
        },
        "cases": [],
        "completed": False,
    }

    def save():
        (output / "results.json").write_text(json.dumps(report, indent=2))

    save()
    # Predetermined controls: never search seeds or lower the quality floor.
    cases = [
        ("original-smoke", False, None, False),
        ("original-clone", True, None, True),
        ("repeat-clone", True, None, True),
        ("seed0-stream", True, 0, True),
        ("seed0-wav", False, 0, True),
        ("seed42-stream", True, 42, True),
        ("seed42-wav", False, 42, True),
        ("seed0-repeat", True, 0, True),
    ]
    server_args = [
        "--served-model-name",
        model,
        "--deploy-config",
        str(deploy),
        "--disable-log-stats",
        "--stage-init-timeout",
        "900",
        "--init-timeout",
        "900",
    ]
    with OmniServer(str(args.model_path.resolve()), server_args, startup_timeout=1200) as server:
        with httpx.Client(base_url=f"http://{server.host}:{server.port}", timeout=400) as client:
            for name, stream, seed, clone in cases:
                request = {
                    "model": model,
                    "input": prompt if clone else "The weather is nice today, perfect for a walk in the park.",
                    "stream": stream,
                    "response_format": "pcm" if stream else "wav",
                }
                if stream:
                    request["stream_format"] = "audio"
                if seed is not None:
                    request["seed"] = seed
                if clone:
                    request.update(ref_audio=reference, ref_text=reference_text)
                started = time.monotonic()
                record = {"name": name, "seed": seed, "stream": stream, "input": request["input"]}
                with client.stream("POST", "/v1/audio/speech", json=request) as response:
                    record.update(status=response.status_code, content_type=response.headers.get("content-type"))
                    chunks = list(response.iter_bytes())
                    raw = b"".join(chunks)
                suffix = request["response_format"]
                (output / f"{name}.{suffix}").write_bytes(raw)
                record.update(bytes=len(raw), chunks=len(chunks), sha256=hashlib.sha256(raw).hexdigest())
                if response.status_code == 200:
                    if stream:
                        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
                        sample_rate = 44100
                        sf.write(output / f"{name}.wav", audio, sample_rate, subtype="PCM_16")
                    else:
                        audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32")
                    record.update(sample_rate=sample_rate, hnr_db=_compute_pcm_hnr_db(audio, sr=sample_rate))
                else:
                    record["http_error"] = raw.decode(errors="replace")[:2000]
                record["elapsed"] = time.monotonic() - started
                report["cases"].append(record)
                save()
                print(json.dumps(record), flush=True)
    # Measure content even for a failing HNR control; retain the original assertion failure separately.
    try:
        for record in report["cases"]:
            if record["status"] != 200:
                continue
            name = record["name"]
            transcript = convert_audio_file_to_text(str(output / f"{name}.wav"))
            record["transcript"] = transcript
            suffix = "pcm" if record["stream"] else "wav"
            response = SimpleNamespace(
                success=True,
                audio_bytes=(output / f"{name}.{suffix}").read_bytes(),
                audio_format=record["content_type"],
                audio_content=transcript,
            )
            config = {
                "input": record["input"],
                "response_format": suffix,
                "min_audio_bytes": 40000,
                "expected_sample_rate": 44100,
                "transcript_pcm_sample_rate": 44100,
            }
            try:
                assert_audio_speech_response(response, config, run_level="advanced_model")
                record["original_assertions_passed"] = True
            except AssertionError as error:
                record.update(original_assertions_passed=False, assertion=str(error))
            save()
    finally:
        release_audio_transcriber()
    report["completed"] = True
    report["meaning"] = (
        "Completed diagnostic collection, not full CI or a quality fix; individual assertions remain recorded."
    )
    save()
    assert len(report["cases"]) == len(cases) and all(case["status"] == 200 for case in report["cases"])


if __name__ == "__main__":
    main()
