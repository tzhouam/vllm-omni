#!/usr/bin/env python3
"""Audit native Windows MiniCPM-o photo/question outputs and speech content."""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import transformers
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from benchmarks.edge_harness.experiments.probe_tts_asr import _word_error_rate


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--asr-model-dir", type=Path, required=True)
    parser.add_argument("--asr-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence_dir
    inputs = json.loads((root / "inputs/input_manifest.json").read_text(encoding="utf-8"))
    input_asr = json.loads((root / "inputs/question_asr.json").read_text(encoding="utf-8"))
    if (sha256(root / "inputs/astronaut_448.jpg")
            != inputs["converted_448_jpeg"]["astronaut"]
            or sha256(root / "inputs/chelsea_cat_448.jpg")
            != inputs["converted_448_jpeg"]["chelsea_cat"]
            or sha256(root / "inputs/question_16k.wav") != inputs["audio_16k_sha256"]
            or input_asr["asr_revision"] != args.asr_revision
            or input_asr["asr_weights_sha256"]
            != sha256(args.asr_model_dir / "model.safetensors")
            or input_asr["wav_sha256"] != inputs["audio_16k_sha256"]
            or input_asr["word_error_rate"] != 0
            or input_asr["transcript"].casefold() != inputs["audio_prompt"].casefold()):
        raise ValueError("spoken image question did not pass the pinned ASR check")
    processor = AutoProcessor.from_pretrained(str(args.asr_model_dir), local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(args.asr_model_dir), local_files_only=True).to("cpu")
    if {p.device.type for p in model.parameters()} != {"cpu"}:
        raise ValueError("ASR model did not stay on CPU")
    rows = []
    for name in ("cpu_astronaut", "cpu_cat", "radeon_astronaut", "radeon_cat"):
        placement, image = name.split("_", 1)
        image_key = "astronaut" if image == "astronaut" else "chelsea_cat"
        expected_placement = "cpu" if placement == "cpu" else "radeon-hybrid"
        report_path = root / f"{name}_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        result = report["complete_request"]
        worker_path = root / f"{name}_worker.log"
        worker = worker_path.read_text(encoding="utf-8", errors="replace")
        wav_path = root / f"{name}.wav"
        with wave.open(str(wav_path), "rb") as wav:
            if (wav.getnchannels() != 1 or wav.getsampwidth() != 2
                    or wav.getframerate() != 24000):
                raise ValueError(f"{name}: output WAV format changed")
            frames = wav.getnframes()
            pcm = wav.readframes(frames)
        if (report["status"] != "passed" or report["placement"] != expected_placement
                or report["model_revision"] != "db25077c33951fe163b42986fba0132e279872a2"
                or report["input_sha256"]["audio_wav"] != inputs["audio_16k_sha256"]
                or report["input_sha256"]["image_jpeg"] != inputs["converted_448_jpeg"][image_key]
                or report["output_limits"] != {"max_output_audio_s": 60,
                                                "max_wav_bytes": 4 << 20}
                or report["memory_budget"]["demands"]["host_ram"] != 21 << 30
                or report["ledger_after_shutdown"]["reserved"]["host_ram"] != 0
                or report["ledger_after_shutdown"]["quarantined"]
                or len(report["measured"]) != 1 or report["warmups"]
                or result["stage_event"]["terminal"] is not True
                or frames != result["audio_metadata"]["pcm_frames"]
                or hashlib.sha256(pcm).hexdigest() != result["audio_metadata"]["pcm_sha256"]
                or len(pcm) > (4 << 20)
                or not result["text"].strip()):
            raise ValueError(f"{name}: model, media, completion or admission gate failed")
        if image == "cat":
            if "cat" not in result["text"].casefold():
                raise ValueError(f"{name}: cat not described")
        elif not any(term in result["text"].casefold()
                     for term in ("astronaut", "shuttle", "space")):
            raise ValueError(f"{name}: astronaut scene not described")
        if placement == "radeon":
            if ("offloaded 37/37 layers to GPU" not in worker
                    or "vision using Vulkan0 backend" not in worker
                    or "Token2Wav device: cpu" not in worker):
                raise ValueError(f"{name}: Radeon language/vision or CPU vocoder not verified")
        elif "vision using CPU backend" not in worker or "Vulkan0" in worker:
            raise ValueError(f"{name}: CPU-only placement not verified")

        audio, sample_rate = sf.read(wav_path, dtype="float32")
        if audio.ndim != 1 or sample_rate != 24000 or not np.isfinite(audio).all():
            raise ValueError(f"{name}: output audio is invalid")
        audio_16k = librosa.resample(audio, orig_sr=24000, target_sr=16000)
        transcripts = []
        # Whisper tiny.en has a 30 s window. Keep every call at most 25 s;
        # concatenate chunk transcripts for the 39 s astronaut response.
        chunk_size = 25 * 16000
        for start in range(0, len(audio_16k), chunk_size):
            chunk = audio_16k[start:start + chunk_size]
            features = processor(chunk, sampling_rate=16000,
                                 return_tensors="pt", return_attention_mask=True)
            with torch.no_grad():
                tokens = model.generate(features.input_features,
                                        attention_mask=features.attention_mask)
            transcripts.append(processor.batch_decode(tokens, skip_special_tokens=True)[0].strip())
        transcript = " ".join(transcripts)
        rows.append({
            "name": name, "placement": expected_placement,
            "image_sha256": inputs["converted_448_jpeg"][image_key],
            "report_sha256": sha256(report_path),
            "worker_log_sha256": sha256(worker_path),
            "driver_log_sha256": sha256(root / f"{name}_driver.log"),
            "wav_sha256": sha256(wav_path),
            "pcm_sha256": result["audio_metadata"]["pcm_sha256"],
            "frames": frames, "duration_s": frames / 24000,
            "complete_request_wall_s": result["wall_s"],
            "text": result["text"], "asr_chunks": transcripts,
            "asr_transcript": transcript,
            "asr_wer_vs_own_text": _word_error_rate(result["text"], transcript),
        })
    output = {
        "status": "four_scoped_natural_image_spoken_question_requests_passed",
        "input_manifest_sha256": sha256(root / "inputs/input_manifest.json"),
        "input_asr_sha256": sha256(root / "inputs/question_asr.json"),
        "asr_model": "openai/whisper-tiny.en",
        "asr_revision": args.asr_revision,
        "asr_weights_sha256": sha256(args.asr_model_dir / "model.safetensors"),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "requests": rows,
        "limits": "Two photos, one synthetic spoken question, one complete request per placement/photo; ASR is an intelligibility proxy, not listening quality or a latency profile.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"],
                      "asr_wer": [row["asr_wer_vs_own_text"] for row in rows]}))


if __name__ == "__main__":
    main()
