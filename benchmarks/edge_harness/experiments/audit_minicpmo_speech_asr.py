#!/usr/bin/env python3
"""Transcribe archived MiniCPM-o speech against its own thinker text."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import transformers
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from benchmarks.edge_harness.experiments.audit_minicpmo_image_suite import archive_path
from benchmarks.edge_harness.experiments.probe_tts_asr import _word_error_rate


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    processor = AutoProcessor.from_pretrained(str(args.model_dir), local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(args.model_dir), local_files_only=True).to("cpu")
    rows = []
    for request in report["requests"]:
        archive = archive_path(args.report, request["audio_archive_path"])
        if sha256(archive) != request["audio_archive_sha256"]:
            raise ValueError(f"waveform hash differs for request {request['index']}")
        waveform = np.load(archive, allow_pickle=False)
        wav_path = archive.with_suffix(".wav")
        audio, sample_rate = sf.read(wav_path, dtype="float32")
        if (sample_rate != 24000 or audio.shape != waveform.shape
                or not np.array_equal(audio, waveform)):
            raise ValueError(f"WAV differs from archived float32 samples for request {request['index']}")
        audio_16k = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        features = processor(audio_16k, sampling_rate=16000, return_tensors="pt",
                             return_attention_mask=True)
        with torch.no_grad():
            token_ids = model.generate(features.input_features,
                                       attention_mask=features.attention_mask)
        transcript = processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()
        rows.append({
            "index": request["index"],
            "image_sha256": request["image_sha256"],
            "audio_input_sha256": request.get("audio_sha256"),
            "generated_wav_sha256": sha256(wav_path),
            "generated_duration_s": audio.size / sample_rate,
            "reference_text": request["text"],
            "transcript": transcript,
            "word_error_rate": _word_error_rate(request["text"], transcript),
        })
    output = {
        "scope": "Whisper tiny.en transcript agreement with generated thinker text; no listening-quality claim",
        "request_report_sha256": sha256(args.report),
        "asr_model": "openai/whisper-tiny.en",
        "asr_revision": args.model_revision,
        "asr_weights_sha256": sha256(args.model_dir / "model.safetensors"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "requests": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"requests": len(rows),
                      "word_error_rates": [row["word_error_rate"] for row in rows]}))


if __name__ == "__main__":
    main()
