#!/usr/bin/env python3
"""Pin two reusable photos and a short human-narration MiniCPM-o fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
import skimage
import soundfile as sf
from PIL import Image
from skimage import data


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = {
        "astronaut.png": data.astronaut(),
        "chelsea_cat.png": data.chelsea(),
    }
    for name, pixels in images.items():
        Image.fromarray(pixels).save(args.output_dir / name)

    source_audio = Path(librosa.ex("libri1"))
    samples, sample_rate = librosa.load(source_audio, sr=16000, mono=True)
    if sample_rate != 16000 or samples.size < 6 * sample_rate or not np.isfinite(samples).all():
        raise RuntimeError("LibriSpeech example lacks six seconds of finite 16 kHz narration")
    audio_path = args.output_dir / "libri1_first_6s.wav"
    sf.write(audio_path, samples[:6 * sample_rate], sample_rate, subtype="PCM_16")

    manifest = {
        "image_source": "https://scikit-image.org/docs/stable/api/skimage.data",
        "images": {
            name: {"sha256": sha256(args.output_dir / name), "shape": list(pixels.shape)}
            for name, pixels in images.items()
        },
        "audio_source": "https://librosa.org/doc/dev/recordings.html",
        "audio_key": "libri1",
        "audio_attribution": "Garth Comira reading Marion Bryce, The Ashiel Mystery; LibriSpeech SLR12, CC-BY-4.0",
        "audio_original_sha256": sha256(source_audio),
        "audio_excerpt": "first six seconds, resampled to 16 kHz mono PCM16",
        "audio_sha256": sha256(audio_path),
        "audio_samples": 6 * sample_rate,
        "librosa": librosa.__version__,
        "skimage": skimage.__version__,
        "soundfile": sf.__version__,
    }
    (args.output_dir / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
