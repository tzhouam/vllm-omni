#!/usr/bin/env python3
"""Extend the pinned two-step rolling-KV fixture to eleven real code chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from vllm_omni.edge.decoder_export import load_code2wav_decoder


REVISION = "85e237c12c027371202489a0ec509ded67b5e4b5"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "codes", "base-fixture", "output", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-codes-sha256", required=True)
    parser.add_argument("--expected-base-fixture-sha256", required=True)
    args = parser.parse_args()
    if (args.model.resolve().name != REVISION
            or sha256(args.codes) != args.expected_codes_sha256
            or sha256(args.base_fixture) != args.expected_base_fixture_sha256):
        raise ValueError("pinned model or source fixture changed")
    with np.load(args.codes, allow_pickle=False) as archive:
        codes = np.ascontiguousarray(archive["codes"])
    if codes.shape != (117, 16):
        raise ValueError("generated-code fixture shape changed")
    with np.load(args.base_fixture, allow_pickle=False) as archive:
        fixture = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    torch.set_num_threads(4)
    decoder = load_code2wav_decoder(str(args.model)).eval()
    source_codes = torch.from_numpy(codes.T.copy()).unsqueeze(0).long()

    def conv_frames(start: int):
        first = max(0, start - 2)
        quantized = decoder.quantizer.decode(source_codes[:, :, first:start + 2])
        return decoder.pre_conv(quantized)[:, :, -2:].transpose(1, 2).numpy()

    with torch.no_grad():
        for index in range(11):
            start = 95 + 2 * index
            actual = np.ascontiguousarray(conv_frames(start))
            if index < 2:
                old = fixture["conv" if index == 0 else "conv_next"]
                if not np.array_equal(old, actual):
                    raise ValueError(f"original convolution fixture changed at {start}")
            else:
                fixture[f"conv_step{index}"] = actual
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **fixture)
    report = {
        "scope": "real generated-code convolution inputs for eleven rolling state steps at frames 95..115",
        "checkpoint_revision": REVISION,
        "decoder_weight_sha256": sha256(args.model / "speech_tokenizer" / "model.safetensors"),
        "codes_sha256": sha256(args.codes),
        "base_fixture_sha256": sha256(args.base_fixture),
        "extended_fixture_sha256": sha256(args.output),
        "unchanged_first_two_convs": True,
        "step_count": 11,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
