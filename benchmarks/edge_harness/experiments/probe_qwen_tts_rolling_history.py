#!/usr/bin/env python3
"""Audit two-frame generated-code windows against one full FP32 Code2Wav decode.

This is a CPU numerical study of explicit rolling input history, not an NPU
stream or a complete text-to-audio profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


REVISION = "85e237c12c027371202489a0ec509ded67b5e4b5"


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def relative_l2(reference, observed) -> float:
    import numpy as np

    a = reference.astype(np.float64)
    b = observed.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "source-report", "codes", "first97-fixture", "report", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")

    import numpy as np
    import torch

    from benchmarks.edge_harness.experiments.export_qwen_tts_vocoder_short_chunk import (
        QuantizedWindowDecoder,
    )
    from vllm_omni.edge.decoder_export import load_code2wav_decoder

    provenance = json.loads(args.source_report.read_text(encoding="utf-8-sig"))
    if (args.model.resolve().name != REVISION
            or provenance["model_revision"] != REVISION
            or sha256(args.codes) != provenance["files"]["codes"]["sha256"]
            or sha256(args.first97_fixture) != provenance["files"]["fixture"]["sha256"]
            or sha256(args.model / "speech_tokenizer" / "model.safetensors")
            != provenance["decoder_weight_sha256"]):
        raise ValueError("generated-code stream or pinned decoder changed")
    with np.load(args.codes, allow_pickle=False) as source:
        codes = np.ascontiguousarray(source["codes"])
    with np.load(args.first97_fixture, allow_pickle=False) as source:
        first97 = np.ascontiguousarray(source["quantized"])
    if (codes.shape != (117, 16) or first97.shape != (1, 512, 97)
            or first97.dtype != np.float32 or not np.issubdtype(codes.dtype, np.integer)):
        raise ValueError("generated-code fixture shape or dtype changed")

    torch.set_num_threads(args.threads)
    decoder = load_code2wav_decoder(str(args.model)).eval()
    model = QuantizedWindowDecoder(decoder, 2).eval()
    hop = model.hop
    if hop != 1920:
        raise ValueError("unexpected Code2Wav hop")
    with torch.inference_mode():
        code_tensor = torch.from_numpy(codes.T.copy()).unsqueeze(0).long()
        quantized = decoder.quantizer.decode(code_tensor).float().detach().numpy()
        full_wave = decoder(code_tensor).reshape(1, -1).detach().numpy()
    if (quantized.shape != (1, 512, 117) or full_wave.shape != (1, 117 * hop)
            or not np.isfinite(quantized).all() or not np.isfinite(full_wave).all()):
        raise ValueError("full decoder output contract failed")
    first97_error = relative_l2(first97, quantized[:, :, :97])
    if first97_error > 1e-6:
        raise ValueError(f"full quantizer disagrees with retained first97: {first97_error}")

    starts = (95, 97, 103, 109, 115)
    rows = []
    outputs = {}
    for start_frame in starts:
        reference = np.ascontiguousarray(full_wave[:, start_frame * hop:(start_frame + 2) * hop])
        outputs[f"reference_{start_frame}"] = reference
        histories = sorted({72, 88, 95, start_frame})
        for history in histories:
            window = np.ascontiguousarray(
                quantized[:, :, start_frame - history:start_frame + 2])
            started = time.perf_counter()
            with torch.inference_mode():
                observed = model(torch.from_numpy(window)).detach().numpy()
            elapsed = time.perf_counter() - started
            if observed.shape != reference.shape or not np.isfinite(observed).all():
                raise ValueError("window decoder output contract failed")
            outputs[f"window_{start_frame}_{history}"] = observed
            rows.append({
                "new_frame_start": start_frame,
                "history_frames": history,
                "window_start": start_frame - history,
                "input_shape": list(window.shape),
                "relative_l2_vs_full_decode": relative_l2(reference, observed),
                "max_abs_vs_full_decode": float(np.max(np.abs(reference - observed))),
                "one_eager_forward_s_not_profile": elapsed,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **outputs)
    report = {
        "scope": "one generated 117-frame stream, CPU FP32 rolling two-frame numerical study; no NPU or request profile",
        "checkpoint_revision": REVISION,
        "decoder_weight_sha256": provenance["decoder_weight_sha256"],
        "codes_sha256": sha256(args.codes),
        "first97_fixture_sha256": sha256(args.first97_fixture),
        "first97_quantized_relative_l2": first97_error,
        "frames": int(codes.shape[0]),
        "hop_samples": hop,
        "threads": args.threads,
        "torch_version": torch.__version__,
        "waveforms_sha256": sha256(args.output),
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"first97_quantized_relative_l2": first97_error,
                      "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
