"""Wire format for codec-token streaming (server → on-device decoder).

Binary frame layout (little-endian):

    header  ``<IHB``  seq:uint32  n_frames:uint16  flags:uint8      (7 bytes)
    body    int16[n_frames * codebooks]  row-major (frame-major)

Flags: ``FLAG_EOS`` (last frame of the utterance), ``FLAG_SEGMENT_END``
(sentence boundary inside a flush), ``FLAG_RESET`` (client must drop its
decoder state before consuming this frame). An EOS frame may carry
``n_frames == 0``.

Codebook ids are ``< 32768`` for every supported codec (Qwen3-TTS: 2048), so
int16 is lossless. Pure functions only; no engine imports.
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

HEADER_FMT = "<IHB"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
FLAG_EOS = 0x1
FLAG_SEGMENT_END = 0x2
FLAG_RESET = 0x4
MAX_FRAMES_PER_PACKET = 0xFFFF
CODEC_STREAM_VERSION = 1


def pack_codec_frame(
    seq: int,
    codes: Any,
    *,
    eos: bool = False,
    segment_end: bool = False,
    reset: bool = False,
    codebooks: int | None = None,
) -> bytes:
    """Pack ``codes`` (``[n_frames, codebooks]`` ints, or ``None``/empty) into one binary frame."""
    if codes is None:
        arr = np.zeros((0, codebooks or 0), dtype=np.int16)
    else:
        if hasattr(codes, "detach"):
            codes = codes.detach().cpu().numpy()
        arr = np.asarray(codes)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1) if arr.size else arr.reshape(0, codebooks or 0)
        if arr.ndim != 2:
            raise ValueError(f"codes must be 2-D [n_frames, codebooks], got shape {arr.shape}")
        if arr.size and (arr.min() < 0 or arr.max() > np.iinfo(np.int16).max):
            raise ValueError("codec ids must fit in int16")
        arr = arr.astype("<i2", copy=False)
    n_frames = int(arr.shape[0])
    if n_frames > MAX_FRAMES_PER_PACKET:
        raise ValueError(f"too many frames for one packet: {n_frames}")
    flags = (FLAG_EOS if eos else 0) | (FLAG_SEGMENT_END if segment_end else 0) | (FLAG_RESET if reset else 0)
    return struct.pack(HEADER_FMT, int(seq) & 0xFFFFFFFF, n_frames, flags) + arr.tobytes(order="C")


def unpack_codec_frame(buf: bytes, codebooks: int) -> tuple[int, int, int, np.ndarray]:
    """Inverse of :func:`pack_codec_frame`: ``(seq, n_frames, flags, int16[n_frames, codebooks])``."""
    if len(buf) < HEADER_SIZE:
        raise ValueError("codec frame shorter than its header")
    seq, n_frames, flags = struct.unpack(HEADER_FMT, buf[:HEADER_SIZE])
    body = buf[HEADER_SIZE:]
    expected = n_frames * codebooks * 2
    if len(body) != expected:
        raise ValueError(f"codec frame body has {len(body)} bytes, expected {expected} for {n_frames}x{codebooks}")
    arr = (
        np.frombuffer(body, dtype="<i2").reshape(n_frames, codebooks)
        if n_frames
        else np.zeros((0, codebooks), dtype=np.int16)
    )
    return seq, n_frames, flags, arr


def accumulate_codec_rows(acc: np.ndarray | None, step: Any) -> np.ndarray:
    """Fold a per-step ``codes.audio`` payload into the growing ``[rows, codebooks]`` matrix.

    DELTA-mode outputs carry only the newest row(s); the final consolidated
    output (and CUMULATIVE mode) carries the whole matrix, possibly with a
    leading placeholder row the deltas never had. A payload whose tail equals
    the rows already accumulated (and that is longer) replaces the matrix; an
    identical payload is a duplicate; anything else is appended.
    """
    if hasattr(step, "detach"):
        step = step.detach().cpu().numpy()
    arr = np.asarray(step)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.size == 0:
        return acc if acc is not None else np.zeros((0, arr.shape[-1] if arr.ndim >= 1 else 0), dtype=np.int64)
    arr = arr.astype(np.int64, copy=False)
    if acc is None or acc.size == 0:
        return arr
    n = acc.shape[0]
    if arr.shape[0] > n and arr.shape[1] == acc.shape[1]:
        # A longer payload that contains the accumulated rows as a contiguous
        # block is a cumulative snapshot (it may add a placeholder row in
        # front and/or newer rows behind): replace.
        for i in range(arr.shape[0] - n + 1):
            if np.array_equal(arr[i : i + n], acc):
                return arr
    if arr.shape == acc.shape and np.array_equal(arr, acc):
        return acc
    return np.concatenate([acc, arr], axis=0)


def align_codec_rows(codes_cum: Any, token_ids_cum: list[int], codebook_size: int) -> np.ndarray:
    """Keep the codec rows that correspond to real codebook-0 tokens.

    Mirrors the talker's full-payload filter: rows are tail-aligned to the
    output token ids, the prefill placeholder row (and any row whose token is
    not a codebook id, e.g. EOS) is dropped, and rows with out-of-range ids
    are dropped. Returns ``int64[n, codebooks]`` (possibly empty).
    """
    if codes_cum is None:
        return np.zeros((0, 0), dtype=np.int64)
    if hasattr(codes_cum, "detach"):
        codes_cum = codes_cum.detach().cpu().numpy()
    arr = np.asarray(codes_cum)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros((0, arr.shape[1] if arr.ndim == 2 else 0), dtype=np.int64)
    ids = list(token_ids_cum)
    while ids and ids[-1] == -1:
        ids.pop()
    n = min(arr.shape[0], len(ids))
    if n <= 0:
        return arr[:0].astype(np.int64)
    rows = arr[-n:]
    toks = np.asarray(ids[-n:])
    tok_ok = (toks >= 0) & (toks < codebook_size)
    row_ok = (rows.min(axis=1) >= 0) & (rows.max(axis=1) < codebook_size)
    return rows[tok_ok & row_ok].astype(np.int64)


def codec_start_message(
    *,
    utterance_index: int,
    sentence_index: int,
    sentence_text: str,
    spec: dict[str, Any],
    model: str | None,
) -> dict[str, Any]:
    return {
        "type": "codec.start",
        "version": CODEC_STREAM_VERSION,
        "utterance_index": utterance_index,
        "sentence_index": sentence_index,
        "sentence_text": sentence_text,
        "model": model,
        "codebooks": int(spec["codebooks"]),
        "codebook_size": int(spec["codebook_size"]),
        "frame_rate_hz": float(spec["frame_rate_hz"]),
        "sample_rate": int(spec["sample_rate"]),
        "decoder_id": spec.get("decoder_id"),
        "header_format": HEADER_FMT,
    }
