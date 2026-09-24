#!/usr/bin/env python3
"""Record file-size and bundled CRC32 checks for a native Qwen3.8 FP8 copy."""

from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path


def crc32(path: Path) -> str:
    value = 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            value = zlib.crc32(chunk, value)
    return f"{value:08x}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--copy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    original = {path.name: path.stat().st_size for path in args.source.iterdir() if path.is_file()}
    copied = {path.name: path.stat().st_size for path in args.copy.iterdir() if path.is_file()}
    if original != copied:
        raise ValueError("source and copy differ in file names or sizes")
    manifest = {}
    for line in (args.source / "crc32.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        manifest[name] = expected.lower()
    rows = []
    for name, expected in manifest.items():
        actual = crc32(args.copy / name)
        source_actual = crc32(args.source / name) if actual != expected else None
        rows.append({"file": name, "bytes": copied[name], "manifest_crc32": expected,
                     "copy_crc32": actual, "source_crc32_if_manifest_differs": source_actual})
    weight_rows = [row for row in rows if row["file"].endswith(".safetensors")]
    mismatches = [row for row in rows if row["manifest_crc32"] != row["copy_crc32"]]
    if len(weight_rows) != 66 or any(row["manifest_crc32"] != row["copy_crc32"] for row in weight_rows):
        raise ValueError("at least one safetensors file failed its bundled CRC32")
    if any(row["source_crc32_if_manifest_differs"] != row["copy_crc32"] for row in mismatches):
        raise ValueError("copy differs from source on a stale-manifest file")
    result = {
        "status": "checked",
        "source": str(args.source),
        "copy": str(args.copy),
        "file_count": len(copied),
        "total_bytes": sum(copied.values()),
        "manifest_entries": len(rows),
        "weight_files_matching_manifest": len(weight_rows),
        "stale_manifest_nonweight_files": [row["file"] for row in mismatches],
        "elapsed_s": time.monotonic() - started,
        "files": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "status", "file_count", "manifest_entries", "weight_files_matching_manifest",
        "stale_manifest_nonweight_files", "elapsed_s"
    )}))


if __name__ == "__main__":
    main()
